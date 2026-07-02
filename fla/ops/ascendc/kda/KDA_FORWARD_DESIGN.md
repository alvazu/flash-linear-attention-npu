# KDA Forward AscendC Operator Design

## 1. Scope

This document describes the forward-only KDA implementation in this PR. The goal is to provide an AscendC operator stack that can execute KDA forward with `gk` while reusing existing GDN forward-h logic where it is semantically identical.

Current PR scope:

- Add `ChunkKdaFwd` AscendC L0/L2 operator for KDA forward.
- Add helper operators `KdaLayoutSwap12` and `KdaGateCumsum`.
- Extend `ChunkGatedDeltaRuleFwdH` with optional `gk`, because KDA reuses GDN state propagation.
- Add PyTorch custom API `npu_chunk_kda_fwd` and reference tests.
- Validate dense BSND/TND compatibility inputs and BNSD/NTD direct inputs for `K=128, V=128, chunk_size=64`.

Out of this PR:

- Backward KDA.
- `V=256` performance template. `V=256` should use a separate template because the UB/L1 budget and cube tiling differ from `V=128`.
- High-throughput non-aligned varlen partial chunks. The public API keeps `cu_seqlens` and `chunk_indices`, but non-chunk-aligned high `K/V` varlen requires a dedicated partial-chunk synchronization path before it should be advertised as optimized.

## 2. Reference Semantics

The implementation follows the KDA forward decomposition used by fla-org:

```text
Aqk[i, j] = tril(q_i * k_j * exp2(g_i - g_j)) * scale
Akk       = inv(I + tril(k_i * k_j * exp2(g_i - g_j) * beta_i, -1))
w         = Akk @ (k * beta * exp2(g))
u         = Akk @ (v * beta)
v_new     = u - w @ h_prev
h_next    = exp2(g_last) * h_prev + kg @ v_new
o         = qg @ h_prev * scale + Aqk @ v_new
```

Where:

- `gk` is the cumulative key gate in log2 space.
- `exp2(x)` is implemented as `exp(x * ln2)` on AscendC vector pipes.
- `final_state` follows fla-org/GDN semantics and is `float32`, even when `q/k/v/o` are `fp16` or `bf16`.

## 3. Public Interface

PyTorch API:

```python
torch.ops.npu.npu_chunk_kda_fwd(
    q,
    k,
    v,
    gk,
    beta,
    scale,
    chunk_size,
    *,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_indices=None,
    return_intermediate=False,
    safe_gate=False,
    transpose_state_layout=False,
)
```

Supported layout intent:

- BSND: `q/k: [B, T, H, K]`, `v: [B, T, HV, V]`, `gk: [B, T, HV, K]`, `beta: [B, T, HV]`.
- BNSD: `q/k: [B, H, T, K]`, `v: [B, HV, T, V]`, `gk: [B, HV, T, K]`, `beta: [B, HV, T]`.
- TND: `q/k: [T, H, K]`, `v: [T, HV, V]`, `gk: [T, HV, K]`, `beta: [T, HV]`.
- NTD: `q/k: [H, T, K]`, `v: [HV, T, V]`, `gk: [HV, T, K]`, `beta: [HV, T]`.

BNSD and NTD are the performance layouts for a pipeline where causal conv has already converted the data. BSND and TND remain compatibility layouts and are converted by `KdaLayoutSwap12` before entering the kernel.

Supported dtype intent:

- `q/k/v/o/Aqk/Akk/w/u/qg/kg/v_new/h`: same dtype as `q` or `v` depending on the tensor role, with `fp16`, `bf16`, and `fp32` operator registrations.
- `gk/beta`: accepted by the PyTorch layer as `fp32` or `bf16`; internally cast to `fp32` before `ChunkKdaFwd`.
- `initial_state/final_state`: `fp32`.

Reserved or rejected:

- `safe_gate=True` is rejected in the PyTorch wrapper for this PR.
- `transpose_state_layout=True` is rejected in the PyTorch wrapper for this PR.

## 4. L2 Composition

The L2 implementation in `aclnn_chunk_kda_fwd.cpp` performs:

1. Make all inputs contiguous.
2. Infer BSND/BNSD/TND/NTD from rank and shape consistency. Ambiguous cases prefer the layout whose head dimension is the shorter dimension.
3. For BNSD/NTD, reshape NTD to a `[1, HV, T, D]` view and enter the kernel directly. No layout swap is issued.
4. For BSND/TND, reshape TND to `[1, T, H, D]` and convert to BNSD through `KdaLayoutSwap12`.
5. Cast `gk/beta` to `fp32` if needed.
6. Dispatch either:
   - split forward path for large half/bfloat16 `chunk_size=64` cases, or
   - monolithic `ChunkKdaFwd` path for smaller or `fp32` cases.
7. For BNSD/NTD, copy split-path temporary outputs back to the same layout. For BSND/TND, convert BNSD intermediates back to the public output layout.

The split path has three `ChunkKdaFwd` stages plus GDN state propagation:

```text
stage 1: prepare qg/kg/w/u/Aqk/Akk inputs and solve intra-chunk terms
GDN fwd_h: update h/v_new/final_state using kg, w, u, gk
stage 2: compute output cube path and final o rows
stage 3: postprocess w/kg/u when post-WU cube is enabled
```

`ChunkGatedDeltaRuleFwdH` is extended to accept optional `gk`. This is a minimal dependency for KDA; the broader GDN operator family is not changed unless required by this forward-h reuse.

For BNSD/NTD split forward, raw `o/Aqk/Akk/w/u/qg/kg/v_new/h` are stored in executor-owned temporary tensors. The final L2 step uses same-layout `ViewCopy` to user outputs. This avoids passing user output tensors as producer-consumer intermediates between custom L0 and elewise L0 ops, which can trigger invalid tiling/workspace inference.

## 5. L0 Kernel Design

`ChunkKdaFwd` contains AIV and AIC work that are paired through cross-core flags.

Important paths:

- Gate product preparation:
  - AIV loads rows of `q/k/gk`.
  - AIV computes `qg = q * exp2(g)`, `w seed = k * exp2(g)`, `kg = k * exp2(-g)`.
  - Double-buffered queues are used for row input/output.
- `Aqk/Akk` raw score:
  - Target `K>=16` path uses Catlass cube GEMM.
  - Scalar fallback exists only as a correctness fallback for non-target shapes and must not be treated as the target performance path.
- `Akk` inverse:
  - Full `BT=64` uses blocked cube-assisted matrix-chain iteration.
  - The solve scratch is stored in `h` workspace slots before state propagation consumes `h`.
- `w/u` post computation:
  - Full `BT=64`, aligned `K/V` uses cube GEMM for `Akk @ w` and `Akk @ v_new`.
  - AIV prepares beta-scaled inputs and finalizes vector postprocess.
- Output:
  - AIC computes `qg @ h` and `Aqk @ v_new`.
  - AIV combines state contribution and local contribution into `o`.

For half/bfloat16 and `K>=16`, tiling launches full AICore block count so every AIC producer has a paired AIV consumer and vice versa. This is required for cross-core flag balance.

## 6. Memory And Layout

The L2 layer uses BNSD internally because it gives contiguous reads/writes on the head and token dimensions used by the kernel:

```text
BSND/TND public input  -> KdaLayoutSwap12 -> BNSD internal tensors
BNSD/NTD direct input  -> reshape/view     -> BNSD internal tensors
```

State layout:

```text
h:           [B, HV, NT, K, V] in the kernel
final_state: [seq_num, HV, K, V], fp32
```

UB principles:

- Use `DataCopy` or `DataCopyPad` for GM/UB transfer.
- Avoid `GetValue` and `SetValue` in target paths.
- Reuse a large UB arena for matrix/vector staging.
- Keep `V=128` and `V=256` templates separate because their UB residency and tile reuse plans are different.

## 7. Validation

Build validation:

- Custom package build for `chunk_kda_fwd`, `chunk_gated_delta_rule_fwd_h`, `kda_layout_swap12`, and `kda_gate_cumsum` passed.

Precision validation:

- Small BSND/TND/BNSD/NTD unit tests passed against `tests/reference/chunk_kda_reference.py`.
- Target sampled BNSD `B=1, H_K=1, H_V=2, T=16384, K=128, V=128, chunk_size=64` passed:
  - `o`: `max_abs=4.26e-4`, `mean_abs=3.78e-5`.
  - `final_state`: `max_abs=1.09e-3`, `mean_abs=1.45e-5`.
- Target sampled BNSD `B=1, H_K=32, H_V=64, T=4096, K=128, V=128, chunk_size=64` passed:
  - `o`: `max_abs=4.63e-4`, `mean_abs=4.01e-5`.
  - `final_state`: `max_abs=1.21e-4`, `mean_abs=9.68e-6`.
- Target sampled NTD `B=1, H_K=1, H_V=2, T=16384, K=128, V=128, chunk_size=64` passed.

Performance validation with `msopprof --aic-metrics=BasicInfo`:

- BNSD `B=1, H_K=1, H_V=2, T=16384, K=128, V=128, chunk_size=64`: relevant KDA+GDN average `3.05 ms`; `KdaLayoutSwap12` count `0`.
- BNSD `B=1, H_K=32, H_V=64, T=4096, K=128, V=128, chunk_size=64`: relevant KDA+GDN average `13.60 ms`; `KdaLayoutSwap12` count `0`.

Known validation boundary:

- Non-chunk-aligned `cu_seqlens` with high `K/V` can hit a kernel timeout in the current prototype. Do not claim optimized varlen high `K/V` support until a dedicated partial-chunk path is implemented and validated.
- `V=256` is intentionally not validated in this PR.

## 8. Extension Plan

Recommended next steps:

1. Add a dedicated partial-chunk varlen path where AIC/AIV flag counts are balanced for each participating subblock, and where cube paths never consume dirty rows outside the valid chunk.
2. Add a `V=256` template with an explicit UB/L1 residency plan instead of stretching the `V=128` template.
3. Add backward KDA as a separate PR, using the same `gk` and state dtype conventions.
4. Add sanitizer runs for race, memory, init, and sync checks once the varlen partial path is stable.
