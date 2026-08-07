"""Golden and route adapters for chunk_bwd_dqkwg."""

from __future__ import annotations

from typing import Any

from tests_new.common.case_loader import chunk_count, resolve_optional_inputs
from tests_new.common.runtime import dtype_from_name, seed_case


def _tensor(torch: Any, shape: tuple[int, ...], dtype: Any, device: Any | None, scale: float):
    data = torch.randn(shape, dtype=torch.float32) * float(scale)
    data = data.to(dtype=dtype).contiguous()
    if device is not None:
        data = data.to(device)
    return data


def _gate(torch: Any, shape: tuple[int, ...], dtype: Any, device: Any | None, span: float):
    data = -torch.sort(torch.rand(shape, dtype=torch.float32) * float(span), dim=-1).values
    data = data.to(dtype=dtype).contiguous()
    if device is not None:
        data = data.to(device)
    return data


def build_inputs(torch: Any, case: dict[str, Any], device: Any | None = None) -> dict[str, Any]:
    seed_case(torch, case)
    shape = case["shape"]
    dtype = case["dtype"]
    generation = case.get("data_generation", {})
    input_scale = float(generation.get("input_scale", 0.05))
    state_scale = float(generation.get("state_scale", 0.02))
    gate_span = float(generation.get("gate_span", 2.0))

    B = int(shape["B"])
    Hk = int(shape["H_k"])
    Hv = int(shape["H_v"])
    T = int(shape["T"])
    K = int(shape["K"])
    V = int(shape["V"])
    Nc = chunk_count(case)
    data_dtype = dtype_from_name(torch, dtype["qkv"])
    gate_dtype = dtype_from_name(torch, dtype["g"])
    state_dtype = dtype_from_name(torch, dtype.get("state", dtype["qkv"]))
    optional = resolve_optional_inputs(case)

    return {
        "q": _tensor(torch, (B, Hk, T, K), data_dtype, device, input_scale),
        "k": _tensor(torch, (B, Hk, T, K), data_dtype, device, input_scale),
        "v": _tensor(torch, (B, Hv, T, V), data_dtype, device, input_scale),
        "g": _gate(torch, (B, Hv, T), gate_dtype, device, gate_span),
        "h": _tensor(torch, (B, Hv, Nc, K, V), state_dtype, device, state_scale),
        "dox": _tensor(torch, (B, Hv, T, V), data_dtype, device, input_scale),
        "dh": _tensor(torch, (B, Hv, Nc, K, V), state_dtype, device, state_scale),
        "dv": _tensor(torch, (B, Hv, T, V), data_dtype, device, input_scale),
        "cu_seqlens": optional.get("cu_seqlens"),
        "chunk_indices": optional.get("chunk_indices"),
    }


def expected_output_shapes(case: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    shape = case["shape"]
    B = int(shape["B"])
    Hk = int(shape["H_k"])
    Hv = int(shape["H_v"])
    T = int(shape["T"])
    K = int(shape["K"])
    return (
        (B, Hk, T, K),
        (B, Hk, T, K),
        (B, Hv, T, K),
        (B, Hv, T),
    )


def _as_cpu_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    cpu_inputs = {}
    for name, value in inputs.items():
        if hasattr(value, "detach"):
            cpu_inputs[name] = value.detach().cpu()
        else:
            cpu_inputs[name] = value
    return cpu_inputs


def run_reference(torch: Any, inputs: dict[str, Any], case: dict[str, Any]):
    cpu_inputs = _as_cpu_inputs(inputs)
    return _chunk_bwd_dqkwg_reference(torch, cpu_inputs, case)


def run_ascendc(torch: Any, inputs: dict[str, Any], case: dict[str, Any]):
    from fla_npu.ops import ascendc as ascendc_ops

    attrs = case["attrs"]
    return ascendc_ops.chunk_bwd_dqkwg(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["h"],
        inputs["dox"],
        inputs["dh"],
        inputs["dv"],
        int(attrs["chunk_size"]),
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
        w=None,
        g_gamma=None,
        scale=float(attrs["scale"]),
        use_exp2=bool(attrs.get("use_exp2", False)),
        transpose_state_layout=bool(attrs.get("transpose_state_layout", False)),
    )


def run_aclnn(torch: Any, inputs: dict[str, Any], case: dict[str, Any]):
    import fla_npu

    fla_npu.load_legacy_torch_ops()
    attrs = case["attrs"]
    return torch.ops.npu.npu_chunk_bwd_dqkwg(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["h"],
        inputs["dox"],
        inputs["dh"],
        inputs["dv"],
        int(attrs["chunk_size"]),
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
        w=None,
        g_gamma=None,
        scale=float(attrs["scale"]),
        use_exp2=bool(attrs.get("use_exp2", False)),
        transpose_state_layout=bool(attrs.get("transpose_state_layout", False)),
    )


def run_direct_launch(torch: Any, inputs: dict[str, Any], case: dict[str, Any]):
    try:
        import ascend_ops  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "direct_launch route requires the fast-kernel ascend_ops extension "
            "built from examples/fast_kernel_launch_example"
        ) from exc

    attrs = case["attrs"]
    return torch.ops.ascend_ops.chunk_bwd_dqkwg(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["h"],
        inputs["dox"],
        inputs["dh"],
        inputs["dv"],
        float(attrs["scale"]),
        int(attrs["chunk_size"]),
        w=None,
        g_gamma=None,
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
    )


def _chunk_bwd_dqkwg_reference(torch: Any, inputs: dict[str, Any], case: dict[str, Any]):
    q = inputs["q"].permute(0, 2, 1, 3).contiguous()
    k = inputs["k"].permute(0, 2, 1, 3).contiguous()
    v = inputs["v"].permute(0, 2, 1, 3).contiguous()
    g = inputs["g"].permute(0, 2, 1).contiguous()
    h = inputs["h"].permute(0, 2, 1, 3, 4).contiguous()
    dox = inputs["dox"].permute(0, 2, 1, 3).contiguous()
    dh = inputs["dh"].permute(0, 2, 1, 3, 4).contiguous()
    dv = inputs["dv"].permute(0, 2, 1, 3).contiguous()
    cu = inputs["cu_seqlens"]
    cu_tensor = None if cu is None else torch.tensor(cu, dtype=torch.long)
    dq, dk, dw, dg = _reference_bthd(
        torch,
        q,
        k,
        v,
        dox,
        h,
        dh,
        g,
        dv,
        float(case["attrs"]["scale"]),
        cu_tensor,
        int(case["attrs"]["chunk_size"]),
    )
    return (
        dq.permute(0, 2, 1, 3).contiguous(),
        dk.permute(0, 2, 1, 3).contiguous(),
        dw.permute(0, 2, 1, 3).contiguous(),
        dg.permute(0, 2, 1).contiguous(),
    )


def _reference_bthd(torch: Any, q, k, v, dox, h, dh, g, dv, scale: float, cu_seqlens, chunk_size: int):
    calc_type = torch.float32
    data_dtype = q.dtype
    gate_dtype = g.dtype
    B, T, Hk, K = q.shape
    Hv = v.shape[2]
    if Hk <= 0 or Hv <= 0 or Hv % Hk != 0:
        raise ValueError(f"invalid GVA head mapping: Hk={Hk}, Hv={Hv}")
    n_ratio = Hv // Hk

    dq_hv = torch.zeros((B, T, Hv, K), dtype=data_dtype)
    dk_hv = torch.zeros((B, T, Hv, K), dtype=data_dtype)
    dw = torch.zeros((B, T, Hv, K), dtype=data_dtype)
    dg = torch.zeros((B, T, Hv), dtype=gate_dtype)

    def process_sequence(b_idx: int, start: int, end: int, chunk_offset: int) -> None:
        num_chunks = (end - start + chunk_size - 1) // chunk_size
        for hv_idx in range(Hv):
            hk_idx = hv_idx // n_ratio
            for chunk_idx in range(num_chunks):
                chunk_start = start + chunk_idx * chunk_size
                chunk_end = min(start + (chunk_idx + 1) * chunk_size, end)
                if chunk_end <= chunk_start:
                    continue
                q_c = q[b_idx, chunk_start:chunk_end, hk_idx, :]
                k_c = k[b_idx, chunk_start:chunk_end, hk_idx, :]
                v_c = v[b_idx, chunk_start:chunk_end, hv_idx, :]
                do_c = dox[b_idx, chunk_start:chunk_end, hv_idx, :]
                dv_c = dv[b_idx, chunk_start:chunk_end, hv_idx, :]
                h_prev = h[b_idx, chunk_offset + chunk_idx, hv_idx, :, :]
                dh_curr = dh[b_idx, chunk_offset + chunk_idx, hv_idx, :, :]

                dq_state = do_c.to(calc_type) @ h_prev.transpose(-1, -2).to(calc_type)
                dq_state = dq_state.to(data_dtype).to(calc_type)
                dk_state = v_c.to(calc_type) @ dh_curr.transpose(-1, -2).to(calc_type)
                dk_state = dk_state.to(data_dtype).to(calc_type)
                dw_c = dv_c.to(calc_type) @ h_prev.transpose(-1, -2).to(calc_type)
                dw[b_idx, chunk_start:chunk_end, hv_idx, :] = -dw_c.to(data_dtype)

                g_c = g[b_idx, chunk_start:chunk_end, hv_idx]
                g_last = g[b_idx, min(chunk_start + chunk_size, end) - 1, hv_idx]
                dg_last = (h_prev.to(calc_type) * dh_curr.to(calc_type)).sum() * torch.exp(g_last.to(calc_type))

                dq_state = dq_state * torch.exp(g_c.to(calc_type))[:, None] * scale
                dk_state = dk_state * torch.exp((-g_c + g_last).to(calc_type))[:, None]

                dg_c = (dq_state * q_c.to(calc_type)).sum(dim=-1).to(data_dtype).to(calc_type)
                dg_c -= (k_c.to(calc_type) * dk_state).sum(dim=-1)
                dg_c = dg_c.to(gate_dtype).to(calc_type)
                dg_last += (dk_state * k_c.to(calc_type)).sum()

                ds = do_c.to(calc_type) @ v_c.transpose(-1, -2).to(calc_type)
                ds = ds.to(data_dtype).to(calc_type)
                length = chunk_end - chunk_start
                rows = torch.arange(length)[:, None]
                cols = torch.arange(length)[None, :]
                mask = rows >= cols
                decay = torch.exp(g_c.to(calc_type)[:, None] - g_c.to(calc_type)[None, :])
                ds = torch.where(mask, ds * decay, torch.zeros_like(ds)) * scale

                qk = q_c.to(calc_type) @ k_c.transpose(-1, -2).to(calc_type)
                qk = qk.to(data_dtype).to(calc_type)
                ds2 = ds * qk
                dg_c += ds2.sum(dim=1)
                dg_c = dg_c.to(gate_dtype).to(calc_type)
                dg_c -= ds2.sum(dim=0)
                dg_c = dg_c.to(gate_dtype)
                dg_c[length - 1] += dg_last.to(gate_dtype)
                dg[b_idx, chunk_start:chunk_end, hv_idx] = dg_c

                dq_intra = (ds.to(calc_type) @ k_c.to(calc_type)).to(data_dtype).to(calc_type)
                dk_intra = (ds.transpose(-1, -2).to(calc_type) @ q_c.to(calc_type)).to(data_dtype).to(calc_type)
                dq_hv[b_idx, chunk_start:chunk_end, hv_idx, :] = (dq_state + dq_intra).to(data_dtype)
                dk_hv[b_idx, chunk_start:chunk_end, hv_idx, :] = (dk_state + dk_intra).to(data_dtype)

    if cu_seqlens is None:
        for b_idx in range(B):
            process_sequence(b_idx, 0, T, b_idx * ((T + chunk_size - 1) // chunk_size))
    else:
        chunk_offset = 0
        for seq_idx in range(cu_seqlens.numel() - 1):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            process_sequence(0, start, end, chunk_offset)
            chunk_offset += (end - start + chunk_size - 1) // chunk_size

    dq = dq_hv.view(B, T, Hk, n_ratio, K).sum(dim=3).to(data_dtype)
    dk = dk_hv.view(B, T, Hk, n_ratio, K).sum(dim=3).to(data_dtype)
    return dq, dk, dw, dg
