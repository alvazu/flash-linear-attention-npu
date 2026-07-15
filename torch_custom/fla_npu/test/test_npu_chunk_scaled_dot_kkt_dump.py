#!/usr/bin/env python3
"""Dual-reference dump test for torch.ops.npu.npu_chunk_scaled_dot_kkt.

The dump format follows the GDN local dump convention:
  storage layout : [B, T, H, *]
  NPU layout     : [B, H, T, *]

This test compares the AscendC output against:
  1. the dumped GPU/framework output with the same dtype, and
  2. an independent CPU fp64 implementation with fp64 accumulation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import ct


DEFAULT_DUMP_ROOT = Path("/home/m00913889/codex04/gva_fix_1")
DEFAULT_ATOL = 5e-3
DEFAULT_RTOL = 5e-3
ZERO_TOL = 1e-6
CPU_BATCH_CHUNKS = int(os.environ.get("CHUNK_SCALED_DOT_KKT_CPU_BATCH_CHUNKS", "8"))
CPU_HEAD_BATCH = int(os.environ.get("CHUNK_SCALED_DOT_KKT_CPU_HEAD_BATCH", "4"))

_BTH_TO_BHT_NAMES = frozenset({"k", "g", "beta", "A"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify chunk_scaled_dot_kkt dump cases on NPU.")
    parser.add_argument(
        "--dump-root",
        default=os.environ.get("CHUNK_SCALED_DOT_KKT_DUMP_ROOT", str(DEFAULT_DUMP_ROOT)),
        help="Dump directory or a single *_chunk_scaled_dot_kkt_fwd.pt file.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("CHUNK_SCALED_DOT_KKT_DEVICE", "npu:0"),
        help="NPU device, for example npu:0.",
    )
    parser.add_argument("--atol", type=float, default=float(os.environ.get("CHUNK_SCALED_DOT_KKT_ATOL", DEFAULT_ATOL)))
    parser.add_argument("--rtol", type=float, default=float(os.environ.get("CHUNK_SCALED_DOT_KKT_RTOL", DEFAULT_RTOL)))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases.",
    )
    parser.add_argument(
        "--compact-ct-report",
        action="store_true",
        default=os.environ.get("CHUNK_SCALED_DOT_KKT_COMPACT_CT", "0") == "1",
        help="Keep ct.dual validation but replace ct's expensive full distribution table with a compact summary.",
    )
    parser.add_argument(
        "--skip-extra-allclose",
        action="store_true",
        default=os.environ.get("CHUNK_SCALED_DOT_KKT_SKIP_EXTRA_ALLCLOSE", "0") == "1",
        help="Skip extra npu-vs-gpu/cpu allclose and zero-region scans after full ct.dual validation.",
    )
    parser.add_argument(
        "--result-json",
        default=os.environ.get("CHUNK_SCALED_DOT_KKT_RESULT_JSON"),
        help="Optional path to write a structured JSON summary.",
    )
    parser.add_argument(
        "--execute-gva-without-full-gpu",
        action="store_true",
        default=os.environ.get("CHUNK_SCALED_DOT_KKT_EXECUTE_GVA_WITHOUT_FULL_GPU", "0") == "1",
        help=(
            "Deprecated compatibility flag. Native GVA dump validation uses the dumped "
            "A=[B,Hk,T,BT] as the GPU reference and selects the first Hk g/beta heads."
        ),
    )
    return parser.parse_args()


def install_compact_ct_report() -> None:
    """Avoid ct's large-array median/variance report while preserving ct.dual checks."""

    ct_cli = sys.modules.get("ct.cli")
    if ct_cli is None:
        return

    def compact_print_dual_report(level: str, cfg: Any, result: dict[str, Any], dtype_header: str | None = None) -> None:
        print(
            "[CT] compact dual report "
            f"level={level} dtype={dtype_header} success={result.get('success')} "
            f"checks={result.get('checks', {})} ratios={result.get('ratios', {})}",
            flush=True,
        )

    ct_cli.print_dual_report = compact_print_dual_report


def bth_to_bht(t: torch.Tensor) -> torch.Tensor:
    if t.ndim < 3:
        return t.detach().cpu()
    return t.transpose(1, 2).contiguous().detach().cpu()


def to_npu_tensor(name: str, value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    out = bth_to_bht(value) if name in _BTH_TO_BHT_NAMES else value.detach().cpu()
    if name == "beta" and out.is_floating_point():
        out = out.float()
    return out


def to_npu_mapping(mapping: dict[str, Any] | None) -> dict[str, Any]:
    if not mapping:
        return {}
    return {name: to_npu_tensor(name, value) for name, value in mapping.items() if value is not None}


def load_dump_for_npu(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dump = torch.load(path, map_location="cpu", weights_only=False)
    meta = dict(dump.get("meta") or {})
    if "inputs_npu" in dump:
        inputs = dump["inputs_npu"]
        outputs = dump.get("outputs_npu") or {}
    else:
        inputs = to_npu_mapping(dump.get("inputs") or {})
        outputs = to_npu_mapping(dump.get("outputs") or {})
    return inputs, meta, outputs


def meta_int_list(meta: dict[str, Any], *names: str) -> list[int] | None:
    for name in names:
        value = meta.get(name)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            return [int(x) for x in value.flatten().tolist()]
        return [int(x) for x in value]
    return None


def iter_chunk_ranges(
    total_t: int,
    chunk_size: int,
    cu_seqlens: list[int] | None = None,
    chunk_indices: list[int] | None = None,
):
    if cu_seqlens is None:
        for start in range(0, total_t, chunk_size):
            yield start, min(start + chunk_size, total_t)
        return

    if chunk_indices is None or len(chunk_indices) % 2 != 0:
        raise ValueError("chunk_indices must be a flat even-length [seq_id, chunk_id] list")
    for idx in range(0, len(chunk_indices), 2):
        seq_idx = int(chunk_indices[idx])
        local_chunk = int(chunk_indices[idx + 1])
        if seq_idx < 0 or seq_idx + 1 >= len(cu_seqlens):
            raise ValueError(f"chunk_indices seq_id={seq_idx} is outside cu_seqlens len={len(cu_seqlens)}")
        bos = int(cu_seqlens[seq_idx])
        eos = int(cu_seqlens[seq_idx + 1])
        start = bos + local_chunk * chunk_size
        end = min(start + chunk_size, eos)
        if start < end:
            yield start, end


def inspect_case(path: Path) -> dict[str, Any]:
    meta_path = path.parent / "case_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return {
            "dtype": meta.get("dtype"),
            "gtype": meta.get("gtype"),
            "varlen": meta.get("varlen"),
            "B": meta.get("B"),
            "T": meta.get("T"),
            "Hk": meta.get("Hk"),
            "Hv": meta.get("Hv"),
            "K": meta.get("K"),
            "chunk_size": meta.get("chunk_size"),
        }

    try:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            dump = torch.load(path, map_location="cpu", weights_only=False)
        k = dump.get("inputs", {}).get("k")
        g = dump.get("inputs", {}).get("g")
        meta = dict(dump.get("meta") or {})
        dtype = str(k.dtype).replace("torch.", "") if isinstance(k, torch.Tensor) else None
        return {
            "dtype": {"float16": "fp16", "bfloat16": "bf16"}.get(dtype, dtype),
            "gtype": str(g.dtype).replace("torch.", "") if isinstance(g, torch.Tensor) else None,
            "varlen": None,
            "B": k.shape[0] if isinstance(k, torch.Tensor) else None,
            "T": k.shape[1] if isinstance(k, torch.Tensor) and k.ndim >= 2 else None,
            "Hk": k.shape[2] if isinstance(k, torch.Tensor) and k.ndim >= 3 else None,
            "Hv": g.shape[2] if isinstance(g, torch.Tensor) and g.ndim >= 3 else None,
            "K": k.shape[3] if isinstance(k, torch.Tensor) and k.ndim >= 4 else None,
            "chunk_size": meta.get("chunk_size"),
        }
    except Exception:
        return {}


def find_cases(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("*_chunk_scaled_dot_kkt_fwd.pt"))


def resolve_cpu_reference_heads(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    gpu_ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Build CPU reference head indices for native GVA dumps.

    q/k use Hk heads while g/beta may use Hv heads. The current KKT dump output
    A is key-head aligned, so the CPU reference produces Hk output heads while
    still receiving the original full-Hv g/beta tensors.
    """

    hk = k.shape[1]
    hv = g.shape[1]
    out_heads = gpu_ref.shape[1]
    if out_heads != hk:
        raise ValueError(
            "Current AscendC op returns [B,Hk,T,BT], but dump output has a different head count: "
            f"k={tuple(k.shape)} A={tuple(gpu_ref.shape)}"
        )
    if g.shape[:1] != k.shape[:1] or g.shape[2] != k.shape[2]:
        raise ValueError(f"g prefix dimensions must match k [B,*,T], got k={tuple(k.shape)} g={tuple(g.shape)}")
    if beta.shape[:1] != k.shape[:1] or beta.shape[2] != k.shape[2]:
        raise ValueError(f"beta prefix dimensions must match k [B,*,T], got k={tuple(k.shape)} beta={tuple(beta.shape)}")
    if beta.shape[1] != hv:
        raise ValueError(f"g/beta head count must match, got g={g.shape[1]} beta={beta.shape[1]}")
    if hv < hk:
        raise ValueError(f"g/beta head count must be >= k head count, got Hk={hk} Hv={hv}")

    k_head_indices = torch.arange(out_heads, dtype=torch.long)
    gate_head_indices = torch.arange(out_heads, dtype=torch.long)
    if hv == hk:
        return k_head_indices, gate_head_indices, "exact"
    return k_head_indices, gate_head_indices, f"gva-hk-output:{hk}-of-{hv}"


def strict_dump_param_mismatch(k: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, gpu_ref: torch.Tensor) -> str | None:
    if gpu_ref.shape[:3] != k.shape[:3] or gpu_ref.shape[3] <= 0:
        return f"output A shape must match k [B,H,T,*] after layout conversion, got k={tuple(k.shape)} A={tuple(gpu_ref.shape)}"
    if g.shape != k.shape[:3]:
        return f"g shape must match k [B,H,T] after layout conversion, got k={tuple(k.shape)} g={tuple(g.shape)}"
    if beta.shape != k.shape[:3]:
        return f"beta shape must match k [B,H,T] after layout conversion, got k={tuple(k.shape)} beta={tuple(beta.shape)}"
    return None


def gva_dump_shape_issue(k: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, gpu_ref: torch.Tensor) -> str | None:
    if k.ndim != 4 or g.ndim != 3 or beta.ndim != 3 or gpu_ref.ndim != 4:
        return "GVA requires k=[B,Hk,T,K], g/beta=[B,Hv,T], A=[B,H,T,BT]"
    bsz, hk, seq_len, _ = k.shape
    hv = g.shape[1]
    if g.shape != beta.shape:
        return f"g and beta must have identical shape, got g={tuple(g.shape)} beta={tuple(beta.shape)}"
    if g.shape[0] != bsz or beta.shape[0] != bsz or g.shape[2] != seq_len or beta.shape[2] != seq_len:
        return f"g/beta must match k B/T, got k={tuple(k.shape)} g={tuple(g.shape)} beta={tuple(beta.shape)}"
    if hk <= 0 or hv <= 0 or hv % hk != 0:
        return f"GVA requires Hv divisible by Hk, got Hk={hk} Hv={hv}"
    if gpu_ref.shape[0] != bsz or gpu_ref.shape[2] != seq_len or gpu_ref.shape[3] <= 0:
        return f"GPU A must match k B/T and have BT>0, got k={tuple(k.shape)} A={tuple(gpu_ref.shape)}"
    if gpu_ref.shape[1] not in (hk, hv):
        return f"GPU A head dimension must be Hk or Hv, got Hk={hk} Hv={hv} A={tuple(gpu_ref.shape)}"
    return None


def cpu_fp64_reference(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    cu_seqlens: list[int] | None = None,
    chunk_indices: list[int] | None = None,
    k_head_indices: torch.Tensor | None = None,
    gate_head_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    k_f = k.to(torch.float64).cpu()
    g_f = g.to(torch.float64).cpu()
    beta_f = beta.to(torch.float64).cpu()
    if k_head_indices is None and gate_head_indices is None:
        if g_f.shape[1] != k_f.shape[1] or beta_f.shape[1] != k_f.shape[1]:
            raise ValueError(
                "CPU reference without explicit GVA head indices requires k/g/beta to have the same head count, "
                f"got k={tuple(k_f.shape)} g={tuple(g_f.shape)} beta={tuple(beta_f.shape)}"
            )
        k_head_indices = torch.arange(k_f.shape[1], dtype=torch.long)
        gate_head_indices = torch.arange(g_f.shape[1], dtype=torch.long)
    elif k_head_indices is None or gate_head_indices is None:
        raise ValueError("k_head_indices and gate_head_indices must be provided together")

    k_idx = k_head_indices.to(torch.long).cpu()
    gate_idx = gate_head_indices.to(torch.long).cpu()
    if k_idx.ndim != 1 or gate_idx.ndim != 1 or k_idx.numel() != gate_idx.numel():
        raise ValueError(
            "CPU GVA head indices must be 1D tensors with identical length, "
            f"got k={tuple(k_idx.shape)} gate={tuple(gate_idx.shape)}"
        )
    if k_idx.numel() == 0:
        raise ValueError("CPU GVA head indices must not be empty")
    if int(k_idx.min()) < 0 or int(k_idx.max()) >= k_f.shape[1]:
        raise ValueError(f"k_head_indices out of range for k shape {tuple(k_f.shape)}: {k_idx.tolist()}")
    if int(gate_idx.min()) < 0 or int(gate_idx.max()) >= g_f.shape[1] or int(gate_idx.max()) >= beta_f.shape[1]:
        raise ValueError(
            f"gate_head_indices out of range for g/beta shapes {tuple(g_f.shape)} / {tuple(beta_f.shape)}: "
            f"{gate_idx.tolist()}"
        )

    k_f = k_f.index_select(1, k_idx)
    g_f = g_f.index_select(1, gate_idx)
    beta_f = beta_f.index_select(1, gate_idx)
    bsz, heads, seq_len, _ = k_f.shape
    out = torch.zeros((bsz, heads, seq_len, chunk_size), dtype=torch.float64)

    if cu_seqlens is None and chunk_indices is None:
        full_chunks = seq_len // chunk_size
        full_t = full_chunks * chunk_size
        if full_chunks:
            mask = torch.tril(torch.ones((chunk_size, chunk_size), dtype=torch.bool), diagonal=-1)
            k_blocks = k_f[:, :, :full_t, :].reshape(bsz, heads, full_chunks, chunk_size, -1)
            g_blocks = g_f[:, :, :full_t].reshape(bsz, heads, full_chunks, chunk_size)
            beta_blocks = beta_f[:, :, :full_t].reshape(bsz, heads, full_chunks, chunk_size)
            batch_chunks = max(1, CPU_BATCH_CHUNKS)
            for chunk_start in range(0, full_chunks, batch_chunks):
                chunk_end = min(chunk_start + batch_chunks, full_chunks)
                k_batch = k_blocks[:, :, chunk_start:chunk_end, :, :]
                scores = torch.matmul(k_batch, k_batch.transpose(-1, -2))
                g_batch = g_blocks[:, :, chunk_start:chunk_end, :]
                gate = torch.exp(torch.clamp(g_batch[..., :, None] - g_batch[..., None, :], -50.0, 50.0))
                scaled = scores * gate
                scaled *= beta_blocks[:, :, chunk_start:chunk_end, :, None]
                scaled.masked_fill_(~mask, 0.0)
                start_t = chunk_start * chunk_size
                end_t = chunk_end * chunk_size
                out[:, :, start_t:end_t, :] = scaled.reshape(bsz, heads, end_t - start_t, chunk_size)
        if full_t == seq_len:
            return out

        start = full_t
        end = seq_len
        valid = end - start
        mask = torch.tril(torch.ones((valid, valid), dtype=torch.bool), diagonal=-1)
        k_block = k_f[:, :, start:end, :]
        scores = torch.matmul(k_block, k_block.transpose(-1, -2))
        gate = torch.exp(torch.clamp(g_f[:, :, start:end, None] - g_f[:, :, None, start:end], -50.0, 50.0))
        scaled = scores * gate * beta_f[:, :, start:end, None]
        out[:, :, start:end, :valid] = torch.where(mask, scaled, torch.zeros_like(scaled))
        return out

    mask_cache: dict[int, torch.Tensor] = {}
    for start, end in iter_chunk_ranges(seq_len, chunk_size, cu_seqlens, chunk_indices):
        valid = end - start
        if valid not in mask_cache:
            mask_cache[valid] = torch.tril(torch.ones((valid, valid), dtype=torch.bool), diagonal=-1)
        mask = mask_cache[valid]

        k_block = k_f[:, :, start:end, :]
        scores = torch.matmul(k_block, k_block.transpose(-1, -2))
        gate = torch.exp(torch.clamp(g_f[:, :, start:end, None] - g_f[:, :, None, start:end], -50.0, 50.0))
        scaled = scores * gate * beta_f[:, :, start:end, None]
        out[:, :, start:end, :valid] = torch.where(mask, scaled, torch.zeros_like(scaled))

    return out


def summarize_dual_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("skipped"):
        return {
            "success": None,
            "skipped": True,
            "reason": result.get("reason"),
            "checks": result.get("checks", {}),
            "ratios": result.get("ratios", {}),
        }
    return {
        "success": bool(result.get("success")),
        "skipped": False,
        "checks": result.get("checks", {}),
        "ratios": result.get("ratios", {}),
    }


def compare_chunk(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict[str, Any]:
    a = actual.float().cpu()
    b = expected.float().cpu()
    diff = (a - b).abs()
    close = diff <= (atol + rtol * b.abs())
    return {
        "ok": bool(close.all().item()),
        "max_err": diff.max().item(),
        "sum_err": diff.sum().item(),
        "count": diff.numel(),
    }


def finalize_compare(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        return {"ok": False, "max_err": float("nan"), "mean_err": float("nan")}
    total = sum(part["count"] for part in parts)
    return {
        "ok": all(part["ok"] for part in parts),
        "max_err": max(part["max_err"] for part in parts),
        "mean_err": sum(part["sum_err"] for part in parts) / total,
    }


def compare_npu_gpu_stream(npu_out: torch.Tensor, gpu_ref: torch.Tensor, chunk_size: int, atol: float, rtol: float) -> dict[str, Any]:
    parts = []
    seq_len = npu_out.shape[2]
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        parts.append(compare_chunk(npu_out[:, :, start:end, :], gpu_ref[:, :, start:end, :], atol, rtol))
    return finalize_compare(parts)


def compare_npu_cpu_stream(
    npu_out: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    k_f = k.float().cpu()
    g_f = g.float().cpu()
    beta_f = beta.float().cpu()
    parts = []
    seq_len = k_f.shape[2]
    mask_cache: dict[int, torch.Tensor] = {}
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        valid = end - start
        if valid not in mask_cache:
            mask_cache[valid] = torch.tril(torch.ones((valid, valid), dtype=torch.bool), diagonal=-1)
        k_block = k_f[:, :, start:end, :]
        scores = torch.matmul(k_block, k_block.transpose(-1, -2))
        gate = torch.exp(torch.clamp(g_f[:, :, start:end, None] - g_f[:, :, None, start:end], -50.0, 50.0))
        ref = torch.zeros((k_f.shape[0], k_f.shape[1], valid, chunk_size), dtype=torch.float32)
        scaled = scores * gate * beta_f[:, :, start:end, None]
        ref[:, :, :, :valid] = torch.where(mask_cache[valid], scaled, torch.zeros_like(scaled))
        parts.append(compare_chunk(npu_out[:, :, start:end, :], ref, atol, rtol))
    return finalize_compare(parts)


def compare_npu_cpu_gva_stream(
    npu_out: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    atol: float,
    rtol: float,
    cu_seqlens: list[int] | None = None,
    chunk_indices: list[int] | None = None,
) -> dict[str, Any]:
    k_f = k.to(torch.float64).cpu()
    g_f = g.to(torch.float64).cpu()
    beta_f = beta.to(torch.float64).cpu()
    npu_f = npu_out.to(torch.float64).cpu()
    bsz, hk, seq_len, _ = k_f.shape
    hv = g_f.shape[1]
    hv_per_hk = hv // hk
    head_batch = max(1, CPU_HEAD_BATCH)
    parts = []
    mask_cache: dict[int, torch.Tensor] = {}
    for start, end in iter_chunk_ranges(seq_len, chunk_size, cu_seqlens, chunk_indices):
        valid = end - start
        if valid not in mask_cache:
            mask_cache[valid] = torch.tril(torch.ones((valid, valid), dtype=torch.bool), diagonal=-1)
        mask = mask_cache[valid]
        for h0 in range(0, hv, head_batch):
            h1 = min(h0 + head_batch, hv)
            hk_indices = torch.arange(h0, h1, dtype=torch.long) // hv_per_hk
            k_block = k_f[:, hk_indices, start:end, :]
            scores = torch.matmul(k_block, k_block.transpose(-1, -2))
            gate = torch.exp(torch.clamp(g_f[:, h0:h1, start:end, None] - g_f[:, h0:h1, None, start:end], -50.0, 50.0))
            scaled = scores * gate * beta_f[:, h0:h1, start:end, None]
            ref = torch.zeros((bsz, h1 - h0, valid, chunk_size), dtype=torch.float64)
            ref[:, :, :, :valid] = torch.where(mask, scaled, torch.zeros_like(scaled))
            parts.append(compare_chunk(npu_f[:, h0:h1, start:end, :], ref, atol, rtol))
    return finalize_compare(parts)


def check_zero_regions(
    out: torch.Tensor,
    chunk_size: int,
    cu_seqlens: list[int] | None = None,
    chunk_indices: list[int] | None = None,
) -> float:
    _, _, seq_len, _ = out.shape
    max_zero = 0.0
    for start, end in iter_chunk_ranges(seq_len, chunk_size, cu_seqlens, chunk_indices):
        valid = end - start
        block = out[:, :, start:end, :].float().cpu()
        upper = torch.triu(block[:, :, :, :valid], diagonal=0)
        max_zero = max(max_zero, upper.abs().max().item())
        if valid < chunk_size:
            max_zero = max(max_zero, block[:, :, :, valid:].abs().max().item())
    return max_zero


def require_npu(device: str) -> torch.device:
    try:
        import torch_npu  # noqa: F401
        import fla_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("NPU dump test requires torch_npu and fla_npu to be importable") from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU device is not available")
    if not hasattr(torch.ops.npu, "npu_chunk_scaled_dot_kkt"):
        raise RuntimeError("torch.ops.npu.npu_chunk_scaled_dot_kkt is not registered")

    dev = torch.device(device)
    if dev.type == "npu":
        torch.npu.set_device(dev.index or 0)
    return dev


def verify_case(
    path: Path,
    device: torch.device,
    atol: float,
    rtol: float,
    skip_extra_allclose: bool = False,
    execute_gva_without_full_gpu: bool = False,
) -> dict[str, Any]:
    inputs, meta, outputs = load_dump_for_npu(path)
    if not {"k", "g", "beta"}.issubset(inputs):
        raise KeyError(f"{path} is missing one of k/g/beta inputs")
    if "A" not in outputs:
        raise KeyError(f"{path} is missing dumped A output")

    chunk_size = int(meta["chunk_size"])
    cu_seqlens = meta_int_list(meta, "cu_seqlens")
    chunk_indices = meta_int_list(meta, "chunk_indices_npu", "chunk_indices")
    if (cu_seqlens is None) != (chunk_indices is None):
        raise ValueError(f"{path} has incomplete varlen metadata: cu_seqlens={cu_seqlens is not None}, chunk_indices={chunk_indices is not None}")
    if chunk_indices is not None and len(chunk_indices) % 2 != 0:
        raise ValueError(f"{path} chunk_indices length must be even, got {len(chunk_indices)}")
    k = inputs["k"].contiguous()
    gpu_out_from_dump = outputs["A"].contiguous()
    g_full = inputs["g"].float().contiguous()
    beta_full = inputs["beta"].float().contiguous()
    gva_issue = gva_dump_shape_issue(k, g_full, beta_full, gpu_out_from_dump)
    if gva_issue is not None:
        mismatch = gva_issue
    else:
        mismatch = None
        if gpu_out_from_dump.shape[:3] != k.shape[:3] or gpu_out_from_dump.shape[3] <= 0:
            mismatch = (
                "output A shape must match k [B,Hk,T,*] after layout conversion, "
                f"got k={tuple(k.shape)} A={tuple(gpu_out_from_dump.shape)}"
            )
    if mismatch is not None:
        return {
            "case": str(path),
            "shape": tuple(gpu_out_from_dump.shape),
            "input_shape": tuple(k.shape),
            "g_shape": tuple(g_full.shape),
            "beta_shape": tuple(beta_full.shape),
            "chunk_size": chunk_size,
            "varlen": cu_seqlens is not None,
            "cu_len": len(cu_seqlens) if cu_seqlens is not None else 0,
            "chunk_pairs": len(chunk_indices) // 2 if chunk_indices is not None else 0,
            "head_map": "strict-mismatch",
            "dtype": None,
            "gpu_dump_dtype": str(gpu_out_from_dump.dtype),
            "cpu_golden_dtype": None,
            "shape_ok": False,
            "dtype_ok": False,
            "max_zero": None,
            "gpu": {},
            "cpu": {},
            "dual": {},
            "ok": False,
            "unsupported": mismatch,
        }

    k_head_indices, gate_head_indices, head_map = resolve_cpu_reference_heads(k, g_full, beta_full, gpu_out_from_dump)
    gpu_reference_mode = "dump-hk"

    op_kwargs: dict[str, Any] = {"chunk_size": chunk_size}
    if cu_seqlens is not None and chunk_indices is not None:
        op_kwargs["cu_seqlens"] = cu_seqlens
        op_kwargs["chunk_indices"] = chunk_indices

    npu_out = torch.ops.npu.npu_chunk_scaled_dot_kkt(
        k.to(device), g_full.to(device), beta_full.to(device), **op_kwargs
    ).cpu()
    cpu_fp64_golden = cpu_fp64_reference(
        k,
        g_full,
        beta_full,
        chunk_size,
        cu_seqlens,
        chunk_indices,
        k_head_indices,
        gate_head_indices,
    )
    print("       running ct.dual(npu_out, cpu_fp64_golden, gpu_out_from_dump, level=\"L1\")", flush=True)
    dual_result = ct.dual(npu_out, cpu_fp64_golden, gpu_out_from_dump, level="L1")
    if skip_extra_allclose:
        gpu_result = {"skipped": True, "ok": None, "max_err": None, "mean_err": None}
        cpu_result = {"skipped": True, "ok": None, "max_err": None, "mean_err": None}
        max_zero = None
    else:
        gpu_result = compare_npu_gpu_stream(npu_out, gpu_out_from_dump, chunk_size, atol, rtol)
        cpu_result = compare_npu_gpu_stream(npu_out, cpu_fp64_golden, chunk_size, atol, rtol)
        max_zero = check_zero_regions(npu_out, chunk_size, cu_seqlens, chunk_indices)
    shape_ok = tuple(npu_out.shape) == tuple(gpu_out_from_dump.shape) == tuple(cpu_fp64_golden.shape)
    cpu_golden_dtype = str(cpu_fp64_golden.dtype)
    dtype_ok = npu_out.dtype == gpu_out_from_dump.dtype

    return {
        "case": str(path),
        "shape": tuple(npu_out.shape),
        "input_shape": tuple(k.shape),
        "g_shape": tuple(inputs["g"].shape),
        "beta_shape": tuple(inputs["beta"].shape),
        "chunk_size": chunk_size,
        "varlen": cu_seqlens is not None,
        "cu_len": len(cu_seqlens) if cu_seqlens is not None else 0,
        "chunk_pairs": len(chunk_indices) // 2 if chunk_indices is not None else 0,
        "head_map": head_map,
        "gpu_reference_mode": gpu_reference_mode,
        "dtype": str(npu_out.dtype),
        "gpu_dump_dtype": str(gpu_out_from_dump.dtype),
        "cpu_golden_dtype": cpu_golden_dtype,
        "shape_ok": shape_ok,
        "dtype_ok": dtype_ok,
        "max_zero": max_zero,
        "gpu": gpu_result,
        "cpu": cpu_result,
        "dual": summarize_dual_result(dual_result),
        "ok": (
            shape_ok
            and dtype_ok
            and (max_zero is None or max_zero <= ZERO_TOL)
            and bool(dual_result.get("success"))
        ),
    }


def print_result(index: int, total: int, result: dict[str, Any]) -> None:
    status = "SKIP" if "unsupported" in result else ("PASS" if result["ok"] else "FAIL")
    print(
        f"[{status}] {index:03d}/{total:03d} {result['case']} "
        f"k={result['input_shape']} g={result['g_shape']} beta={result['beta_shape']} "
        f"out={result['shape']} dtype={result['dtype']} gpu_dtype={result['gpu_dump_dtype']} "
        f"cpu_dtype={result['cpu_golden_dtype']} BT={result['chunk_size']} head_map={result['head_map']} "
        f"gpu_ref_mode={result.get('gpu_reference_mode')} "
        f"varlen={result['varlen']} cu_len={result['cu_len']} chunk_pairs={result['chunk_pairs']}",
        flush=True,
    )
    if "unsupported" in result:
        print(f"       reason={result['unsupported']}", flush=True)
        return
    if result["gpu"].get("available") is False:
        print(f"       npu_vs_gpu unavailable: {result['gpu']['reason']}", flush=True)
        print(
            f"       npu_vs_cpu max_err={result['cpu']['max_err']:.9f} "
            f"mean_err={result['cpu']['mean_err']:.9f} allclose={result['cpu']['ok']} "
            f"shape_ok={result['shape_ok']} dtype_ok={result['dtype_ok']} max_zero={result['max_zero']:.3e}",
            flush=True,
        )
    elif result["gpu"].get("skipped"):
        print("       npu_vs_gpu extra_allclose=skipped (covered by ct.dual)", flush=True)
        print(
            "       npu_vs_cpu extra_allclose=skipped (covered by ct.dual) "
            f"shape_ok={result['shape_ok']} dtype_ok={result['dtype_ok']} max_zero=skipped",
            flush=True,
        )
    else:
        print(
            f"       npu_vs_gpu max_err={result['gpu']['max_err']:.9f} "
            f"mean_err={result['gpu']['mean_err']:.9f} allclose={result['gpu']['ok']}",
            flush=True,
        )
        print(
            f"       npu_vs_cpu max_err={result['cpu']['max_err']:.9f} "
            f"mean_err={result['cpu']['mean_err']:.9f} allclose={result['cpu']['ok']} "
            f"shape_ok={result['shape_ok']} dtype_ok={result['dtype_ok']} max_zero={result['max_zero']:.3e}",
            flush=True,
        )
    if result["dual"].get("skipped"):
        print(f"       ct.dual L1 skipped: {result['dual'].get('reason')}", flush=True)
    else:
        print(
            f"       ct.dual L1 success={result['dual']['success']} "
            f"checks={result['dual']['checks']} ratios={result['dual']['ratios']}",
            flush=True,
        )


def cleanup_case(device: torch.device) -> None:
    gc.collect()
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()


def main() -> int:
    args = parse_args()
    if args.compact_ct_report:
        install_compact_ct_report()
    root = Path(args.dump_root)
    cases = find_cases(root)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        print(f"[ERROR] no chunk_scaled_dot_kkt dump cases found under {root}")
        return 1

    device = require_npu(args.device)
    print(f"chunk_scaled_dot_kkt dump root={root} cases={len(cases)} device={device} atol={args.atol} rtol={args.rtol}", flush=True)

    passed = 0
    skipped = 0
    failures: list[str] = []
    results: list[dict[str, Any]] = []
    for idx, path in enumerate(cases, start=1):
        try:
            result = verify_case(
                path,
                device,
                args.atol,
                args.rtol,
                args.skip_extra_allclose,
                args.execute_gva_without_full_gpu,
            )
            print_result(idx, len(cases), result)
            results.append(result)
            if "unsupported" in result:
                skipped += 1
            elif result["ok"]:
                passed += 1
            else:
                failures.append(result["case"])
        except Exception as exc:
            failures.append(str(path))
            print(f"[FAIL] {idx:03d}/{len(cases):03d} {path} error={type(exc).__name__}: {exc}", flush=True)
            results.append({"case": str(path), "ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        finally:
            cleanup_case(device)

    failed = len(cases) - passed - skipped
    print("=" * 90, flush=True)
    print(f"Summary: PASS={passed} FAIL={failed} SKIP={skipped} TOTAL={len(cases)}", flush=True)
    if failures:
        print("Failed cases:", flush=True)
        for case in failures:
            print(f"  {case}", flush=True)
    if args.result_json:
        summary = {
            "dump_root": str(root),
            "device": str(device),
            "atol": args.atol,
            "rtol": args.rtol,
            "total": len(cases),
            "pass": passed,
            "fail": failed,
            "skip": skipped,
            "failures": failures,
            "results": results,
        }
        Path(args.result_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
