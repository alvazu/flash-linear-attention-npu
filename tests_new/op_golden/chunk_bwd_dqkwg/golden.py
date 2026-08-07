"""chunk_bwd_dqkwg 的输入构造、CPU 标杆和三条 route 适配。

这个文件是每个算子需要新增的主要内容：
- build_inputs: 根据简化后的 cases.yaml 构造输入；
- run_reference: 生成 CPU 对照/float64 标杆；
- run_ascendc/run_aclnn/run_direct_launch: 适配不同调用通路。
"""

from __future__ import annotations

from typing import Any

from tests_new.common.case_loader import chunk_count, data_generation_config, resolve_optional_inputs
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
    """按 case 构造输入。

    cases.yaml 只写 shape/dtype/attrs/optional_inputs，随机数据范围从 config.yaml/common 读取。
    """

    seed_case(torch, case)
    shape = case["shape"]
    dtype = case["dtype"]
    generation = data_generation_config(case)
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


def run_reference(torch: Any, inputs: dict[str, Any], case: dict[str, Any], *, benchmark: bool = False):
    """运行 CPU 标杆。

    benchmark=False: 按算子输入 dtype 进行普通精度对照；
    benchmark=True: 使用 float64 计算，作为 ct.dual 的真值输入。
    """

    cpu_inputs = _as_cpu_inputs(inputs)
    return _chunk_bwd_dqkwg_reference(torch, cpu_inputs, case, benchmark=benchmark)


def run_ascendc(torch: Any, inputs: dict[str, Any], case: dict[str, Any]):
    """默认稳定入口：fla_npu.ops.ascendc.chunk_bwd_dqkwg。"""

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
    """aclnn/op_api 兼容入口：torch.ops.npu.npu_chunk_bwd_dqkwg。"""

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
        use_exp2=False,
        transpose_state_layout=False,
    )


def run_direct_launch(torch: Any, inputs: dict[str, Any], case: dict[str, Any]):
    """直接 <<<>>> route，需要 fast-kernel 示例扩展 ascend_ops。"""

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


def _chunk_bwd_dqkwg_reference(torch: Any, inputs: dict[str, Any], case: dict[str, Any], *, benchmark: bool):
    """把 BNSD 输入转成 CPU 参考实现使用的 BTHD 布局，再转回 BNSD。"""

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
        benchmark,
    )
    return (
        dq.permute(0, 2, 1, 3).contiguous(),
        dk.permute(0, 2, 1, 3).contiguous(),
        dw.permute(0, 2, 1, 3).contiguous(),
        dg.permute(0, 2, 1).contiguous(),
    )


def _reference_bthd(torch: Any, q, k, v, dox, h, dh, g, dv, scale: float, cu_seqlens, chunk_size: int, benchmark: bool):
    """CPU 参考实现。

    benchmark=True 时尽量沿用旧测试里的 float64 计算方式，避免用低精度标杆放宽误差。
    """

    calc_type = torch.float64 if benchmark else torch.float32
    data_dtype = torch.float64 if benchmark else q.dtype
    gate_dtype = torch.float64 if benchmark else g.dtype
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
                q_c = q[b_idx, chunk_start:chunk_end, hk_idx, :].to(calc_type)
                k_c = k[b_idx, chunk_start:chunk_end, hk_idx, :].to(calc_type)
                v_c = v[b_idx, chunk_start:chunk_end, hv_idx, :].to(calc_type)
                do_c = dox[b_idx, chunk_start:chunk_end, hv_idx, :].to(calc_type)
                dv_c = dv[b_idx, chunk_start:chunk_end, hv_idx, :].to(calc_type)
                h_prev = h[b_idx, chunk_offset + chunk_idx, hv_idx, :, :].to(calc_type)
                dh_curr = dh[b_idx, chunk_offset + chunk_idx, hv_idx, :, :].to(calc_type)

                dq_state = do_c @ h_prev.transpose(-1, -2)
                dq_state = dq_state.to(data_dtype).to(calc_type)
                dk_state = v_c @ dh_curr.transpose(-1, -2)
                dk_state = dk_state.to(data_dtype).to(calc_type)
                dw_c = dv_c @ h_prev.transpose(-1, -2)
                dw[b_idx, chunk_start:chunk_end, hv_idx, :] = -dw_c.to(data_dtype)

                g_c = g[b_idx, chunk_start:chunk_end, hv_idx].to(calc_type)
                g_last = g[b_idx, min(chunk_start + chunk_size, end) - 1, hv_idx].to(calc_type)
                dg_last = (h_prev * dh_curr).sum() * torch.exp(g_last)

                dq_state = dq_state * torch.exp(g_c)[:, None] * scale
                dk_state = dk_state * torch.exp(-g_c + g_last)[:, None]

                dg_c = (dq_state * q_c).sum(dim=-1).to(data_dtype).to(calc_type)
                dg_c -= (k_c * dk_state).sum(dim=-1)
                dg_c = dg_c.to(gate_dtype).to(calc_type)
                dg_last += (dk_state * k_c).sum()

                ds = do_c @ v_c.transpose(-1, -2)
                ds = ds.to(data_dtype).to(calc_type)
                length = chunk_end - chunk_start
                rows = torch.arange(length)[:, None]
                cols = torch.arange(length)[None, :]
                mask = rows >= cols
                decay = torch.exp(g_c[:, None] - g_c[None, :])
                ds = torch.where(mask, ds * decay, torch.zeros_like(ds)) * scale

                qk = q_c @ k_c.transpose(-1, -2)
                qk = qk.to(data_dtype).to(calc_type)
                ds2 = ds * qk
                dg_c += ds2.sum(dim=1)
                dg_c = dg_c.to(gate_dtype).to(calc_type)
                dg_c -= ds2.sum(dim=0)
                dg_c = dg_c.to(gate_dtype)
                dg_c[length - 1] += dg_last.to(gate_dtype)
                dg[b_idx, chunk_start:chunk_end, hv_idx] = dg_c

                dq_intra = (ds @ k_c).to(data_dtype).to(calc_type)
                dk_intra = (ds.transpose(-1, -2) @ q_c).to(data_dtype).to(calc_type)
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
