"""ctypes backed aclnn calls for FLA NPU Ascend C operators."""

from __future__ import annotations

import ctypes
from collections import deque
from typing import Iterable, Optional, Sequence


ACL_SUCCESS = 0
ACL_FORMAT_NCHW = 0
ACL_FORMAT_ND = 2
ACL_FORMAT_NCDHW = 30
ACL_FORMAT_NCL = 47

_ACL_FORMAT_BY_NAME = {
    "NCHW": ACL_FORMAT_NCHW,
    "ND": ACL_FORMAT_ND,
    "NCDHW": ACL_FORMAT_NCDHW,
    "NCL": ACL_FORMAT_NCL,
}

_RECENT_LAUNCH_STORAGE = deque(maxlen=128)

_GET_WORKSPACE_ARGTYPES = {
    "aclnnSolveTri": [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ],
}


def _dtype_to_acl(dtype) -> int:
    import torch

    mapping = {
        torch.float32: 0,  # ACL_FLOAT
        torch.float16: 1,  # ACL_FLOAT16
        torch.int8: 2,  # ACL_INT8
        torch.int32: 3,  # ACL_INT32
        torch.uint8: 4,  # ACL_UINT8
        torch.int16: 6,  # ACL_INT16
        torch.int64: 9,  # ACL_INT64
        torch.float64: 11,  # ACL_DOUBLE
        torch.bool: 12,  # ACL_BOOL
        torch.bfloat16: 27,  # ACL_BF16
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported dtype for aclnn tensor descriptor: {dtype}") from exc


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _stride(tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.stride())


def _storage_numel(tensor) -> int:
    try:
        nbytes = tensor.untyped_storage().nbytes()
    except AttributeError:
        nbytes = tensor.storage().nbytes()
    return int(nbytes // tensor.element_size())


def _storage_data_ptr(tensor) -> int:
    try:
        return int(tensor.untyped_storage().data_ptr())
    except AttributeError:
        return int(tensor.storage().data_ptr())


def _format(tensor) -> int:
    try:
        import torch_npu

        npu_format = torch_npu.get_npu_format(tensor)
    except Exception:
        npu_format = None

    if isinstance(npu_format, str):
        acl_format = _ACL_FORMAT_BY_NAME.get(npu_format)
        if acl_format is not None:
            return acl_format
    elif npu_format is not None:
        try:
            return int(npu_format)
        except (TypeError, ValueError):
            pass

    dim = tensor.dim()
    if dim == 3:
        return ACL_FORMAT_NCL
    if dim == 4:
        return ACL_FORMAT_NCHW
    if dim == 5:
        return ACL_FORMAT_NCDHW
    return ACL_FORMAT_ND


def _ensure_npu_tensor(tensor, name: str):
    if tensor is None:
        return None
    if not hasattr(tensor, "device") or tensor.device.type != "npu":
        raise TypeError(f"{name} must be a torch NPU tensor, got {type(tensor)!r}.")
    return tensor


def _optional_bool(value, default: bool) -> bool:
    return default if value is None else bool(value)


def _optional_int(value, default: int) -> int:
    return default if value is None else int(value)


def _optional_float(value, default: float) -> float:
    return default if value is None else float(value)


def _chunk_num(total_tokens: int, chunk_size: int, chunk_indices: Optional[Sequence[int]]) -> int:
    if chunk_indices is not None:
        return len(chunk_indices) // 2
    return (total_tokens + chunk_size - 1) // chunk_size


def _current_stream_ptr() -> int:
    import torch

    stream = torch.npu.current_stream()
    return int(getattr(stream, "npu_stream"))


def _empty_like(tensor, *, dtype=None):
    import torch

    dtype = dtype or tensor.dtype
    return torch.empty_like(tensor, dtype=dtype)


def _empty(shape: Iterable[int], like, *, dtype=None):
    import torch

    return torch.empty(tuple(int(dim) for dim in shape), device=like.device, dtype=dtype or like.dtype)


def _zeros(shape: Iterable[int], like, *, dtype=None):
    import torch

    return torch.zeros(tuple(int(dim) for dim in shape), device=like.device, dtype=dtype or like.dtype)


class _AclTensor:
    def __init__(self, runtime: "_AclnnRuntime", tensor):
        tensor = _ensure_npu_tensor(tensor, "tensor")
        self._runtime = runtime
        self._tensor = tensor
        self._shape = (ctypes.c_int64 * tensor.dim())(*_shape(tensor))
        self._stride = (ctypes.c_int64 * tensor.dim())(*_stride(tensor))
        self._storage_shape = (ctypes.c_int64 * 1)(_storage_numel(tensor))
        self.ptr = runtime.acl_create_tensor(
            self._shape,
            ctypes.c_uint64(tensor.dim()),
            ctypes.c_int(_dtype_to_acl(tensor.dtype)),
            self._stride,
            ctypes.c_int64(int(tensor.storage_offset())),
            ctypes.c_int(_format(tensor)),
            self._storage_shape,
            ctypes.c_uint64(1),
            ctypes.c_void_p(_storage_data_ptr(tensor)),
        )
        if not self.ptr:
            raise RuntimeError("aclCreateTensor returned nullptr.")

    def destroy(self) -> None:
        if self.ptr:
            self._runtime.acl_destroy_tensor(self.ptr)
        self.ptr = None


class _AclIntArray:
    def __init__(self, runtime: "_AclnnRuntime", values: Optional[Sequence[int]]):
        self._runtime = runtime
        self.ptr = None
        if values is None:
            return
        values = tuple(int(value) for value in values)
        if not values:
            return
        self._values = (ctypes.c_int64 * len(values))(*values)
        self.ptr = runtime.acl_create_int_array(self._values, ctypes.c_uint64(len(values)))
        if not self.ptr:
            raise RuntimeError("aclCreateIntArray returned nullptr.")

    def destroy(self) -> None:
        if self.ptr:
            self._runtime.acl_destroy_int_array(self.ptr)
        self.ptr = None


class _CallContext:
    def __init__(self, runtime: "_AclnnRuntime"):
        self.runtime = runtime
        self.resources = []
        self.keepalive_tensors = []

    def tensor(self, tensor, name: str = "tensor") -> ctypes.c_void_p:
        if tensor is None:
            return ctypes.c_void_p()
        desc = _AclTensor(self.runtime, _ensure_npu_tensor(tensor, name))
        self.resources.append(desc)
        return ctypes.c_void_p(desc.ptr)

    def int_array(self, values: Optional[Sequence[int]]) -> ctypes.c_void_p:
        desc = _AclIntArray(self.runtime, values)
        self.resources.append(desc)
        return ctypes.c_void_p(desc.ptr or 0)

    def int_tensor(self, values: Optional[Sequence[int]], device) -> ctypes.c_void_p:
        if values is None:
            return ctypes.c_void_p()
        import torch

        tensor = torch.as_tensor(tuple(int(value) for value in values), dtype=torch.int64, device=device)
        self.keepalive_tensors.append(tensor)
        return self.tensor(tensor, "int tensor")

    def destroy(self) -> None:
        for resource in reversed(self.resources):
            resource.destroy()
        self.resources.clear()


class _AclnnRuntime:
    def __init__(self):
        import fla_npu

        self._libraries = fla_npu.load_ascendc_opapi_libraries()
        self._symbols = {}
        self.acl_create_tensor = self.symbol("aclCreateTensor")
        self.acl_create_tensor.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.acl_create_tensor.restype = ctypes.c_void_p

        self.acl_destroy_tensor = self.symbol("aclDestroyTensor")
        self.acl_destroy_tensor.argtypes = [ctypes.c_void_p]
        self.acl_destroy_tensor.restype = ctypes.c_int

        self.acl_create_int_array = self.symbol("aclCreateIntArray")
        self.acl_create_int_array.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64]
        self.acl_create_int_array.restype = ctypes.c_void_p

        self.acl_destroy_int_array = self.symbol("aclDestroyIntArray")
        self.acl_destroy_int_array.argtypes = [ctypes.c_void_p]
        self.acl_destroy_int_array.restype = ctypes.c_int

    def symbol(self, name: str):
        if name in self._symbols:
            return self._symbols[name]
        for library in self._libraries:
            try:
                symbol = getattr(library, name)
            except AttributeError:
                continue
            self._symbols[name] = symbol
            return symbol
        raise AttributeError(f"Unable to resolve aclnn symbol {name}.")

    def call(self, name: str, args: Sequence[object], outputs: Sequence[object]):
        get_workspace = self.symbol(f"{name}GetWorkspaceSize")
        launch = self.symbol(name)
        get_workspace.restype = ctypes.c_int
        if name in _GET_WORKSPACE_ARGTYPES:
            get_workspace.argtypes = _GET_WORKSPACE_ARGTYPES[name]
        launch.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
        launch.restype = ctypes.c_int
        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()

        ret = get_workspace(*args, ctypes.byref(workspace_size), ctypes.byref(executor))
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"{name}GetWorkspaceSize failed with aclnnStatus={ret}.")

        workspace = None
        workspace_ptr = ctypes.c_void_p()
        if workspace_size.value:
            import torch

            device = outputs[0].device
            workspace = torch.empty((int(workspace_size.value),), dtype=torch.uint8, device=device)
            workspace_ptr = ctypes.c_void_p(int(workspace.data_ptr()))

        ret = launch(
            workspace_ptr,
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(_current_stream_ptr()),
        )
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"{name} failed with aclnnStatus={ret}.")
        return workspace


_RUNTIME: Optional[_AclnnRuntime] = None


def _runtime() -> _AclnnRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = _AclnnRuntime()
    return _RUNTIME


def _finalize(outputs, workspace, keepalive_tensors):
    _RECENT_LAUNCH_STORAGE.append((tuple(outputs), workspace, tuple(keepalive_tensors)))


def _call_aclnn(name: str, build_args, outputs):
    runtime = _runtime()
    ctx = _CallContext(runtime)
    outputs_tuple = outputs if isinstance(outputs, tuple) else (outputs,)
    try:
        args = build_args(ctx)
        workspace = runtime.call(name, args, outputs_tuple)
    finally:
        ctx.destroy()
    _finalize(outputs_tuple, workspace, ctx.keepalive_tensors)
    return outputs


def npu_fast_gelu_custom(self):
    out = _empty_like(self)
    return _call_aclnn(
        "aclnnFastGelu",
        lambda ctx: [ctx.tensor(self, "self"), ctx.tensor(out, "out")],
        out,
    )


def npu_fast_gelu_custom_backward(grad, self):
    out = _empty_like(grad)
    return _call_aclnn(
        "aclnnFastGeluBackward",
        lambda ctx: [ctx.tensor(grad, "grad"), ctx.tensor(self, "self"), ctx.tensor(out, "out")],
        out,
    )


def npu_prepare_wy_repr_bwd_full(
    k,
    v,
    beta,
    A,
    dA,
    dw,
    du,
    g,
    chunk_size,
    *,
    cu_seqlens=None,
    chunk_indices=None,
):
    dk = _empty_like(k)
    dv = _empty_like(v)
    dbeta = _empty_like(beta)
    dg = _empty_like(g)
    outputs = (dk, dv, dbeta, dg)
    return _call_aclnn(
        "aclnnPrepareWyReprBwdFull",
        lambda ctx: [
            ctx.tensor(k, "k"),
            ctx.tensor(v, "v"),
            ctx.tensor(beta, "beta"),
            ctx.tensor(A, "A"),
            ctx.tensor(dA, "dA"),
            ctx.tensor(dw, "dw"),
            ctx.tensor(du, "du"),
            ctx.tensor(g, "g"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_int64(int(chunk_size)),
            ctx.tensor(dk, "dk"),
            ctx.tensor(dv, "dv"),
            ctx.tensor(dbeta, "dbeta"),
            ctx.tensor(dg, "dg"),
        ],
        outputs,
    )


def npu_chunk_gated_delta_rule_bwd_dhu(
    q,
    k,
    w,
    d_o,
    dv,
    scale,
    chunk_size,
    *,
    g=None,
    gK=None,
    h0=None,
    dht=None,
    cu_seqlens=None,
    chunk_indices=None,
    use_exp2=False,
    transpose_state_layout=False,
):
    q_shape = _shape(q)
    dv_shape = _shape(dv)
    B, _, T, K = q_shape
    Hv, V = dv_shape[1], dv_shape[3]
    NT = _chunk_num(T, int(chunk_size), chunk_indices)
    dh = _empty((B, Hv, NT, K, V), q)
    dh0 = _empty((B, Hv, NT, K, V), q) if h0 is not None else None
    dv2 = _empty_like(dv)
    outputs = (dh, dh0, dv2)
    return _call_aclnn(
        "aclnnChunkGatedDeltaRuleBwdDhu",
        lambda ctx: [
            ctx.tensor(q, "q"),
            ctx.tensor(k, "k"),
            ctx.tensor(w, "w"),
            ctx.tensor(d_o, "d_o"),
            ctx.tensor(dv, "dv"),
            ctx.tensor(g, "g"),
            ctx.tensor(gK, "gK"),
            ctx.tensor(h0, "h0"),
            ctx.tensor(dht, "dht"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_double(float(scale)),
            ctypes.c_int64(int(chunk_size)),
            ctx.tensor(dh, "dh"),
            ctx.tensor(dh0, "dh0"),
            ctx.tensor(dv2, "dv2"),
        ],
        outputs,
    )


def npu_chunk_bwd_dv_local(
    q,
    k,
    d_o,
    g,
    scale,
    chunk_size,
    *,
    g_gamma=None,
    A=None,
    cu_seqlens=None,
    chunk_indices=None,
):
    out = _empty_like(d_o)
    return _call_aclnn(
        "aclnnChunkBwdDvLocal",
        lambda ctx: [
            ctx.tensor(q, "q"),
            ctx.tensor(k, "k"),
            ctx.tensor(d_o, "d_o"),
            ctx.tensor(g, "g"),
            ctx.tensor(g_gamma, "g_gamma"),
            ctx.tensor(A, "A"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_double(float(scale)),
            ctypes.c_int64(int(chunk_size)),
            ctx.tensor(out, "out"),
        ],
        out,
    )


def npu_prepare_wy_repr_bwd_da(
    k,
    v,
    beta,
    A,
    dw,
    du,
    g,
    *,
    chunk_size,
    cu_seqlens=None,
    chunk_indices=None,
):
    out = _empty_like(A)
    return _call_aclnn(
        "aclnnPrepareWyReprBwdDa",
        lambda ctx: [
            ctx.tensor(k, "k"),
            ctx.tensor(v, "v"),
            ctx.tensor(beta, "beta"),
            ctx.tensor(A, "A"),
            ctx.tensor(dw, "dw"),
            ctx.tensor(du, "du"),
            ctx.tensor(g, "g"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_int64(int(chunk_size)),
            ctx.tensor(out, "dA"),
        ],
        out,
    )


def npu_chunk_bwd_dqkwg(
    q,
    k,
    v,
    g,
    h,
    dox,
    dh,
    dv,
    chunk_size,
    *,
    cu_seqlens=None,
    chunk_indices=None,
    w=None,
    g_gamma=None,
    scale=None,
    use_exp2=None,
    transpose_state_layout=None,
):
    q_shape = _shape(q)
    value_num_heads = int(v.shape[1])
    dq = _empty_like(q)
    dk = _empty_like(k)
    dw = _empty((q_shape[0], value_num_heads, q_shape[2], q_shape[3]), q)
    dg = _empty_like(g)
    outputs = (dq, dk, dw, dg)
    return _call_aclnn(
        "aclnnChunkBwdDqkwg",
        lambda ctx: [
            ctx.tensor(q, "q"),
            ctx.tensor(k, "k"),
            ctx.tensor(v, "v"),
            ctx.tensor(g, "g"),
            ctx.tensor(h, "h"),
            ctx.tensor(dox, "dox"),
            ctx.tensor(dh, "dh"),
            ctx.tensor(dv, "dv"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctx.tensor(w, "w"),
            ctx.tensor(g_gamma, "g_gamma"),
            ctypes.c_float(_optional_float(scale, 1.0)),
            ctypes.c_int64(int(chunk_size)),
            ctypes.c_bool(_optional_bool(use_exp2, False)),
            ctypes.c_bool(_optional_bool(transpose_state_layout, False)),
            ctx.tensor(dq, "dq"),
            ctx.tensor(dk, "dk"),
            ctx.tensor(dw, "dw"),
            ctx.tensor(dg, "dg"),
        ],
        outputs,
    )


def npu_chunk_fwd_o(
    q,
    k,
    v,
    h,
    scale,
    *,
    g=None,
    g_gamma=None,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=None,
    transpose_state_layout=False,
):
    del g_gamma, transpose_state_layout
    chunk_size = _optional_int(chunk_size, 64)
    out = _empty_like(v)
    return _call_aclnn(
        "aclnnChunkFwdO",
        lambda ctx: [
            ctx.tensor(q, "q"),
            ctx.tensor(k, "k"),
            ctx.tensor(v, "v"),
            ctx.tensor(h, "h"),
            ctx.tensor(g, "g"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_double(float(scale)),
            ctypes.c_int64(chunk_size),
            ctx.tensor(out, "out"),
        ],
        out,
    )


def npu_chunk_gated_delta_rule_fwd_h(
    k,
    w,
    u,
    g=None,
    *,
    gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=None,
    save_new_value=True,
    cu_seqlens=None,
    chunk_indices=None,
    use_exp2=False,
    transpose_state_layout=False,
):
    if g is None:
        raise RuntimeError("npu_chunk_gated_delta_rule_fwd_h: g cannot be None.")
    save_new_value = _optional_bool(save_new_value, True)
    use_exp2 = _optional_bool(use_exp2, False)
    transpose_state_layout = _optional_bool(transpose_state_layout, False)
    if not save_new_value:
        raise RuntimeError("npu_chunk_gated_delta_rule_fwd_h: save_new_value must be True.")
    if use_exp2:
        raise RuntimeError("npu_chunk_gated_delta_rule_fwd_h: use_exp2 must be False.")
    if transpose_state_layout:
        raise RuntimeError("npu_chunk_gated_delta_rule_fwd_h: transpose_state_layout must be False.")

    output_final_state = _optional_bool(output_final_state, False)
    chunk_size = _optional_int(chunk_size, 64)
    B, _, T, K = _shape(k)
    _, HV, _, V = _shape(u)
    NT = _chunk_num(T, chunk_size, chunk_indices)
    h_out = _zeros((B, HV, NT, K, V), k)
    v_new_out = _empty_like(u)
    if output_final_state:
        N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
        like = initial_state if initial_state is not None else h_out
        final_state_out = _empty((N, HV, K, V), like)
    else:
        final_state_out = _empty((1,), k)
    outputs = (h_out, v_new_out, final_state_out if output_final_state else None)
    return _call_aclnn(
        "aclnnChunkGatedDeltaRuleFwdH",
        lambda ctx: [
            ctx.tensor(k, "k"),
            ctx.tensor(w, "w"),
            ctx.tensor(u, "u"),
            ctx.tensor(g, "g"),
            ctx.tensor(gk, "gk"),
            ctx.tensor(initial_state, "initial_state"),
            ctypes.c_bool(output_final_state),
            ctypes.c_int64(chunk_size),
            ctypes.c_bool(save_new_value),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_bool(use_exp2),
            ctypes.c_bool(transpose_state_layout),
            ctx.tensor(h_out, "h"),
            ctx.tensor(v_new_out, "v_new"),
            ctx.tensor(final_state_out, "final_state"),
        ],
        outputs,
    )


def npu_recompute_w_u_fwd(
    k,
    v,
    beta,
    A,
    chunk_size,
    *,
    g=None,
    gk=None,
    cu_seqlens=None,
    chunk_indices=None,
):
    w_shape = list(_shape(v))
    w_shape[3] = int(k.shape[3])
    w_out = _empty(w_shape, v, dtype=k.dtype)
    u_out = _empty_like(v)
    outputs = (w_out, u_out)
    return _call_aclnn(
        "aclnnRecomputeWUFwd",
        lambda ctx: [
            ctx.tensor(k, "k"),
            ctx.tensor(v, "v"),
            ctx.tensor(beta, "beta"),
            ctx.tensor(A, "A"),
            ctx.tensor(g, "g"),
            ctx.tensor(gk, "gk"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            ctypes.c_int64(int(chunk_size)),
            ctx.tensor(w_out, "w"),
            ctx.tensor(u_out, "u"),
        ],
        outputs,
    )


def _infer_causal_conv1d_y(x, head_num: int, run_mode: int):
    x_dim = x.dim()
    if run_mode == 0 and head_num > 0:
        if x_dim == 3:
            b, s, d_model = _shape(x)
            return _empty((b, head_num, s, d_model // head_num), x)
        if x_dim == 2:
            s, d_model = _shape(x)
            return _empty((head_num, s, d_model // head_num), x)
    return _empty_like(x)


def npu_causal_conv1d(
    x,
    weight,
    bias=None,
    conv_states=None,
    *,
    query_start_loc=None,
    cache_indices=None,
    initial_state_mode=None,
    num_accepted_tokens=None,
    activation_mode=0,
    pad_slot_id=-1,
    run_mode=0,
    head_num=0,
):
    out = _infer_causal_conv1d_y(x, int(head_num), int(run_mode))
    return _call_aclnn(
        "aclnnCausalConv1d",
        lambda ctx: [
            ctx.tensor(x, "x"),
            ctx.tensor(weight, "weight"),
            ctx.tensor(bias, "bias"),
            ctx.tensor(conv_states, "conv_states"),
            ctx.int_tensor(query_start_loc, x.device),
            ctx.int_tensor(cache_indices, x.device),
            ctx.int_tensor(initial_state_mode, x.device),
            ctx.int_tensor(num_accepted_tokens, x.device),
            ctypes.c_int64(int(activation_mode)),
            ctypes.c_int64(int(pad_slot_id)),
            ctypes.c_int64(int(run_mode)),
            ctypes.c_int64(int(head_num)),
            ctx.tensor(out, "out"),
        ],
        out,
    )


def npu_causal_conv1d_bwd(
    x,
    y,
    weight,
    dy,
    initial_state=None,
    dht=None,
    *,
    query_start_loc=None,
    activation=0,
    input_layout="BSND",
):
    input_layout = str(input_layout)
    width, dim = int(weight.shape[0]), int(weight.shape[1])
    if input_layout == "BNSD":
        batch = int(x.shape[0])
        dx_shape = _shape(x)
    elif input_layout in {"NTD", "TND"}:
        if query_start_loc is None:
            raise RuntimeError(f"query_start_loc is required for {input_layout} input.")
        batch = len(query_start_loc) - 1
        dx_shape = _shape(x)
    else:
        batch = int(x.shape[0])
        dx_shape = _shape(x)
    dx = _empty(dx_shape, x)
    dw = _empty((width, dim), weight)
    db = _empty((dim,), weight)
    dh0 = _empty((batch, width, dim), x)
    outputs = (dx, dw, db, dh0)
    layout_buffer = ctypes.create_string_buffer(input_layout.encode("utf-8"))
    return _call_aclnn(
        "aclnnCausalConv1dBwd",
        lambda ctx: [
            ctx.tensor(x, "x"),
            ctx.tensor(y, "y"),
            ctx.tensor(weight, "weight"),
            ctx.tensor(dy, "dy"),
            ctx.tensor(initial_state, "initial_state"),
            ctx.tensor(dht, "dht"),
            ctx.int_array(query_start_loc),
            ctypes.c_int64(int(activation)),
            ctypes.cast(layout_buffer, ctypes.c_char_p),
            ctx.tensor(dx, "dx"),
            ctx.tensor(dw, "dw"),
            ctx.tensor(db, "db"),
            ctx.tensor(dh0, "dh0"),
        ],
        outputs,
    )


def npu_solve_tri(x, *, cu_seqlens=None, chunk_indices=None, layout="bsnd"):
    x_contig = x.contiguous()
    out = _empty_like(x_contig)
    layout_arg = ctypes.c_char_p(str(layout).encode("utf-8"))
    return _call_aclnn(
        "aclnnSolveTri",
        lambda ctx: [
            ctx.tensor(x_contig, "x"),
            ctx.int_array(cu_seqlens),
            ctx.int_array(chunk_indices),
            layout_arg,
            ctx.tensor(out, "out"),
        ],
        out,
    )


ASCENDC_CTYPES_OPS = {
    name: value
    for name, value in globals().items()
    if name.startswith("npu_") and callable(value)
}
