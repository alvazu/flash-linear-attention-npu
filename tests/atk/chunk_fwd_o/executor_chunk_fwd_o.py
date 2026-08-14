"""ATK executor for chunk_fwd_o."""

from __future__ import annotations

import json
import math

import torch

try:
    import numpy as np

    torch.serialization.add_safe_globals(
        [
            np.core.multiarray.scalar,
            np.dtype,
            type(np.dtype(np.float32)),
            type(np.dtype(np.float64)),
            type(np.dtype(np.int32)),
            type(np.dtype(np.int64)),
        ]
    )
except (AttributeError, ImportError):
    pass


_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}") from exc


def _seed(spec: dict, offset: int = 0) -> int:
    return int(spec.get("seed", 20260813)) + offset


def _rand(shape, dtype, device, seed, *, low=-0.08, high=0.08):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.rand(tuple(int(x) for x in shape), generator=generator, dtype=torch.float32)
    value = value.mul(float(high) - float(low)).add(float(low)).to(dtype)
    return value.to(device).contiguous()


def _normal(shape, dtype, device, seed, *, scale=0.08, bias=0.0):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.randn(tuple(int(x) for x in shape), generator=generator, dtype=torch.float32)
    value = value.mul(float(scale)).add(float(bias)).to(dtype)
    return value.to(device).contiguous()


def _cu(spec: dict) -> list[int] | None:
    value = str(spec.get("cu_seqlens", "")).strip()
    if not value:
        return None
    return [int(item) for item in value.split(",")]


def _chunk_indices(spec: dict) -> list[int] | None:
    cu = _cu(spec)
    if cu is None or not _as_bool(spec.get("explicit_chunk_indices", False)):
        return None
    chunk_size = int(spec.get("chunk_size", 64))
    indices: list[int] = []
    for seq_id, (start, end) in enumerate(zip(cu, cu[1:])):
        for chunk_id in range((end - start + chunk_size - 1) // chunk_size):
            indices.extend((seq_id, chunk_id))
    return indices


def _chunk_count(spec: dict) -> int:
    cu = _cu(spec)
    chunk_size = int(spec.get("chunk_size", 64))
    if cu is None:
        b = int(spec.get("B", 1))
        t = int(spec.get("T", 128))
        return b * ((t + chunk_size - 1) // chunk_size)
    return sum((end - start + chunk_size - 1) // chunk_size for start, end in zip(cu, cu[1:]))


def _finite_tuple(outputs) -> tuple[torch.Tensor, ...]:
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    visible = []
    for output in outputs:
        if isinstance(output, torch.Tensor):
            tensor = output.to(torch.float32) if output.is_floating_point() else output
            if not torch.isfinite(tensor.to(torch.float32)).all().item():
                raise RuntimeError("operator output contains NaN or Inf")
            visible.append(tensor.contiguous())
    if not visible:
        raise RuntimeError("operator returned no tensor outputs")
    return tuple(visible)


def _layout_qkv(spec: dict, device):
    b = int(spec.get("B", 1))
    h = int(spec.get("H", 4))
    hv = int(spec.get("HV", h))
    t = int(spec.get("T", 128))
    k_dim = int(spec.get("K", 128))
    v_dim = int(spec.get("V", 128))
    dtype = _dtype(spec.get("dtype", "bf16"))
    q = _normal((b, h, t, k_dim), dtype, device, _seed(spec, 1), scale=0.04)
    k = _normal((b, h, t, k_dim), dtype, device, _seed(spec, 2), scale=0.04)
    v = _normal((b, hv, t, v_dim), dtype, device, _seed(spec, 3), scale=0.04)
    g = _normal((b, hv, t), _dtype(spec.get("g_dtype", "fp32")), device, _seed(spec, 4), scale=0.02)
    beta = _rand((b, hv, t), _dtype(spec.get("beta_dtype", "fp32")), device, _seed(spec, 5), low=0.0, high=1.0)
    return q, k, v, g, beta


def prepare_call_args(spec: dict, low_marker: torch.Tensor):
    device = low_marker.device
    q, k, v, g, _beta = _layout_qkv(spec, device)
    b = int(spec.get("B", 1))
    hv = int(spec.get("HV", spec.get("H", 4)))
    k_dim = int(spec.get("K", 128))
    v_dim = int(spec.get("V", 128))
    chunks = _chunk_count(spec)
    dtype = _dtype(spec.get("dtype", "bf16"))
    h_state = _rand((b, hv, chunks, k_dim, v_dim), dtype, device, _seed(spec, 6))
    scale = float(spec.get("scale", 1.0 / math.sqrt(k_dim)))
    return q, k, v, h_state, scale, g, None, _cu(spec), _chunk_indices(spec), int(spec.get("chunk_size", 64)), False


def reference_outputs(spec: dict, low_marker: torch.Tensor) -> tuple[torch.Tensor, ...]:
    device = low_marker.device
    dtype = _dtype(spec.get("dtype", "bf16"))
    b = int(spec.get("B", 1))
    hv = int(spec.get("HV", spec.get("H", 4)))
    t = int(spec.get("T", 128))
    v_dim = int(spec.get("V", 128))
    return (torch.zeros((b, hv, t, v_dim), device=device, dtype=dtype),)


def run_npu(public_api: str, spec: dict, low_marker: torch.Tensor):
    from fla_npu.ops import ascendc

    args = prepare_call_args(spec, low_marker)
    op = getattr(ascendc, public_api)
    return op(*args[:5], g=args[5], g_gamma=args[6], cu_seqlens=args[7], chunk_indices=args[8], chunk_size=args[9], transpose_state_layout=args[10])


from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("executor_chunk_fwd_o")
class FunctionApi(BaseApi):
    op_name = "chunk_fwd_o"
    public_api = "chunk_fwd_o"

    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.spec = None
        self.is_benchmark_task = bool(task_result.is_benchmark_task)

    def init_by_input_data(self, input_data: InputDataset):
        self.spec = json.loads(str(input_data.kwargs["case_spec"]))

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        del with_output
        if getattr(self, "spec", None) is None:
            self.init_by_input_data(input_data)
        low_marker = input_data.kwargs["low_precision_marker"]
        if self.device == "npu":
            outputs = run_npu(self.public_api, self.spec, low_marker)
        else:
            outputs = reference_outputs(self.spec, low_marker)
        return _finite_tuple(outputs)

    def export_custom_data(self, input_data: InputDataset):
        del input_data
        return {
            "case_key": str(self.spec.get("case_key", "")),
            "soc": str(self.spec.get("soc", "")),
            "route": str(self.spec.get("route", "")),
            "seed": int(self.spec.get("seed", 0)),
        }
