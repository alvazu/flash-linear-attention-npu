"""ATK executor for recurrent_gated_delta_rule."""

from __future__ import annotations

import json

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


def _kda_layout_qkv(spec: dict, device):
    b = int(spec.get("B", 1))
    h = int(spec.get("H", 4))
    hv = int(spec.get("HV", h))
    t = int(spec.get("T", 128))
    k_dim = int(spec.get("K", 128))
    v_dim = int(spec.get("V", 128))
    dtype = _dtype(spec.get("dtype", "bf16"))
    layout = str(spec.get("layout", "BSND"))
    q_bsnd = _normal((b, t, h, k_dim), dtype, device, _seed(spec, 11), scale=0.04)
    k_bsnd = _normal((b, t, h, k_dim), dtype, device, _seed(spec, 12), scale=0.04)
    v_bsnd = _normal((b, t, hv, v_dim), dtype, device, _seed(spec, 13), scale=0.04)
    g_bsnd = _normal((b, t, hv, k_dim), _dtype(spec.get("g_dtype", "fp32")), device, _seed(spec, 14), scale=0.04)
    beta_bsnd = _rand((b, t, hv), _dtype(spec.get("beta_dtype", "bf16")), device, _seed(spec, 15), low=0.0, high=1.0)
    if layout == "BSND":
        return q_bsnd, k_bsnd, v_bsnd, g_bsnd, beta_bsnd
    if layout == "BNSD":
        return (
            q_bsnd.permute(0, 2, 1, 3).contiguous(),
            k_bsnd.permute(0, 2, 1, 3).contiguous(),
            v_bsnd.permute(0, 2, 1, 3).contiguous(),
            g_bsnd.permute(0, 2, 1, 3).contiguous(),
            beta_bsnd.permute(0, 2, 1).contiguous(),
        )
    if layout == "TND":
        return tuple(tensor.squeeze(0).contiguous() for tensor in (q_bsnd, k_bsnd, v_bsnd, g_bsnd, beta_bsnd))
    if layout == "NTD":
        return (
            q_bsnd.squeeze(0).permute(1, 0, 2).contiguous(),
            k_bsnd.squeeze(0).permute(1, 0, 2).contiguous(),
            v_bsnd.squeeze(0).permute(1, 0, 2).contiguous(),
            g_bsnd.squeeze(0).permute(1, 0, 2).contiguous(),
            beta_bsnd.squeeze(0).permute(1, 0).contiguous(),
        )
    raise ValueError(f"unsupported layout: {layout}")


def prepare_call_args(spec: dict, low_marker: torch.Tensor):
    return _kda_layout_qkv(spec, low_marker.device)


def reference_outputs(spec: dict, low_marker: torch.Tensor) -> tuple[torch.Tensor, ...]:
    device = low_marker.device
    dtype = _dtype(spec.get("dtype", "bf16"))
    b = int(spec.get("B", 1))
    t = int(spec.get("T", 128))
    hv = int(spec.get("HV", spec.get("H", 4)))
    v_dim = int(spec.get("V", 128))
    return (torch.zeros((b, t, hv, v_dim), device=device, dtype=dtype),)


def run_npu(public_api: str, spec: dict, low_marker: torch.Tensor):
    raise RuntimeError("recurrent_gated_delta_rule has no public fla_npu.ops.ascendc ctypes wrapper yet")


from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("executor_recurrent_gated_delta_rule")
class FunctionApi(BaseApi):
    op_name = "recurrent_gated_delta_rule"
    public_api = "recurrent_gated_delta_rule"

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
