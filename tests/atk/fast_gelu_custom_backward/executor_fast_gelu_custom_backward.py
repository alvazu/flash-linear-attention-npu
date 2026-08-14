"""ATK executor for fast_gelu_custom_backward."""

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


def prepare_call_args(spec: dict, low_marker: torch.Tensor):
    device = low_marker.device
    return (
        _rand(spec["shape"], _dtype(spec["dtype"]), device, _seed(spec, 1), low=-1.0, high=1.0),
        _rand(spec["shape"], _dtype(spec["dtype"]), device, _seed(spec, 2), low=-3.0, high=3.0),
    )


def reference_outputs(spec: dict, low_marker: torch.Tensor) -> tuple[torch.Tensor, ...]:
    device = low_marker.device
    grad = _rand(spec["shape"], _dtype(spec["dtype"]), device, _seed(spec, 1), low=-1.0, high=1.0)
    x = _rand(spec["shape"], _dtype(spec["dtype"]), device, _seed(spec, 2), low=-3.0, high=3.0).requires_grad_(True)
    y = torch.nn.functional.gelu(x.to(torch.float32), approximate="tanh")
    dx, = torch.autograd.grad(y, x, grad.to(torch.float32))
    return (dx.to(grad.dtype),)


def run_npu(public_api: str, spec: dict, low_marker: torch.Tensor):
    from fla_npu.ops import ascendc

    args = prepare_call_args(spec, low_marker)
    op = getattr(ascendc, public_api)
    return op(*args)


from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("executor_fast_gelu_custom_backward")
class FunctionApi(BaseApi):
    op_name = "fast_gelu_custom_backward"
    public_api = "fast_gelu_custom_backward"

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
