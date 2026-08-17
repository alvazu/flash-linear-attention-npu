"""chunk_kda_bwd_intra 的 ATK executor。

输入生成、CPU 标杆、run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _RCP_LN2,
    _calc_dtype,
    _case_spec,
    _chunks,
    _finite_tuple,
    _gate,
    _int_tensor,
    _kda_gate,
    _marker_device,
    _num_chunks,
    _orig_dtype,
    _rand,
    _randn,
    _zeros,
)


OP_NAME = "chunk_kda_bwd_intra"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    calc_dtype = torch.float64 if high_precision else torch.bfloat16
    seed = int(spec.get("seed", 20260817))
    B, T, H, K, chunk_size = (int(spec[x]) for x in ("B", "T", "H", "K", "chunk_size"))
    beta_dtype = str(spec.get("beta_dtype", "bf16")).lower()
    beta_calc = torch.float64 if high_precision else _orig_dtype(beta_dtype)
    return {
        "q": _randn((B, T, H, K), "bf16", calc_dtype, device, seed + 1),
        "k": _randn((B, T, H, K), "bf16", calc_dtype, device, seed + 2),
        "gk": _kda_gate((B, T, H, K), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 3),
        "beta": _rand((B, T, H), beta_dtype, beta_calc, device, seed + 4, 0.1, 0.9),
        "dAqk": _randn((B, T, H, chunk_size), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 5),
        "dAkk": _randn((B, T, H, chunk_size), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 6),
        "dq": _randn((B, T, H, K), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 7),
        "dk": _randn((B, T, H, K), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 8),
        "db": _randn((B, T, H), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 9),
        "dg": _randn((B, T, H, K), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 10),
        "chunk_size": chunk_size,
        "layout": str(spec.get("layout", "BSND")),
    }


def _chunk_kda_bwd_intra_ref(inputs):
    return inputs["dq"].clone(), inputs["dk"].clone(), inputs["db"].clone(), inputs["dg"].clone()


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """运行 CPU 同精度或 fp64 高精度标杆。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _chunk_kda_bwd_intra_ref(inputs)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.chunk_kda_bwd_intra(inputs["q"], inputs["k"], inputs["gk"], inputs["beta"], inputs["dAqk"], inputs["dAkk"], inputs["dq"], inputs["dk"], inputs["db"], inputs["dg"], cu_seqlens=None, chunk_indices=None, chunk_size=inputs["chunk_size"], safe_gate=True, layout=inputs["layout"])


@register("executor_chunk_kda_bwd_intra")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = self.device == "cpu" and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs)
