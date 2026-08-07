"""Runtime helpers shared by route, accuracy and perf scripts."""

from __future__ import annotations

import importlib
import os
from typing import Any


def get_torch():
    return importlib.import_module("torch")


def require_npu(device_id: int = 0):
    torch = get_torch()
    try:
        importlib.import_module("torch_npu")
    except Exception as exc:  # pragma: no cover - host dependent
        raise RuntimeError("torch_npu is required for tests_new NPU operator tests") from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU device is not available")
    torch.npu.set_device(int(device_id))
    return torch.device(f"npu:{int(device_id)}")


def dtype_from_name(torch: Any, name: str):
    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    try:
        return mapping[str(name)]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype in case manifest: {name}") from exc


def seed_case(torch: Any, case: dict[str, Any]) -> None:
    torch.manual_seed(int(case.get("seed", 0)))


def selected_device_id(default: int = 0) -> int:
    return int(os.environ.get("TEST_DEVICE_ID", default))
