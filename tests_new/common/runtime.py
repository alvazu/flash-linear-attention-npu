"""NPU 运行时辅助函数。"""

from __future__ import annotations

import importlib
import os
from typing import Any


def get_torch():
    """延迟导入 torch，避免纯静态检查时过早触发 NPU 依赖。"""

    return importlib.import_module("torch")


def require_npu(device_id: int = 0):
    """检查 torch_npu/NPU 是否可用，并切到指定 device。"""

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
    """把 YAML 中的 dtype 字符串转换成 torch dtype。"""

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
    """每个 case 使用固定 seed，方便问题复现。"""

    torch.manual_seed(int(case.get("seed", 0)))


def selected_device_id(default: int = 0) -> int:
    """读取测试 device，保留给单脚本直接调用使用。"""

    return int(os.environ.get("TEST_DEVICE_ID", default))
