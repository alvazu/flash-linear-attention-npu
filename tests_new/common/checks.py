"""tests_new 的通用输出检查。

这里故意只保留两个层次：
- shape/finite：所有 route、accuracy、perf 都会做的轻量检查；
- ct.dual：accuracy 可选的精度对比，和旧 chunk_bwd_dqkwg 测试习惯保持一致。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def flatten_outputs(outputs: Any) -> list[Any]:
    """把 tuple/list 形式的算子输出拍平成 tensor 列表。"""

    if outputs is None:
        return []
    if isinstance(outputs, (tuple, list)):
        flat = []
        for item in outputs:
            flat.extend(flatten_outputs(item))
        return flat
    return [outputs]


def assert_shapes(outputs: Any, expected_shapes: Sequence[Sequence[int]]) -> None:
    """检查输出 shape 是否符合算子契约。"""

    tensors = flatten_outputs(outputs)
    if len(tensors) < len(expected_shapes):
        raise AssertionError(f"expected {len(expected_shapes)} outputs, got {len(tensors)}")
    for index, expected in enumerate(expected_shapes):
        actual = tuple(int(value) for value in tensors[index].shape)
        expected_tuple = tuple(int(value) for value in expected)
        if actual != expected_tuple:
            raise AssertionError(f"output[{index}] shape {actual} != {expected_tuple}")


def assert_finite(torch: Any, outputs: Any) -> None:
    """检查浮点输出不含 NaN/Inf。"""

    for index, tensor in enumerate(flatten_outputs(outputs)):
        if tensor.is_floating_point() and tensor.numel():
            if not bool(torch.isfinite(tensor).all().item()):
                raise AssertionError(f"output[{index}] contains NaN or Inf")


def assert_dual_outputs(
    torch: Any,
    outputs: Any,
    truth_float64: Any,
    control_outputs: Any,
    compare_dtype: str,
) -> None:
    """使用 ct.dual 做精度对比。

    对每个输出执行：
        ct.dual(actual.to(torch.xxx), truth_float64.to(torch.xxx), control)['success']

    actual 是待测 NPU 输出；truth_float64 是 float64 标杆计算结果；
    control_outputs 是普通精度 CPU 对照值，用于 ct.dual 的三方比较。
    """

    import ct

    compare_torch_dtype = getattr(torch, compare_dtype)
    actual_tensors = flatten_outputs(outputs)
    truth_tensors = flatten_outputs(truth_float64)
    control_tensors = flatten_outputs(control_outputs)
    if len(actual_tensors) != len(truth_tensors) or len(actual_tensors) != len(control_tensors):
        raise AssertionError(
            "output/control/truth count mismatch: "
            f"{len(actual_tensors)}, {len(control_tensors)}, {len(truth_tensors)}"
        )

    for index, (actual, truth, control) in enumerate(zip(actual_tensors, truth_tensors, control_tensors)):
        actual_cmp = actual.detach().cpu().to(compare_torch_dtype)
        truth_cmp = truth.detach().cpu().to(compare_torch_dtype)
        control_cmp = control.detach().cpu()
        result = ct.dual(actual_cmp, truth_cmp, control_cmp)
        if not result["success"]:
            raise AssertionError(f"ct.dual failed for output[{index}]: {result}")
