"""通过 fla_npu.ops.ascendc.<op> 运行 accuracy/generalization 用例。"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests_new.common.case_loader import accuracy_config, select_cases
from tests_new.common.checks import assert_dual_outputs, assert_finite, assert_shapes
from tests_new.common.runtime import require_npu


def _parse_args() -> argparse.Namespace:
    """解析命令行参数；--case 可重复，也支持逗号多选。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--soc")
    parser.add_argument("--skip-compare", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = require_npu(args.device)
    torch = importlib.import_module("torch")
    golden = importlib.import_module(f"tests_new.op_golden.{args.op}.golden")

    cases = select_cases(
        args.op,
        test_type="accuracy",
        case_ids=args.case,
        soc=args.soc,
    )
    if not cases:
        raise RuntimeError(f"no {args.op} accuracy cases selected")

    for case in cases:
        inputs = golden.build_inputs(torch, case, device)
        outputs = golden.run_ascendc(torch, inputs, case)
        torch.npu.synchronize()
        assert_shapes(outputs, golden.expected_output_shapes(case))
        assert_finite(torch, outputs)
        if not args.skip_compare:
            # control 是普通精度 CPU 对照；truth 是 float64 标杆。
            control = golden.run_reference(torch, inputs, case, benchmark=False)
            truth = golden.run_reference(torch, inputs, case, benchmark=True)
            compare_dtype = accuracy_config(case).get("compare_dtype", "float16")
            assert_dual_outputs(torch, outputs, truth, control, compare_dtype)
        print(f"[PASS] {args.op}/{case['id']} accuracy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
