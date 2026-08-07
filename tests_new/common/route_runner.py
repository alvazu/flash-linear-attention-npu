"""三条调用通路的公共 smoke runner。"""

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


def parse_route_args(route: str) -> argparse.Namespace:
    """route 脚本共用的参数解析。"""

    parser = argparse.ArgumentParser(description=f"Run {route} route smoke cases")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--soc")
    parser.add_argument("--compare-golden", action="store_true")
    return parser.parse_args()


def run_route(op_name: str, route: str, args: argparse.Namespace) -> int:
    """运行单条 route。

    routes 未显式传 --case 时只取 config.yaml 中 route case 列表的第一个；
    显式传 --case 时按用户给出的多个 case 运行。
    """

    device = require_npu(args.device)
    torch = importlib.import_module("torch")
    golden = importlib.import_module(f"tests_new.op_golden.{op_name}.golden")
    cases = select_cases(
        op_name,
        test_type="routes",
        case_ids=args.case,
        default_first=True,
        soc=args.soc,
    )
    if not cases:
        raise RuntimeError(f"no {op_name} route cases selected for route={route}")

    route_func = getattr(golden, f"run_{route}")
    for case in cases:
        inputs = golden.build_inputs(torch, case, device)
        outputs = route_func(torch, inputs, case)
        torch.npu.synchronize()
        assert_shapes(outputs, golden.expected_output_shapes(case))
        assert_finite(torch, outputs)
        if args.compare_golden:
            control = golden.run_reference(torch, inputs, case, benchmark=False)
            truth = golden.run_reference(torch, inputs, case, benchmark=True)
            compare_dtype = accuracy_config(case).get("compare_dtype", "float16")
            assert_dual_outputs(torch, outputs, truth, control, compare_dtype)
        print(f"[PASS] {op_name}/{case['id']} route={route}")
    return 0
