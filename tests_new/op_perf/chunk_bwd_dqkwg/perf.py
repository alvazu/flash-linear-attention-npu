"""chunk_bwd_dqkwg 的同步 wall-time smoke 性能脚本。"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests_new.common.case_loader import perf_config, select_cases
from tests_new.common.checks import assert_finite, assert_shapes
from tests_new.common.runtime import require_npu


def _parse_args() -> argparse.Namespace:
    """perf 也支持 --case 多选；不指定时跑 config.yaml 中 perf.case_ids 的全部。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--soc")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = require_npu(args.device)
    torch = importlib.import_module("torch")
    golden = importlib.import_module("tests_new.op_golden.chunk_bwd_dqkwg.golden")
    cases = select_cases(
        "chunk_bwd_dqkwg",
        test_type="perf",
        case_ids=args.case,
        soc=args.soc,
    )
    if not cases:
        raise RuntimeError("no chunk_bwd_dqkwg perf cases selected")

    for case in cases:
        cfg = perf_config(case)
        warmup = int(cfg.get("warmup", 3))
        repeats = int(cfg.get("repeats", 10))
        inputs = golden.build_inputs(torch, case, device)
        for _ in range(warmup):
            outputs = golden.run_ascendc(torch, inputs, case)
            torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            outputs = golden.run_ascendc(torch, inputs, case)
            torch.npu.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
        assert_shapes(outputs, golden.expected_output_shapes(case))
        assert_finite(torch, outputs)
        print(f"[PERF] chunk_bwd_dqkwg/{case['id']}: {elapsed_ms:.3f} ms/run repeats={repeats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
