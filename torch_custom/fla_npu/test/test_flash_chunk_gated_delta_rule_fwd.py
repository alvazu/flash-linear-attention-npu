#!/usr/bin/env python3
"""Tests for examples.flash_gated_delta_rule.flash_chunk_gated_delta_rule_fwd."""

from __future__ import annotations

import ast
import os
import re
import sys
import unittest
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_SOURCE = REPO_ROOT / "examples" / "flash_gated_delta_rule.py"
ASCENDC_WRAPPER_SOURCE = REPO_ROOT / "torch_custom" / "fla_npu" / "fla_npu" / "ops" / "ascendc" / "__init__.py"
OPAPI_SOURCE = REPO_ROOT / "torch_custom" / "fla_npu" / "op_plugin" / "ops" / "opapi" / "FLANpuOpApi.cpp"

ASCENDC_OPAPI = {
    "npu_chunk_local_cumsum": "aclnnChunkLocalCumsum",
    "npu_chunk_scaled_dot_kkt": "aclnnChunkScaledDotKkt",
    "npu_solve_tri": "aclnnSolveTri",
    "npu_recompute_w_u_fwd": "aclnnRecomputeWUFwd",
    "npu_chunk_gated_delta_rule_fwd_h": "aclnnChunkGatedDeltaRuleFwdH",
    "npu_chunk_fwd_o": "aclnnChunkFwdO",
}

ASCENDC_PYTHON_WRAPPERS = tuple(ASCENDC_OPAPI)

FLASH_FWD_REQUIRED_CALLS = (
    "ascendc_chunk_local_cumsum",
    "ascendc_chunk_scaled_dot_kkt",
    "solve_tri_auto",
    "recompute_w_u",
    "ascendc_chunk_gated_delta_rule_fwd_h",
    "ascendc_chunk_fwd_o",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_source(path: Path, name: str) -> str:
    source = _read(path)
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"Function {name!r} was not found in {path}")


class FlashChunkGatedDeltaRuleFwdAscendCTest(unittest.TestCase):
    maxDiff = None

    def test_python_entrypoint_routes_to_expected_ascendc_ops(self):
        fwd_source = _function_source(EXAMPLE_SOURCE, "flash_chunk_gated_delta_rule_fwd")
        missing = [call for call in FLASH_FWD_REQUIRED_CALLS if call not in fwd_source]
        self.assertFalse(missing, f"flash_chunk_gated_delta_rule_fwd is missing calls: {missing}")

        solve_tri_source = _function_source(EXAMPLE_SOURCE, "solve_tri_ascendc")
        self.assertIn("ascendc_solve_tri", solve_tri_source)

        recompute_source = _function_source(EXAMPLE_SOURCE, "recompute_w_u")
        self.assertIn("ascendc_recompute_w_u_fwd", recompute_source)

        wrapper_source = _read(ASCENDC_WRAPPER_SOURCE)
        for op_name in ASCENDC_PYTHON_WRAPPERS:
            self.assertIn(f'"{op_name}"', wrapper_source)
        self.assertIn("return _get_torch_op(name)(*args, **kwargs)", wrapper_source)

    def test_opapi_functions_call_aclnn_ascendc_interfaces(self):
        cpp_source = _read(OPAPI_SOURCE)
        function_positions = {
            op_name: cpp_source.find(f"{op_name}(")
            for op_name in ASCENDC_OPAPI
        }
        missing_defs = [op_name for op_name, pos in function_positions.items() if pos < 0]
        self.assertFalse(missing_defs, f"Missing C++ opapi function definitions: {missing_defs}")

        ordered_positions = sorted(function_positions.items(), key=lambda item: item[1])
        for idx, (op_name, start) in enumerate(ordered_positions):
            end = ordered_positions[idx + 1][1] if idx + 1 < len(ordered_positions) else len(cpp_source)
            body = cpp_source[start:end]
            aclnn_name = ASCENDC_OPAPI[op_name]
            self.assertIn("EXEC_NPU_CMD_EXT", body, f"{op_name} is not dispatched through EXEC_NPU_CMD_EXT")
            self.assertRegex(
                body,
                re.compile(rf"EXEC_NPU_CMD_EXT\s*\(\s*{re.escape(aclnn_name)}\b", re.S),
                f"{op_name} does not call {aclnn_name}",
            )

    def test_runtime_forward_does_not_use_triton_solve_fallback(self):
        try:
            import torch
            import torch_npu  # noqa: F401
            import fla_npu  # noqa: F401
        except Exception as exc:
            raise unittest.SkipTest(f"runtime test requires torch_npu and fla_npu: {exc}") from exc

        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise unittest.SkipTest("runtime test requires an available NPU device")

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from examples import flash_gated_delta_rule as flash_gdr

        device_id = int(os.environ.get("TEST_DEVICE_ID", "0"))
        torch.npu.utils.set_device(device_id)

        missing_ops = [op_name for op_name in ASCENDC_OPAPI if not hasattr(torch.ops.npu, op_name)]
        self.assertFalse(missing_ops, f"torch.ops.npu is missing custom ops: {missing_ops}")

        B, H, T, K, V, BT = 1, 2, 64, 128, 128, 64
        torch.manual_seed(2026)
        dtype = torch.float16
        q = (torch.randn(B, H, T, K, device="npu", dtype=dtype) * 0.02).contiguous()
        k = (torch.randn(B, H, T, K, device="npu", dtype=dtype) * 0.02).contiguous()
        v = (torch.randn(B, H, T, V, device="npu", dtype=dtype) * 0.02).contiguous()
        g = (torch.randn(B, T, H, device="npu", dtype=torch.float32) * 0.01).contiguous()
        beta = torch.sigmoid(torch.randn(B, T, H, device="npu", dtype=torch.float32)).contiguous()

        flash_gdr._SOLVE_TRI_ASCENDC_AVAILABLE = None
        flash_gdr._SOLVE_TRI_ASCENDC_UNAVAILABLE_REASON = ""

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            g_out, o, A, final_state = flash_gdr.flash_chunk_gated_delta_rule_fwd(
                q,
                k,
                v,
                g,
                beta,
                K**-0.5,
                None,
                True,
                chunk_size=BT,
            )
            torch.npu.synchronize()

        self.assertEqual(tuple(g_out.shape), (B, T, H))
        self.assertEqual(tuple(o.shape), (B, T, H, V))
        self.assertEqual(tuple(A.shape), (B, H, T, BT))
        self.assertEqual(tuple(final_state.shape), (B, H, K, V))
        self.assertEqual(g_out.dtype, torch.float32)
        self.assertEqual(o.dtype, dtype)
        self.assertEqual(A.dtype, dtype)
        self.assertEqual(final_state.dtype, dtype)

        fallback_warnings = [
            str(item.message)
            for item in caught
            if "falling back to Triton solve_tril_npu" in str(item.message)
        ]
        self.assertFalse(
            fallback_warnings,
            "flash_chunk_gated_delta_rule_fwd used the Triton solve fallback; "
            f"expected all runtime ops to stay on AscendC. warnings={fallback_warnings}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
