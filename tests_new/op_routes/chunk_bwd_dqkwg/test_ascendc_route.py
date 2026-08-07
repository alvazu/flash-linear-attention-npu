"""Smoke test the stable fla_npu.ops.ascendc.chunk_bwd_dqkwg route."""

from tests_new.common.route_runner import parse_route_args, run_route


if __name__ == "__main__":
    raise SystemExit(run_route("chunk_bwd_dqkwg", "ascendc", parse_route_args("ascendc")))
