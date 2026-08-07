"""Smoke test the fast-kernel direct <<<>>> route through ascend_ops."""

from tests_new.common.route_runner import parse_route_args, run_route


if __name__ == "__main__":
    raise SystemExit(run_route("chunk_bwd_dqkwg", "direct_launch", parse_route_args("direct_launch")))
