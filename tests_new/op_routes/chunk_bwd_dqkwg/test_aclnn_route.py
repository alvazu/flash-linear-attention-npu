"""Smoke test the generated ACLNN op_api route through torch.ops.npu."""

from tests_new.common.route_runner import parse_route_args, run_route


if __name__ == "__main__":
    raise SystemExit(run_route("chunk_bwd_dqkwg", "aclnn", parse_route_args("aclnn")))
