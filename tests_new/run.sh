#!/usr/bin/env bash
set -euo pipefail

TEST_DEVICE=0
OP_NAME=""
TEST_TARGETS="routes,accuracy,perf"
CASE_ARGS=()
SOC_ARG=()

usage() {
    cat <<'EOF'
Usage:
  bash tests_new/run.sh --device=0 --op=chunk_bwd_dqkwg --test=routes
  bash tests_new/run.sh --device=0 --op=chunk_bwd_dqkwg --test=accuracy
  bash tests_new/run.sh --device=0 --op=chunk_bwd_dqkwg --test=perf
  bash tests_new/run.sh --device=0 --op=chunk_bwd_dqkwg --test=routes/accuracy/perf

Options:
  --device=N        NPU device id. Default: 0
  --op=NAME        Operator name. Currently adapted: chunk_bwd_dqkwg
  --test=TARGETS   routes, accuracy, perf, all, or slash/comma separated list
  --case=ID        Restrict selected cases. May be repeated.
  --soc=SOC        Restrict by SOC in cases.yaml.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --device=*)
            TEST_DEVICE="${arg#*=}"
            ;;
        --op=*)
            OP_NAME="${arg#*=}"
            ;;
        --test=*)
            TEST_TARGETS="${arg#*=}"
            ;;
        --case=*)
            CASE_ARGS+=("--case" "${arg#*=}")
            ;;
        --soc=*)
            SOC_ARG=("--soc" "${arg#*=}")
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$OP_NAME" ]]; then
    echo "--op is required" >&2
    usage
    exit 2
fi

if [[ "$OP_NAME" != "chunk_bwd_dqkwg" ]]; then
    echo "Operator '$OP_NAME' is not adapted in tests_new yet." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TEST_DEVICE_ID="${TEST_DEVICE}"

normalized="${TEST_TARGETS//\//,}"
if [[ "$normalized" == "all" ]]; then
    normalized="routes,accuracy,perf"
fi
IFS=',' read -ra TARGETS <<< "$normalized"

run_python() {
    local script="$1"
    shift
    python3 "$script" --device "$TEST_DEVICE" "${CASE_ARGS[@]}" "${SOC_ARG[@]}" "$@"
}

for target in "${TARGETS[@]}"; do
    case "$target" in
        routes)
            run_python "${SCRIPT_DIR}/op_routes/${OP_NAME}/test_ascendc_route.py"
            run_python "${SCRIPT_DIR}/op_routes/${OP_NAME}/test_aclnn_route.py"
            run_python "${SCRIPT_DIR}/op_routes/${OP_NAME}/test_direct_launch_route.py"
            ;;
        accuracy)
            run_python "${SCRIPT_DIR}/common/run_accuracy.py" --op "$OP_NAME"
            ;;
        perf)
            run_python "${SCRIPT_DIR}/op_perf/${OP_NAME}/perf.py"
            ;;
        "")
            ;;
        *)
            echo "Unknown --test target: $target" >&2
            usage
            exit 2
            ;;
    esac
done
