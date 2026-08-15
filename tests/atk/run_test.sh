#!/usr/bin/env bash
set -euo pipefail

# 单算子 ATK 一键验证脚本。
# 本脚本只负责调度 ATK 与 mssanitizer 包裹的 ATK 命令，不直接调用算子私有测试入口。

show_usage() {
  cat <<'EOF'
用法：
  bash tests/atk/run_test.sh -op=chunk_kda_fwd -npu_device_id=<物理NPU卡号> -gpu_host=<GPU节点地址> -gpu_port=<GPU节点端口>

常用参数：
  -op=chunk_kda_fwd            当前仅支持 chunk_kda_fwd
  -npu_device_id=0             物理 NPU 卡号；脚本会设置 ASCEND_RT_VISIBLE_DEVICES，并在 ATK 中使用逻辑设备 0
  -gpu_host=127.0.0.1          GPU ATK server 的宿主机地址
  -gpu_port=9090               GPU ATK server 对外端口
  -gpu_device_id=0             GPU ATK server 内的逻辑设备号，默认 0
  -soc=ascend910b              可选：ascend910b/A2 或 ascend950/A5；默认自动识别，识别失败时按 A2
  -scope=all                   可选：all、accuracy、performance、determinism、mssanitizer

常用环境变量：
  ATK_ENV                      ATK 虚拟环境目录，设置后自动 source "$ATK_ENV/bin/activate"
  CANN_ENV                     CANN set_env.sh 路径，设置后自动 source
  FLA_NPU_ENV                  fla_npu_transformer set_env.bash 路径，设置后自动 source
  ATK_OUTPUT_ROOT              输出根目录，默认写入当前算子的 atk_output/run_test_<时间戳>
  ATK_TIMEOUT                  ATK 超时时间，默认 2000
  PERFORMANCE_DATA             性能参数，默认 20,100,80
  DC_LOOP_NUMS                 确定性循环次数，默认使用 ATK 默认值
  MSS_TOOLS                    mssanitizer 工具列表，默认 memcheck racecheck initcheck synccheck

示例：
  bash tests/atk/run_test.sh -op=chunk_kda_fwd -npu_device_id=0 -gpu_host=10.0.0.8 -gpu_port=9090
EOF
}

log_info() {
  echo "[ATK一键验证] $*"
}

die() {
  echo "[ATK一键验证] 错误：$*" >&2
  exit 1
}

source_env_file() {
  local label="$1"
  local file_path="$2"
  if [[ -n "$file_path" && -f "$file_path" ]]; then
    log_info "加载${label}：${file_path}"
    set +u
    # shellcheck source=/dev/null
    source "$file_path"
    set -u
  fi
}

should_run() {
  local stage="$1"
  [[ "$RUN_SCOPE" == "all" || "$RUN_SCOPE" == "$stage" ]]
}

OP=""
NPU_DEVICE_ID="${npu_device_id:-${NPU_DEVICE_ID:-0}}"
GPU_HOST="${gpu_host:-${GPU_HOST:-}}"
GPU_PORT="${gpu_port:-${GPU_HOST_PORT:-${GPU_PORT:-}}}"
GPU_DEVICE_ID="${gpu_device_id:-${GPU_DEVICE_ID:-0}}"
SOC="${soc:-${SOC:-auto}}"
RUN_SCOPE="${run_scope:-${RUN_SCOPE:-all}}"
ATK_TIMEOUT="${ATK_TIMEOUT:-2000}"
PERFORMANCE_DATA="${PERFORMANCE_DATA:-20,100,80}"
MSS_TOOLS="${MSS_TOOLS:-memcheck racecheck initcheck synccheck}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -op=*)
      OP="${1#-op=}"
      ;;
    -op)
      shift
      OP="${1:-}"
      ;;
    --op=*)
      OP="${1#--op=}"
      ;;
    --op)
      shift
      OP="${1:-}"
      ;;
    -npu_device_id=*)
      NPU_DEVICE_ID="${1#-npu_device_id=}"
      ;;
    -npu_device_id)
      shift
      NPU_DEVICE_ID="${1:-}"
      ;;
    --npu_device_id=*)
      NPU_DEVICE_ID="${1#--npu_device_id=}"
      ;;
    --npu_device_id)
      shift
      NPU_DEVICE_ID="${1:-}"
      ;;
    -gpu_host=*)
      GPU_HOST="${1#-gpu_host=}"
      ;;
    -gpu_host)
      shift
      GPU_HOST="${1:-}"
      ;;
    --gpu_host=*)
      GPU_HOST="${1#--gpu_host=}"
      ;;
    --gpu_host)
      shift
      GPU_HOST="${1:-}"
      ;;
    -gpu_port=*)
      GPU_PORT="${1#-gpu_port=}"
      ;;
    -gpu_port)
      shift
      GPU_PORT="${1:-}"
      ;;
    --gpu_port=*)
      GPU_PORT="${1#--gpu_port=}"
      ;;
    --gpu_port)
      shift
      GPU_PORT="${1:-}"
      ;;
    -gpu_device_id=*)
      GPU_DEVICE_ID="${1#-gpu_device_id=}"
      ;;
    -gpu_device_id)
      shift
      GPU_DEVICE_ID="${1:-}"
      ;;
    --gpu_device_id=*)
      GPU_DEVICE_ID="${1#--gpu_device_id=}"
      ;;
    --gpu_device_id)
      shift
      GPU_DEVICE_ID="${1:-}"
      ;;
    -soc=*)
      SOC="${1#-soc=}"
      ;;
    -soc)
      shift
      SOC="${1:-}"
      ;;
    --soc=*)
      SOC="${1#--soc=}"
      ;;
    --soc)
      shift
      SOC="${1:-}"
      ;;
    -scope=*)
      RUN_SCOPE="${1#-scope=}"
      ;;
    -scope)
      shift
      RUN_SCOPE="${1:-}"
      ;;
    --scope=*)
      RUN_SCOPE="${1#--scope=}"
      ;;
    --scope)
      shift
      RUN_SCOPE="${1:-}"
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    *)
      show_usage
      die "未知参数：$1"
      ;;
  esac
  shift
done

[[ -n "$OP" ]] || die "必须传入 -op=chunk_kda_fwd"
[[ "$OP" == "chunk_kda_fwd" ]] || die "当前脚本仅支持 -op=chunk_kda_fwd，实际收到：${OP}"
[[ -n "$NPU_DEVICE_ID" ]] || die "必须设置 -npu_device_id"

case "$RUN_SCOPE" in
  all|accuracy|performance|determinism|mssanitizer)
    ;;
  *)
    die "不支持的执行范围：${RUN_SCOPE}"
    ;;
esac

if should_run accuracy || should_run determinism; then
  [[ -n "$GPU_HOST" ]] || die "必须设置 -gpu_host，双标杆精度和确定性验证需要 GPU ATK server"
  [[ -n "$GPU_PORT" ]] || die "必须设置 -gpu_port，双标杆精度和确定性验证需要 GPU ATK server"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OP_DIR="${SCRIPT_DIR}/${OP}"

[[ -d "$OP_DIR" ]] || die "找不到 ATK 算子目录：${OP_DIR}"
[[ -f "${OP_DIR}/atk_${OP}.json" ]] || die "找不到 ATK case 文件：${OP_DIR}/atk_${OP}.json"
[[ -f "${OP_DIR}/executor_${OP}.py" ]] || die "找不到 ATK executor：${OP_DIR}/executor_${OP}.py"

source_env_file "ATK虚拟环境" "${ATK_ENV:-}/bin/activate"
source_env_file "CANN环境" "${CANN_ENV:-}"
source_env_file "fla_npu_transformer环境" "${FLA_NPU_ENV:-${FLA_NPU_OPP_ENV:-}}"

export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICE_ID"
export PYTHONPATH="${REPO_ROOT}/torch_custom/fla_npu:${REPO_ROOT}:${PYTHONPATH:-}"

if [[ "$SOC" == "auto" ]]; then
  SOC_INFO=""
  if command -v npu-smi >/dev/null 2>&1; then
    SOC_INFO="$(npu-smi info -i "$NPU_DEVICE_ID" 2>/dev/null || true)"
  fi
  if echo "$SOC_INFO" | grep -Eiq 'ascend950|(^|[^0-9])950([^0-9]|$)'; then
    SOC="ascend950"
  elif echo "$SOC_INFO" | grep -Eiq 'ascend910b|910b'; then
    SOC="ascend910b"
  else
    SOC="ascend910b"
    log_info "警告：未能自动识别 SOC，默认使用 A2/ascend910b 用例范围；如需 A5 请传 -soc=ascend950"
  fi
fi

case "$SOC" in
  a2|A2|ascend910b)
    SOC="ascend910b"
    DEFAULT_PERFORMANCE_CASES="[0,16]"
    DEFAULT_DETERMINISM_CASES="[4,18]"
    DEFAULT_MSS_CASES="[8,16]"
    ;;
  a5|A5|ascend950)
    SOC="ascend950"
    DEFAULT_PERFORMANCE_CASES="[250,266]"
    DEFAULT_DETERMINISM_CASES="[254,268]"
    DEFAULT_MSS_CASES="[258,266]"
    ;;
  *)
    die "不支持的 SOC：${SOC}，请使用 ascend910b/A2 或 ascend950/A5"
    ;;
esac

PERFORMANCE_CASES="${PERFORMANCE_CASES:-$DEFAULT_PERFORMANCE_CASES}"
DETERMINISM_CASES="${DETERMINISM_CASES:-$DEFAULT_DETERMINISM_CASES}"
MSS_CASES="${MSS_CASES:-$DEFAULT_MSS_CASES}"

if should_run accuracy; then
  read -r JSON_ACCURACY_START JSON_ACCURACY_END JSON_ACCURACY_COUNT < <(
    python3 - "$OP_DIR/atk_${OP}.json" "$SOC" <<'PY'
import json
import sys

case_file, soc = sys.argv[1], sys.argv[2]
with open(case_file, encoding="utf-8") as f:
    cases = json.load(f)

def input_value(case, name):
    return next(item["range_values"] for item in case["inputs"] if item["name"] == name)

ids = [
    case["id"]
    for case in cases
    if input_value(case, "soc") == soc and case.get("expected_error_msg") is None
]
if not ids:
    print("0 0 0")
else:
    print(min(ids), max(ids) + 1, len(ids))
PY
  )
  [[ "$JSON_ACCURACY_COUNT" -gt 0 ]] || die "case JSON 中没有 ${SOC} 的正向精度用例"
  ACCURACY_START="${ACCURACY_START:-$JSON_ACCURACY_START}"
  ACCURACY_END="${ACCURACY_END:-$JSON_ACCURACY_END}"
fi

ATK_BIN="$(command -v atk || true)"
[[ -n "$ATK_BIN" ]] || die "找不到 atk，请先安装并激活 ATK 环境"

OUTPUT_ROOT="${ATK_OUTPUT_ROOT:-${OP_DIR}/atk_output/run_test_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_ROOT"
cd "$OP_DIR"

log_info "算子：${OP}"
log_info "SOC：${SOC}"
log_info "物理 NPU 卡号：${NPU_DEVICE_ID}；ATK 逻辑 NPU 设备号：0"
if should_run accuracy; then
  log_info "精度 case 范围：-s ${ACCURACY_START} -e ${ACCURACY_END}"
fi
if should_run accuracy || should_run determinism; then
  log_info "GPU ATK server：${GPU_HOST}:${GPU_PORT}，GPU 逻辑设备号：${GPU_DEVICE_ID}"
fi
log_info "输出根目录：${OUTPUT_ROOT}"
log_info "ATK 路径：${ATK_BIN}"
"$ATK_BIN" --version || die "atk --version 执行失败"

if should_run accuracy; then
  log_info "开始双标杆精度验证：GPU 高精度真值 + GPU 同精度对照"
  "$ATK_BIN" node --name npu_dut --backend npu \
      --devices 0 \
      --output_path "${OUTPUT_ROOT}/accuracy_dual_gpu" \
    node --name gpu_reference --backend gpu \
      --host "$GPU_HOST" \
      --port "$GPU_PORT" \
      --devices "$GPU_DEVICE_ID" \
      --is_compare true \
      --output_path "${OUTPUT_ROOT}/accuracy_dual_gpu" \
    task \
      -c "./atk_${OP}.json" \
      --task accuracy \
      --bm_device gpu \
      -p "./executor_${OP}.py" \
      -s "$ACCURACY_START" \
      -e "$ACCURACY_END" \
      --syc_dataset \
      -mt 1 \
      -to "$ATK_TIMEOUT"
  log_info "完成双标杆精度验证"
fi

if should_run performance; then
  log_info "开始性能对比验证：使用 ATK performance_device 与 device profiler"
  "$ATK_BIN" node --name npu_dut --backend npu \
      --devices 0 \
      --output_path "${OUTPUT_ROOT}/performance_device" \
    task \
      -c "./atk_${OP}.json" \
      --task performance_device \
      -p "./executor_${OP}.py" \
      -wl "$PERFORMANCE_CASES" \
      --performance_data "$PERFORMANCE_DATA" \
      --save_data profile \
      -sp \
      -to "$ATK_TIMEOUT"
  log_info "完成性能对比验证"
fi

if should_run determinism; then
  log_info "开始确定性验证：使用 ATK accuracy_dc，并开启 --gm_init_flag"
  DC_ARGS=()
  if [[ -n "${DC_LOOP_NUMS:-}" ]]; then
    DC_ARGS=(--dc_loop_nums "$DC_LOOP_NUMS")
  fi
  "$ATK_BIN" node --name npu_dut --backend npu \
      --devices 0 \
      --output_path "${OUTPUT_ROOT}/determinism_gm_init" \
    node --name gpu_reference --backend gpu \
      --host "$GPU_HOST" \
      --port "$GPU_PORT" \
      --devices "$GPU_DEVICE_ID" \
      --is_compare true \
      --output_path "${OUTPUT_ROOT}/determinism_gm_init" \
    task \
      -c "./atk_${OP}.json" \
      --task accuracy_dc \
      --bm_device gpu \
      -p "./executor_${OP}.py" \
      -wl "$DETERMINISM_CASES" \
      --gm_init_flag \
      "${DC_ARGS[@]}" \
      --syc_dataset \
      -mt 1 \
      -to "$ATK_TIMEOUT"
  log_info "完成确定性验证"
fi

if should_run mssanitizer; then
  command -v mssanitizer >/dev/null 2>&1 || die "找不到 mssanitizer，请先加载支持 sanitizer 的 CANN/调试环境"
  for MSS_TOOL in $MSS_TOOLS; do
    log_info "开始 mssanitizer 内存检测：${MSS_TOOL}"
    mssanitizer --tool="$MSS_TOOL" --log-file "${OUTPUT_ROOT}/mssanitizer_${MSS_TOOL}.log" -- \
      "$ATK_BIN" node --name npu_dut --backend npu \
        --devices 0 \
        --output_path "${OUTPUT_ROOT}/mssanitizer_${MSS_TOOL}" \
      task \
        -c "./atk_${OP}.json" \
        --task run \
        -p "./executor_${OP}.py" \
        -wl "$MSS_CASES" \
        -sp \
        -to "$ATK_TIMEOUT"
    log_info "完成 mssanitizer 内存检测：${MSS_TOOL}"
  done
fi

log_info "全部请求的 ATK 验证动作执行完成"
