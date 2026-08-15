#!/usr/bin/env bash
set -euo pipefail

# 单算子 ATK CPU 标杆一键验证脚本。
# 所有测试动作均由 ATK 发起；内存检测由 mssanitizer 包裹 ATK run 任务。

show_usage() {
  cat <<'EOF'
用法：
  bash tests/atk/run_test_cpu.sh -op=<算子名> -npu_device_id=<NPU卡号>

常用参数：
  -op=chunk_kda_fwd              ATK 算子目录名
  -npu_device_id=6               传给 ATK node --devices 的 NPU 卡号
  -soc=ascend910b                可选：ascend910b/A2、ascend910_93/A3、ascend950/A5
  -scope=all                     可选：all、accuracy、performance、determinism、mssanitizer

常用环境变量：
  ATK_ENV                        ATK 虚拟环境目录，设置后 source "$ATK_ENV/bin/activate"
  CANN_ENV                       CANN set_env.sh 路径，设置后 source
  FLA_NPU_ENV                    fla_npu_transformer set_env.bash 路径，设置后 source
  ATK_OUTPUT_ROOT                输出根目录，默认 ./atk_output
  ATK_TIMEOUT                    精度阶段超时，默认 14400
  PERFORMANCE_TIMEOUT            性能阶段超时，默认 2000
  CASE_START/CASE_END            通用 case 顺序范围，默认 0/1，ATK 执行 [start, end)
  ACCURACY_START/ACCURACY_END    精度与 NaN 检测 case 范围
  PERFORMANCE_START/END          性能 case 范围
  DETERMINISM_START/END          确定性 case 范围
  MSS_START/MSS_END              mssanitizer case 范围
  MSS_TOOL                       mssanitizer 工具，默认 memcheck
  MSS_LOG_PATH                   ATK -msl 日志路径，默认使用脚本内置绝对路径

示例：
  bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=6
  CASE_START=0 CASE_END=1 bash tests/atk/run_test_cpu.sh -op=chunk_bwd_dqkwg -npu_device_id=6
EOF
}

log_info() {
  echo "[ATK CPU标杆验证] $*"
}

die() {
  echo "[ATK CPU标杆验证] 错误：$*" >&2
  exit 1
}

source_env_file() {
  local label="$1"
  local file_path="$2"
  if [[ -f "$file_path" ]]; then
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
NPU_DEVICE_ID="${NPU_DEVICE_ID:-}"
SOC="${SOC:-auto}"
RUN_SCOPE="${RUN_SCOPE:-all}"
ATK_TIMEOUT="${ATK_TIMEOUT:-14400}"
PERFORMANCE_TIMEOUT="${PERFORMANCE_TIMEOUT:-2000}"
CASE_START="${CASE_START:-0}"
CASE_END="${CASE_END:-1}"
MSS_TOOL="${MSS_TOOL:-memcheck}"
MSS_LOG_PATH="${MSS_LOG_PATH:-/home/huangjunzhe/gdn/github/alvazu-atk/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/tests/ATK/log.txt}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -op=*) OP="${1#-op=}" ;;
    -op)
      shift
      [[ $# -gt 0 ]] || die "参数 -op 需要取值"
      OP="$1"
      ;;
    --op=*) OP="${1#--op=}" ;;
    --op)
      shift
      [[ $# -gt 0 ]] || die "参数 --op 需要取值"
      OP="$1"
      ;;
    -npu_device_id=*) NPU_DEVICE_ID="${1#-npu_device_id=}" ;;
    -npu_device_id)
      shift
      [[ $# -gt 0 ]] || die "参数 -npu_device_id 需要取值"
      NPU_DEVICE_ID="$1"
      ;;
    --npu_device_id=*) NPU_DEVICE_ID="${1#--npu_device_id=}" ;;
    --npu_device_id)
      shift
      [[ $# -gt 0 ]] || die "参数 --npu_device_id 需要取值"
      NPU_DEVICE_ID="$1"
      ;;
    -soc=*) SOC="${1#-soc=}" ;;
    -soc)
      shift
      [[ $# -gt 0 ]] || die "参数 -soc 需要取值"
      SOC="$1"
      ;;
    --soc=*) SOC="${1#--soc=}" ;;
    --soc)
      shift
      [[ $# -gt 0 ]] || die "参数 --soc 需要取值"
      SOC="$1"
      ;;
    -scope=*) RUN_SCOPE="${1#-scope=}" ;;
    -scope)
      shift
      [[ $# -gt 0 ]] || die "参数 -scope 需要取值"
      RUN_SCOPE="$1"
      ;;
    --scope=*) RUN_SCOPE="${1#--scope=}" ;;
    --scope)
      shift
      [[ $# -gt 0 ]] || die "参数 --scope 需要取值"
      RUN_SCOPE="$1"
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

[[ -n "$OP" ]] || die "必须传入 -op=<算子名>"
[[ -n "$NPU_DEVICE_ID" ]] || die "必须传入 -npu_device_id=<NPU卡号>"

case "$RUN_SCOPE" in
  all|accuracy|performance|determinism|mssanitizer) ;;
  *) die "不支持的执行范围：${RUN_SCOPE}" ;;
esac

case "$SOC" in
  auto) ;;
  a2|A2|ascend910b) SOC="ascend910b" ;;
  a3|A3|ascend910_93) SOC="ascend910_93" ;;
  a5|A5|ascend950) SOC="ascend950" ;;
  *) die "不支持的 SOC：${SOC}，请使用 ascend910b/A2、ascend910_93/A3 或 ascend950/A5" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OP_DIR="${SCRIPT_DIR}/${OP}"

[[ -d "$OP_DIR" ]] || die "找不到 ATK 算子目录：${OP_DIR}"
[[ -f "${OP_DIR}/atk_${OP}.json" ]] || die "找不到 ATK 用例文件：${OP_DIR}/atk_${OP}.json"
[[ -f "${OP_DIR}/executor_${OP}.py" ]] || die "找不到 ATK 执行器：${OP_DIR}/executor_${OP}.py"

if [[ -n "${ATK_ENV:-}" ]]; then
  source_env_file "ATK虚拟环境" "${ATK_ENV}/bin/activate"
fi
if [[ -n "${CANN_ENV:-}" ]]; then
  source_env_file "CANN环境" "$CANN_ENV"
fi
if [[ -n "${FLA_NPU_ENV:-${FLA_NPU_OPP_ENV:-}}" ]]; then
  source_env_file "fla_npu_transformer环境" "${FLA_NPU_ENV:-${FLA_NPU_OPP_ENV:-}}"
fi

export PYTHONPATH="${REPO_ROOT}/torch_custom/fla_npu:${REPO_ROOT}:${PYTHONPATH:-}"

ATK_BIN="$(command -v atk || true)"
[[ -n "$ATK_BIN" ]] || die "找不到 atk，请先安装并激活 ATK 环境"

ACCURACY_START="${ACCURACY_START:-$CASE_START}"
ACCURACY_END="${ACCURACY_END:-$CASE_END}"
PERFORMANCE_START="${PERFORMANCE_START:-$CASE_START}"
PERFORMANCE_END="${PERFORMANCE_END:-$CASE_END}"
DETERMINISM_START="${DETERMINISM_START:-$CASE_START}"
DETERMINISM_END="${DETERMINISM_END:-$CASE_END}"
MSS_START="${MSS_START:-$CASE_START}"
MSS_END="${MSS_END:-$CASE_END}"

cd "$OP_DIR"
ATK_OUTPUT_ROOT="${ATK_OUTPUT_ROOT:-./atk_output}"
mkdir -p "${ATK_OUTPUT_ROOT}/cpu_dual_reference" "${ATK_OUTPUT_ROOT}/perf"

log_info "算子：${OP}"
log_info "SOC：${SOC}"
log_info "NPU 设备号：${NPU_DEVICE_ID}"
log_info "ATK 路径：${ATK_BIN}"
log_info "输出根目录：${ATK_OUTPUT_ROOT}"
"$ATK_BIN" --version || die "atk --version 执行失败"

if should_run accuracy; then
  log_info "开始精度与 NaN 检测：accuracy + CPU高精度标杆 + CPU同精度标杆 + --gm_init_flag"
  "$ATK_BIN" node --name npu_dut --backend npu --devices "$NPU_DEVICE_ID" \
      --output_path "${ATK_OUTPUT_ROOT}/cpu_dual_reference" \
    node --name cpu_reference --backend cpu \
      --output_path "${ATK_OUTPUT_ROOT}/cpu_dual_reference" \
    task \
      -c "./atk_${OP}.json" \
      --task accuracy \
      --bm_device cpu \
      -p "./executor_${OP}.py" \
      -s "$ACCURACY_START" \
      -e "$ACCURACY_END" \
      --gm_init_flag \
      -sp \
      -mt 1 \
      -to "$ATK_TIMEOUT"
  log_info "完成精度与 NaN 检测"
fi

if should_run performance; then
  log_info "开始性能测试：performance_device"
  "$ATK_BIN" node --name npu_dut --backend npu --devices "$NPU_DEVICE_ID" \
      --output_path "${ATK_OUTPUT_ROOT}/perf" \
    task \
      -c "atk_${OP}.json" \
      --task performance_device \
      -p "executor_${OP}.py" \
      -s "$PERFORMANCE_START" \
      -e "$PERFORMANCE_END" \
      --save_data profile \
      -sp \
      -to "$PERFORMANCE_TIMEOUT"
  log_info "完成性能测试"
fi

if should_run determinism; then
  log_info "开始确定性测试：accuracy_dc"
  "$ATK_BIN" node --name npu_dut --backend npu --devices "$NPU_DEVICE_ID" \
    task \
      -c "atk_${OP}.json" \
      -p "executor_${OP}.py" \
      --task accuracy_dc \
      -s "$DETERMINISM_START" \
      -e "$DETERMINISM_END"
  log_info "完成确定性测试"
fi

if should_run mssanitizer; then
  command -v mssanitizer >/dev/null 2>&1 || die "找不到 mssanitizer，请先加载支持 sanitizer 的 CANN/调试环境"
  log_info "开始内存检测：mssanitizer ${MSS_TOOL}"
  log_info "ATK mssanitizer 日志：${MSS_LOG_PATH}"
  mssanitizer --tool="$MSS_TOOL" -- \
    "$ATK_BIN" node --name npu_dut --backend npu --devices "$NPU_DEVICE_ID" \
    task \
      -c "atk_${OP}.json" \
      -p "executor_${OP}.py" \
      --task run \
      --mssanitizer \
      -msl "$MSS_LOG_PATH" \
      -s "$MSS_START" \
      -e "$MSS_END"
  log_info "完成内存检测"
fi

log_info "请求的 ATK 测试动作已执行完成"
