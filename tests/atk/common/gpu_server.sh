#!/usr/bin/env bash
set -euo pipefail

# GPU 端 ATK server 一键启动脚本。
# 职责：在 GPU 宿主机上启动 Docker 容器，容器内激活 ATK 环境、校验 CUDA/Triton，
#       随后前台启动 ATK server，监听 0.0.0.0:<port>，供 NPU 端 accuracy_gpu 远程调用。
# action=gpu_server_start 在 GPU 宿主机执行：启动容器并前台运行 ATK server。
# action=npu 在 NPU 机器执行：构造 atk node(本地 NPU) + node(远程 GPU) + task 链，
#       对接已启动的 GPU server，跑 accuracy_gpu 精度对拍。
# action=test_connection_from_npu 在 NPU 机器执行：测试到 GPU server 的连通性。
# 容器内 ATK server 与 NPU 端 action=npu / run_test_cpu.sh -scope=accuracy_gpu 配对使用。

show_usage() {
  cat <<'EOF'
用法：
  bash tests/atk/common/gpu_server.sh -op=<算子名> [选项]

  默认（action=gpu_server_start）在 GPU 宿主机执行：启动容器并前台运行 ATK server。
  action=npu 在 NPU 机器执行：对接远程 GPU server 跑精度对拍（atk node ... task）。
  action=test_connection_from_npu 在 NPU 机器执行：测试到 GPU server 的连通性。

必选参数：
  -op=chunk_kda_fwd              ATK 算子目录名（与 NPU 端 run_test_cpu.sh -op 一致）

常用参数：
  -gpu_host=                    GPU server 宿主机地址；action=test_connection_from_npu 必选
  -gpu_host_port=9090           容器 9090 映射到宿主机的端口，默认 9090；须与 NPU 端 -gpu_host_port 一致
  -gpu_device_id=6               物理 GPU 卡号，默认 6；容器内重新编号为逻辑设备 0
  -gpu_container=fla_gpu_atk    容器名，默认 fla_gpu_atk
  -gpu_image=                   GPU 基础镜像，必选（如 pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel）
  -gpu_image_tar=              本地镜像 tar 文件路径；设置后先 docker load 再用加载出的镜像名创建容器
  -gpu_repo_root=               仓库在容器内的挂载根目录，必选（如 /workspace/flash-linear-attention-npu）
  -action=gpu_server_start       gpu_server_start：启动容器并前台运行 ATK server；
                                  gpu_server_stop：停止并删除容器；
                                  npu：在 NPU 机器构造 atk node+node+task，对接远程 GPU server 跑精度对拍；
                                  test_connection_from_npu：在 NPU 机器测试到 GPU server 连通性
  -npu_device_id=7               NPU 物理卡号，action=npu 时使用，默认 7
  -npu_gpu_device_id=0           远程 GPU 参考节点 --devices，action=npu 时使用，默认 0
  -npu_output_path=./atk_output/kda_remote
                                  action=npu 输出目录，默认 ./atk_output/kda_remote
  -npu_task=accuracy             action=npu 任务类型，默认 accuracy
  -npu_mt=1                      action=npu -mt 值，默认 1
  -npu_task_timeout=2000         action=npu task 超时，默认 2000
  -npu_atk_config=               action=npu atk 配置 json；未设置时用 ./atk_${op}.json
  -npu_executor=                 action=npu executor 文件；未设置时用 ./executor_${op}.py
  -atk_env=                     容器内 ATK 虚拟环境目录，设置后 source "$ATK_ENV/bin/activate"
  -triton_root=                 兼容 Triton 源码根（加入 PYTHONPATH）；未设置则跳过
  -atk_server_timeout=8000      ATK server 单任务超时，默认 8000
  -h|--help                      显示帮助

示例：
  # 启动 GPU server（前台，不要退出终端）
  bash tests/atk/common/gpu_server.sh -op=chunk_kda_fwd \
      -gpu_image=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel \
      -gpu_repo_root=/workspace/flash-linear-attention-npu

  # 用本地镜像 tar 启动（先 docker load 再创建容器）
  bash tests/atk/common/gpu_server.sh -op=chunk_kda_fwd \
      -gpu_image_tar=/data/gpu_atk.tar \
      -gpu_repo_root=/workspace/flash-linear-attention-npu

  # 复用已有容器时仅启动 ATK server
  bash tests/atk/common/gpu_server.sh -op=chunk_kda_fwd \
      -gpu_container=fla_gpu_atk -gpu_repo_root=/workspace/flash-linear-attention-npu

  # 停止并删除容器
  bash tests/atk/common/gpu_server.sh -op=chunk_kda_fwd -action=gpu_server_stop

  # 在 NPU 机器跑精度对拍（对接已启动的远程 GPU server）
  bash tests/atk/common/gpu_server.sh -op=chunk_kda_fwd -action=npu \
      -gpu_host=10.10.10.10 -gpu_host_port=9090 -npu_device_id=7

  # 在 NPU 机器测试到 GPU server 的连通性（TCP + ATK server 响应）
  bash tests/atk/common/gpu_server.sh -op=chunk_kda_fwd \
      -action=test_connection_from_npu -gpu_host=10.10.10.10 -gpu_host_port=9090

NPU 端对应调用：
  bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -scope=accuracy_gpu \
      -gpu_host=<本机宿主机地址> -gpu_host_port=9090
EOF
}

log_info() {
  echo "[GPU ATK server] $*"
}

die() {
  echo "[GPU ATK server][错误] $*" >&2
  exit 1
}

OP=""
GPU_HOST="${GPU_HOST:-}"
GPU_DEVICE_ID="${GPU_DEVICE_ID:-6}"
GPU_HOST_PORT="${GPU_HOST_PORT:-9090}"
GPU_CONTAINER="${GPU_CONTAINER:-fla_gpu_atk}"
GPU_IMAGE=""
GPU_IMAGE_TAR="${GPU_IMAGE_TAR:-}"
GPU_REPO_ROOT=""
ACTION="${ACTION:-gpu_server_start}"
ATK_ENV="${ATK_ENV:-}"
TRITON_ROOT="${TRITON_ROOT:-}"
ATK_SERVER_TIMEOUT="${ATK_SERVER_TIMEOUT:-8000}"
# action=npu 专用参数
NPU_DEVICE_ID="${NPU_DEVICE_ID:-7}"
NPU_GPU_DEVICE_ID="${NPU_GPU_DEVICE_ID:-0}"
NPU_OUTPUT_PATH="${NPU_OUTPUT_PATH:-./atk_output/kda_remote}"
NPU_TASK="${NPU_TASK:-accuracy}"
NPU_MT="${NPU_MT:-1}"
NPU_TASK_TIMEOUT="${NPU_TASK_TIMEOUT:-2000}"
NPU_ATK_CONFIG="${NPU_ATK_CONFIG:-}"
NPU_EXECUTOR="${NPU_EXECUTOR:-}"

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
    -gpu_host=*) GPU_HOST="${1#-gpu_host=}" ;;
    -gpu_host)
      shift
      [[ $# -gt 0 ]] || die "参数 -gpu_host 需要取值"
      GPU_HOST="$1"
      ;;
    --gpu_host=*) GPU_HOST="${1#--gpu_host=}" ;;
    --gpu_host)
      shift
      [[ $# -gt 0 ]] || die "参数 --gpu_host 需要取值"
      GPU_HOST="$1"
      ;;
    -gpu_device_id=*) GPU_DEVICE_ID="${1#-gpu_device_id=}" ;;
    -gpu_device_id)
      shift
      [[ $# -gt 0 ]] || die "参数 -gpu_device_id 需要取值"
      GPU_DEVICE_ID="$1"
      ;;
    --gpu_device_id=*) GPU_DEVICE_ID="${1#--gpu_device_id=}" ;;
    --gpu_device_id)
      shift
      [[ $# -gt 0 ]] || die "参数 --gpu_device_id 需要取值"
      GPU_DEVICE_ID="$1"
      ;;
    -gpu_host_port=*) GPU_HOST_PORT="${1#-gpu_host_port=}" ;;
    -gpu_host_port)
      shift
      [[ $# -gt 0 ]] || die "参数 -gpu_host_port 需要取值"
      GPU_HOST_PORT="$1"
      ;;
    --gpu_host_port=*) GPU_HOST_PORT="${1#--gpu_host_port=}" ;;
    --gpu_host_port)
      shift
      [[ $# -gt 0 ]] || die "参数 --gpu_host_port 需要取值"
      GPU_HOST_PORT="$1"
      ;;
    -gpu_container=*) GPU_CONTAINER="${1#-gpu_container=}" ;;
    -gpu_container)
      shift
      [[ $# -gt 0 ]] || die "参数 -gpu_container 需要取值"
      GPU_CONTAINER="$1"
      ;;
    --gpu_container=*) GPU_CONTAINER="${1#--gpu_container=}" ;;
    --gpu_container)
      shift
      [[ $# -gt 0 ]] || die "参数 --gpu_container 需要取值"
      GPU_CONTAINER="$1"
      ;;
    -gpu_image=*) GPU_IMAGE="${1#-gpu_image=}" ;;
    -gpu_image)
      shift
      [[ $# -gt 0 ]] || die "参数 -gpu_image 需要取值"
      GPU_IMAGE="$1"
      ;;
    --gpu_image=*) GPU_IMAGE="${1#--gpu_image=}" ;;
    --gpu_image)
      shift
      [[ $# -gt 0 ]] || die "参数 --gpu_image 需要取值"
      GPU_IMAGE="$1"
      ;;
    -gpu_image_tar=*) GPU_IMAGE_TAR="${1#-gpu_image_tar=}" ;;
    -gpu_image_tar)
      shift
      [[ $# -gt 0 ]] || die "参数 -gpu_image_tar 需要取值"
      GPU_IMAGE_TAR="$1"
      ;;
    --gpu_image_tar=*) GPU_IMAGE_TAR="${1#--gpu_image_tar=}" ;;
    --gpu_image_tar)
      shift
      [[ $# -gt 0 ]] || die "参数 --gpu_image_tar 需要取值"
      GPU_IMAGE_TAR="$1"
      ;;
    -gpu_repo_root=*) GPU_REPO_ROOT="${1#-gpu_repo_root=}" ;;
    -gpu_repo_root)
      shift
      [[ $# -gt 0 ]] || die "参数 -gpu_repo_root 需要取值"
      GPU_REPO_ROOT="$1"
      ;;
    --gpu_repo_root=*) GPU_REPO_ROOT="${1#--gpu_repo_root=}" ;;
    --gpu_repo_root)
      shift
      [[ $# -gt 0 ]] || die "参数 --gpu_repo_root 需要取值"
      GPU_REPO_ROOT="$1"
      ;;
    -action=*) ACTION="${1#-action=}" ;;
    -action)
      shift
      [[ $# -gt 0 ]] || die "参数 -action 需要取值"
      ACTION="$1"
      ;;
    --action=*) ACTION="${1#--action=}" ;;
    --action)
      shift
      [[ $# -gt 0 ]] || die "参数 --action 需要取值"
      ACTION="$1"
      ;;
    -atk_env=*) ATK_ENV="${1#-atk_env=}" ;;
    -atk_env)
      shift
      [[ $# -gt 0 ]] || die "参数 -atk_env 需要取值"
      ATK_ENV="$1"
      ;;
    --atk_env=*) ATK_ENV="${1#--atk_env=}" ;;
    --atk_env)
      shift
      [[ $# -gt 0 ]] || die "参数 --atk_env 需要取值"
      ATK_ENV="$1"
      ;;
    -triton_root=*) TRITON_ROOT="${1#-triton_root=}" ;;
    -triton_root)
      shift
      [[ $# -gt 0 ]] || die "参数 -triton_root 需要取值"
      TRITON_ROOT="$1"
      ;;
    --triton_root=*) TRITON_ROOT="${1#--triton_root=}" ;;
    --triton_root)
      shift
      [[ $# -gt 0 ]] || die "参数 --triton_root 需要取值"
      TRITON_ROOT="$1"
      ;;
    -atk_server_timeout=*) ATK_SERVER_TIMEOUT="${1#-atk_server_timeout=}" ;;
    -atk_server_timeout)
      shift
      [[ $# -gt 0 ]] || die "参数 -atk_server_timeout 需要取值"
      ATK_SERVER_TIMEOUT="$1"
      ;;
    --atk_server_timeout=*) ATK_SERVER_TIMEOUT="${1#--atk_server_timeout=}" ;;
    --atk_server_timeout)
      shift
      [[ $# -gt 0 ]] || die "参数 --atk_server_timeout 需要取值"
      ATK_SERVER_TIMEOUT="$1"
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
    -npu_gpu_device_id=*) NPU_GPU_DEVICE_ID="${1#-npu_gpu_device_id=}" ;;
    -npu_gpu_device_id)
      shift
      [[ $# -gt 0 ]] || die "参数 -npu_gpu_device_id 需要取值"
      NPU_GPU_DEVICE_ID="$1"
      ;;
    --npu_gpu_device_id=*) NPU_GPU_DEVICE_ID="${1#--npu_gpu_device_id=}" ;;
    --npu_gpu_device_id)
      shift
      [[ $# -gt 0 ]] || die "参数 --npu_gpu_device_id 需要取值"
      NPU_GPU_DEVICE_ID="$1"
      ;;
    -npu_output_path=*) NPU_OUTPUT_PATH="${1#-npu_output_path=}" ;;
    -npu_output_path)
      shift
      [[ $# -gt 0 ]] || die "参数 -npu_output_path 需要取值"
      NPU_OUTPUT_PATH="$1"
      ;;
    --npu_output_path=*) NPU_OUTPUT_PATH="${1#--npu_output_path=}" ;;
    --npu_output_path)
      shift
      [[ $# -gt 0 ]] || die "参数 --npu_output_path 需要取值"
      NPU_OUTPUT_PATH="$1"
      ;;
    -npu_task=*) NPU_TASK="${1#-npu_task=}" ;;
    -npu_task)
      shift
      [[ $# -gt 0 ]] || die "参数 -npu_task 需要取值"
      NPU_TASK="$1"
      ;;
    --npu_task=*) NPU_TASK="${1#--npu_task=}" ;;
    --npu_task)
      shift
      [[ $# -gt 0 ]] || die "参数 --npu_task 需要取值"
      NPU_TASK="$1"
      ;;
    -npu_mt=*) NPU_MT="${1#-npu_mt=}" ;;
    -npu_mt)
      shift
      [[ $# -gt 0 ]] || die "参数 -npu_mt 需要取值"
      NPU_MT="$1"
      ;;
    --npu_mt=*) NPU_MT="${1#--npu_mt=}" ;;
    --npu_mt)
      shift
      [[ $# -gt 0 ]] || die "参数 --npu_mt 需要取值"
      NPU_MT="$1"
      ;;
    -npu_task_timeout=*) NPU_TASK_TIMEOUT="${1#-npu_task_timeout=}" ;;
    -npu_task_timeout)
      shift
      [[ $# -gt 0 ]] || die "参数 -npu_task_timeout 需要取值"
      NPU_TASK_TIMEOUT="$1"
      ;;
    --npu_task_timeout=*) NPU_TASK_TIMEOUT="${1#--npu_task_timeout=}" ;;
    --npu_task_timeout)
      shift
      [[ $# -gt 0 ]] || die "参数 --npu_task_timeout 需要取值"
      NPU_TASK_TIMEOUT="$1"
      ;;
    -npu_atk_config=*) NPU_ATK_CONFIG="${1#-npu_atk_config=}" ;;
    -npu_atk_config)
      shift
      [[ $# -gt 0 ]] || die "参数 -npu_atk_config 需要取值"
      NPU_ATK_CONFIG="$1"
      ;;
    --npu_atk_config=*) NPU_ATK_CONFIG="${1#--npu_atk_config=}" ;;
    --npu_atk_config)
      shift
      [[ $# -gt 0 ]] || die "参数 --npu_atk_config 需要取值"
      NPU_ATK_CONFIG="$1"
      ;;
    -npu_executor=*) NPU_EXECUTOR="${1#-npu_executor=}" ;;
    -npu_executor)
      shift
      [[ $# -gt 0 ]] || die "参数 -npu_executor 需要取值"
      NPU_EXECUTOR="$1"
      ;;
    --npu_executor=*) NPU_EXECUTOR="${1#--npu_executor=}" ;;
    --npu_executor)
      shift
      [[ $# -gt 0 ]] || die "参数 --npu_executor 需要取值"
      NPU_EXECUTOR="$1"
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
case "$ACTION" in
  gpu_server_start|gpu_server_stop|npu|test_connection_from_npu) ;;
  *) die "不支持的 action：${ACTION}，请使用 gpu_server_start、gpu_server_stop、npu 或 test_connection_from_npu" ;;
esac

# ---------------------------------------------------------------------------
# test_connection_from_npu：在 NPU 机器测试到 GPU server 的连通性
# ---------------------------------------------------------------------------
if [[ "$ACTION" == "test_connection_from_npu" ]]; then
  [[ -n "$GPU_HOST" ]] || die "action=test_connection_from_npu 必须传入 -gpu_host=<GPU server 地址>"
  log_info "从 NPU 机器测试到 GPU server 连通性：${GPU_HOST}:${GPU_HOST_PORT}"

  FAIL=0

  # 1. TCP 端口可达性
  log_info "[1/2] 测试 TCP 连通性：${GPU_HOST}:${GPU_HOST_PORT}"
  if python3 - "$GPU_HOST" "$GPU_HOST_PORT" <<'PYEOF'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(5)
try:
    sock.connect((host, port))
    print("OK")
    sys.exit(0)
except Exception as e:
    print(f"FAIL: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    sock.close()
PYEOF
  then
    log_info "  TCP 连通：${GPU_HOST}:${GPU_HOST_PORT} 可达"
  else
    log_info "  TCP 连通失败：无法连接 ${GPU_HOST}:${GPU_HOST_PORT}，请检查 GPU 容器是否启动、端口映射、防火墙"
    FAIL=1
  fi

  # 2. ATK server HTTP 响应
  log_info "[2/2] 测试 ATK server HTTP 响应：http://${GPU_HOST}:${GPU_HOST_PORT}"
  set +e
  HTTP_BODY="$(python3 - "$GPU_HOST" "$GPU_HOST_PORT" 2>&1 <<'PYEOF'
import urllib.request, sys
host, port = sys.argv[1], int(sys.argv[2])
url = f"http://{host}:{port}/"
try:
    with urllib.request.urlopen(url, timeout=10) as resp:
        body = resp.read().decode(errors="replace")
        print(f"STATUS={resp.status}")
        print(f"BODY={body[:500]}")
        sys.exit(0)
except urllib.error.HTTPError as e:
    # HTTP 4xx/5xx 也说明 server 在响应（如 404），视为连通
    print(f"STATUS={e.code}")
    print(f"BODY={e.read().decode(errors='replace')[:500]}")
    sys.exit(0)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
PYEOF
)"
  HTTP_RC=$?
  set -e
  if [[ $HTTP_RC -eq 0 ]]; then
    log_info "  ATK server 响应正常"
    echo "$HTTP_BODY" | sed 's/^/    /'
  else
    log_info "  ATK server 无 HTTP 响应（可能 server 未启动或端口映射异常）"
    echo "$HTTP_BODY" | sed 's/^/    /'
    FAIL=1
  fi

  # 汇总
  echo ""
  if [[ "$FAIL" -eq 0 ]]; then
    log_info "连通性测试通过：${GPU_HOST}:${GPU_HOST_PORT} TCP 可达且 ATK server 有响应"
  else
    die "连通性测试失败：${GPU_HOST}:${GPU_HOST_PORT}（详见上方日志）"
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# npu：在 NPU 机器构造 atk node(本地 NPU) + node(远程 GPU) + task 链，跑精度对拍
# ---------------------------------------------------------------------------
if [[ "$ACTION" == "npu" ]]; then
  [[ -n "$GPU_HOST" ]] || die "action=npu 必须传入 -gpu_host=<GPU server 地址>"
  [[ -n "$GPU_HOST_PORT" ]] || die "action=npu 必须传入 -gpu_host_port=<GPU server 端口>"

  ATK_BIN="$(command -v atk || true)"
  [[ -n "$ATK_BIN" ]] || die "找不到 atk，请先安装并激活 ATK 环境"

  # 从脚本自身路径解析仓库根目录（向上 3 级：common/atk/tests -> 仓库根）
  HOST_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
  OP_TEST_DIR="${HOST_REPO_ROOT}/tests/atk/${OP}"

  # 默认 atk 配置 / executor 文件名按算子名推导
  [[ -n "$NPU_ATK_CONFIG" ]] || NPU_ATK_CONFIG="./atk_${OP}.json"
  [[ -n "$NPU_EXECUTOR" ]] || NPU_EXECUTOR="./executor_${OP}.py"

  # 预检：executor 与 atk 配置文件必须存在
  HOST_EXECUTOR="${OP_TEST_DIR}/executor_${OP}.py"
  [[ -f "$HOST_EXECUTOR" ]] || die "未找到 executor 文件：${HOST_EXECUTOR}（仓库根目录：${HOST_REPO_ROOT}）"
  HOST_ATK_CONFIG="${OP_TEST_DIR}/${NPU_ATK_CONFIG#./}"
  [[ -f "$HOST_ATK_CONFIG" ]] || die "未找到 atk 配置：${HOST_ATK_CONFIG}（仓库根目录：${HOST_REPO_ROOT}）"

  log_info "action=npu：在 NPU 机器对接远程 GPU server 跑精度对拍"
  log_info "算子：${OP}"
  log_info "NPU 设备号：${NPU_DEVICE_ID}"
  log_info "GPU 远程 server：${GPU_HOST}:${GPU_HOST_PORT}（参考节点 devices=${NPU_GPU_DEVICE_ID}）"
  log_info "ATK 路径：${ATK_BIN}"
  log_info "输出目录：${NPU_OUTPUT_PATH}"
  log_info "task=${NPU_TASK} bm_device=gpu mt=${NPU_MT} timeout=${NPU_TASK_TIMEOUT}"

  mkdir -p "$NPU_OUTPUT_PATH"
  cd "$OP_TEST_DIR"

  "$ATK_BIN" \
    node \
      --name npu_dut \
      --backend npu \
      --devices "$NPU_DEVICE_ID" \
      --output_path "$NPU_OUTPUT_PATH" \
    node \
      --name gpu_reference \
      --backend gpu \
      --host "$GPU_HOST" \
      --port "$GPU_HOST_PORT" \
      --devices "$NPU_GPU_DEVICE_ID" \
      --is_compare true \
      --output_path "$NPU_OUTPUT_PATH" \
    task \
      -c "$NPU_ATK_CONFIG" \
      --task "$NPU_TASK" \
      --bm_device gpu \
      -p "$NPU_EXECUTOR" \
      --syc_dataset \
      -mt "$NPU_MT" \
      -to "$NPU_TASK_TIMEOUT"

  log_info "完成精度对拍"
  exit 0
fi

# ---------------------------------------------------------------------------
# gpu_server_stop：停止并删除容器
# ---------------------------------------------------------------------------
if [[ "$ACTION" == "gpu_server_stop" ]]; then
  log_info "停止并删除容器：${GPU_CONTAINER}"
  docker rm -f "$GPU_CONTAINER" >/dev/null 2>&1 || true
  log_info "容器 ${GPU_CONTAINER} 已删除"
  exit 0
fi

# ---------------------------------------------------------------------------
# gpu_server_start：启动容器 + ATK server
# ---------------------------------------------------------------------------
[[ -n "$GPU_REPO_ROOT" ]] || die "必须传入 -gpu_repo_root=<容器内仓库根目录>"

OP_TEST_DIR="${GPU_REPO_ROOT}/tests/atk/${OP}"
EXECUTOR_FILE="executor_${OP}.py"

# 从脚本自身路径解析宿主机仓库根目录（向上 3 级：common/atk/tests -> 仓库根）
HOST_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# 预检：宿主机上 executor 文件必须存在
HOST_EXECUTOR="${HOST_REPO_ROOT}/tests/atk/${OP}/${EXECUTOR_FILE}"
if [[ ! -f "$HOST_EXECUTOR" ]]; then
  die "宿主机未找到 executor 文件：${HOST_EXECUTOR}（仓库根目录：${HOST_REPO_ROOT}）"
fi
log_info "宿主机仓库根：${HOST_REPO_ROOT}"
log_info "宿主机 executor：${HOST_EXECUTOR}"

log_info "算子：${OP}"
log_info "物理 GPU 卡号：${GPU_DEVICE_ID}（容器内逻辑设备 0）"
log_info "宿主机映射端口：${GPU_HOST_PORT} -> 容器 9090"
log_info "容器名：${GPU_CONTAINER}"
log_info "容器内仓库根：${GPU_REPO_ROOT}"
log_info "算子测试目录：${OP_TEST_DIR}"
if [[ -n "$GPU_IMAGE_TAR" ]]; then
  log_info "GPU 镜像 tar：${GPU_IMAGE_TAR}"
fi
if [[ -n "$GPU_IMAGE" ]]; then
  log_info "GPU 镜像：${GPU_IMAGE}"
fi

# 容器已存在则复用，否则创建新容器
if docker inspect "$GPU_CONTAINER" >/dev/null 2>&1; then
  log_info "容器 ${GPU_CONTAINER} 已存在，复用"
  docker start "$GPU_CONTAINER" >/dev/null 2>&1 || true
else
  # 确定镜像名：优先用 -gpu_image_tar 加载，其次用 -gpu_image
  RUN_IMAGE=""
  if [[ -n "$GPU_IMAGE_TAR" ]]; then
    [[ -f "$GPU_IMAGE_TAR" ]] || die "镜像 tar 文件不存在：${GPU_IMAGE_TAR}"
    log_info "从 tar 加载镜像：${GPU_IMAGE_TAR}"
    LOAD_OUTPUT="$(docker load -i "$GPU_IMAGE_TAR")" || die "docker load 失败：${GPU_IMAGE_TAR}"
    # docker load 输出形如 "Loaded image: repo:tag" 或 "Loaded image ID: sha256:xxx"
    RUN_IMAGE="$(echo "$LOAD_OUTPUT" | grep -oP 'Loaded image: \K.*' | head -n1)"
    if [[ -z "$RUN_IMAGE" ]]; then
      # 旧格式只给了 image ID，回退到 -gpu_image 或用 ID
      RUN_IMAGE="$GPU_IMAGE"
      [[ -n "$RUN_IMAGE" ]] || die "docker load 未返回镜像名，且未传入 -gpu_image，无法创建容器。load 输出：${LOAD_OUTPUT}"
    fi
    log_info "加载得到镜像：${RUN_IMAGE}"
  else
    RUN_IMAGE="$GPU_IMAGE"
  fi
  [[ -n "$RUN_IMAGE" ]] || die "容器不存在，必须传入 -gpu_image=<镜像名> 或 -gpu_image_tar=<tar路径> 创建新容器"
  log_info "创建新容器：${GPU_CONTAINER}（镜像 ${RUN_IMAGE}）"
  docker run -d \
    --name "$GPU_CONTAINER" \
    --gpus "\"device=${GPU_DEVICE_ID}\"" \
    -p "${GPU_HOST_PORT}:9090" \
    -v "${HOST_REPO_ROOT}:${GPU_REPO_ROOT}" \
    -w "$OP_TEST_DIR" \
    "$RUN_IMAGE" \
    sleep infinity
fi

# 校验容器内 GPU 可见性
log_info "校验容器内 GPU 可见性"
docker exec "$GPU_CONTAINER" nvidia-smi -L || die "容器内 nvidia-smi 不可用，检查 --gpus 或 CUDA_VISIBLE_DEVICES"
GPU_COUNT="$(docker exec "$GPU_CONTAINER" bash -c 'nvidia-smi -L | wc -l' || echo "0")"
if [[ "$GPU_COUNT" -lt 1 ]]; then
  die "容器内未检测到 GPU，请检查 --gpus 设备映射"
fi
log_info "容器内可见 GPU 数：${GPU_COUNT}"

# 构造容器内执行命令
INNER_CMDS=()
INNER_CMDS+=("set -euo pipefail")
INNER_CMDS+=("export CUDA_VISIBLE_DEVICES=0")
if [[ -n "$ATK_ENV" ]]; then
  INNER_CMDS+=("source '${ATK_ENV}/bin/activate'")
fi
INNER_CMDS+=("export PYTHONPATH='${GPU_REPO_ROOT}:${TRITON_ROOT}:${PYTHONPATH:-}'")
INNER_CMDS+=("unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY")
INNER_CMDS+=("cd '${OP_TEST_DIR}'")
INNER_CMDS+=("atk --version")
INNER_CMDS+=("python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.device_count())'")
INNER_CMDS+=("test -f '${EXECUTOR_FILE}' || { echo '未找到 executor: ${OP_TEST_DIR}/${EXECUTOR_FILE}（容器内）'; exit 1; }")
INNER_CMDS+=("echo '[GPU ATK server] 开始启动 ATK server，监听 0.0.0.0:9090'")
INNER_CMDS+=("atk server --host 0.0.0.0 --port 9090 --devices 0 --name gpu_reference --output_path ./atk_output/gpu_server --plugin_path './${EXECUTOR_FILE}' --timeout '${ATK_SERVER_TIMEOUT}'")

INNER_SCRIPT="$(IFS=$'\n'; echo "${INNER_CMDS[*]}")"

log_info "在容器内激活环境并前台启动 ATK server"
log_info "NPU 端应使用 -gpu_host=<本机宿主机地址> -gpu_host_port=${GPU_HOST_PORT} 连接"
docker exec -i "$GPU_CONTAINER" bash -c "$INNER_SCRIPT"
