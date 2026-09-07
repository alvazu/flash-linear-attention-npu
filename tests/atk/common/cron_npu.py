#!/usr/bin/env python3
"""NPU 侧定时脚本：环境准备 → 编译安装 → 逐算子对接 GPU server 跑精度检查。

运行环境：NPU 机器。
与 cron_gpu.py 配对使用：本脚本完成环境准备后，对每个算子：
  1. 轮询等待 GPU server 端口可达
  2. 调用 gpu_server.sh -action=npu 执行精度检查
  3. SSH 到 GPU 机器执行 gpu_server_stop（杀掉容器，使 cron_gpu 的阻塞调用返回）
  4. 继续下一个算子

用法示例：
  python3 tests/atk/common/cron_npu.py \
      --conda-env jisihuai \
      --clone-dir /data/cron/fla \
      --gpu-host 141.61.21.62 \
      --gpu-host-port 9090 \
      --gpu-ssh-user root \
      --gpu-clone-dir /data/cron/fla

前置条件：
  - NPU 机器可通过 SSH 免密登录 GPU 机器（ssh-keygen / ssh-copy-id）
  - NPU 机器已安装 conda
  - NPU 机器已配置 CANN 环境（source set_env.sh）
"""

from __future__ import annotations

import argparse
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

FLA_REPO = "https://github.com/flashserve/flash-linear-attention-npu.git"
WORK_TOOL_REPO = "https://gitcode.com/chen-linxin4/work_tool.git"
WORK_TOOL_BRANCH = "atk-test"


def log(msg: str) -> None:
    print(f"[cron_npu] {msg}", flush=True)


def run(cmd: str | list[str], *, check: bool = True, cwd: Path | None = None,
        env: dict | None = None, timeout: int | None = None,
        capture: bool = False) -> subprocess.CompletedProcess:
    """执行命令，统一封装。"""
    shell = isinstance(cmd, str)
    log(f"$ {cmd if shell else ' '.join(cmd)}")
    return subprocess.run(
        cmd, shell=shell, check=check, cwd=cwd, env=env,
        timeout=timeout, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


# ---------------------------------------------------------------------------
# 步骤 0：环境准备
# ---------------------------------------------------------------------------

def activate_conda(conda_env: str) -> dict:
    """激活 conda 环境，返回带激活后 PATH 的 env 字典。

    在非交互式 shell 中，`conda activate` 需要先初始化 conda 的 shell 函数，
    否则会报 "CommandNotFoundError: Your shell has not been properly
    configured to use 'conda activate'"。通过 `conda shell.bash hook` 注入。
    """
    env = __import__("os").environ.copy()
    if not conda_env:
        log("未指定 conda 环境，使用当前环境")
        return env
    result = run(
        f'eval "$(conda shell.bash hook)" && conda activate {conda_env} && env',
        shell=True, check=True, capture=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            env[key] = val
    # 确认 Python 可执行文件
    py = run("which python", shell=True, check=True, env=env, capture=True)
    log(f"Python: {py.stdout.strip()}")
    return env


# ---------------------------------------------------------------------------
# 步骤 1：git clone
# ---------------------------------------------------------------------------

def clone_repos(clone_dir: Path) -> Path:
    """克隆仓库，返回 fla 仓库根目录。"""
    clone_dir.mkdir(parents=True, exist_ok=True)
    fla_dir = clone_dir / "flash-linear-attention-npu"
    if not fla_dir.exists():
        run(f"git clone {FLA_REPO} {fla_dir}", cwd=clone_dir)
    else:
        log(f"已存在，跳过 clone: {fla_dir}")
        run("git pull --rebase", cwd=fla_dir, check=False)

    work_tool_dir = clone_dir / "work_tool"
    if not work_tool_dir.exists():
        run(f"git clone -b {WORK_TOOL_BRANCH} {WORK_TOOL_REPO} {work_tool_dir}",
            cwd=clone_dir)
    else:
        log(f"已存在，跳过 clone: {work_tool_dir}")
        run("git pull --rebase", cwd=work_tool_dir, check=False)

    return fla_dir


# ---------------------------------------------------------------------------
# 步骤 2-3：编译 + 安装
# ---------------------------------------------------------------------------

def build_and_install(fla_dir: Path, env: dict) -> None:
    """编译 fla wheel 并安装（取构建输出最后一行作为安装命令）。"""
    log("步骤 2: 编译 fla wheel")
    build_env = env.copy()
    build_env["FLA_NPU_SOC"] = "ascend950"
    result = run(
        "python scripts/build_wheel.py",
        cwd=fla_dir, env=build_env, capture=True,
    )
    build_output = result.stdout.strip()
    log(build_output)

    # 取最后一行作为安装命令
    lines = [ln.strip() for ln in build_output.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("build_wheel.py 无输出，无法确定安装命令")
    install_cmd = lines[-1]
    # 过滤掉非安装命令的日志行
    if not install_cmd.startswith(("pip", sys.executable)):
        # 从输出中查找包含 "pip install" 的行
        install_lines = [ln for ln in lines if "pip install" in ln]
        if not install_lines:
            raise RuntimeError(f"未在构建输出中找到安装命令，最后一行: {install_cmd}")
        install_cmd = install_lines[-1]

    log(f"步骤 3: 安装 -> {install_cmd}")
    run(install_cmd, shell=True, cwd=fla_dir, env=env)


# ---------------------------------------------------------------------------
# 步骤 4：逐算子精度检查
# ---------------------------------------------------------------------------

def detect_operators(atk_dir: Path) -> list[str]:
    """扫描 atk 目录，返回有 executor 和 atk 配置的算子名列表（排除 common）。"""
    ops: list[str] = []
    for sub in sorted(atk_dir.iterdir()):
        if not sub.is_dir() or sub.name in ("common", "__pycache__"):
            continue
        has_executor = any(sub.glob(f"executor_{sub.name}.py"))
        has_config = any(sub.glob(f"atk_{sub.name}.json"))
        if has_executor and has_config:
            ops.append(sub.name)
    return ops


def wait_server_up(host: str, port: int, timeout: int = 1800) -> bool:
    """轮询等待 GPU server 端口可达。返回 True 表示可达。"""
    log(f"等待 GPU server 可达: {host}:{port}（超时 {timeout}s）")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5):
                log(f"GPU server 已就绪: {host}:{port}")
                return True
        except OSError:
            time.sleep(10)
    log(f"超时: GPU server 在 {timeout}s 内未就绪")
    return False


def ssh_stop_gpu_server(ssh_target: str, gpu_clone_dir: Path,
                        op: str, gpu_container: str) -> None:
    """SSH 到 GPU 机器执行 gpu_server_stop。"""
    gpu_server_sh = gpu_clone_dir / "flash-linear-attention-npu" / "tests" / "atk" / "common" / "gpu_server.sh"
    cmd = (
        f"bash {shlex.quote(str(gpu_server_sh))} "
        f"-op={op} -action=gpu_server_stop "
        f"-gpu_container={gpu_container}"
    )
    log(f"SSH 停止 GPU server: ssh {ssh_target} '{cmd}'")
    run(["ssh", "-o", "StrictHostKeyChecking=no", ssh_target, cmd],
        check=False, timeout=120)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conda-env", required=True,
                        help="NPU 侧 conda 环境名（如 jisihuai）")
    parser.add_argument("--clone-dir", type=Path, required=True,
                        help="NPU 侧仓库 clone 根目录")
    parser.add_argument("--gpu-host", required=True,
                        help="GPU server 宿主机地址")
    parser.add_argument("--gpu-host-port", default="9090",
                        help="GPU server 端口（默认 9090）")
    parser.add_argument("--gpu-ssh-user", default="",
                        help="SSH 登录 GPU 机器的用户名（默认当前用户）")
    parser.add_argument("--gpu-clone-dir", type=Path, required=True,
                        help="GPU 侧仓库 clone 根目录（用于 SSH 执行 gpu_server_stop）")
    parser.add_argument("--npu-device-id", default="7",
                        help="NPU 物理卡号（默认 7）")
    parser.add_argument("--npu-gpu-device-id", default="0",
                        help="远程 GPU 参考节点 devices 值（默认 0）")
    parser.add_argument("--gpu-container", default="fla_gpu_atk",
                        help="GPU Docker 容器名（默认 fla_gpu_atk）")
    parser.add_argument("--npu-output-path", default="./atk_output/kda_remote",
                        help="精度检查输出目录（默认 ./atk_output/kda_remote）")
    parser.add_argument("--npu-task-timeout", default="2000",
                        help="精度检查 task 超时（默认 2000）")
    parser.add_argument("--operators", default="",
                        help="逗号分隔的算子名列表；为空时自动扫描")
    parser.add_argument("--server-wait-timeout", type=int, default=1800,
                        help="等待 GPU server 可达的超时（秒，默认 1800）")
    args = parser.parse_args()

    ssh_target = f"{args.gpu_ssh_user}@{args.gpu_host}" if args.gpu_ssh_user else args.gpu_host

    # 步骤 0: 激活 conda 环境
    log("步骤 0: 激活 conda 环境")
    env = activate_conda(args.conda_env)

    # 步骤 1: git clone
    log("步骤 1: git clone 仓库")
    fla_dir = clone_repos(args.clone_dir)
    atk_dir = fla_dir / "tests" / "atk"
    gpu_server_sh = atk_dir / "common" / "gpu_server.sh"

    # 步骤 2-3: 编译 + 安装
    build_and_install(fla_dir, env)

    # 确定算子列表
    if args.operators:
        ops = [o.strip() for o in args.operators.split(",") if o.strip()]
    else:
        ops = detect_operators(atk_dir)
    if not ops:
        log("未检测到算子，退出")
        return 1
    log(f"算子列表（{len(ops)}）: {', '.join(ops)}")

    # 步骤 4: 逐算子精度检查
    port = int(args.gpu_host_port)
    for idx, op in enumerate(ops, 1):
        log(f"===== [{idx}/{len(ops)}] {op} =====")

        # 4a. 等待 GPU server 可达
        if not wait_server_up(args.gpu_host, port, args.server_wait_timeout):
            log(f"跳过 {op}：GPU server 未就绪")
            continue

        # 4b. 执行精度检查
        npu_cmd = [
            "bash", str(gpu_server_sh),
            f"-op={op}",
            f"-action=npu",
            f"-gpu_host={args.gpu_host}",
            f"-gpu_host_port={args.gpu_host_port}",
            f"-npu_device_id={args.npu_device_id}",
            f"-npu_gpu_device_id={args.npu_gpu_device_id}",
            f"-npu_output_path={args.npu_output_path}",
            f"-npu_task_timeout={args.npu_task_timeout}",
        ]
        log(f"执行精度检查: {' '.join(npu_cmd)}")
        try:
            run(npu_cmd, check=False, cwd=atk_dir / op, env=env)
        except Exception as e:
            log(f"精度检查异常: {e}")

        # 4c. SSH 停止 GPU server（使 cron_gpu 阻塞调用返回）
        ssh_stop_gpu_server(ssh_target, args.gpu_clone_dir, op, args.gpu_container)

        # 等待端口关闭，确保下一个算子的 server 能干净启动
        time.sleep(10)
        log(f"[{idx}/{len(ops)}] {op} 完成")

    log("所有算子处理完毕")
    return 0


if __name__ == "__main__":
    sys.exit(main())
