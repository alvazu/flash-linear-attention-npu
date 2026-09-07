#!/usr/bin/env python3
"""GPU 侧定时脚本：按算子逐个启动 ATK GPU server，等待 NPU 端完成精度检查后停止。

运行环境：GPU 宿主机。
与 cron_npu.py 配对使用：cron_npu 完成精度检查后通过 SSH 杀掉容器，
本脚本的 gpu_server_start 阻塞调用随之返回，随后执行 gpu_server_stop 清理，
继续下一个算子。

用法示例：
  python3 tests/atk/common/cron_gpu.py \
      --clone-dir /data/cron/fla \
      --gpu-image pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel \
      --gpu-repo-root /workspace/flash-linear-attention-npu
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

FLA_REPO = "https://github.com/flashserve/flash-linear-attention-npu.git"
WORK_TOOL_REPO = "https://gitcode.com/chen-linxin4/work_tool.git"
WORK_TOOL_BRANCH = "atk-test"


def log(msg: str) -> None:
    print(f"[cron_gpu] {msg}", flush=True)


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
        run(f"git pull --rebase", cwd=work_tool_dir, check=False)

    return fla_dir


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clone-dir", type=Path, required=True,
                        help="仓库 clone 根目录")
    parser.add_argument("--conda-env", default="",
                        help="GPU 侧 conda 环境名（可选）")
    parser.add_argument("--gpu-host-port", default="9090",
                        help="GPU server 端口（默认 9090）")
    parser.add_argument("--gpu-device-id", default="6",
                        help="物理 GPU 卡号（默认 6）")
    parser.add_argument("--gpu-container", default="fla_gpu_atk",
                        help="Docker 容器名（默认 fla_gpu_atk）")
    parser.add_argument("--gpu-image", default="",
                        help="GPU 基础镜像名")
    parser.add_argument("--gpu-image-tar", default="",
                        help="本地镜像 tar 路径")
    parser.add_argument("--gpu-repo-root", default="/workspace/flash-linear-attention-npu",
                        help="仓库在容器内挂载根目录")
    parser.add_argument("--atk-env", default="",
                        help="容器内 ATK 虚拟环境目录（可选）")
    parser.add_argument("--operators", default="",
                        help="逗号分隔的算子名列表；为空时自动扫描")
    parser.add_argument("--server-wait-timeout", type=int, default=14400,
                        help="gpu_server_start 最大阻塞时长（秒，默认 14400）")
    args = parser.parse_args()

    # 1. 克隆仓库
    log("步骤 1: 克隆仓库")
    fla_dir = clone_repos(args.clone_dir)
    atk_dir = fla_dir / "tests" / "atk"
    gpu_server_sh = atk_dir / "common" / "gpu_server.sh"

    # 2. 确定算子列表
    if args.operators:
        ops = [o.strip() for o in args.operators.split(",") if o.strip()]
    else:
        ops = detect_operators(atk_dir)
    if not ops:
        log("未检测到算子，退出")
        return 1
    log(f"算子列表（{len(ops)}）: {', '.join(ops)}")

    # 3. 逐算子启动 server
    for idx, op in enumerate(ops, 1):
        log(f"===== [{idx}/{len(ops)}] {op} =====")

        # 启动 GPU server（阻塞，NPU 端 SSH 杀容器后返回）
        cmd = [
            "bash", str(gpu_server_sh),
            f"-op={op}",
            f"-action=gpu_server_start",
            f"-gpu_host_port={args.gpu_host_port}",
            f"-gpu_device_id={args.gpu_device_id}",
            f"-gpu_container={args.gpu_container}",
            f"-gpu_repo_root={args.gpu_repo_root}",
        ]
        if args.gpu_image:
            cmd.append(f"-gpu_image={args.gpu_image}")
        if args.gpu_image_tar:
            cmd.append(f"-gpu_image_tar={args.gpu_image_tar}")
        if args.atk_env:
            cmd.append(f"-atk_env={args.atk_env}")

        log(f"启动 GPU server: {' '.join(cmd)}")
        try:
            run(cmd, check=False, timeout=args.server_wait_timeout)
        except subprocess.TimeoutExpired:
            log(f"GPU server 超时 ({args.server_wait_timeout}s)，强制停止")
        except Exception as e:
            log(f"GPU server 异常退出: {e}")

        # 清理容器
        log(f"清理容器: {args.gpu_container}")
        stop_cmd = [
            "bash", str(gpu_server_sh),
            f"-op={op}",
            f"-action=gpu_server_stop",
            f"-gpu_container={args.gpu_container}",
        ]
        run(stop_cmd, check=False)
        log(f"[{idx}/{len(ops)}] {op} 完成")

    log("所有算子处理完毕")
    return 0


if __name__ == "__main__":
    sys.exit(main())
