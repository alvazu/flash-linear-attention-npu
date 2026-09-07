#!/usr/bin/env python3
"""融合定时脚本：仅在 NPU 侧运行，通过 SSH 驱动 GPU 侧 start/npu/stop 全流程。

取代原来需要分别在 GPU/NPU 两台机器各跑一个 cron（cron_gpu.py + cron_npu.py）
的双脚本方案。本脚本只在 NPU 机器运行，通过 SSH 远程驱动 GPU 侧：

  1. NPU 本地：激活 conda → 设置代理 → clone 仓库 → unset 代理 → build wheel + install
  2. SSH 到 GPU：clone 仓库（启用代理，clone 完成后 unset）
  3. ping GPU IP，不通直接报错退出
  4. 逐算子循环：
     a. SSH 后台启动 gpu_server_start（nohup ... &，SSH 立即返回）
     b. NPU 轮询 GPU server 端口可达
     c. NPU 本地设置 PYTHONHOME / ASCEND_CUSTOM_OPP_PATH 后执行 action=npu 跑精度对拍
     d. SSH 到 GPU 执行 gpu_server_stop（杀掉容器，后台 start 进程随之退出）

前置条件：
  - NPU 机器可通过 SSH 免密登录 GPU 机器（ssh-keygen / ssh-copy-id）
  - NPU 机器已安装 conda、已配置 CANN 环境（source set_env.sh）
  - GPU 机器已安装 docker，且 NPU 的 SSH 用户有 docker 权限

用法示例：
  python3 tests/atk/common/cron.py \
      --conda-env jisihuai \
      --clone-dir /data/cron/fla \
      --http-proxy http://127.0.0.1:7890 \
      --gpu-host 141.61.21.62 \
      --gpu-host-port 9090 \
      --gpu-ssh-user root \
      --gpu-ssh-password 'secret123' \
      --gpu-clone-dir /data/cron/fla \
      --gpu-image pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel \
      --gpu-repo-root /workspace/flash-linear-attention-npu

  # 用私钥认证
  python3 tests/atk/common/cron.py ... --gpu-ssh-key ~/.ssh/id_rsa
  # 用默认免密（agent/keychain/known_hosts）
  python3 tests/atk/common/cron.py ...
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

FLA_REPO = "https://github.com/flashserve/flash-linear-attention-npu.git"
WORK_TOOL_REPO = "https://gitcode.com/chen-linxin4/work_tool.git"
WORK_TOOL_BRANCH = "atk-test"


def log(msg: str) -> None:
    print(f"[cron] {msg}", flush=True)


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
# 代理 / 网络检查 / NPU ATK 环境变量
# ---------------------------------------------------------------------------

def make_proxy_env(env: dict | None, proxy: str) -> dict:
    """返回带 http_proxy/https_proxy 设置的 env 副本。

    仅用于 git clone 阶段；调用方在 clone 完成后使用未带代理的 env 即可，
    等效于 `unset http_proxy; unset https_proxy`。
    """
    new_env = (env or {}).copy()
    if proxy:
        new_env["http_proxy"] = proxy
        new_env["https_proxy"] = proxy
        new_env["HTTP_PROXY"] = proxy
        new_env["HTTPS_PROXY"] = proxy
        log(f"启用代理: {proxy}")
    return new_env


def ping_host(host: str, count: int = 3, wait: int = 5) -> bool:
    """ping 主机，返回可达 True/False。"""
    cmd = ["ping", "-c", str(count), "-W", str(wait), host]
    log(f"$ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        log("ping 命令不存在，跳过连通性检查")
        return True
    except subprocess.TimeoutExpired:
        log(f"ping {host} 超时")
        return False
    if result.returncode == 0:
        log(f"ping {host} 可达")
        return True
    log(f"ping {host} 不可达 (rc={result.returncode})")
    if result.stdout.strip():
        for ln in result.stdout.strip().splitlines()[-3:]:
            log(f"  {ln}")
    if result.stderr.strip():
        for ln in result.stderr.strip().splitlines()[-3:]:
            log(f"  {ln}")
    return False


def query_python_paths(env: dict) -> tuple[str, str]:
    """查询当前 python 的 sys.prefix（PYTHONHOME）与 site-packages 路径。"""
    py_prefix = run(
        "python -c 'import sys; print(sys.prefix)'",
        shell=True, check=True, env=env, capture=True,
    )
    pythonhome = py_prefix.stdout.strip()
    site_pkg = run(
        "python -c 'import site; print(site.getsitepackages()[0])'",
        shell=True, check=True, env=env, capture=True,
    )
    site_packages = site_pkg.stdout.strip()
    if not pythonhome or not site_packages:
        raise RuntimeError(f"无法确定 PYTHONHOME/site-packages: "
                           f"prefix={pythonhome!r} site={site_packages!r}")
    log(f"PYTHONHOME: {pythonhome}")
    log(f"site-packages: {site_packages}")
    return pythonhome, site_packages


def apply_npu_atk_env(env: dict, pythonhome: str, site_packages: str) -> dict:
    """设置 NPU 精度检查所需的 PYTHONHOME 与 ASCEND_CUSTOM_OPP_PATH。

    等效于：
      export PYTHONHOME=<conda 环境位置>
      export ASCEND_CUSTOM_OPP_PATH="<site-packages>/fla_npu/opp/vendors/fla_npu_transformer:${ASCEND_CUSTOM_OPP_PATH:-}"
    """
    new_env = env.copy()
    new_env["PYTHONHOME"] = pythonhome
    opp_path = f"{site_packages}/fla_npu/opp/vendors/fla_npu_transformer"
    existing = new_env.get("ASCEND_CUSTOM_OPP_PATH", "")
    new_env["ASCEND_CUSTOM_OPP_PATH"] = (
        f"{opp_path}:{existing}" if existing else opp_path
    )
    log(f"PYTHONHOME={new_env['PYTHONHOME']}")
    log(f"ASCEND_CUSTOM_OPP_PATH={new_env['ASCEND_CUSTOM_OPP_PATH']}")
    return new_env


# ---------------------------------------------------------------------------
# SSH 认证（密码 / 私钥 / 默认免密）
# ---------------------------------------------------------------------------

def build_ssh_prefix(args: argparse.Namespace) -> list[str]:
    """返回 SSH 命令前缀（不含 ssh_target 与远程命令部分）。

    三选一：
      - --gpu-ssh-password 指定：通过 sshpass 传密码，强制 password 认证、禁用 pubkey
      - --gpu-ssh-key 指定：通过 -i 指定私钥，加 IdentitiesOnly=yes 避免 agent 干扰
      - 都不传：走默认免密（agent/keychain/known_hosts）

    使用密码时会校验 sshpass 是否安装；缺失则给出安装提示并报错。
    """
    if args.gpu_ssh_password:
        if not shutil.which("sshpass"):
            raise RuntimeError(
                "使用 --gpu-ssh-password 需要安装 sshpass；"
                "Ubuntu/Debian: apt-get install -y sshpass；"
                "CentOS/RHEL: yum install -y sshpass"
            )
        return [
            "sshpass", "-p", args.gpu_ssh_password,
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "PreferredAuthentications=password",
            "-o", "PubkeyAuthentication=no",
        ]
    prefix = ["ssh", "-o", "StrictHostKeyChecking=no"]
    if args.gpu_ssh_key:
        prefix += ["-i", args.gpu_ssh_key, "-o", "IdentitiesOnly=yes"]
    return prefix


def ssh_run(ssh_prefix: list[str], ssh_target: str, remote_cmd: str,
            *, timeout: int = 600, check: bool = False) -> subprocess.CompletedProcess:
    """统一封装的 SSH 调用：ssh_prefix + [ssh_target, remote_cmd]。"""
    cmd = ssh_prefix + [ssh_target, remote_cmd]
    return run(cmd, check=check, timeout=timeout)


# ---------------------------------------------------------------------------
# 步骤 0：NPU 侧环境准备
# ---------------------------------------------------------------------------

def activate_conda(conda_env: str) -> dict:
    """激活 conda 环境，返回带激活后 PATH 的 env 字典。

    在非交互式 shell 中，`conda activate` 需要先初始化 conda 的 shell 函数，
    否则会报 "CommandNotFoundError: Your shell has not been properly
    configured to use 'conda activate'"。通过 `conda shell.bash hook` 注入。
    """
    import os
    env = os.environ.copy()
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
    py = run("which python", shell=True, check=True, env=env, capture=True)
    log(f"Python: {py.stdout.strip()}")
    return env


# ---------------------------------------------------------------------------
# 通用：git clone（NPU 本地或 SSH 到 GPU 远程）
# ---------------------------------------------------------------------------

def clone_repos_local(clone_dir: Path, proxy: str = "",
                      base_env: dict | None = None) -> Path:
    """在本地 clone 仓库，返回 fla 仓库根目录。

    clone 期间启用 proxy（http_proxy/https_proxy）；调用方传入的 base_env 不会被
    修改，clone 完成后继续使用 base_env 即等效于 `unset http_proxy/https_proxy`。
    """
    clone_dir.mkdir(parents=True, exist_ok=True)
    proxy_env = make_proxy_env(base_env, proxy)
    # env=None 时 subprocess 继承父进程环境；显式传入以应用代理
    run_env = proxy_env or None

    fla_dir = clone_dir / "flash-linear-attention-npu"
    if not fla_dir.exists():
        run(f"git clone {FLA_REPO} {fla_dir}", cwd=clone_dir, env=run_env)
    else:
        log(f"已存在，跳过 clone: {fla_dir}")
        run("git pull --rebase", cwd=fla_dir, check=False, env=run_env)

    work_tool_dir = clone_dir / "work_tool"
    if not work_tool_dir.exists():
        run(f"git clone -b {WORK_TOOL_BRANCH} {WORK_TOOL_REPO} {work_tool_dir}",
            cwd=clone_dir, env=run_env)
    else:
        log(f"已存在，跳过 clone: {work_tool_dir}")
        run(f"git pull --rebase", cwd=work_tool_dir, check=False, env=run_env)

    if proxy:
        log("本地 clone 完成，后续命令不再使用代理（等效 unset http_proxy/https_proxy）")
    return fla_dir


def ssh_clone_repos(ssh_prefix: list[str], ssh_target: str,
                    gpu_clone_dir: Path, proxy: str = "") -> None:
    """SSH 到 GPU 机器 clone 仓库，确保 gpu_server.sh 与 executor 文件存在。

    在远程脚本中先 `export http_proxy/https_proxy`，clone 完成后 `unset`，
    避免 SSH 通道后续命令残留代理（每条 SSH 命令独立 shell，但仍显式 unset）。
    """
    clone_dir_str = str(gpu_clone_dir)
    fla_subdir = "flash-linear-attention-npu"
    work_tool_subdir = "work_tool"

    proxy_prefix = ""
    proxy_suffix = ""
    if proxy:
        proxy_prefix = (
            f"export http_proxy={shlex.quote(proxy)}; "
            f"export https_proxy={shlex.quote(proxy)}; "
        )
        proxy_suffix = "unset http_proxy https_proxy; "

    # 一条 SSH 命令完成 mkdir / clone-or-pull / unset，保证幂等
    remote_script = (
        f"set -e; "
        f"{proxy_prefix}"
        f"mkdir -p {shlex.quote(clone_dir_str)}; "
        f"cd {shlex.quote(clone_dir_str)}; "
        f"if [ -d {fla_subdir} ]; then "
        f"  echo 'fla 已存在，pull'; git -C {fla_subdir} pull --rebase || true; "
        f"else git clone {FLA_REPO} {fla_subdir}; fi; "
        f"if [ -d {work_tool_subdir} ]; then "
        f"  echo 'work_tool 已存在，pull'; git -C {work_tool_subdir} pull --rebase || true; "
        f"else git clone -b {WORK_TOOL_BRANCH} {WORK_TOOL_REPO} {work_tool_subdir}; fi; "
        f"{proxy_suffix}"
        f"echo 'SSH clone 完成'"
    )
    cmd = ssh_prefix + [ssh_target, remote_script]
    log(f"SSH clone 仓库到 GPU: {ssh_target}:{clone_dir_str}")
    run(cmd, check=False, timeout=600)


# ---------------------------------------------------------------------------
# 步骤 2-3：NPU 本地编译 + 安装
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

    lines = [ln.strip() for ln in build_output.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("build_wheel.py 无输出，无法确定安装命令")
    install_cmd = lines[-1]
    if not install_cmd.startswith(("pip", sys.executable)):
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


def build_gpu_server_start_cmd(gpu_server_sh_remote: str, op: str,
                               args: argparse.Namespace) -> str:
    """构造 GPU 侧 gpu_server_start 命令字符串（在 GPU 宿主机本地执行）。"""
    parts = [
        "bash", shlex.quote(gpu_server_sh_remote),
        f"-op={op}",
        "-action=gpu_server_start",
        f"-gpu_host_port={args.gpu_host_port}",
        f"-gpu_device_id={args.gpu_device_id}",
        f"-gpu_container={args.gpu_container}",
        f"-gpu_repo_root={args.gpu_repo_root}",
    ]
    if args.gpu_image:
        parts.append(f"-gpu_image={args.gpu_image}")
    if args.gpu_image_tar:
        parts.append(f"-gpu_image_tar={args.gpu_image_tar}")
    if args.atk_env:
        parts.append(f"-atk_env={args.atk_env}")
    if args.triton_root:
        parts.append(f"-triton_root={args.triton_root}")
    return " ".join(parts)


def ssh_start_gpu_server(ssh_prefix: list[str], ssh_target: str,
                         gpu_clone_dir: Path, op: str,
                         args: argparse.Namespace) -> None:
    """SSH 到 GPU 后台启动 gpu_server_start（nohup ... &，SSH 立即返回）。

    gpu_server_start 本身前台阻塞运行 ATK server；通过 nohup + & 放到后台，
    日志重定向到 GPU 宿主机文件，SSH 调用立即返回，NPU 端随后轮询端口。
    """
    gpu_server_sh = (
        gpu_clone_dir / "flash-linear-attention-npu" / "tests" / "atk" /
        "common" / "gpu_server.sh"
    )
    gpu_server_sh_remote = str(gpu_server)
    start_cmd = build_gpu_server_start_cmd(gpu_server_sh_remote, op, args)
    log_file = f"/tmp/cron_gpu_server_{op}.log"
    # nohup + < /dev/null + & 确保 SSH 立即返回，进程在 GPU 侧常驻
    remote_cmd = (
        f"nohup {start_cmd} > {shlex.quote(log_file)} 2>&1 < /dev/null & "
        f"echo \"GPU server 已后台启动，PID=$!，日志: {log_file}\""
    )
    cmd = ssh_prefix + [ssh_target, remote_cmd]
    log(f"SSH 后台启动 GPU server: {ssh_target} '{start_cmd} &'")
    run(cmd, check=False, timeout=120)


def ssh_stop_gpu_server(ssh_prefix: list[str], ssh_target: str,
                        gpu_clone_dir: Path, op: str,
                        gpu_container: str) -> None:
    """SSH 到 GPU 执行 gpu_server_stop（杀掉容器，后台 start 进程随之退出）。"""
    gpu_server_sh = (
        gpu_clone_dir / "flash-linear-attention-npu" / "tests" / "atk" /
        "common" / "gpu_server.sh"
    )
    cmd_str = (
        f"bash {shlex.quote(str(gpu_server_sh))} "
        f"-op={op} -action=gpu_server_stop "
        f"-gpu_container={gpu_container}"
    )
    log(f"SSH 停止 GPU server: {ssh_target} '{cmd_str}'")
    cmd = ssh_prefix + [ssh_target, cmd_str]
    run(cmd, check=False, timeout=120)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # NPU 侧参数
    parser.add_argument("--conda-env", required=True,
                        help="NPU 侧 conda 环境名（如 jisihuai）")
    parser.add_argument("--clone-dir", type=Path, required=True,
                        help="NPU 侧仓库 clone 根目录")
    parser.add_argument("--http-proxy", default="",
                        help="git clone 时使用的 HTTP 代理 "
                             "（如 http://127.0.0.1:7890）；clone 完成后自动 unset")
    parser.add_argument("--npu-device-id", default="7",
                        help="NPU 物理卡号（默认 7）")
    parser.add_argument("--npu-gpu-device-id", default="0",
                        help="远程 GPU 参考节点 devices 值（默认 0）")
    parser.add_argument("--npu-output-path", default="./atk_output/kda_remote",
                        help="精度检查输出目录（默认 ./atk_output/kda_remote）")
    parser.add_argument("--npu-task-timeout", default="2000",
                        help="精度检查 task 超时（默认 2000）")

    # GPU 侧参数（通过 SSH 驱动）
    parser.add_argument("--gpu-host", required=True,
                        help="GPU server 宿主机地址")
    parser.add_argument("--gpu-host-port", default="9090",
                        help="GPU server 端口（默认 9090）")
    parser.add_argument("--gpu-ssh-user", default="",
                        help="SSH 登录 GPU 机器的用户名（默认当前用户）")
    parser.add_argument("--gpu-ssh-password", default="",
                        help="SSH 登录密码（需要 NPU 机器安装 sshpass；"
                             "指定后强制 password 认证、禁用 pubkey）")
    parser.add_argument("--gpu-ssh-key", default="",
                        help="SSH 私钥文件路径（如 ~/.ssh/id_rsa）；"
                             "指定后用 -i 加载并设 IdentitiesOnly=yes")
    parser.add_argument("--gpu-clone-dir", type=Path, required=True,
                        help="GPU 侧仓库 clone 根目录（SSH clone + gpu_server.sh 路径）")
    parser.add_argument("--gpu-device-id", default="6",
                        help="物理 GPU 卡号（默认 6）")
    parser.add_argument("--gpu-container", default="fla_gpu_atk",
                        help="GPU Docker 容器名（默认 fla_gpu_atk）")
    parser.add_argument("--gpu-image", default="",
                        help="GPU 基础镜像名")
    parser.add_argument("--gpu-image-tar", default="",
                        help="本地镜像 tar 路径（GPU 宿主机上）")
    parser.add_argument("--gpu-repo-root", default="/workspace/flash-linear-attention-npu",
                        help="仓库在容器内挂载根目录")
    parser.add_argument("--atk-env", default="",
                        help="容器内 ATK 虚拟环境目录（可选）")
    parser.add_argument("--triton-root", default="",
                        help="兼容 Triton 源码根（加入 PYTHONPATH，可选）")

    # 通用参数
    parser.add_argument("--operators", default="",
                        help="逗号分隔的算子名列表；为空时自动扫描")
    parser.add_argument("--server-wait-timeout", type=int, default=1800,
                        help="等待 GPU server 可达的超时（秒，默认 1800）")
    parser.add_argument("--skip-gpu-clone", action="store_true",
                        help="跳过 SSH clone GPU 侧仓库（假定已存在且最新）")
    args = parser.parse_args()

    ssh_target = (
        f"{args.gpu_ssh_user}@{args.gpu_host}" if args.gpu_ssh_user else args.gpu_host
    )

    # 构造 SSH 前缀（密码 / 私钥 / 默认免密三选一）
    try:
        ssh_prefix = build_ssh_prefix(args)
    except RuntimeError as e:
        log(f"SSH 认证配置错误: {e}")
        return 1
    if args.gpu_ssh_password:
        log("SSH 认证：密码（sshpass）")
    elif args.gpu_ssh_key:
        log(f"SSH 认证：私钥 {args.gpu_ssh_key}")
    else:
        log("SSH 认证：默认免密（agent/keychain）")

    # 步骤 0: 激活 conda 环境
    log("步骤 0: 激活 conda 环境")
    env = activate_conda(args.conda_env)

    # 步骤 1a: NPU 本地 git clone（启用代理）
    log("步骤 1a: NPU 本地 git clone 仓库")
    fla_dir = clone_repos_local(args.clone_dir, proxy=args.http_proxy, base_env=env)
    atk_dir = fla_dir / "tests" / "atk"
    gpu_server_sh = atk_dir / "common" / "gpu_server.sh"

    # 步骤 1b: SSH 到 GPU clone 仓库（启用代理，clone 后 unset）
    if not args.skip_gpu_clone:
        log("步骤 1b: SSH 到 GPU clone 仓库")
        ssh_clone_repos(ssh_prefix, ssh_target, args.gpu_clone_dir,
                        proxy=args.http_proxy)
    else:
        log("步骤 1b: 跳过 GPU 侧 clone（--skip-gpu-clone）")

    # 步骤 1c: ping GPU IP，不通直接报错退出
    log("步骤 1c: ping GPU 主机检查连通性")
    if not ping_host(args.gpu_host):
        log(f"错误: ping GPU 主机 {args.gpu_host} 不通，请检查网络后重试，退出")
        return 1

    # 步骤 2-3: NPU 本地编译 + 安装（使用未带代理的 env）
    build_and_install(fla_dir, env)

    # 步骤 3.5: 查询 python 路径，设置 PYTHONHOME / ASCEND_CUSTOM_OPP_PATH
    log("步骤 3.5: 设置 NPU 精度检查环境变量 PYTHONHOME / ASCEND_CUSTOM_OPP_PATH")
    pythonhome, site_packages = query_python_paths(env)
    npu_env = apply_npu_atk_env(env, pythonhome, site_packages)

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

        # 4a. SSH 后台启动 GPU server
        try:
            ssh_start_gpu_server(ssh_prefix, ssh_target, args.gpu_clone_dir,
                                 op, args)
        except Exception as e:
            log(f"SSH 启动 GPU server 异常: {e}，跳过 {op}")
            continue

        # 4b. 等待 GPU server 可达
        if not wait_server_up(args.gpu_host, port, args.server_wait_timeout):
            log(f"GPU server 未就绪，跳过 {op}")
            ssh_stop_gpu_server(ssh_prefix, ssh_target, args.gpu_clone_dir,
                                op, args.gpu_container)
            continue

        # 4c. NPU 本地执行精度检查
        npu_cmd = [
            "bash", str(gpu_server_sh),
            f"-op={op}",
            "-action=npu",
            f"-gpu_host={args.gpu_host}",
            f"-gpu_host_port={args.gpu_host_port}",
            f"-npu_device_id={args.npu_device_id}",
            f"-npu_gpu_device_id={args.npu_gpu_device_id}",
            f"-npu_output_path={args.npu_output_path}",
            f"-npu_task_timeout={args.npu_task_timeout}",
        ]
        log(f"执行精度检查: {' '.join(npu_cmd)}")
        try:
            run(npu_cmd, check=False, cwd=atk_dir / op, env=npu_env)
        except Exception as e:
            log(f"精度检查异常: {e}")

        # 4d. SSH 停止 GPU server（杀掉容器，后台 start 进程随之退出）
        ssh_stop_gpu_server(ssh_prefix, ssh_target, args.gpu_clone_dir,
                            op, args.gpu_container)

        # 等待端口关闭，确保下一个算子的 server 能干净启动
        time.sleep(10)
        log(f"[{idx}/{len(ops)}] {op} 完成")

    log("所有算子处理完毕")
    return 0


if __name__ == "__main__":
    sys.exit(main())
