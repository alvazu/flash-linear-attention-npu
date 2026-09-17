#!/usr/bin/env bash
# 用法: apply_branch_protection.sh [branch] [profile]
#   branch  默认 main
#   profile 默认 main；legacy 用于 v26.9.0 / v26.6.0 等使用
#          "NPU CI / 手动验证" + "NPU CI / 精度检查" 门禁的分支
set -euo pipefail

repo="${GITHUB_REPOSITORY:-flashserve/flash-linear-attention-npu}"
branch="${1:-main}"
profile="${2:-main}"
api_url="${GITHUB_API_URL:-https://api.github.com}"
token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"

if [[ -z "$token" ]]; then
    echo "GITHUB_TOKEN or GH_TOKEN with repository administration permission is required." >&2
    exit 2
fi

case "${profile}" in
    main)
        contexts=(
            "NPU CI / A2+A5 / 01 环境、wheel 与运行时契约"
            "NPU CI / A2+A5 / 02 全量 OPP 构建"
            "NPU CI / A2+A5 / 03 torch_custom wheel 与 OPP 布局"
            "NPU CI / A2+A5 / 04 OPP 安装与 PyTorch 适配"
            "NPU CI / A2+A5 / 05 GDR Example/ST"
            "NPU CI / A2+A5 / 06 chunk_fwd_o 局部覆盖安装"
            "NPU CI / A2+A5 / 07 报告与 commit 校验"
            "CI 契约测试"
        )
        ;;
    legacy)
        contexts=(
            "NPU CI / 手动验证"
            "NPU CI / 精度检查"
        )
        ;;
    *)
        echo "Unknown context profile: ${profile} (supported: main, legacy)" >&2
        exit 2
        ;;
esac

payload="$(python3 - "${contexts[@]}" <<'PY'
import json
import sys

contexts = sys.argv[1:]
protection = {
    "required_status_checks": {
        "strict": True,
        "contexts": contexts,
    },
    "enforce_admins": True,
    "required_pull_request_reviews": {
        "dismissal_restrictions": {},
        "dismiss_stale_reviews": True,
        "require_code_owner_reviews": True,
        "require_last_push_approval": True,
        "required_approving_review_count": 2,
        "bypass_pull_request_allowances": {
            "users": [
                "weinachuan"
            ],
            "teams": [],
            "apps": [],
        },
    },
    "restrictions": None,
    "required_conversation_resolution": True,
    "allow_force_pushes": False,
    "allow_deletions": False,
    "block_creations": False,
}
print(json.dumps(protection, ensure_ascii=False))
PY
)"

curl -fsSL \
    -X PUT \
    -H "Authorization: Bearer ${token}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "${api_url}/repos/${repo}/branches/${branch}/protection" \
    -d "${payload}"

echo "Applied ${profile} branch protection to ${repo}:${branch}."
