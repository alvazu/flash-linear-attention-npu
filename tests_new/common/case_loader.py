"""tests_new 的配置和 case 加载工具。

设计目标是让每个算子的 cases.yaml 尽量小：
- cases.yaml 只描述“怎么构造这个 case”；
- tests_new/config.yaml 统一描述“哪些 case 用于 routes/accuracy/perf”；
- 命令行 --case 或环境变量 FLA_NPU_CASE_IDS 可以显式多选任意 case。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_NEW_ROOT = REPO_ROOT / "tests_new"
CASE_ROOT = TESTS_NEW_ROOT / "op_cases"
CONFIG_PATH = TESTS_NEW_ROOT / "config.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    """读取一个 YAML 文件，并保证顶层是 mapping。"""

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: YAML 顶层必须是 mapping")
    return data


def load_config() -> dict[str, Any]:
    """读取 tests_new 根配置。

    根配置集中保存公共随机数据策略、精度比较策略、perf 默认参数，以及每个算子的覆盖矩阵。
    """

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"测试框架配置不存在: {CONFIG_PATH}")
    return _read_yaml(CONFIG_PATH)


def operator_config(op_name: str) -> dict[str, Any]:
    """返回某个算子的集中配置。"""

    config = load_config()
    try:
        return config["operators"][op_name]
    except KeyError as exc:
        raise KeyError(f"tests_new/config.yaml 中没有算子配置: {op_name}") from exc


def common_config() -> dict[str, Any]:
    """返回公共默认配置。"""

    return load_config().get("common", {})


def load_manifest(op_name: str) -> dict[str, Any]:
    """读取某个算子的 cases.yaml。"""

    path = CASE_ROOT / op_name / "cases.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"case manifest not found: {path}")
    manifest = _read_yaml(path)
    if manifest.get("op") != op_name:
        raise ValueError(f"{path}: op 必须是 {op_name!r}，实际是 {manifest.get('op')!r}")
    return manifest


def _split_csv(value: str | None) -> list[str]:
    """把逗号分隔的 case ID 拆开，保留用户给出的顺序。"""

    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _explicit_case_ids(case_ids: Iterable[str] = ()) -> list[str]:
    """合并命令行 --case 和环境变量 FLA_NPU_CASE_IDS。"""

    result: list[str] = []
    for value in case_ids:
        result.extend(_split_csv(str(value)))
    result.extend(_split_csv(os.environ.get("FLA_NPU_CASE_IDS")))
    return result


def has_explicit_case_selection(case_ids: Iterable[str] = ()) -> bool:
    """判断用户是否显式指定了 case。"""

    return bool(_explicit_case_ids(case_ids))


def _coverage_case_ids(op_name: str, test_type: str) -> list[str]:
    """读取某个测试类型的默认 case 列表。"""

    coverage = operator_config(op_name).get("coverage_requirements", {})
    test_config = coverage.get(test_type, {})
    return list(test_config.get("case_ids", ()))


def _case_map(op_name: str) -> dict[str, dict[str, Any]]:
    """按 ID 建立 case 索引，同时检查重复 ID。"""

    cases = load_manifest(op_name).get("cases", [])
    mapping: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = case.get("id")
        if not case_id:
            raise ValueError(f"{op_name}: case 缺少 id")
        if case_id in mapping:
            raise ValueError(f"{op_name}: case id 重复: {case_id}")
        mapping[case_id] = case
    return mapping


def _with_runtime_defaults(op_name: str, case: dict[str, Any]) -> dict[str, Any]:
    """给 case 注入运行期默认配置，避免在 cases.yaml 重复公共字段。"""

    selected = dict(case)
    selected["_common_config"] = common_config()
    selected["_op_config"] = operator_config(op_name)
    return selected


def select_cases(
    op_name: str,
    *,
    test_type: str,
    case_ids: Iterable[str] = (),
    default_first: bool = False,
    soc: str | None = None,
) -> list[dict[str, Any]]:
    """按测试类型选择 case。

    未显式指定 --case 时，从 config.yaml 的 coverage_requirements 中取默认列表；
    显式指定 --case 时，直接从 op_cases/<op>/cases.yaml 中按 ID 多选。
    routes 调用方会传 default_first=True，因此默认只跑第一个 route case。
    """

    op_cfg = operator_config(op_name)
    requested_soc = soc or os.environ.get("FLA_NPU_SOC")
    if requested_soc and requested_soc not in op_cfg.get("capability", {}).get("soc", []):
        raise ValueError(f"{op_name}: config.yaml 未声明支持 SOC {requested_soc}")

    mapping = _case_map(op_name)
    ids = _explicit_case_ids(case_ids)
    if not ids:
        ids = _coverage_case_ids(op_name, test_type)
    if not ids:
        ids = list(mapping)
    if default_first and not has_explicit_case_selection(case_ids):
        ids = ids[:1]

    missing = [case_id for case_id in ids if case_id not in mapping]
    if missing:
        raise KeyError(f"{op_name}: 未找到 case: {', '.join(missing)}")
    return [_with_runtime_defaults(op_name, mapping[case_id]) for case_id in ids]


def chunk_indices_from_cu_seqlens(cu_seqlens: Iterable[int], chunk_size: int) -> list[int]:
    """按 sequence-major 顺序从 cu_seqlens 推导 chunk_indices。"""

    values = [int(value) for value in cu_seqlens]
    result: list[int] = []
    for seq_idx, (start, end) in enumerate(zip(values[:-1], values[1:])):
        length = end - start
        if length < 0:
            raise ValueError(f"cu_seqlens 必须单调非降，实际为 {values}")
        for chunk_idx in range((length + int(chunk_size) - 1) // int(chunk_size)):
            result.extend((seq_idx, chunk_idx))
    return result


def resolve_optional_inputs(case: dict[str, Any]) -> dict[str, Any]:
    """返回可选输入；其中 chunk_indices: derived 会自动推导为 list[int]。"""

    optional = dict(case.get("optional_inputs") or {})
    chunk_size = int(case["attrs"]["chunk_size"])
    cu_seqlens = optional.get("cu_seqlens")
    chunk_indices = optional.get("chunk_indices")
    if cu_seqlens is not None and chunk_indices == "derived":
        optional["chunk_indices"] = chunk_indices_from_cu_seqlens(cu_seqlens, chunk_size)
    return optional


def chunk_count(case: dict[str, Any]) -> int:
    """计算状态张量的 chunk 数。"""

    if "N_c" in case.get("shape", {}):
        return int(case["shape"]["N_c"])
    optional = resolve_optional_inputs(case)
    chunk_size = int(case["attrs"]["chunk_size"])
    cu_seqlens = optional.get("cu_seqlens")
    if cu_seqlens is not None:
        return len(chunk_indices_from_cu_seqlens(cu_seqlens, chunk_size)) // 2
    time = int(case["shape"]["T"])
    return (time + chunk_size - 1) // chunk_size


def data_generation_config(case: dict[str, Any]) -> dict[str, Any]:
    """读取输入生成参数，默认来自 config.yaml/common/data_generation。"""

    return dict(case.get("_common_config", {}).get("data_generation", {}))


def accuracy_config(case: dict[str, Any]) -> dict[str, Any]:
    """读取 accuracy 公共配置。"""

    return dict(case.get("_common_config", {}).get("accuracy", {}))


def perf_config(case: dict[str, Any]) -> dict[str, Any]:
    """读取 perf 公共配置。"""

    return dict(case.get("_common_config", {}).get("perf", {}))
