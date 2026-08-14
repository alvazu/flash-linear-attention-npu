# RecurrentGatedDeltaRule ATK 精度测试归档

本目录归档 `recurrent_gated_delta_rule` 算子的 ATK 看护资产。算子仅覆盖 A2
`ascend910b` 的 `ascendc` route，冻结 1 条 smoke/精度 case，使用同机 CPU
提供 Torch FP64 真值和同精度 Torch 对照，不依赖 GPU 或独立 ATK server。

通用版本、case 范围、精度标准和复检规则见 [`../README.md`](../README.md)。

> 注：算子已在 OPP 中注册，但 `fla_npu.ops.ascendc` 暂无公开的 ctypes wrapper。
> NPU 路径当前无法调用，精度任务只能以 CPU 同精度对照 + CPU Torch FP64 golden
> 双标杆形式空跑；待公开 wrapper 后再补 NPU DUT 执行结果。

## 1. 范围

| 项 | 值 |
| --- | --- |
| 公开 Python API | `fla_npu.ops.ascendc.recurrent_gated_delta_rule`（暂未公开 ctypes wrapper） |
| ACLNN 名称 | `RecurrentGatedDeltaRule` |
| 源算子 | `fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule` |
| SOC | `ascend910b`（A2） |
| route | `ascendc` |
| 冻结 case | 1 条 smoke/精度 |
| 精度标准 | ATK `cv_fused_double_benchmark`：`max_re_ratio=5`、`avg_re_ratio=1.5`、`root_mean_squared_ratio=1.5` |
| 性能 | `perf: not_key` |

冻结 case 矩阵：

| case ID | seed | B | H | HV | K | V | T | dtype | layout | varlen | cu_seqlens | explicit_chunk_indices | tags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 20260813 | 1 | 4 | 4 | 128 | 128 | 128 | bf16 | BSND | false | "" | false | accuracy,smoke |

唯一 case 覆盖 `B=1`、`H=HV=4`、`K=V=128`、`T=128`、`dtype=bf16`、`layout=BSND`、
固定长度，验证 Gated Delta Rule 递归前向在最小 shape 下的输出。由于 NPU 路径暂未
公开 ctypes wrapper，本 case 当前仅作 CPU 双标杆空跑与 case 资产冻结，不作为
NPU 精度通过结论。

## 2. 文件

```text
atk_recurrent_gated_delta_rule.json
recurrent_gated_delta_rule.yaml
executor_recurrent_gated_delta_rule.py
gen_recurrent_gated_delta_rule.py
README.md
```

`executor_recurrent_gated_delta_rule.py` 注册为 `executor_recurrent_gated_delta_rule`，复用
`_shared/atk_executor_common.py` 的 `GeneratedAtkApiMixin`，NPU 路径调用仓内
Ascend C ctypes wrapper（当前未公开），CPU 路径提供确定性 PyTorch 参考实现。

## 3. CPU 双标杆拓扑

CPU 双标杆不依赖 GPU，也不需要单独启动 ATK server：

```text
NPU host
  atk task
  |-- local NPU DUT（当前不可用：ctypes wrapper 未公开）
  |-- local CPU same-precision control
  `-- local CPU Torch FP64 golden
```

普通 CPU 任务使用与算子输入一致的 dtype，并按模型计算边界量化中间结果；
`cpu_benchmark` 任务使用 Torch FP64 计算真值。ATK 仍使用 YAML 中配置的
`cv_fused_double_benchmark` 原生双标杆标准，不在 executor 中另设精度阈值。

### 3.1 NPU 环境准备

在 NPU 机器加载 ATK、CANN、当前构建的 OPP 和仓内 Python 包：

```bash
source "$ATK_ENV/bin/activate"
source <cann_install_path>/set_env.sh
source <fla_npu_install_path>/vendors/fla_npu_transformer/bin/set_env.bash

export ASCEND_RT_VISIBLE_DEVICES=<physical_npu_device>
export PYTHONPATH="$REPO_ROOT/torch_custom/fla_npu:$REPO_ROOT:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR=<writable_cache_dir>

cd "$REPO_ROOT/test/recurrent_gated_delta_rule"
atk --version
python -c 'import fla_npu; print(fla_npu.__file__)'
npu-smi info -i <physical_npu_device>
```

先构建匹配的 OPP 包：

```bash
bash build.sh --pkg --soc=ascend910b \
  --vendor_name=fla_npu --ops=recurrent_gated_delta_rule
```

`atk --version` 应为 `26.7.8`。若设置 `ASCEND_RT_VISIBLE_DEVICES`，后续 ATK
`node --devices` 传映射后的逻辑编号，不要同时传物理编号。

> 当前 `fla_npu.ops.ascendc.recurrent_gated_delta_rule` 未公开 ctypes wrapper，
> NPU DUT 路径会在调用阶段报错。在 wrapper 公开前，仅以 CPU 双标杆空跑验证
> case 资产与 executor 逻辑，不作为 NPU 精度结论。

### 3.2 执行命令

```bash
atk node --name npu_dut --backend npu --devices 0 \
    --output_path ./atk_output/cpu_dual_reference \
  node --name cpu_reference --backend cpu \
    --output_path ./atk_output/cpu_dual_reference \
  task \
    -c ./atk_recurrent_gated_delta_rule.json \
    --task accuracy \
    --bm_device cpu \
    -p ./executor_recurrent_gated_delta_rule.py \
    -s 0 \
    -e 1 \
    -sp \
    -mt 1 \
    -to 14400
```

`-s 0 -e 1` 覆盖全部 1 条 case。`-sp` 使 case 复用 CPU 标杆缓存；NPU DUT 仍
独立执行（当前因 wrapper 未公开会失败）。

只有最终报告同时满足 `Total Task: 1, success 1, failed 0` 和
`acc_pass_result: Pass` 才能作为全量精度通过结论。仅执行成功或输出全为有限值
不等价于精度通过。在 ctypes wrapper 公开前，NPU DUT 路径无法贡献通过结论。

## 4. 生成与校验

```bash
cd "$REPO_ROOT/test/recurrent_gated_delta_rule"
atk case \
  -f ./recurrent_gated_delta_rule.yaml \
  -p ./gen_recurrent_gated_delta_rule.py \
  -dt 1 \
  -en 0 \
  -s 20260813

python3 ./gen_recurrent_gated_delta_rule.py \
  --output ./atk_recurrent_gated_delta_rule.generated.json \
  --summary
```

相同 YAML、gen 和 seed 必须生成稳定 case ID 与稳定结构。生成后检查 ATK schema、
case 数量、SOC、route、dtype、layout 和覆盖摘要，不要静默覆盖已评审的
`atk_recurrent_gated_delta_rule.json`。

## 5. 单 case 定位

定位某一条 case 时保留 `--save_data output`，并分析三路结果：

```bash
atk node --name npu_dut --backend npu --devices 0 \
    --output_path ./atk_output/case_debug \
  node --name cpu_reference --backend cpu \
    --output_path ./atk_output/case_debug \
  task \
    -c ./atk_recurrent_gated_delta_rule.json \
    --task accuracy \
    --bm_device cpu \
    -p ./executor_recurrent_gated_delta_rule.py \
    -s <case_index> \
    -e <case_index_plus_one> \
    --save_data output \
    -sp \
    -mt 1 \
    -to 14400
```

保存输出后可用 ATK 配套 CT 工具对失败输出做三路可视化：

```bash
ct viz \
  <npu_output_0.pt> \
  <cpu_benchmark_output_0.pt> \
  <cpu_control_output_0.pt> \
  --out_dir <viz_output_dir> \
  --name recurrent_gated_delta_rule_out \
  --spatial
```

## 6. 常见失败

| 现象 | 原因与处理 |
| --- | --- |
| NPU 报 `No module named 'fla_npu'` | `PYTHONPATH` 未包含 `torch_custom/fla_npu`，或当前 OPP/Python 包不是同一提交。按第 3.1 节重新加载。 |
| NPU 报算子不存在 | 未构建/安装匹配 SOC 的 OPP 包。按第 3.1 节构建 `ascend910b` 的 `recurrent_gated_delta_rule` OPP。 |
| NPU 报 ctypes wrapper 未公开 | `fla_npu.ops.ascendc.recurrent_gated_delta_rule` 尚未公开。当前仅 CPU 双标杆空跑可用，待 wrapper 公开后补 NPU DUT 结果。 |
| 设备号不可用 | 物理设备经 `ASCEND_RT_VISIBLE_DEVICES` 后会重新编号；ATK 使用映射后的逻辑编号。 |
| 执行成功但 `acc_pass_result: Failed` | 检查 ATK 报告中每条 case 的 `max_re_ratio`、`avg_re_ratio`、`root_mean_squared_ratio` 是否超出阈值；保存输出用 `ct viz` 定位结构性差异。 |
| 生成结果与已评审 JSON 不一致 | 比较生成器、YAML 和 seed；不要静默覆盖 `atk_recurrent_gated_delta_rule.json`。 |

结果归档和公开 PR/issue 只记录测试项、case 范围、通过/失败结论和必要的非敏感错误摘要，
不得记录服务器地址、账号、绝对路径、容器名、token 或内部日志路径。
