# FastGelu ATK 精度测试归档

本目录归档 `fast_gelu_custom` 算子的 ATK 看护资产。算子仅覆盖 A2
`ascend910b` 的 `ascendc` route，冻结 3 条 smoke/精度 case，使用同机 CPU
提供 Torch FP64 真值和同精度 Torch 对照，不依赖 GPU 或独立 ATK server。

通用版本、case 范围、精度标准和复检规则见 [`../README.md`](../README.md)。

## 1. 范围

| 项 | 值 |
| --- | --- |
| 公开 Python API | `fla_npu.ops.ascendc.fast_gelu_custom` |
| ACLNN 名称 | `FastGelu` |
| 源算子 | `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py` |
| SOC | `ascend910b`（A2） |
| route | `ascendc` |
| 冻结 case | 3 条 smoke/精度 |
| 精度标准 | ATK `cv_fused_double_benchmark`：`max_re_ratio=5`、`avg_re_ratio=1.5`、`root_mean_squared_ratio=1.5` |
| 性能 | `perf: not_key` |

冻结 case 矩阵：

| case ID | seed | shape_key | dtype | shape | tags |
| --- | --- | --- | --- | --- | --- |
| 0 | 20260813 | vector_fp32 | fp32 | [1024] | accuracy,smoke |
| 1 | 20260814 | matrix_bf16 | bf16 | [8, 256] | accuracy,smoke |
| 2 | 20260815 | matrix_fp16 | fp16 | [4, 512] | accuracy,smoke |

三条 case 共享 `cu_seqlens=""`、`explicit_chunk_indices=false`，分别覆盖
1-D 向量（`vector_fp32`）与 2-D 矩阵（`matrix_bf16`、`matrix_fp16`）三种 dtype
下的 FastGelu 元素级激活输出。

## 2. 文件

```text
atk_fast_gelu_custom.json
fast_gelu_custom.yaml
executor_fast_gelu_custom.py
gen_fast_gelu_custom.py
README.md
```

`executor_fast_gelu_custom.py` 注册为 `executor_fast_gelu_custom`，复用
`_shared/atk_executor_common.py` 的 `GeneratedAtkApiMixin`，NPU 路径调用仓内
Ascend C ctypes wrapper，CPU 路径提供确定性 PyTorch 参考实现。

## 3. CPU 双标杆拓扑

CPU 双标杆不依赖 GPU，也不需要单独启动 ATK server：

```text
NPU host
  atk task
  |-- local NPU DUT
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

cd "$REPO_ROOT/test/fast_gelu_custom"
atk --version
python -c 'import fla_npu; from fla_npu.ops.ascendc import fast_gelu_custom; print(fla_npu.__file__)'
npu-smi info -i <physical_npu_device>
```

先构建匹配的 OPP 包：

```bash
bash build.sh --pkg --soc=ascend910b \
  --vendor_name=fla_npu --ops=fast_gelu_custom
```

`atk --version` 应为 `26.7.8`。若设置 `ASCEND_RT_VISIBLE_DEVICES`，后续 ATK
`node --devices` 传映射后的逻辑编号，不要同时传物理编号。

### 3.2 执行命令

```bash
atk node --name npu_dut --backend npu --devices 0 \
    --output_path ./atk_output/cpu_dual_reference \
  node --name cpu_reference --backend cpu \
    --output_path ./atk_output/cpu_dual_reference \
  task \
    -c ./atk_fast_gelu_custom.json \
    --task accuracy \
    --bm_device cpu \
    -p ./executor_fast_gelu_custom.py \
    -s 0 \
    -e 3 \
    -sp \
    -mt 1 \
    -to 14400
```

`-s 0 -e 3` 覆盖全部 3 条 case。`-sp` 使三条 case 复用 CPU 标杆缓存；每条 NPU
DUT 仍独立执行。

只有最终报告同时满足 `Total Task: 3, success 3, failed 0` 和
`acc_pass_result: Pass` 才能作为全量精度通过结论。仅执行成功或输出全为有限值
不等价于精度通过。

## 4. 生成与校验

```bash
cd "$REPO_ROOT/test/fast_gelu_custom"
atk case \
  -f ./fast_gelu_custom.yaml \
  -p ./gen_fast_gelu_custom.py \
  -dt 1 \
  -en 0 \
  -s 20260813

python3 ./gen_fast_gelu_custom.py \
  --output ./atk_fast_gelu_custom.generated.json \
  --summary
```

相同 YAML、gen 和 seed 必须生成稳定 case ID 与稳定结构。生成后检查 ATK schema、
case 数量、SOC、route、shape_key、dtype 和覆盖摘要，不要静默覆盖已评审的
`atk_fast_gelu_custom.json`。

## 5. 单 case 定位

定位某一条 case 时保留 `--save_data output`，并分析三路结果：

```bash
atk node --name npu_dut --backend npu --devices 0 \
    --output_path ./atk_output/case_debug \
  node --name cpu_reference --backend cpu \
    --output_path ./atk_output/case_debug \
  task \
    -c ./atk_fast_gelu_custom.json \
    --task accuracy \
    --bm_device cpu \
    -p ./executor_fast_gelu_custom.py \
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
  --name fast_gelu_custom_out \
  --spatial
```

## 6. 常见失败

| 现象 | 原因与处理 |
| --- | --- |
| NPU 报 `No module named 'fla_npu'` | `PYTHONPATH` 未包含 `torch_custom/fla_npu`，或当前 OPP/Python 包不是同一提交。按第 3.1 节重新加载。 |
| NPU 报算子不存在 | 未构建/安装匹配 SOC 的 OPP 包。按第 3.1 节构建 `ascend910b` 的 `fast_gelu_custom` OPP。 |
| 设备号不可用 | 物理设备经 `ASCEND_RT_VISIBLE_DEVICES` 后会重新编号；ATK 使用映射后的逻辑编号。 |
| 执行成功但 `acc_pass_result: Failed` | 检查 ATK 报告中每条 case 的 `max_re_ratio`、`avg_re_ratio`、`root_mean_squared_ratio` 是否超出阈值；保存输出用 `ct viz` 定位结构性差异。 |
| 生成结果与已评审 JSON 不一致 | 比较生成器、YAML 和 seed；不要静默覆盖 `atk_fast_gelu_custom.json`。 |

结果归档和公开 PR/issue 只记录测试项、case 范围、通过/失败结论和必要的非敏感错误摘要，
不得记录服务器地址、账号、绝对路径、容器名、token 或内部日志路径。
