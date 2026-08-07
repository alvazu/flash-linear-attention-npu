# tests_new

`tests_new` 是一个尽量轻量的算子测试框架。每个算子的 `cases.yaml` 只保留构造用例必须的信息，公共配置和覆盖矩阵统一放在根目录 [config.yaml](C:/github/flash-linear-attention-npu/tests_new/config.yaml)。

## 目录结构

| 路径 | 用途 |
| --- | --- |
| `config.yaml` | 公共依赖说明、随机数据默认参数、accuracy/perf 默认配置、各算子的覆盖矩阵。 |
| `op_cases/<op>/cases.yaml` | 算子用例，只写 shape、dtype、attrs、optional_inputs、seed。 |
| `op_golden/<op>/` | 算子输入构造、CPU 标杆、三条调用通路适配。 |
| `op_routes/<op>/` | `aclnn`、直接 `<<<>>>`、`fla_npu.ops.ascendc.<op>` 三条通路 smoke 脚本。 |
| `op_perf/<op>/` | 算子性能 smoke 脚本。 |
| `common/` | 公共加载、运行时、shape/finite 检查和 `ct.dual` 对比。 |

## Requirements

运行前请准备：

- `pyyaml`
- `torch`
- `torch_npu`
- 已安装当前仓库构建出的 `fla_npu`
- `ct` 精度对比工具

`accuracy` 使用如下形式做精度对比：

```python
import ct

assert ct.dual(
    actual.cpu().to(torch.float16),
    truth_float64.cpu().to(torch.float16),
    control.cpu(),
)["success"]
```

其中 `actual` 是待测 NPU 输出，`truth_float64` 是 float64 标杆计算结果，`control` 是普通精度 CPU 对照值。比较 dtype 默认在 `config.yaml/common/accuracy/compare_dtype` 中配置。

## 命令

```bash
bash tests_new/run.sh --device=0 --op=chunk_bwd_dqkwg --test=routes
bash tests_new/run.sh --device=0 --op=chunk_bwd_dqkwg --test=accuracy
bash tests_new/run.sh --device=0 --op=chunk_bwd_dqkwg --test=perf
bash tests_new/run.sh --device=0 --op=chunk_bwd_dqkwg --test=routes/accuracy/perf
```

选择 case：

```bash
bash tests_new/run.sh --device=0 --op=chunk_bwd_dqkwg --test=accuracy --case=chunk_bwd_dqkwg_dense_fp16
bash tests_new/run.sh --device=0 --op=chunk_bwd_dqkwg --test=perf --case=chunk_bwd_dqkwg_dense_fp16,chunk_bwd_dqkwg_varlen_tail
```

规则：

- `--case` 可重复，也可用逗号传多个 ID。
- 未指定 `--case` 时，accuracy/perf 运行 `config.yaml` 中对应列表的全部 case。
- routes 未指定 `--case` 时，只运行 `config.yaml` 中 routes 列表的第一个 case。
- 显式指定 `--case` 时，直接从 `op_cases/<op>/cases.yaml` 按 ID 选择，不依赖 tags、soc 或 run_on。

## chunk_bwd_dqkwg

当前已适配：

- `op_routes/chunk_bwd_dqkwg/test_ascendc_route.py`：调用 `fla_npu.ops.ascendc.chunk_bwd_dqkwg`。
- `op_routes/chunk_bwd_dqkwg/test_aclnn_route.py`：调用 `torch.ops.npu.npu_chunk_bwd_dqkwg`。
- `op_routes/chunk_bwd_dqkwg/test_direct_launch_route.py`：调用 fast-kernel 扩展暴露的 `torch.ops.ascend_ops.chunk_bwd_dqkwg`。
- `op_perf/chunk_bwd_dqkwg/perf.py`：可使用 `--case` 从 `op_cases` 中选择任意 case 做同步 wall-time smoke。

直接 `<<<>>>` route 需要先构建并安装 `examples/fast_kernel_launch_example` 中的 `ascend_ops` 扩展。
