# KdaGateCumsum ATK 工程

本目录提供 `kda_gate_cumsum` 的 ATK 单算子工程，包含 `executor_kda_gate_cumsum.py`、`gen_kda_gate_cumsum.py`、`kda_gate_cumsum.yaml`、`atk_kda_gate_cumsum.json`。

## 输入约束

- `g` 可按 rank 4 `[B,T,HV,K]` 或 `[B,HV,T,K]`，也可按 rank 3 `[T,HV,K]` 或 `[HV,T,K]` 输入；layout 由 wrapper 传入的逻辑属性解释。
- 逻辑 `B/T/HV/K` 均必须为正数，且 `K <= 256`。
- `g` 支持 `FLOAT/BFLOAT16/FLOAT16`；输出为 FP32 gate cumsum，用于后续 `exp2(gk)`。
- `chunk_size` 按时间维分块；`use_gate_in_kernel=true` 时必须提供 `A_log`，`dt_bias` 可选。
- `safe_gate=true` 时使用 `lower_bound * sigmoid(...)`，`lower_bound` 需要为负值；`use_gate_in_kernel=false` 时 `g` 被视作已激活 gate。
- 变长模式通过 `cu_seqlens` 描述序列累计长度；当前 ATK 用例不传 `A_log/dt_bias/cu_seqlens`。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

tests/operators/_shared/chunk_kda_backend.py; fla/ops/ascendc/kda/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_kda_gate_cumsum.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 1, "HV": 1, "T": 16, "K": 128, "chunk_size": 64, "op": "kda_gate_cumsum", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`
- FP16 用例：`{"dtype": "fp16", "B": 1, "HV": 1, "T": 16, "K": 128, "chunk_size": 64, "op": "kda_gate_cumsum", "case_id": 1, "seed": 20260818, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=kda_gate_cumsum -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=kda_gate_cumsum -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=kda_gate_cumsum -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=kda_gate_cumsum -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=kda_gate_cumsum -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=kda_gate_cumsum -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
