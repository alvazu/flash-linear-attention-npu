# ChunkKdaBwdIntra ATK 工程

本目录提供 `chunk_kda_bwd_intra` 的 ATK 单算子工程，包含 `executor_chunk_kda_bwd_intra.py`、`gen_chunk_kda_bwd_intra.py`、`chunk_kda_bwd_intra.yaml`、`atk_chunk_kda_bwd_intra.json`。

## 输入约束

- dense `BNSD` 语义下，`q/k/gk/dq/dk/dg` 为 `[B,H,T,K]`，`beta/db` 为 `[B,H,T]`，`dAqk/dAkk` 为 `[B,H,T,chunk_size]`。
- 兼容 `BSND` 输入时，executor 在本目录内完成与算子约定一致的维度排列。
- `q/k/gk/dq/dk/dg` 形状必须一致，`beta/db` 形状必须一致，`dAqk/dAkk` 形状必须一致。
- `q/k` 仅支持 `BFLOAT16`；`beta` 支持 `BFLOAT16/FLOAT`；首版 kernel 要求 `safe_gate=true`。
- `chunk_size` 固定为 `64`；dense 模式支持 `K=64/128/256`，varlen 模式支持 `K=128`。
- varlen `TND` 模式需要 `cu_seqlens` 和 packed chunk metadata，且 `cu_seqlens` 元素数不超过 `65`。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_intra.py; fla/ops/ascendc/kda/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_chunk_kda_bwd_intra.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

- `bf16_beta_bf16`: `{"name": "bf16_beta_bf16", "dtype": "bf16", "beta_dtype": "bf16", "B": 1, "T": 16, "H": 1, "K": 128, "chunk_size": 64, "layout": "BSND", "op": "chunk_kda_bwd_intra", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`
- `bf16_beta_fp32`: `{"name": "bf16_beta_fp32", "dtype": "bf16", "beta_dtype": "fp32", "B": 1, "T": 16, "H": 1, "K": 128, "chunk_size": 64, "layout": "BSND", "op": "chunk_kda_bwd_intra", "case_id": 1, "seed": 20260818, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_intra -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_intra -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_intra -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_intra -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_intra -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_intra -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
