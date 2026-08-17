# ChunkFwdO ATK 工程

本目录提供 `chunk_fwd_o` 的 ATK 单算子工程，包含 `executor_chunk_fwd_o.py`、`gen_chunk_fwd_o.py`、`chunk_fwd_o.yaml`、`atk_chunk_fwd_o.json`。

## 输入约束

q/k=[B,HK,T,K], v/o=[B,HV,T,V], h=[B,HV,num_chunks,K,V], g=[B,HV,T]; B=1, HK=1, HV=1, T=16, K=128, V=128, chunk_size=64.

默认用例均使用定长小 shape，用于提升精度、性能、确定性和内存检测速度。

## 标杆来源

torch_custom/fla_npu/test/test_fwd_o.py; fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_chunk_fwd_o.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

- `bf16_small`: `{"name": "bf16_small", "dtype": "bf16", "B": 1, "HK": 1, "HV": 1, "T": 16, "K": 128, "V": 128, "chunk_size": 64, "op": "chunk_fwd_o", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`
- `fp16_small`: `{"name": "fp16_small", "dtype": "fp16", "B": 1, "HK": 1, "HV": 1, "T": 16, "K": 128, "V": 128, "chunk_size": 64, "op": "chunk_fwd_o", "case_id": 1, "seed": 20260818, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
