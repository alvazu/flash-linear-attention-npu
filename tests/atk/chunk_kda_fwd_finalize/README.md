# chunk_kda_fwd_finalize ATK 测试

本目录只依赖八个公开输入：

```text
qg_scaled, aqk, v_new, h, cu_seqlens, chunk_indices,
output_layout, state_v_first
```

冻结 JSON 是 ATK 原生直接输入；没有 marker tensor、序列化的
`case_spec` 或隐藏测试开关。`qg_scaled/Aqk` 使用 Prepare 的
head-major BF16 形状，packed 时 rank-3；`v_new/h` 使用 FwdH 的
rank-4/rank-5 BF16 形状，h 为 NT-first `[B,C,HV,128,128]`，packed 时首维仍为 1。`Aqk`
已经乘过 scale，CPU 和 NPU 都不再重复缩放。

## 资产

| 文件 | 用例 | 覆盖 |
| --- | ---: | --- |
| `atk_chunk_kda_fwd_finalize.json` | 200 | 保留 main 的 25 个边界/shape/变长结构 × 4 个输出 layout × 2 种 state 轴顺序，仅更新 h shape |
| `atk_chunk_kda_fwd_finalize_perf.json` | 10 | 模型大 shape、dense/varlen，供单算 profiling |
| `atk_chunk_kda_fwd_finalize_mss.json` | 12 | 4 个输出 layout × 2 种 state 轴顺序的尾块输入，以及 AIV 搬运模板的 dense/packed、KV/VK 确定性输入 |

精度包括 T 为 `1/15/16/17/31/32/33/63/64/65` 的边界、
多 chunk、严格递增变长序列、显式/自动 canonical chunk indices、
不同 HV、`B=2` dense，以及 `B=384,HV=5,T=17` 和由 382 个单
token 序列加 33/64 行序列组成的 AIV 搬运模板 dense/varlen 边界。这些
用例同时覆盖不足 4 个 head 的尾组、显式 `chunk_indices` 及
`Aqk` 的 1/2/3/4 个 16-row 数据块。
packed 组中 `qg_scaled/Aqk` 是 rank-3，
`v_new` 是 rank-4。独立 Finalize 无 HK 输入，不把这些 case
记作 GVA 映射验证；需要在三阶段组合链路另测 HK<HV。

## Tiling 模板覆盖

| `USE_AIV_INPUT_MOVER` | 选择条件 | 精度普通用例 | 精度边界用例 | `_mss.json` 用例 | 适用 SoC | 预期核类型 |
| --- | --- | --- | --- | --- | --- | --- |
| `false` | A2/A3；或 A5 的 head 被拆分；或最小单核负载不足 8 个 chunk | 64-71 | 0-63、72-87、92-191 | 0-7 | A2/A3/A5 | AIC-only |
| `true` | A5、每个 work item 包含完整 value head，且最小单核负载至少 8 个 chunk | 192-199 | 88-91、192-199 | 8-11 | A5 | MIX AIC 1:2 |

case 88-91 是 dense 的 17 行尾块；case 192-199 在同一变长输入中
包含 1/33/64 行序列并覆盖四种 layout。两组均使用非 4 倍数 HV，
因此会进入第二个 head group 并验证不足 4 个 head 的尾组。上述
模板组合同时通过 `ASCENDC_TPL_KERNEL_TYPE_SEL` 声明核类型，不在
Kernel 内按数值 TilingKey 分支。NT-first 修改前的历史 runtime profile 中，性能
case 2 命中 `_0`/cube 实例，耗时 908.545 us；性能 case 3 命中
`_1_mix_aic`/mix 实例，耗时 1794.889 us，AIC/AIV block 为
28/56。性能结论以性能 JSON 对应的大 shape profiling 为准。

CPU golden 把四个直接输入恢复为 BF16 后，使用 FP64 两项矩阵乘与求和，
不提前舍入为 BF16；executor 最后将 FP64 结果转为 FP32 比较载荷，
匹配 ATK 原生 `mixed_tolerance_bm` 支持的 BF16 DUT / FP32 golden 类型对。
`run_cpu` 自身仍返回 FP64，供独立高精度检查。
NPU 节点用本 executor 的窄 aclnn
直调适配器验证设备实现。公开
`fla_npu.ops.ascendc.chunk_kda_fwd_finalize` 稳定入口由共享 Python
wrapper 提供，其接口验证不属于本 ATK 目录。

## 生成与静态核对

```bash
python3 tests/atk/chunk_kda_fwd_finalize/gen_chunk_kda_fwd_finalize.py \
  --output-dir tests/atk/chunk_kda_fwd_finalize --summary

cd tests/atk/chunk_kda_fwd_finalize
atk case -f ./chunk_kda_fwd_finalize.yaml \
  -p ./gen_chunk_kda_fwd_finalize.py -en 0 -s 20260914
```

`dtype_numbers: 200` 与 `shape_distributions: [[0,1.0]]` 在目标 ATK
生成 200 条；生成器覆盖 YAML 临时形状，并在导出前检查八个输入
顺序。冻结 JSON 不因 `atk case` 的物理 case ID 重新排序而覆盖。

```bash
python3 -m py_compile \
  tests/atk/chunk_kda_fwd_finalize/gen_chunk_kda_fwd_finalize.py \
  tests/atk/chunk_kda_fwd_finalize/executor_chunk_kda_fwd_finalize.py
```

## 精度

加载与待测算子配套的 custom OPP、ATK 和 Python runtime 后：

```bash
cd tests/atk/chunk_kda_fwd_finalize
atk node --backend npu --devices 0 -o ./atk_output/accuracy \
  node --backend cpu task \
  -c ./atk_chunk_kda_fwd_finalize.json \
  --task accuracy -p ./executor_chunk_kda_fwd_finalize.py \
  -sp -to 60
```

单条超过 60 秒判定超时。只有最终报告确认总任务 200、执行失败
0、精度结论通过，才能声称全量通过；不能仅依据 shell 退出码。
任意 case 未通过时先使用 `--save_data output` 保留实值，再定位
shape/索引/尾块问题或执行精度复检，不能改输入 range 或阈值掩盖失败。

## 性能与确定性

大 shape 只跑 NPU 单算。先核对目标 case name，再在单条 ATK
`run` 外包 `msopprof`；性能只读取目标 kernel 的 duration，不使用
Python wall time。例如性能集第 2 条（从 0 开始）：

```bash
msopprof \
  --application="atk node --backend npu --devices 0 task -c ./atk_chunk_kda_fwd_finalize_perf.json --task run -p ./executor_chunk_kda_fwd_finalize.py -s 2 -e 3 -sp -to 60" \
  --output=./atk_output/profile_case_2 \
  --aic-metrics=BasicInfo --launch-count=1 --warm-up=0 --kill=off
```

确定性使用 `_mss.json` 的十二条输入，覆盖八种输出布局/状态组合和
AIV 搬运模板的 dense/packed、KV/VK 四条分支，精确逐元素比较实际
可见结果；内存检查须先构建 sanitizer 对象，并确认运行时真正加载，
仅有 `IsFinite` 等辅助 kernel 的 sanitizer 记录不能视为 Finalize 通过。

```bash
atk node --backend npu --devices 0 task \
  -c ./atk_chunk_kda_fwd_finalize_mss.json \
  --task accuracy_dc -p ./executor_chunk_kda_fwd_finalize.py \
  -s 0 -e 12 -sp -to 60
```

## 2026-10-12 A2/A3 单 Stage 0 优化验证

基线为 `bead79ae45fca5d10a8414b215d4f762cc66538d`。只改变 A2/A3 的
chunk/head 分核：最终固定 `CG=1`，`Nwork=Nbase*HV`，grid-stride 遍历。
四项输入仍为 BF16，两项 MMAD 先初始化、再原位累加到同一 FP32 L0C，
一次 Fixpipe `F322BF16` 直接写 GM。无 Vector stage、UB 或 GM relay；
详见[设计](../../../fla/ops/ascendc/kda/chunk_kda_fwd_finalize/docs/design.md)。
本次 A2 全部精度 case 0–199 与 `_mss.json` 0–11 均使用
`USE_AIV_INPUT_MOVER=false`；前述 MIX 模板覆盖属于 A5。

### 正常构建性能与 CG 选择

实机为 `root@192.168.9.234` 的 device 1，Ascend910B3，20 AIC/40 AIV、
192 MiB L2，CANN 9.1.0、Python 3.10、torch 2.12.0+cu130、
torch_npu 2.12.0、ATK 26.8.8。所有候选在独立
`/data/huangjunzhe/GDN/work/kda_finalize_env` 中轮换安装同一算子的正常包，
trace/sanitizer 关闭。冻结性能 JSON 不变，`--disable_id_seed` 使用各 case
的 `default_seed`，`ATK_GM_INIT_MODE=off`，warm-up 3 次、采样 10 次。
以下仅统计目标 `aclnnChunkKdaFwdFinalize` 的 profiler
`Task Duration(us)` 中位数，不计 CPU、Python、安装和编译时间。

| case | B/HV/T、场景 | 基线 µs | CG4 µs | CG3 µs | CG2 µs | CG1 µs | CG1 耗时降低 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1/16/11264 dense | 247.6750 | 251.8950 | 280.4460 | 233.9940 | 226.7845 | 8.43% |
| 1 | 1/32/11264 dense | 527.8010 | 513.1205 | 514.4210 | 513.0300 | 512.8505 | 2.83% |
| 2 | 1/96/8192 dense | 1195.8640 | 1117.6730 | 1116.8925 | 1121.5620 | 1118.2925 | 6.49% |
| 3 | 1/96/16384 varlen | 2263.3850 | 2215.4240 | 2206.4340 | 2216.3240 | 2205.9340 | 2.54% |
| 4 | 2/16/11264 dense | 529.2210 | 513.1905 | 558.8615 | 511.5900 | 513.2110 | 3.03% |
| 5 | 4/96/128 dense | 36.6005 | 37.7710 | 36.7110 | 36.9210 | 36.4005 | 0.55% |
| 6 | 1/32/160 dense | 7.6100 | 8.9300 | 7.7805 | 7.8100 | 7.3100 | 3.94% |
| 7 | 6/6/1084 dense | 34.4705 | 40.1510 | 31.4310 | 30.2410 | 30.2505 | 12.24% |
| 8 | 1/8/32768 dense | 384.0375 | 378.8675 | 374.9570 | 372.8070 | 373.5980 | 2.72% |
| 9 | 1/32/65536 varlen | 3006.9400 | 2978.8595 | 3050.1605 | 2963.1995 | 2959.8590 | 1.57% |

CG1 在六个 case 最快，其余与最快候选相差 0.03%–0.32%，低于对应采样
波动；相比 CG4，它消除了 case 6/7 的约 17%/16% 退化。因此采用统一 CG1。
全部 10 个 CG1 case 各有 10 个有效样本，任务 10/10 成功；均为
`AI_CORE`、BlockDim 20、MixBlockDim 0，四项输入/输出为 `DT_BF16`，
AIV total/vector 时间均为 0。case 5/6 的小耗时收益与采样波动相近，
表格只记录本次中位数，不据此保证小 shape 的固定加速比。

复现正常性能采样：

```bash
ATK_GM_INIT_MODE=off OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
atk node --backend npu --devices 1 task \
  -c tests/atk/chunk_kda_fwd_finalize/atk_chunk_kda_fwd_finalize_perf.json \
  --task performance_device --performance_data 3,10,10 --save_data profile \
  --disable_id_seed -p tests/atk/chunk_kda_fwd_finalize/executor_chunk_kda_fwd_finalize.py \
  -s 0 -e 10 -sp -to 60
```

### 精度、同步与架构隔离

正常 CG1 包对原 200 条精度用例逐条运行三组固定种子，实际通过
600/600，执行失败 0。每条 case 的实际 seed 为
`20260914 + case_id + offset`，offset 分别为 0、100000、200000；
三组各 200 个不同 `(case_id, seed)` 组合。使用 `--disable_id_seed`，
dev4/dev5，CPU 标杆单线程、GM init 关闭；仅临时 JSON 副本的
`default_seed` 改动，冻结 shape、数据域、golden、standard 和 executor 均未修改。
原始 XLSX 的 summary、失败表及每条 `mixed_tolerance_bm` 结果分别核验，
600 条均为 True。每组 case 192–199 的 metadata cell 达到 Excel 的
32767 字符上限；可见 seed/standard 和 inputs 前缀与冻结调用 JSON 核对，
完整输入由实际调用 JSON 保留，精度结果列本身未截断。

稳定公开入口 `fla_npu.ops.ascendc.chunk_kda_fwd_finalize` 另测两条：
dense `B2/HV3/T65, BSND, KV` 和 packed `HV5/T99, TND, VK,
cu_seqlens=[0,1,34,99]`、自动 chunk indices。seed 为 20261012/20261013，
使用原生 BF16 `mixed_tolerance_bm` 对 FP64 golden，均 matched_ratio=1、
abs/ULP 失败 0，保留输入、DUT 和 golden。

独立 trace worktree/environment 在 device6 验证原有 case 88–91、192–199、
80–87，共 20/20 精度通过。每个 case 的 runtime BlockDim=20，全部
20 个 core 实际连续处理多个 work 并回绕 L0C slot；8 个 HV48/T129 case
还验证同 core 的 head 切换及 64/64/1 行 chunk。core0 的一组实际记录为：
`work=[0,20,40,60,80,100,120,140]`、
`head=[0,20,40,12,32,4,24,44]`、`slot=[0,1,0,1,0,1,0,1]`。
trace 插桩未修改正式版本的 34 个同步语句，所有事件 credit/Wait 与退出 drain 检查通过。
trace 仅存在于独立工作区，正式源码及正常性能包无 trace、额外 workspace 或同步。

A5 的 `__CCE_AICORE__=310` 预处理 token 与基线完全一致（2787 tokens，
SHA256 `3aac387caee64ec560cb6518c9cb44e7b344c6db1e70ea02f6f7b7858b3c3637`）；
Host 默认 `headGroupSize=0` 的 60,480 个参数组合与原版五项调度结果一致，
其中 18,016 个合法、42,464 个拒绝输入。新增 HostUT 用
`g++ -std=c++17 -Wall -Wextra -Werror` 编译运行通过，覆盖 CG1 dense/varlen、
小工作域、单 head、候选尾组和 uint32 work 溢出边界。A3 `ascend910_93`
单算子构建通过；本机器只有 A2，未进行 A3 实机精度、同步或性能验证。

### 构建与原始证据

证据目录：`/data/huangjunzhe/GDN/work/kda_finalize_20261012`。
最终源文件冻结后重构建 A2，全部 6 个库/kernel 二进制与 CG1 profiling 包
及当前 normalenv OPP 的 MD5/SHA256 完全一致。对应记录为
`final_production_source_manifest.json`、`final_rebuilt_a2_binary_comparison.json`、
`final_normal_runtime_verification.json`；未用旧装包或跳过 kernel 比对。
正常 A2 包 `final_normal.run` 的 SHA256 为
`f3498c42d56df9670f3f5a555e3504d4f69506215027b9197b8d2482f8e157b6`。
A3 包 `final_a3.run` 的 SHA256 为
`50956062d8018ca6ed46ce23b8bc923b8779d0d51ef1310413e128a0e360b3b9`。

`final_cg_sweep.json/.csv/.md` 保留各 case 的 shape、seed、全部采样和
50 个原始 profiler CSV/5 个 ATK XLSX 路径。基线与 CG2 早期脚本在 ATK
完成后出现尾部 shell 127，原始退出码保留；两者原始报告仍明确各 10 个
任务成功、0 失败及每 case 10 个完整采样。最终 CG1 精度/性能退出码均 0。
600 seed 证据在 `seed_sweep_runs/cg1/seed_sweep_summary.json` 和各组
`report_validation.json`；稳定入口见 `cg1_stable_api/`；同步实测见
`trace_evidence_summary.json` 与 `trace_logs/`。

### 正式四项门禁

使用仓库原始 `run_test_cpu.sh` 的对应 scope 完成全部门禁，正常包负责
accuracy/performance/determinism，独立的无 trace sanitizer 包负责
mssanitizer。两种包来自同一份冻结 CG1 正式源码。

| 门禁 | 包与设备 | 实际结果 |
| --- | --- | --- |
| accuracy | normalenv / dev4 | 200/200 执行成功、`mixed_tolerance_bm` 全部 True，0 失败 |
| performance | normalenv / dev4 | 10/10 任务成功；冻结 standard 为 `not_key`，性能收益按前述同卡 profiler 表判定 |
| determinism | normalenv / dev4 | 12/12 全部 True；每条 50 次执行、49 次 `torch.equal` 精确比较，共 600 次执行/588 次比较，无数值容差 |
| mssanitizer memcheck | trace_env 中安装无 trace sanitizer 包 / dev6 | 12/12 memory Pass、0 执行失败；目标 kernel Start 12/Finished 12/ERROR 0 |

每项均读取真实原始报告，未依赖 shell 退出码或 runner 的“未找到报告（跳过）”
提示。确定性原始报告实际位于 workdir 的
`trace_repo/tests/atk/chunk_kda_fwd_finalize/atk_output/`，12 条
`确定性计算结果` 与结果详情逐条为 True。路径、实际统计和检查证据统一见
`final_formal_validation.json`；正常流程退出码 `normal_gates.exit=0`，
内存检测退出码 `sanitizer_gates.exit=0`。

sanitizer 包以 `--bisheng_flags=ccec_g,sanitizer` 构建并确认实际安装
kernel `.o` MD5 匹配。日志明确检测 `ChunkKdaFwdFinalize_*_0`，
不是只有 `IsFinite` 等辅助 kernel。初次将 sanitizer 包直接用于未注入的
普通 precision 流程时，case0 的 fault PC 落在
`__sanitizer_report_set_ffts_base_addr+0x1c8`，触发 scalar internal-buffer
越界；这次失败记录保留在 `sanitizer_all.log.without_injection` 与
`sanitizer_plain_run_diagnosis.json/.md`。同一包在正确注入下的 case0、case25
均 1/1 memory Pass，正式 12 条全部通过；未为此修改正式 kernel。
本机 msSanitizer 的 `--log-file` 必须与 ATK `-msl` 指向同一实际文件。

复现时正常包使用三个普通 scope；sanitizer 包仅在 msSanitizer 下启动：

```bash
# 已加载正常包及其配套 OPP/runtime；CPU 标杆线程和 GM init 设置同上。
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd_finalize -soc=A2 -npu_device_id=4 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd_finalize -soc=A2 -npu_device_id=4 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd_finalize -soc=A2 -npu_device_id=4 -scope=determinism

# 已独立安装 sanitizer 包、加载其配套 OPP/runtime；不得替换正常性能测量包。
export INJ_LOG_LEVEL=0
FINALIZE_MSS_LOG=/tmp/chunk_kda_fwd_finalize_mss.log
mssanitizer --tool=memcheck --log-level=info --log-file="$FINALIZE_MSS_LOG" -- \
  atk node --backend npu --devices 6 task \
  -c tests/atk/chunk_kda_fwd_finalize/atk_chunk_kda_fwd_finalize_mss.json \
  --task run -p tests/atk/chunk_kda_fwd_finalize/executor_chunk_kda_fwd_finalize.py \
  --mssanitizer -msl "$FINALIZE_MSS_LOG" -s 0 -e 12 -sp -to 60
```
