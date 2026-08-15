# 验证方法

## 验证分层

按改动风险选择验证范围，不要把“能编译”当作“功能正确”：

- 静态检查：`git diff --check`、schema/文档一致性检查、生成物检查。
- 环境检查：`python scripts/check_npu_env.py --build-only`。
- 构建验证：按目标 SOC 生成 wheel 或 OPP run 包。
- 打包验证：检查一体化 wheel、standalone wheel 和 run 包覆盖后的包名、import 面与 OPP 布局。
- 单算子验证：运行对应 `torch_custom/fla_npu/test/test_npu_<op>.py` 或 `test.sh --op <name>`。
- 端到端验证：运行 `examples/flash_gated_delta_rule.py` 或 `ci/run_example_st_cases.py`。
- 精度验证：对比参考实现，覆盖关键 shape、dtype、layout、dense/varlen 和边界 case。
- 性能验证：使用合适 profiling 工具，不用 Python wall time 直接作为性能结论。
- 内存/同步验证：疑似越界、未初始化、流水 hazard 或同步问题时，使用对应 sanitizer/profiling 方法验证。

## 测试矩阵

测试矩阵应覆盖语义路径、调度路径和值域路径，而不是只堆几个默认 shape：

- fixed length 和 varlen 都要覆盖；varlen 场景要覆盖 `cu_seqlens` 与 `chunk_indices` 成对出现、尾 chunk、短序列和多 chunk。
- head 关系要覆盖一一对应和 GVA/grouped 场景，例如 `H_out` 是 `H_qk` 的 1、2、4 倍，确认 head 映射和 workspace slot 没有串头。
- 目标维度要覆盖关键模板组合，例如 `chunkSize=64/128`、`V=128/256`、主 dtype 为 `fp16/bf16`，以及 gate/scale 等辅助输入与主输入 dtype 不同的 mixed 场景。
- 可选但当前不支持的参数要有反向用例，确认代码会明确拦截，而不是静默忽略或在 kernel 内崩溃。
- 输出支持非连续视图时，要验证最终 `ViewCopy` 或等价路径；不要只测 contiguous 输出。
- 多阶段 AIC/AIV 协同算子要覆盖长序列、多 chunk、多 head ratio，让同一个 core 连续处理多个 task，触发 workspace slot 复用和 ready/free flag 协议。

## 构建矩阵

常用 SOC 映射：

- A2：`ascend910b`
- A3：`ascend910_93`
- A5：`ascend950`

修改公共接口、公共 kernel 组件或跨平台逻辑时，应考虑多 SOC 编译和必要运行验证。若当前环境无法覆盖某个 SOC，应在结果中明确说明未覆盖原因。

## 打包和安装验证

一体化 wheel 和 `torch_custom/fla_npu` standalone wheel 都应使用 pip 项目名
`flash-linear-attention-npu`，安装后公开 import 名为 `fla_npu`。验证时至少确认：

- `python -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl` 可安装。
- `python scripts/check_packaged_wheel_api.py` 通过。
- 安装后的 wheel 不依赖顶层 `fla` 包；Ascend C 入口是 `fla_npu.ops.ascendc`，Triton 入口是 `fla_npu.ops.triton`。
- standalone wheel + run 包 `--full` 或 `--install` 后，`site-packages/fla_npu/opp/vendors/fla_npu_transformer` 下能看到当前 run 包覆盖后的 op_api、tiling、kernel 和配置产物。
- wheel 和 run 包都要覆盖重复安装：同一个 wheel 连续强制安装两次、同一个 scoped run 包连续覆盖两次，最终 OPP 内容、`RECORD`、`set_env.bash` 和动态库选择保持一致。
- 源码或 Python 适配修改后重新构建 wheel，再覆盖旧 wheel 两次；新增公开适配时同时检查 API 可发现性和主要动态库加载通路。
- run 包覆盖后再强制安装新 wheel，确认 run 包增加的文件由更新后的 `RECORD` 清理，新 wheel 的 OPP 内容与归档完全一致。
- 使用 `python scripts/check_install_workflows.py --help` 查看统一看护入口。源码检查环境可用 `--skip-runtime-load`，但不能据此声明动态库加载通过；算子精度、泛化和性能仍使用各算子的专用测试。

## 精度问题处理

精度失败先分类，再决定处理方式：

- 如果误差呈结构性错位、整片符号/幅值异常、维度映射错误，优先回到索引、layout、任务分配、数据搬运和写回路径定位。
- 如果误差集中在无效区或 padding 区，先确认该区域的语义和测试后处理。
- 如果是随机数值误差，固定 shape/layout/属性后做多轮复检，再判断是否稳定劣于参考实现。

不要通过收窄输入 range、删除失败 case、降低覆盖强度或放宽阈值来制造通过结论。

## 单算子 ATK 一键验证

仓内单算子 NPU 看护使用 `tests/atk/run_test.sh` 统一调度。脚本只支持
`-op=chunk_kda_fwd`，所有测试动作都通过 ATK 发起；mssanitizer 阶段也只是在外层包裹
同一条 ATK `task` 命令。

### 前置准备

NPU 节点需要先准备 ATK、CANN、当前构建的 OPP 和仓内 Python 包：

```bash
cd <repo_root>
source <atk_venv>/bin/activate
source <cann_install_path>/set_env.sh
source <fla_npu_install_path>/vendors/fla_npu_transformer/bin/set_env.bash

export PYTHONPATH="<repo_root>/torch_custom/fla_npu:<repo_root>:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR=<writable_cache_dir>
atk --version
npu-smi info -i <physical_npu_device>
```

`atk --version` 应与 `tests/atk/README.md` 中锁定版本一致。脚本会根据
`-npu_device_id=<physical_npu_device>` 设置 `ASCEND_RT_VISIBLE_DEVICES`，所以后续
ATK 命令固定使用映射后的逻辑设备 `--devices 0`。不要在外部再把
`ASCEND_RT_VISIBLE_DEVICES` 设置成另一张卡。

双标杆精度和确定性验证需要可达的 GPU ATK server。GPU 节点加载 ATK、CUDA Torch、Triton
和同提交的 `chunk_kda_fwd` ATK 资产后启动：

```bash
cd <repo_root>/tests/atk/chunk_kda_fwd
source <gpu_atk_venv>/bin/activate
export CUDA_VISIBLE_DEVICES=<physical_gpu_device>
export PYTHONPATH="<triton_kda_source_root>:<repo_root>:${PYTHONPATH:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

atk server \
  --host 0.0.0.0 \
  --port <gpu_server_port> \
  --devices 0 \
  --name gpu_reference \
  --output_path ./atk_output/gpu_server \
  --plugin_path ./executor_chunk_kda_fwd.py \
  --timeout 8000
```

GPU server 的 `--devices` 使用 `CUDA_VISIBLE_DEVICES` 映射后的逻辑设备号。发起端传给
`run_test.sh` 的 `-gpu_host` 和 `-gpu_port` 必须是 NPU 节点能够访问到的地址和端口。

mssanitizer 阶段需要使用带 sanitizer 信息的 debug OPP 包。构建时确认 `opc` 命令包含
`--op_debug_level=1 --op_debug_config=dump_cce,sanitizer`，执行前抽查目标对象中存在
sanitizer 符号：

```bash
nm <chunk_kda_fwd_object> | grep sanitizer
```

### 一键执行

在仓库根目录执行：

```bash
bash tests/atk/run_test.sh \
  -op=chunk_kda_fwd \
  -npu_device_id=<physical_npu_device> \
  -gpu_host=<gpu_server_host> \
  -gpu_port=<gpu_server_port>
```

默认执行 `all`，顺序为双标杆精度、性能、确定性和 mssanitizer。需要单独跑某一项时使用
`-scope=accuracy`、`-scope=performance`、`-scope=determinism` 或
`-scope=mssanitizer`。常用覆盖参数：

```bash
ATK_TIMEOUT=2000
PERFORMANCE_DATA=20,100,80
DC_LOOP_NUMS=50
MSS_TOOLS="memcheck racecheck initcheck synccheck"
```

脚本默认自动识别 SOC。自动识别失败时按 A2 `ascend910b` 执行；A5 可显式传入
`-soc=ascend950`。精度范围由脚本从 `atk_chunk_kda_fwd.json` 中按 SOC 和正向用例自动计算，
也可以通过 `ACCURACY_START` 和 `ACCURACY_END` 覆盖。当前 `chunk_kda_fwd` 固定专项 case：

| 平台 | 性能 case | 确定性 case | mssanitizer case |
| --- | --- | --- | --- |
| A2 `ascend910b` | `-wl '[0,16]'` | `-wl '[4,18]'` | `-wl '[8,16]'` |
| A5 `ascend950` | `-wl '[250,266]'` | `-wl '[254,268]'` | `-wl '[258,266]'` |

### 脚本覆盖的 ATK 动作

双标杆精度验证使用 GPU 高精度真值和 GPU 同精度对照。`<accuracy_start>` 与
`<accuracy_end>` 默认来自当前 JSON 中对应 SOC 的正向用例范围：

```bash
atk node --name npu_dut --backend npu --devices 0 \
  node --name gpu_reference --backend gpu --host <gpu_host> --port <gpu_port> --devices 0 --is_compare true \
  task -c ./atk_chunk_kda_fwd.json --task accuracy --bm_device gpu -p ./executor_chunk_kda_fwd.py \
  -s <accuracy_start> -e <accuracy_end> --syc_dataset -mt 1 -to <timeout>
```

性能对比验证只使用 ATK `performance_device` 的 device profiler：

```bash
atk node --name npu_dut --backend npu --devices 0 \
  task -c ./atk_chunk_kda_fwd.json --task performance_device -p ./executor_chunk_kda_fwd.py \
  -wl '<performance_cases>' --performance_data 20,100,80 --save_data profile -sp -to <timeout>
```

确定性验证使用 ATK `accuracy_dc`，并开启 `--gm_init_flag`：

```bash
atk node --name npu_dut --backend npu --devices 0 \
  node --name gpu_reference --backend gpu --host <gpu_host> --port <gpu_port> --devices 0 --is_compare true \
  task -c ./atk_chunk_kda_fwd.json --task accuracy_dc --bm_device gpu -p ./executor_chunk_kda_fwd.py \
  -wl '<determinism_cases>' --gm_init_flag --syc_dataset -mt 1 -to <timeout>
```

内存检测由 `mssanitizer --tool=<tool>` 包裹 ATK `run` 任务，不叠加 ATK 自身的 `-ms`：

```bash
mssanitizer --tool=memcheck --log-file ./mssanitizer_memcheck.log -- \
  atk node --name npu_dut --backend npu --devices 0 \
  task -c ./atk_chunk_kda_fwd.json --task run -p ./executor_chunk_kda_fwd.py \
  -wl '<mssanitizer_cases>' -sp -to <timeout>
```

`racecheck`、`initcheck` 和 `synccheck` 只替换 `--tool` 与日志名。每一项都必须同时检查
ATK 总任务数、失败数、精度或专项结论，以及 mssanitizer 日志是否真正命中目标 kernel。
没有命中 sanitizer 或报告中存在 failed case 时，本次验证不能记为通过。

## 结果记录

对外描述测试结果时，只写测试项和结果，不写本地机器、账号、绝对路径、临时目录或日志路径。若没有执行某项验证，写清楚原因，例如缺少 NPU、缺少 CANN 环境或依赖版本不满足。
