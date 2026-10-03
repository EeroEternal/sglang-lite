# Qwen3.8 GPU-only 优化记录（2026-10-03）

## 结论与验收状态

目标是独立的 Qwen3.8-Flash-Next RadixArk NVFP4 引擎，在同机、同权重、
相同 GPU 资源预算下超过分别调优后的标准 SGLang。允许不同 TP/EP/DP/PP，
但必须披露结构、保证生成正确，并分别记录端到端与纯 decode 时长。

**目前没有验收通过的 lite 吞吐成绩，也没有超过 SGLang 的结论。**
标准 SGLang 最佳已测结果为 256-token **114.45 tok/s**。
独立原型已完成 48 层加载、CUDA graph 捕获/释放和首轮八卡持久张量审计；
完整数值验收、长上下文、状态恢复/复用与 serving 集成仍未完成。
默认服务入口继续保留 V4，不把原型冒充已支持模型。

## 可复现环境与来源

- 硬件：8×RTX 5090；所有跨 GPU peer-access 配对均不可用，不能假定 CUDA
  IPC/P2P 直接读取可用。
- 权重：`RadixArk/Qwen3.8-Flash-Next-NVFP4`，ModelScope 镜像；
  safetensors index 引用 206 个 shard，合计 135,195,303,851 bytes。
  基线记录 config/index SHA256 和 shard 大小，尚未完成所有 shard 的完整哈希校验。
- 标准环境：SGLang 0.5.20、Torch 2.13.0+cu130、FlashInfer 0.6.18、
  sglang-kernel 0.4.7。依赖和缓存均隔离，不替换共享 Torch 环境。
- SGLang 源码参考：`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`。
- Strata 研究参考：`99f3dbd0b21d1401b3769e0c0d963913607f380b`。
  不移植 CPU/SSD 专家执行或 speculative decoding。
- 实现与许可证来源：[QWEN38_SOURCES.md](vendor/QWEN38_SOURCES.md)；
  运行命令：[runbook.md](runbook.md)。

## 1. 先建立真实基线

新增 `scripts/qwen38_sglang_baseline.py`，与独立 engine 分离。
明确关闭 PLE/权重 CPU offload、unified memory 和 speculation；采用 BF16
dense + ModelOpt NVFP4 + FlashInfer CUTLASS。
固定 greedy 输入、`ignore_eos=True` 和输出长度，短输出直接拒绝，不计入吞吐。
记录三次 warm 原始时长、中位数、完整输出及哈希。

Qwen4Exp 的上游默认会启用 PLE CPU offload，因此显式设置
`ple_offload_embedding=False`。0.5.20 不支持该模型的 language-only 参数，
标准基线保留 vision tower，但仅发送文本请求；lite 不实现 vision。

启动修复包括独立 FlashInfer cubin/workspace、兼容依赖和 ninja PATH。
没有关闭版本检查、启用 offload 或降低输出长度来绕过失败。

固定输入：`Write a short explanation of why the sky is blue.`
下表均为**单请求端到端生成**，包含 prefill/调度，并非纯 decode。

| 配置 | 输出 token | 三次 warm 原始时长（秒） | 中位时长（秒） | tok/s |
|---|---:|---|---:|---:|
| TP8/EP8，attention DP1 | 64 | 0.744063 / 0.725411 / 0.746597 | 0.744063 | 86.01 |
| TP8/EP8，attention DP1 | 128 | 1.229212 / 1.239238 / 1.255027 | 1.239238 | 103.29 |
| TP8/EP8，attention DP1 | 256 | 2.226090 / 2.236805 / 2.244826 | 2.236805 | 114.45 |
| TP8/EP8，attention DP2 | 64 | 0.949939 / 0.949267 / 0.960169 | 0.949939 | 67.37 |
| TP8/EP8，attention DP2 | 128 | 1.652043 / 1.653563 / 1.643285 | 1.652043 | 77.48 |
| TP8/EP8，attention DP2 | 256 | 3.036018 / 3.032625 / 3.031594 | 3.032625 | 84.42 |

attention DP2 把 attention TP 降到 4，但保持全局 TP/EP8；该实验更慢，
暂不采用。不能因为 lite 尚无成绩就宣称降低 TP 已经改善吞吐，也不能假定
上游存在 PP 参数就代表 Qwen4Exp 的 PP 前向已经可用。

原始 artifact 名称：
`qwen38-sglang-tp8-ep8-attempt05.json`、
`qwen38-sglang-tp8-ep8-attndp2.json`。
大权重、trace 和远端环境不提交到 Git。

## 2. 驻留、重复性与 bottleneck

新增 `scripts/qwen38_sglang_diagnose.py`。标准路径八个 rank 各遍历 2,024 个
参数、buffer 和持久池张量，没有非 CUDA 权重或浮点持久状态。
审计不枚举所有临时张量，也不排除 NCCL host staging。

标准 greedy 的五次 64-token 请求在零基位置 43 起分歧，清缓存与不清缓存
均如此；cached_tokens 均为 0，因此不能直接作为固定 golden。
确定性模式的默认 DeepGEMM 在 `M=8,N=2060,K=2560` 遇到 TMA stride
对齐断言。使用官方
`SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0` 选择 GPU Triton 后，
五次 64-token IDs 一致，八卡驻留审计继续通过。
这仅是一个短 prompt 的稳定参考，不是全模型正确性证明，也不是上述速度配置。

32-token TP0 trace 的累计 kernel duration：

| 类别 | 累计时间 | 占累计 kernel 时间 |
|---|---:|---:|
| NCCL AllReduce | 165.57 ms | 44.75% |
| HC mix | 41.98 ms | 11.35% |
| 总计 | 370.00 ms | 100% |

这不是墙钟占比，不能直接据此计算 speedup。
在该机器上，通信结构与 HC 融合值得优先研究，不能未经验证使用 IPC one-shot。

## 3. 独立 runner 的实现与修复

`engine/qwen38_runner/` 自持模型与单序列 GPU 状态，不 runtime import
SGLang/vLLM，使用 Torch/Triton/FlashInfer 叶子核：

- `config.py`：对 TP8/EP8、context≤4096、模型结构与 NVFP4 参数 fail closed。
- `weights.py`：CPU 映射仅用于加载；显式 TP/EP 切片，保留 NVFP4 六组 scale。
- `model.py`：GDN/QSA/PLE、hyperconnection、routed/shared experts 与 logits。
- `ops.py`：GPU GDN 更新、dense/sparse attention、FP8 分片查表和 RoPE。
- `scripts/qwen38_lite_probe.py`：自由生成、teacher forcing、graph 与驻留审计。
- wheel 显式包含新子包与 Apache license；新增 `qwen38` 叶子依赖 extra。

真机迭代中已修复：

1. safetensors 0-D scale 不能 `[:]`，改为 scalar tensor 加载并补回归测试。
2. FlashInfer blockscale interleave 需要 uint8 字节视图，不是数值 FP8 转换；
   用 GPU 测试逐字节核对布局。
3. CUTLASS SwiGLU 要求 **[Up, Gate]**，权重和 blockscale 同时调换，
   不能沿用普通 [Gate, Up] 拼接。
4. FP8 PLE 查表必须乘 checkpoint scalar scale，不能只上转 BF16。
5. Triton FP8 masked load 的填充值采用浮点 0；动态 attention 长度统一 int32。
6. GDN convolution 在 FP32 激活后再舍入 BF16。
7. `graph.reset()` 先于 NCCL communicator teardown，避免退出保留显存。
   一个已完成旧探针未释放显存曾导致下一次 OOM，已停止并验证 GPU 释放；
   没有用 offload、扩大 GPU 预算或修改共享环境掩盖问题。

原型并行结构：attention TP8、expert EP8/MoE TP1、DP1/PP1；
共享专家仅 rank0 执行，再参加全局 reduction，区别于标准引擎共享专家 TP 分片。

首轮 owned 持久张量审计：rank0 为 1,650，其余 rank 各 1,458，全部 CUDA。
自由生成前 43 token 匹配确定性参考，零基位置 43 分歧。
64 步 teacher forcing 的 top1 匹配 63/64，最大所选 token logprob 差为
0.279，**完整数值验收未通过**。

该探针计时排除 prefill，含逐 token host 同步，实际执行 N-1 次 decode；
不能按 N 除以时长，与标准 SGLang 的端到端 warm tok/s 混用。
对应 artifact：`qwen38-owned-graph-64-attempt03.json`、
`qwen38-owned-teacher-64.json` 及逐 rank audit JSON。

## 4. 同次提交中的 V4/legacy 工作

按用户确认，本次一起提交原有 V4 与 hybrid-cache 未提交修改。
它们作为历史/兼容资产保留，不改变 Qwen 为唯一 P0 的决定：

- V4 grouped FP4 MoE、expert gather 与 graph-safe 固定形状 decode。
- graph 内 GPU token feedback 和位置递增，计时循环不逐 token 同步 CPU。
- torchrun rank/device 映射修复，以及 DeepGEMM enabled/armed 门禁。
- 通用开发 benchmark 增加重复 warm 中位数和输出样例。
- hybrid cache 保留 GDN/conv state，而不是错误地转换成普通 K/V。
  多序列并发 batching 与快照回收问题仍需修复，不能作为生产完成证据。

早期合成 graph 测试的 256-step 中位时长为 **13.84→3.16 ms**，token 和
位置符合预期。它不是真实 V4 或 Qwen 模型 throughput，也不计入主 KPI。
当前未重新进行 V4 全权重验收；本机无对应权重，不能把合成结果代替真实对照。
IPC one-shot 在无 P2P 的 5090 环境不可用，NCCL 保留；禁止尝试已知会挂起的
host-mapped system-atomic 路径。

## 5. 验证与后续门禁

本次实现已验证：Qwen CPU/源码边界回归、五项 GPU 叶子测试、真实 48 层加载、
graph 捕获/释放、八卡持久张量审计，以及 wheel 子包/license 内容。
本地缺少 Torch，因此 GPU 测试必须在隔离远端环境执行，不能把 skip 当作通过。
提交前扩展 legacy 回归时发现缓存 GPT2 tokenizer 可加载但返回空 token，
导致三项 cache 测试尚未进入引擎路径就失败；fixture 增加非空编码检查，
回退到已有的离线 BPE tokenizer，不修改生产 tokenizer 或放宽测试断言。
修复后，隔离远端的 Qwen、V4 graph/MoE、DynamicCache 和 prefix/cancel 回归
合计 **28 项通过**，包含 CUDA graph feedback 与五项 GPU 叶子测试。
Qwen 新文件 Ruff 格式检查与 lint 通过，所有变更 Python 文件 AST 检查通过。
本地 Qwen/V4 graph 共 17 项通过，6 项因缺少 Torch/CUDA 跳过。
未重跑真实模型吞吐、完整 Rust 检查和完整仓库测试；它们不是本次发布验证结果。

后续按顺序进行：

1. 定位 teacher-forced 数值差异，扩大 prompt 与上下文覆盖。
2. 验证超过 QSA budget 的稀疏选择、EOS/PLE 历史、reset/恢复/复用。
3. 定义并验证实际 serving、调度与请求生命周期的最小闭环。
4. 针对已测瓶颈优化，测多次真实 warm 中位数。
5. 同资源分别调优两边，并保留相同结构的受控对照；验收后才迁移默认入口。

## 6. Speed-only 迭代：同步清理与受控优化（2026-10-03）

本轮按用户要求只优化速度，不增加服务、模型或协议功能。默认入口不变。
新增 `scripts/qwen38_lite_bench.py`，将正确性探针与吞吐计时分开：

- graph 内完成 token feedback、位置递增和 GPU 输出写入；没有逐 token
  `.item()` / logprob / CPU token 交接。
- 每次请求完成后仅传回一次生成 ID。11-token prefill、tokenizer 和输出 decode
  计入 runner generation；reset、rank barrier、加载和编译不计入。
  **不包含 serving scheduler，不能称作服务端端到端吞吐。**
- N 个输出对应 N-1 次纯 decode replay，GPU event 按 N-1 计数。
  输出长度不足时拒绝作为吞吐样本；记录重复 ID 是否一致。
- TP8/EP8、attention DP1、PP1、共享专家 rank0、context capacity 4096
  不变；同 checkpoint、GPU 预算、prompt、固定输出长度，每项三次 warm。
  `--full-logits` 仅供 benchmark 做通信路径受控对照。
- 最终 profile 在 position 266 开始，测四次 replay；修改过计时口径或
  profile 起始上下文的中间试验不能直接归因到某个内核。

### 6.1 无同步优化前基线

Artifact：`results/qwen38-owned-speed-before.json` 及 `.trace.json`，
均位于隔离远端 workspace，不把大 trace 或模型权重提交进 Git。

| 输出数 | runner generation warm 秒 | 中位 tok/s | 纯 decode 次数 | decode 中位秒 | decode tok/s |
|---|---|---:|---:|---:|---:|
| 64 | 0.9266221 / 0.9124288 / 0.9233177 | 69.32 | 63 | 0.7779587 | 80.98 |
| 128 | 1.7169757 / 1.7082654 / 1.7121925 | 74.76 | 127 | 1.5777056 | 80.50 |
| 256 | 3.2519639 / 3.1812654 / 3.1822585 | 80.45 | 255 | 3.0521995 | 83.55 |

三项重复 ID 一致。这只说明 owned runner 重复性，**不等于匹配 SGLang**。
既有 SGLang 请求端 warm 中位数仍为 86.01 / 103.29 / 114.45 tok/s，
未在本轮重跑；计时范围不同，不能用新 runner 数字宣称 KPI 已达成。

lite rank0 四次 replay 的 kernel 时间合计 46.832 ms，不是请求 wall time：
98 次 AllReduce/token，共 8.223 ms（17.56%）；主要 GEMV 家族
5.747 ms（12.27%）；NVFP4 CUTLASS GEMM 4.033 ms（8.61%）。
完整 logits AllGather 共 326.685 µs，约 81.67 µs/token（0.70%）。
这说明通信不是唯一瓶颈，不能把标准引擎 profile 的占比套到 lite。

### 6.2 保留与撤回的优化

保留：

1. 拆出本地 logits 计算，`step()` 保留完整 logits 的数值诊断接口；
   `step_greedy()` 仅交换各 rank 的 `(最高分, 全局 token ID)`。
   原来每卡发送 31,040 个 FP32 logits，现在发送两个 FP32 值。
   checkpoint 的 token ID 小于 $2^{24}$，FP32 表示精确；本地及跨 rank
   同分时保持最低全局 ID，与完整 argmax 一致。测试覆盖本地和跨卡分片同分。
2. GDN immutable conv 权重加载后缓存 FP32，避免每步重复 dtype conversion；
   保留原有 FP32 multiply/sum/SiLU、BF16 舍入顺序和 BF16 history。
   没有合并会改变浮点计算的算子，没有 host/offload 缓存。

受控候选交换单项试验（`qwen38-owned-speed-candidates-only.json`）：
runner warm 中位数为 74.00 / 79.16 / 81.51 tok/s；
三项输出与优化前逐 ID 完全一致。这个中间快照的 GDN 尚未缓存。

撤回：

- 将 GDN conv、SiLU 和 history 更新全部 `torch.compile` 融合：
  GPU 随机叶子测试虽通过，真实模型在零基位置 43 改变了生成结果。
  256-token 测到 83.87 tok/s，但**不作为可接受收益，也未保留代码**。
  Artifact：`qwen38-owned-speed-candidates-conv.json`。
- FP32 history 与 conv 权重同时缓存：输出和 teacher-forced logprob
  与原型完全一致，但引入混合 dtype concat；短请求比仅候选交换慢约 1.5%。
  最终缩小为只缓存 immutable 权重。中间 artifact：
  `qwen38-owned-speed-candidates-cache.json`、
  `qwen38-owned-speed-final.json`、`qwen38-owned-speed-final-full-logits.json`。

FP32-history 中间快照的同上下文 profile 显示候选 AllGather 约
11.44 µs/token，完整 logits 约 80.20 µs/token；候选本地 max 约
6.54 µs/token。请求级收益并不与带宽缩减比例相等：
候选路径的 64/128-token 请求仍慢于同快照的完整 logits 路径，
256-token 为 82.74 对 82.04 tok/s。保留这些反例，不宣称所有长度均胜出。

### 6.3 数值与验证纪律

最终权重缓存快照已对照旧的 64-step teacher-forcing：
top1 ID 和每项 selected-token logprob 完全一致，最大变化 **0.0**。
与确定性 SGLang 的差异仍为 **63/64 top1、最大 logprob 误差
0.2788197994**，零基位置 43 的既有问题没有被这轮性能优化解决。
新增候选 buffer 后持久张量 rank0 为 1,651，其余各 1,459，全部 CUDA；
审计不覆盖 NCCL host staging 或临时张量。
最终 artifact：`qwen38-owned-teacher-64-weight-cache.json` 与八个 rank audit。

Qwen GPU 叶子、benchmark 计数、源码边界、外部基线和 diagnostic 回归
合计 **25 项通过**；本地相应轻量子集 18 项通过，7 项因无 CUDA 跳过。
Ruff lint/format 已运行。完整 Rust、仓库全套、长上下文稀疏状态恢复
不在本次验证范围，完整数值门禁仍未通过。

### 6.4 最终测速：只缓存权重，不改 history dtype

Artifact：`qwen38-owned-speed-weight-cache.json` 与同上下文 `.trace.json`。

| 输出数 | runner generation warm 秒 | 中位 tok/s | 相对初始 runner | 纯 decode warm 秒（N-1 次） | decode tok/s |
|---|---|---:|---:|---|---:|
| 64 | 0.8530269 / 0.8521775 / 0.8520035 | 75.10 | +8.35% | 0.7262848 / 0.7263850 / 0.7262028 | 86.74 |
| 128 | 1.5933816 / 1.5938759 / 1.5935447 | 80.32 | +7.45% | 1.4672737 / 1.4677755 / 1.4676895 | 86.53 |
| 256 | 3.0951279 / 3.0950799 / 3.0947784 | 82.71 | +2.82% | 2.9686921 / 2.9688213 / 2.9685879 | 85.90 |

全部 warm 重复 ID 一致，并与初始 runner 的对应 64/128/256 序列完全一致。
同一最终实现的 `--full-logits` 对照
（`qwen38-owned-speed-weight-cache-full-logits.json`）：

| 输出数 | 完整 logits 路径 warm 秒 | runner 中位 tok/s | decode 中位 tok/s |
|---|---|---:|---:|
| 64 | 0.9306963 / 0.9127594 / 0.9057774 | 70.12 | 81.03 |
| 128 | 1.6125985 / 1.6067015 / 1.6086596 | 79.57 | 85.71 |
| 256 | 3.1257884 / 3.1232718 / 3.1264890 | 81.90 | 85.04 |

此对照输出也完全一致。64-token 完整 logits 路径原始时长存在约 2.7%
跨度，且中间快照出现过反向结果；收益应视作这台机器上的测量，
不是所有长度、温度/频率、上下文或拓扑下的保证。
两种最终路径均在 position 266 profile；候选路径 kernel 时间合计
45.735 ms，初始为 46.832 ms。kernel 合计不是请求 wall time。
模型测速进程结束后八卡显存均回到 2 MiB，没有留下 owned runner 占用。

**结论：小幅变快，仍慢于既有标准 SGLang，未通过主 KPI 或完整数值门禁。**
后续速度工作应集中在已测的 HC/GDN GEMV、小算子 launch、专家执行与
98 次/token reduction，不再把 host 同步或完整 logits 当成唯一原因。
