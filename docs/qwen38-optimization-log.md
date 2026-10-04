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

## 7. 有长度上界的短上下文 QSA 剪枝（2026-10-03）

上一轮已推送为 `8a023b2`。本轮仍仅处理速度，不新增模型、服务或采样功能。

先尝试合并 GDN 的 QKV/Z/B/A 投影，从四次 GEMV 改为一次
M=1、N=2060、K=2560 的 BF16 投影。严格 GPU 叶子对照发现 Z 的
768 项中一项不一致，最大绝对差 0.0625；因此撤回打包实现，
没有放宽断言、保留失败代码或把未通过数值检查的时长记作收益。

改为消除短上下文不会使用的 QSA 工作：

- 原 attention 在 position<2048 时读取所有已有 KV，不使用 indexer
  的 selected slots；但之前每个 QSA 层仍执行 IQ norm/RoPE、压缩键
  score GEMV、ReLU/reduction、mask、top-k 和 selected-slot 构建。
- runner 可接收显式 `execution_limit`。只有该上界≤2048 时，跳过
  上述评分/选择，给 attention 一个 GPU dummy slot，并专门化 dense 分支。
  未提供上界或上界>2048 时保持原有动态 dense/sparse 路径。
- **不缩小 KV capacity，也不删除索引状态。** index-QK 投影、raw index
  更新、每四 token 的压缩/归一化/RoPE 与 compressed 写入完全保留；
  不切割投影维度，以免再次改变 GEMV 的 BF16 舍入。
- dense kernel 显式检查 `0≤position<execution_limit`，强制启用 Triton
  device assertion。超过声明上界会失败，不静默继续使用错误的 dense 图。
  非整数、非正数或超出 capacity 的上界在加载前拒绝。
- benchmark 从 prompt、最大输出数、warmup/capture 和四次 profile
  replay 计算安全上界，本轮为 271；容量仍为 4096。正确性探针同样
  从自身完整执行范围推导上界。`--full-qsa-selection` 只作为 benchmark
  的原路径对照，不是 serving 功能开关。

新增 GPU 回归覆盖：dense 专门化与原 kernel 的逐位输出一致、
完整 QSA 八步输出及 keys/values/index_raw/compressed 状态完全一致、
2048 以上上界走原路径，以及非法上界的加载前拒绝。
另外在独立子进程测试 GPU 越界 assertion，避免负例污染主测试 CUDA context。

### 7.1 真实权重 A/B 结果

保持 TP8/EP8、attention DP1/PP1、相同 checkpoint、4096 KV capacity、
11-token prompt 和三次 warm。两个 profile 均从 position 266 开始。
Artifact：`qwen38-owned-speed-bounded-qsa.json`、
`qwen38-owned-speed-full-qsa-control.json` 及各自 `.trace.json`。

| 输出数 | 有界 QSA runner warm 秒 | 中位 tok/s | 相对上一轮 8a023b2 | decode warm 秒（N-1 次） | decode tok/s |
|---|---|---:|---:|---|---:|
| 64 | 0.7888706 / 0.7877996 / 0.7884440 | 81.17 | +8.08% | 0.6717875 / 0.6715131 / 0.6720558 | 93.78 |
| 128 | 1.4739556 / 1.4737807 / 1.4739572 | 86.84 | +8.11% | 1.3571497 / 1.3572131 / 1.3575284 | 93.57 |
| 256 | 2.8617650 / 2.8615802 / 2.8615952 | 89.46 | +8.16% | 2.7447957 / 2.7449451 / 2.7448940 | 92.90 |

同期 `--full-qsa-selection` 原路径对照：

| 输出数 | runner warm 秒 | 中位 tok/s | decode warm 秒（N-1 次） | decode tok/s |
|---|---|---:|---|---:|
| 64 | 0.8522895 / 0.8522907 / 0.8519205 | 75.09 | 0.7263524 / 0.7263553 / 0.7261352 | 86.73 |
| 128 | 1.6462050 / 1.6464677 / 1.5953206 | 77.75 | 1.5158798 / 1.5160681 / 1.4678901 | 83.78 |
| 256 | 3.0951939 / 3.0949350 / 3.0948220 | 82.72 | 2.9689194 / 2.9688989 / 2.9686184 | 85.89 |

128-token 对照存在时长波动，其中一次恢复上一轮约 1.59 秒的水平；
不隐去原始样本或把中位数变化全归因到改动。
64/256-token 对照与上一轮基本一致。全部 warm 重复 ID 一致，两个
新路径的 64/128/256 序列都与上一轮对应序列逐 ID 完全一致；
prompt、config/index hash、并行结构和容量也已核对一致。

四次 replay 的 rank0 kernel 数 **18,248→16,040（-12.10%）**，
kernel 时间合计 **45.731→42.158 ms（-7.81%）**。
top-k kernel 次数 **240→192**，保留 48 MoE 层每步的 routing top-k，
仅消除 12 QSA 层每步不会被 dense attention 使用的选择。
kernel 合计仍不是请求 wall time。

### 7.2 数值、驻留与限制

`qwen38-owned-teacher-64-bounded-qsa.json` 对照
`qwen38-owned-teacher-64-weight-cache.json`：
64-step top1 ID、每项 selected-token logprob 完全一致，最大变化 **0.0**。
与确定性 SGLang 的既有差异不变：**63/64 top1、最大 logprob 误差
0.2788197994**；此优化不代表完整正确性验收。
八卡持久 tensor audit 全部 CUDA：rank0 为 1,663，其余各 1,471，
相比上一轮各增加 12 个 dummy-selection GPU tensor；模型运行退出后
八卡均回到 2 MiB，没有新增 CPU/SSD offload 或统一内存路径。

最终聚焦回归 **29 项通过**（含 11 项 GPU 叶子与负例）；
本地 18 项通过、11 项无 CUDA 跳过；Ruff lint/format 与 diff 检查通过。
完整 Rust/仓库套件、跨 QSA budget 的真实长上下文、状态恢复/复用仍未验收。
默认 serving gate 不变。新速度只适用于有正确执行上界的单序列路径，
未经声明的 caller 默认保留原实现，不能把有界图拿去执行更长请求。

**目前 runner 256-token 89.46 tok/s，仍低于既有 SGLang 请求端参考
114.45 tok/s；计时范围不同，主 KPI 尚未达成。**

## 8. 延续实验：GDN/MoE、attention 拓扑与 QSA 状态写入（2026-10-04）

本节接续上轮未完成的会话；所有 lite 测速仍是单序列 runner generation，
含 prefill、tokenizer 和输出 decode，但**不含 serving 调度**。
同一 checkpoint、8×5090、TP8/EP8、PP1、4096 KV capacity、
11-token prompt、三次 warm；除特别注明外 attention TP8。
完整数值正确性与服务端 KPI 均未验收。

1. GDN 四项卷积求和按原 CUDA 归约顺序融合，SiLU 保留 eager；
   immutable Gemma RMS scale 预缓存。64-step teacher forcing 的 ID 与
   selected-token logprob 与此前完全一致。单项对照的 256-token runner
   由 89.48 升至约 92.02 tok/s。打包 GDN 投影曾出现一项 BF16 差异，
   未保留。
2. 48 层复用单 token MoE 的约 1.6 MB GPU workspace，routing 的归一化
   和 top-k ID 转换合成一次 GPU kernel；256-token runner 约
   92.90 tok/s，teacher-forced ID/logprob 不变。MoE autotune 曾测到约
   95.42 tok/s，但重复输出不一致，**不得采用**。
3. 保持 expert EP8，尝试 attention TP4/TP2（分别有 2/4 组复制的
   attention 计算）；其 64/128/256-token runner 为
   81.17/86.82/92.12 与 77.34/83.40/87.13 tok/s，
   都不优于 TP8 的 84.34/90.20/92.90。三种结构的 teacher forcing
   对确定性参考均是 63/64 top1、第 44 token 首次分歧；
   不据此宣称跨结构逐位相等。默认继续 TP8。
4. 标准 SGLang **独立重测** attention DP1/DP2/DP4 的
   64/128/256-token 请求端 warm，分别为
   82.13/96.78/105.77、57.05/66.62/79.81、
   49.64/75.28/82.61 tok/s。DP4 的 64-token 原始时长
   1.615549/1.289401/0.967142 秒，波动明显，不作精确速度归因。
   此次 DP1 低于早期 114.45 tok/s 的 256-token 记录；两次均披露，
   不用较慢的重测成绩替代已有较优基线，也不将 runner 与服务端时长混为一谈。

继续在默认 attention TP8 上做 QSA 的固定开销削减：

- 缓存每个 QSA 层 4 个 GPU 槽位偏移，不再每步调用 `torch.arange(4)`。
  四次 replay 的 rank0 profile 少 48 个 `arange` kernel，总 kernel
  数由 14,768 降至 14,720；256-token 中位数
  **92.90→92.88 tok/s**，单独没有可确认的墙钟收益。
- 将 compressed index 每四 token 的条件写入改成一个 Triton kernel，
  不再对旧块做 `index_select` / `where` / `index_copy_`。
  每次调用仍计算相同的压缩值，只在原来会写的 token 更新 GPU 状态。
  相比紧邻的槽位缓存对照，四次 replay 的 rank0 kernel 数
  **14,720→14,528**、累计 kernel 时间 **40.606→40.369 ms**；
  kernel 时间不是请求墙钟。64/128/256-token runner 原始 warm 秒
  分别为 **0.781280/0.781232/0.781324**、
  **1.461797/1.461338/1.461170**、
  **2.740230/2.739924/2.739776**；中位数
  **81.92/87.59/93.43 tok/s**。紧邻槽位对照是
  **81.46/87.13/92.88 tok/s**，256-token 改善约 0.6%；
  早一批 MoE 对照的 64/128-token 数字更快，环境波动存在，
  不能宣称所有长度相对那批都有收益。

三组 64/128/256-token 输出 ID 与更早 MoE 对照逐项一致，warm 重复
ID 一致；64-step teacher-forced 的 ID 和每项 selected-token logprob
也完全一致，既有 **63/64 top1、第 44 token 分歧**没有修复。
新 QSA 叶子测试逐步对照原条件写入的全部压缩状态；全 Qwen 聚焦测试
**33 项通过**，本地不带 Torch 的 CPU 子集 **18 项通过**，Ruff 和
`git diff --check` 通过。持久张量审计 rank0 为 1,676、其余各 1,484，
全部 CUDA；新增 12 个槽位偏移张量/每 rank。进程退出后八卡各 2 MiB。
审计不覆盖临时张量/NCCL host staging；跨 QSA budget、KV 恢复复用、
完整数值验收与服务端吞吐仍待验证。

新远端 artifact：`qwen38-owned-speed-slot-offsets.json`、
`qwen38-owned-speed-compressed-write.json` 及其 `.trace.json`、
`qwen38-owned-teacher-64-compressed-write.json` 和八个 rank audit；
拓扑对照：`qwen38-owned-speed-attntp2.json`、
`qwen38-owned-speed-attntp4.json`、
`qwen38-sglang-refresh-attndp{1,2,4}.json`。

## 9. QSA 压缩索引四行均值融合（2026-10-04）

前一轮提交 `f074b36` 后继续缩减 QSA 的小算子：对 GPU `index_raw` 的
相邻四行做 FP32 均值并转 BF16，取代每层每步的
`arange → clamp → index_select → float → mean → BF16`。
位置和缓存仍在 GPU，四行未写入的值仍按原逻辑从已有状态读取；
compressed index 每四 token 写一次的条件和 dense/sparse 边界不变。
叶子测试对照原表达式的多个块起止及容量末端；真实权重的 64-step
teacher forcing 与改动前的 ID、每项 selected-token logprob 完全一致。
既有 **63/64 top1、第 44 token 分歧**仍未解决。

拒绝的候选：将 dense attention 的 Triton block 32 改为 64 虽缩短单核
时长，但 1008 组随机输入有 96 组不逐位一致，未修改默认内核。

在同一远端、同一权重、TP8/EP8/attention TP8、4096 KV capacity、
相同 prompt、三次 warm 的连续“旧实现→新实现”对照如下。
两边均启用 profile，仅在测速完成后采集四次 replay；这是**单序列
runner generation**，不包含 serving scheduler，不能与 SGLang 服务端
端到端请求直接对比：

| 输出数 | 旧实现原始 warm 秒 | 新实现原始 warm 秒 | runner 中位 tok/s（旧→新） |
|---:|---|---|---:|
| 64 | 0.780795 / 0.781341 / 0.781912 | 0.745340 / 0.745526 / 0.745586 | 81.91→85.85 |
| 128 | 1.461035 / 1.460818 / 1.436580 | 1.411011 / 1.393228 / 1.393009 | 87.62→91.87 |
| 256 | 2.739483 / 2.739447 / 2.739231 | 2.706863 / 2.707008 / 2.707331 | 93.45→94.57 |

128-token 的旧实现一次样本较快，不隐去；独立于配对测试的首次新实现
256-token 中位为 94.62 tok/s，结论不依赖该单次较快样本。三组输出
ID 全部逐项相等，各自 warm 重复 ID 一致。相同 profile 位置的四次
rank0 replay kernel 数 **14,528→14,240**，累计 kernel 时间
**40.354→39.850 ms**；这不是整请求墙钟。

新增 GPU 叶子回归后远端聚焦测试 **34 项通过**，本地 CPU 子集
**18 项通过**；八卡持久张量审计均为 CUDA，rank0 1,676、其余各
1,484。结束时各卡 2 MiB。长上下文、跨 QSA budget、状态恢复与
服务端正确性仍未验收；前述确定性 SGLang 数值差异未修复。
配对原始 artifact：
`qwen38-owned-speed-qsa-mean-control-paired.json` 和
`qwen38-owned-speed-qsa-mean-paired.json`（各有 `.trace.json`）；
数值 artifact：`qwen38-owned-teacher-64-qsa-mean.json` 及八个 rank audit。

## 10. 第 44 token 的 GDN gate 修正与成本（2026-10-04）

在同一 prompt/token 前缀上捕获 SGLang 和 owned 的第 0 层中间张量：
layer input 与 attention hyperconnection mix 逐位一致，最早可靠差异
出现在 local GDN 输出（相对 RMS 约 0.00505）。上游 packed decode 核
将 FP32 `sigmoid(b)` 舍入到 BF16 gate 后再用于 FP32 recurrent state；
owned 原先直接使用 FP32 gate。只在 `torch.compile` 内添加
BF16→FP32 cast 会被 Inductor 消去，GPU 测试确认其行为仍是未舍入。
因此需要显式 BF16 tensor 边界；为避免额外启动一次 kernel，最终将
gate 舍入并入原有 GDN convolution sum Triton kernel。测试对比
FP32 convolution、BF16 history 和 BF16 sigmoid gate 的原表达式；
真实权重 64-step teacher forcing 与独立 gate kernel 候选的输出 ID
及每项 selected-token logprob **完全相等**。自由生成前 49 个 token
与确定性参考一致，第 50 token 起仍分歧。

原先零基索引 43 的第 44 token 错误选择 1204，修正后选择确定性
SGLang 参考的 **69377**。但这不是完整数值修复：首次分歧移至零基
索引 **49**（第 50 token），仍为 **63/64 top1**，最大 selected-token
logprob 绝对误差约 **0.323413**；第 0 层 GDN 输出整体的参考差异
未消失。预填充和 packed decode 的状态/舍入路径尚需继续定位，
长上下文、QSA budget 边界和服务端验收均未通过。

同机同权重单序列 runner、TP8/EP8/attention TP8，3 次 warm 的对照
如下。原先 QSA 均值优化的旧快照与本轮测试不是严格交替的配对运行；
仅比较紧邻的两个 gate 方案来判断额外 kernel 的影响。所有时间均含
runner prefill/逐步生成，但不含 serving scheduler，不能直接对比
SGLang 的 Engine.generate 请求端 KPI。

| 输出数 | 旧 QSA 均值快照（秒） | 独立 gate kernel（秒） | gate 融合进 conv（秒） | 融合后 runner tok/s |
|---:|---|---|---|---:|
| 64 | 0.745340 / 0.745526 / 0.745586 | 0.772954 / 0.772457 / 0.772747 | 0.772867 / 0.772685 / 0.772153 | 82.83 |
| 128 | 1.411011 / 1.393228 / 1.393009 | 1.445842 / 1.446461 / 1.444573 | 1.444695 / 1.405732 / 1.395677 | 91.06 |
| 256 | 2.706863 / 2.707008 / 2.707331 | 2.713215 / 2.713795 / 2.713347 | 2.709625 / 2.709507 / 2.709283 | 94.48 |

128-token 新方案的三次时长存在波动，不把中位数差当作稳定的提速。
256-token 相比旧快照的 94.57 tok/s **没有可确认提升**。四次
rank0 replay 的 profile kernel 数：旧 QSA 快照 14,240、
独立 gate 14,384、融合后恢复到 14,240；对应 kernel 累计时间
39.850/40.304/40.238 ms，不代表整请求时间。

另试将 QSA 当前索引行写入与四行均值合并：叶子状态与 64-step
teacher-forced ID/logprob 全部相等，但紧邻的 256-token runner
只从 94.348 到 94.430 tok/s，短输出略慢，**已撤回**。
在同一个编译图内用 FP32 位运算模拟 BF16 舍入的试验虽通过随机
叶子测试，真实权重第 44 token 又分歧，**也已撤回**。
不以 kernel 数下降或随机输入单测替代模型级正确性门禁。

原始 artifact：`qwen38-owned-teacher-64-beta-boundary.json`、
`qwen38-owned-teacher-64-beta-conv-fused.json`、
`qwen38-owned-speed-beta-boundary.json`、
`qwen38-owned-speed-beta-conv-fused.json` 及各 `.trace.json`；
撤回试验为 `qwen38-owned-teacher-64-beta-qsa-fused.json` 和
`qwen38-owned-teacher-64-beta-inline.json`。
隔离八卡的 Qwen 聚焦 GPU/源码/benchmark/基线测试 **35 项通过**，
本地 CPU 子集 **18 项通过**；Ruff 格式和 lint、`git diff --check`
通过。真实权重自由生成 artifact：
`qwen38-owned-free-64-beta-conv-fused.json`。没有进行服务端测速，
也没有通过跨 QSA budget、状态复用和完整数值验收。

## 11. Decode 时间归因与共享专家双 stream（2026-10-04）

先拆开 elapsed time，避免将 rank0 的 kernel 总时长误当成请求时间。
修正 GDN gate 后的 256-token runner 例子：单请求 warm 中位
**2.709507 s**（94.48 tok/s），其中 255 次 decode replay 的
CUDA event 中位 **2.599243 s**，即 **10.193 ms/step**；
差约 **0.110 s**，还包含 11-token prefill、首 token、
主机 tokenizer/输出读取等，不能全归于 prefill。该 runner 没有
serving scheduler，也没有 HTTP 请求开销。

同位置 rank0 四次 graph replay 的原 profile 有 **392 次**
BF16 NCCL AllReduce，即 **98 次/token**，kernel 累计约
8.08 ms / 四次 replay（约 20% 的 rank0 累计 kernel 时间）。
同时存在约 1,932 次 GEMV kernel 和 384 次 NVFP4 routed-expert
CUTLASS GEMM；因此仅消除 logits 通信不足以解决 decode 延迟。
新增 benchmark 的 `--profile-all-ranks` 同时保存各 rank trace。
该八卡 profile 中，rank0 比非共享专家 rank 多约 **480 个
kernel/token**；但同时采集 profiler 明显扰动 NCCL 等待时间，
不能把不同 rank 的累计 kernel 时间差换算成墙钟加速。

保留所有 98 次/token AllReduce 和原有浮点运算顺序，仅让 rank0
的 BF16 共享专家在独立 CUDA stream 上与 routed experts 并行，
在两条分支使用同一个 `value` 前等待主 stream，合并前等待共享
stream；只建一个 runner 级 side stream，48 层顺序复用。
CUDA graph 多 stream 叶子测试与串行结果逐位相同；真实权重
64-step teacher forcing 的 ID、每项 selected-token logprob
与串行对照完全一致，三种长度的自由生成 ID 及各组 warm 重复
也全部一致。默认 runner 启用双 stream；诊断脚本提供
`--serial-shared-expert` 复测旧路径。

同机同权重、8×5090、TP8/EP8/attention TP8、11-token prompt、
每项三次 warm。先串行→双 stream，再反向双 stream→串行，
两个顺序都先完整运行一遍冷启动；profile 均在测速之后才采集：

| 次序/输出 | 串行三次 warm 秒 | 双 stream 三次 warm 秒 | 中位 runner tok/s（串行→双 stream） |
|---|---|---|---:|
| 正向/64 | 0.772897 / 0.772952 / 0.773473 | 0.735014 / 0.735226 / 0.735072 | 82.80→87.07 |
| 正向/128 | 1.446266 / 1.446170 / 1.445860 | 1.375211 / 1.375181 / 1.375429 | 88.51→93.08 |
| 正向/256 | 2.711878 / 2.710046 / 2.710137 | 2.564737 / 2.565191 / 2.564710 | 94.46→99.82 |
| 反向/64 | 0.746113 / 0.746196 / 0.746304 | 0.705341 / 0.705411 / 0.705597 | 85.77→90.73 |
| 反向/128 | 1.395462 / 1.395280 / 1.395383 | 1.319922 / 1.319873 / 1.320005 | 91.73→96.98 |
| 反向/256 | 2.709440 / 2.709292 / 2.709828 | 2.562772 / 2.562293 / 2.563116 | 94.48→99.89 |

256-token 纯 decode 分别由 **10.195→9.647 ms/step** 和
**10.193→9.639 ms/step**，两个方向均改善约 0.55 ms/step，
runner throughput 相对提升约 5.7%。64/128-token 两轮的绝对
速度不同，故仅比较各轮紧邻对照，不混用最快单次结果。
正向 rank0 profile 的 kernel 数均为 14,240，AllReduce 均为
392 次；累计 kernel 时间甚至从 40.277 增至 43.110 ms，
但四次 replay 的 rank0 观测跨度从约 46.275 降至 44.093 ms。
这是**重叠执行**而非减少 kernel 或通信，GPU profile 不能代替
上表的请求墙钟计时。

输出尚在第 50 token 与确定性 SGLang 分歧，完整模型数值验收
仍未通过；runner 99.89 tok/s 与 SGLang 曾有的 114.45 tok/s
也不属于同口径服务 KPI，不能声称已追平标准服务。
artifact：`qwen38-rank-profile-beta-conv-fused.rank*.trace.json`，
`qwen38-shared-stream-{control-paired,paired,reverse-optimized,reverse-control}.json`
及各 `.trace.json`；数值探针
`qwen38-teacher-64-shared-stream.json`。默认双 stream 再次跑
64-step teacher/free：teacher 与串行快照 ID/logprob 全相等，
free 首次分歧仍在零基位置 49。远端聚焦 GPU/源码/benchmark/基线
**37 项通过**，本地 CPU 子集 **19 项通过**；Ruff、格式和 diff
检查通过，远端退出后八卡各 2 MiB。默认路径探针 artifact：
`qwen38-shared-stream-default-{teacher,free}.json`。

## 12. 第 50 token：GDN recurrent 输出顺序（2026-10-04）

继续对照上游 SGLang packed-decode GDN 叶子核与 lite：用相同
BF16 Q/K/V/gate、FP32 初态和权重进行 GPU 随机测试，旧的
`torch.compile` 状态更新在 FP32 state 上仅有微小误差，但其
BF16 recurrent 输出约四分之一元素不一致，足以在后续层改变
接近的 logits。单独用相同输入测试 gated RMSNorm 后结果逐位
一致，因此不能将该差异归因于输出 norm。

将上游的 Q/K 归一化、gate/decay、按 32 个 value 通道切块的
FP32 state 更新与缩放输出 reduction 顺序，收敛为 owned
Triton GPU 叶子核；保留已验证的 BF16 gate 及共享专家双 stream。
随机四组输入对上游 packed-decode kernel 的 FP32 最终 state、
BF16 输出和 gated norm 结果**逐位一致**；上游仅用于隔离
测试，不成为 lite runtime 依赖。对应源码来源、Apache-2.0
和 commit 见 [QWEN38_SOURCES.md](vendor/QWEN38_SOURCES.md)。

在原 11-token prompt、确定性 SGLang 64-token 参考上，真实权重
CUDA graph 的 teacher forcing **64/64 top1**；自由生成
**64/64 token ID 完全一致**，包含此前分歧的第 50 个 token。
selected-token logprob 最大绝对误差仍为 **0.144327**，不等于
全模型逐位相等。另用相同确定性设置重新生成 128-token
SGLang 参考，其前 64 token 与原 golden 完全一致；
128-token teacher forcing **128/128 top1**，自由生成
**128/128 token ID**，但最大 selected-token logprob 绝对
误差约 **0.274001**。这仍只是同一个 prompt，当前还没有
验证其他 prompt、长上下文、
跨 QSA budget、状态恢复/复用与 serving；不能将单 prompt
通过等同完整正确性验收。

开启 rank0 共享专家双 stream 后，新算子的 runner
64/128/256-token warm 原始时长分别为
**0.734714/0.736235/0.735101**、
**1.374729/1.374645/1.375295**、
**2.584553/2.566968/2.567697** 秒；中位
**87.06/93.11/99.70 tok/s**。此前同配置的中位为
**87.07/93.08/99.82 tok/s**，没有可确认的加速或减速；
256-token 有一次偏慢样本，不隐去。runner 与标准
SGLang Engine.generate 的请求端仍不是同口径 KPI。

隔离 artifact：`qwen38-reference-recurrence-teacher64.json`、
`qwen38-reference-recurrence-free64.json`、
`qwen38-reference-recurrence-speed.json` 及 `.trace.json`、
`qwen38-diagnostic-128-triton.json`、
`qwen38-clean-recurrence-{teacher128,free128}.json`。
清理旧 GDN 实现后，隔离八卡 GPU/源码/benchmark/基线聚焦回归
**37 项通过**；本地 CPU 子集 **19 项通过**。参考引擎首次临时
诊断尝试因脚本使用标准输入，随后因遗漏多进程 main guard 而
启动失败；改为带 main guard 的远端隔离脚本后成功生成 128-token
参考且退出释放全部 GPU。代码无需也没有依赖这些远端脚本。

## 13. QSA 单 token RoPE 融合（2026-10-04）

在通过 128-token 数值对照的 GDN 版本上继续只做速度优化。
QSA 每步对 BF16 query/key 和 compressed index key 旋转 32+32
维；旧表达式以 FP32 分段乘加、BF16 舍入，再拼回未旋转通道，
每次需要多个 Torch launch/concat。改为一个 owned Triton kernel，
禁用 FMA 融合以保留乘法、加减和 BF16 舍入次序；未旋转通道
直接复制。默认 TP8 及备选 attention TP4/TP2 的 1/3/4/6/12 行
叶子形状逐位匹配旧表达式。真实权重 128-token graph teacher
forcing 的所有 ID 和每项 selected-token logprob 与旧 runner
**完全相等**，自由生成仍与确定性参考 **128/128 ID** 相等。

只交换 RoPE 算子，其他代码、权重、并行结构、prompt、11-token
prefill 和 `--profile` 设置不变。每次先 cold 再三次 warm，
先串行旧 RoPE→融合 RoPE，再融合→旧 RoPE：

| 次序/长度 | 旧 RoPE 三次 warm 秒 | 融合 RoPE 三次 warm 秒 | runner 中位 tok/s（旧→新） |
|---|---|---|---:|
| 正向/64 | 0.735713 / 0.735815 / 0.735505 | 0.684299 / 0.684944 / 0.685229 | 86.99→93.44 |
| 正向/128 | 1.375109 / 1.375113 / 1.375525 | 1.281426 / 1.316721 / 1.333820 | 93.08→97.21 |
| 正向/256 | 2.565423 / 2.565757 / 2.566307 | 2.528245 / 2.489118 / 2.488527 | 99.78→102.85 |
| 反向/64 | 0.707277 / 0.707232 / 0.707312 | 0.713355 / 0.713490 / 0.713597 | 90.49→89.70 |
| 反向/128 | 1.323061 / 1.323031 / 1.322847 | 1.333826 / 1.333891 / 1.333453 | 96.75→95.96 |
| 反向/256 | 2.569036 / 2.567934 / 2.567773 | 2.486797 / 2.486916 / 2.486564 | 99.69→102.94 |

256-token 两个方向均约提升 **3%**，decode GPU event
**9.650→9.362 ms/step** 与 **9.658→9.352 ms/step**。
64/128-token 在反向测试并不更快，**不能将正向短输出收益宣称为
稳定结论**。相同位置四次 replay 的 rank0 kernel 数
**13,952→12,512**，其中融合 RoPE 运行 144 次；
kernel 累计时间正向约 **43.008→41.900 ms**，
反向 **42.986→41.865 ms**；它不等于请求墙钟。
输出长度的每组三次 warm ID 都相等，两个版本的输出 ID 逐项
相等。这仍是**单序列 runner generation**，没有 serving scheduler，
不能与 SGLang Engine.generate 数字当作等口径 KPI。

原始 artifact：
`qwen38-rope-{control-paired,fused-paired,reverse-fused,reverse-control}.json`
及各 `.trace.json`；
`qwen38-rope-fused-{teacher128,free128}.json`。
远端 Qwen 聚焦 GPU/源码/benchmark/基线 **38 项通过**，
本地 CPU 子集 **19 项通过**，结束后八卡各 2 MiB。
其他 prompt、超过 QSA budget 的路径、状态复用与实际服务仍未验收。

## 14. GDN 卷积与 SiLU 合核（2026-10-04）

在保持第 13 节的融合 RoPE、GDN recurrent 和共享专家双 stream
不变的前提下，将 36 个 GDN 层的卷积 FP32 四项求和与其后单独的
`F.silu`/BF16 转换合入原卷积 Triton kernel，BF16 beta gate
仍在该 kernel 中计算。单纯用 Triton `sigmoid` 或
`libdevice.exp` 加普通除法时，随机叶子测试在 BF16
舍入边界出现一处不相等；最终使用
`tl.div_rn(result, 1.0 + libdevice.exp(-result))`
保留 eager SiLU 的舍入结果。多组逐位测试覆盖 1280/2560/5120
通道各 16 次输入、history 更新和 BF16 gate；真实权重 graph
128-token teacher 的 **所有 ID 和 selected-token logprob** 与
第 13 节版本逐项相等，自由生成 ID 也逐项相等，并保持与确定性
参考 **128/128** 匹配。

同机、同权重、11-token prompt、TP8/EP8/attention TP8、4096
capacity、相同 `--profile` 和 runner 操作；每次 cold 后取三次
warm。基线→合核与反向合核→基线分别完整加载模型并测试：

| 次序/长度 | 基线三次 warm 秒 | 合核三次 warm 秒 | runner 中位 tok/s（旧→新） |
|---|---|---|---:|
| 正向/64 | 0.712267 / 0.712621 / 0.712321 | 0.708353 / 0.707920 / 0.708076 | 89.85→90.39 |
| 正向/128 | 1.332273 / 1.332802 / 1.328953 | 1.324270 / 1.324152 / 1.321356 | 96.08→96.67 |
| 正向/256 | 2.486024 / 2.485865 / 2.486101 | 2.468800 / 2.468867 / 2.468169 | 102.98→103.69 |
| 反向/64 | 0.685218 / 0.684846 / 0.684907 | 0.707939 / 0.708091 / 0.707753 | 93.44→90.40 |
| 反向/128 | 1.280617 / 1.280803 / 1.280205 | 1.323729 / 1.323506 / 1.323241 | 99.95→96.71 |
| 反向/256 | 2.486961 / 2.487486 / 2.486789 | 2.482342 / 2.467903 / 2.467898 | 102.94→103.73 |

两种顺序下 256-token 都约 **+0.7%**，decode GPU event
**9.350→9.285** 和 **9.354→9.282 ms/step**；四次 rank0
replay 的 kernel 数 **12,512→12,224**（每次少 72 个），
累计 kernel 时间正向 **41.826→41.655 ms**、反向
**41.762→41.707 ms**，累计时间不能当作端到端收益。
64/128-token 的反向测试旧版本明显更快；虽然合核的短输出
两轮自身时长接近，**没有证据说明短输出稳定提速**，也不以
两轮平均值掩盖该矛盾。三种长度的基线/合核 ID 和各组 warm ID
完全一致。仍只有单 prompt、单序列 runner，没有 serving
scheduler，不能拿 runner tok/s 当服务 KPI。

原始 artifact：
`qwen38-gdn-silu-{control-paired,candidate-paired,reverse-candidate,reverse-control}.json`
及各 `.trace.json`，
`qwen38-gdn-activated-{teacher128,free128}.json`。
远端 8×5090 聚焦 GPU/源码/benchmark/基线回归 **39 项通过**，
退出后八卡各 2 MiB。跨 QSA budget、更多 prompt、状态复用
与请求端同口径测速仍未完成。

## 15. AllReduce 配置试验与 GDN 小 GEMV 打包（2026-10-04）

第 14 节基线 rank0 四次 graph replay 有 **392 次 AllReduce**，
GPU kernel 累计约 **8.96 ms**；同时按 GEMV 名称汇总约
**2,700 次、9.75 ms**。同机 8×5090 无 NVLink，部分 GPU
间经跨 NUMA 的 `SYS` 路径。两类累计 kernel 时间均**不是**
可直接节省的墙钟；48 层串行依赖限制相邻层的跨层合并。

先独立测试 NCCL 通信参数，未改模型代码。强制
`NCCL_ALGO=Tree` 使已存在的 AllGather 报
`no algorithm/protocol available`，无法完成数值验收；
`NCCL_PROTO=LL128` 和 `NCCL_PROTO=Simple` 均能完成
128-token graph teacher forcing，并保持 **128/128 top1**，
但相对默认协议各有 **123/128 项 selected-token logprob
不相等**，最大差异 **0.325581**。不能用只匹配 token ID
替代数值门槛，**三个配置均不采用，也不声明 AllReduce
次数或时间得到优化**。

针对小 GEMV，每个 GDN 层原有两个依赖同一 `value` 的
BF16 六行投影 `in_proj_b` 和 `in_proj_a`；在权重加载时
按 b、a 顺序打包为十二行，将两次 `F.linear` 合为一次，
其余投影、gate、recurrent、MoE 与通信调用不变。
GPU 随机 128 组输入在 2560/10240 输入宽度的两个
BF16 六行输出均逐位相等；真实权重 128-token graph
teacher forcing 的 ID、每项 selected-token logprob 与
原版逐项相同，自由生成亦逐项相同，二者均与确定性参考
**128/128 ID** 匹配。另检查 rank0 共享专家两个 640
行投影打包，随机输入出现 **8 项 BF16 不一致**，未采用。

同机同权重、11-token prompt、TP8/EP8/attention TP8、
4096 capacity，默认 NCCL 配置、相同 `--profile`，
每次 cold 后三次 warm；依次旧版→打包版与打包版→旧版：

| 次序/长度 | 旧版三次 warm 秒 | 打包版三次 warm 秒 | runner 中位 tok/s（旧→新） |
|---|---|---|---:|
| 正向/64 | 0.707870 / 0.708631 / 0.709303 | 0.672524 / 0.672467 / 0.672489 | 90.32→95.17 |
| 正向/128 | 1.312078 / 1.271563 / 1.271418 | 1.257577 / 1.306122 / 1.275635 | 100.66→100.34 |
| 正向/256 | 2.469954 / 2.469322 / 2.469110 | 2.442731 / 2.442775 / 2.442804 | 103.67→104.80 |
| 反向/64 | 0.680004 / 0.679751 / 0.680121 | 0.700099 / 0.700699 / 0.702536 | 94.12→91.34 |
| 反向/128 | 1.314300 / 1.322874 / 1.322846 | 1.310551 / 1.263627 / 1.257816 | 96.76→101.30 |
| 反向/256 | 2.469232 / 2.469250 / 2.468834 | 2.443485 / 2.443557 / 2.443443 | 103.68→104.77 |

两轮 256-token 约 **+1.1%**，decode GPU event
**9.287→9.187**、**9.286→9.190 ms/step**。rank0 四次
replay kernel 数 **12,224→12,080**（每 token 减少
36 次小 GEMV），累计时间正向 **41.631→41.183 ms**、
反向 **41.571→41.212 ms**；AllReduce 仍 **392 次**
（每 token 98 次）。64/128-token 正反结果矛盾，
不宣称短输出稳定改善。旧/新版各长度 ID 和各组 warm
重复 ID 均相等。这些是单序列 runner 结果，不是与
SGLang Engine.generate 同口径的请求端 KPI。

原始 artifact：`qwen38-gdn-ab-{control-paired,candidate-paired,reverse-candidate,reverse-control}.json`
及 `.trace.json`、`qwen38-gdn-packed-ab-{teacher128,free128}.json`、
`qwen38-nccl-{ll128,simple}-teacher128.json`。
远端八卡聚焦 GPU/源码/benchmark/基线 **40 项通过**，
退出后八卡各 2 MiB。仍未验收更多 prompt、长上下文、
跨 QSA budget、状态复用及服务端吞吐。

## 16. QSA K/V GEMV 打包（2026-10-04）

在第 15 节未提交的 GDN a/b 打包版本上，继续减少 12 个
QSA 层各自的小型 GEMV。QSA 的 K/V 权重各有 256 行、相同
2560 列和同一个输入：加载时以 K、V 顺序拼成 512 行，
把两次 `F.linear` 改为一次，再切回原 K/V。
没有打包 512+512 行的其他 GEMV：随机输入测试曾出现
**16 项 BF16 差异**。当前 K/V 256+256 行在 128 组 GPU
随机输入逐位等于两次原始计算；真实权重的 128-token
graph teacher forcing **所有 ID 和 selected-token logprob**
以及自由生成 ID/logprob 均与 GDN-only 版本逐项相等，
两种路径仍与确定性参考 **128/128 ID** 匹配。

同机同权重、11-token prompt、TP8/EP8/attention TP8、
4096 capacity、默认 NCCL、相同 `--profile`，每次先 cold
再三次 warm；依次旧版→K/V 打包与反向 K/V 打包→旧版：

| 次序/长度 | GDN-only 三次 warm 秒 | 再打包 QSA 三次 warm 秒 | runner 中位 tok/s（旧→新） |
|---|---|---|---:|
| 正向/64 | 0.702963 / 0.701588 / 0.701436 | 0.670040 / 0.670084 / 0.670380 | 91.22→95.51 |
| 正向/128 | 1.311397 / 1.311555 / 1.285169 | 1.301528 / 1.305901 / 1.305809 | 97.61→98.02 |
| 正向/256 | 2.443262 / 2.442855 / 2.443368 | 2.434223 / 2.434800 / 2.434681 | 104.78→105.15 |
| 反向/64 | 0.673083 / 0.672917 / 0.672914 | 0.698327 / 0.698502 / 0.698838 | 95.11→91.63 |
| 反向/128 | 1.258610 / 1.258625 / 1.258640 | 1.272403 / 1.252744 / 1.253083 | 101.70→102.15 |
| 反向/256 | 2.443723 / 2.443432 / 2.444197 | 2.433836 / 2.433478 / 2.433822 | 104.76→105.18 |

两轮 256-token runner 约 **+0.36%**，decode event
**9.189→9.157**、**9.191→9.153 ms/step**；rank0
四次 replay kernel 数 **12,080→12,032**，每步少
12 个 GEMV，累计 kernel 时间正向 **41.318→41.180 ms**、
反向 **41.337→41.148 ms**。64-token 正反方向的墙钟
结论相反；128-token 两组中位虽略快，也不把这一幅度视为
已验证的稳定短输出收益。所有基线/候选及各自 warm ID
逐项相同。AllReduce 仍是 **98 次/token**，本轮只优化
12 个 QSA GEMV；GPU kernel 累计时间不等于 runner
或服务端请求时长。

原始 artifact：
`qwen38-qsa-kv-{control-paired,candidate-paired,reverse-candidate,reverse-control}.json`
及各 `.trace.json`，
`qwen38-qsa-packed-kv-{teacher128,free128}.json`。
远端八卡聚焦 GPU/源码/benchmark/基线 **41 项通过**，
进程退出后八卡各 2 MiB。仍只验证单 prompt、
短上下文与单序列 runner；跨 QSA budget、更多 prompt、
状态复用和请求端同口径 KPI 未验收。

## 17. QSA K/V RoPE 与缓存写入合核（2026-10-04）

在第 16 节已提交的 QSA K/V GEMV 打包版本上，进一步消除
每个 QSA 层的 K 旋转后独立 `index_copy_` 和 V 的
`index_copy_`：一个 owned Triton kernel 按原 FP32
乘减/乘加及 BF16 舍入次序将 K 写到 GPU cache 当前行，
并复制 V 到另一块 cache 的当前行。保留既有通用 RoPE
kernel 给 Q 与 compressed index 使用。QSA 位置保持
GPU tensor，核内检查容量边界；未改 attention 计算、
AllReduce 或 KV 生命周期。

叶子测试在多个 cache 位置逐位检查 **全部 K/V 状态**
与旧 RoPE→两次 index_copy 结果；真实权重 128-token
graph teacher/free 的 ID 与**每项 selected-token logprob**
分别与已提交版本完全相等，仍为确定性参考 **128/128 ID**。
另筛选 GDN QKV/Z 与 QSA query/index 的同输入 GEMV
合并：128 组随机 BF16 输入各出现 **25/12 项不一致**，
均未加入实现。

同机同权重、11-token prompt、TP8/EP8/attention TP8、
4096 capacity、默认 NCCL、相同 `--profile`，
每次 cold 后三次 warm；依次旧版→合核、反向合核→旧版：

| 次序/长度 | 旧版三次 warm 秒 | 合核三次 warm 秒 | runner 中位 tok/s（旧→新） |
|---|---|---|---:|
| 正向/64 | 0.696811 / 0.696922 / 0.696792 | 0.668080 / 0.668652 / 0.673517 | 91.85→95.72 |
| 正向/128 | 1.283328 / 1.253169 / 1.252484 | 1.302421 / 1.302042 / 1.302378 | 102.14→98.28 |
| 正向/256 | 2.433976 / 2.433480 / 2.433087 | 2.428402 / 2.427879 / 2.428335 | 105.20→105.42 |
| 反向/64 | 0.669834 / 0.670034 / 0.670152 | 0.668068 / 0.668514 / 0.668338 | 95.52→95.76 |
| 反向/128 | 1.253740 / 1.253661 / 1.253840 | 1.250255 / 1.249856 / 1.249739 | 102.09→102.41 |
| 反向/256 | 2.434619 / 2.434504 / 2.434192 | 2.428552 / 2.428391 / 2.428299 | 105.16→105.42 |

256-token 两轮约 **+0.22%**，decode GPU event
**9.152→9.132** 与 **9.155→9.133 ms/step**。
rank0 四次 replay kernel 数 **12,032→11,936**，
每次少 24 个启动；累计 kernel 时间正向
**41.272→41.058 ms**、反向 **41.161→41.073 ms**，
并非可直接扣除的请求时长。正向 128-token 合核
**明显更慢**，不能以反向较快样本声称稳定短输出收益。
所有长度旧/新版生成 ID 和三次 warm 的重复 ID 相同。
AllReduce 保持每 token 98 次；本阶段仍仅为单序列 runner，
不是 SGLang Engine.generate 同口径服务 KPI。

原始 artifact：
`qwen38-qsa-kv-write-{control-paired,candidate-paired,reverse-candidate,reverse-control}.json`
及各 `.trace.json`，
`qwen38-qsa-kv-write-fused-{teacher128,free128}.json`。
远端八卡 GPU/源码/benchmark/基线聚焦回归 **42 项通过**，
进程退出后各卡 2 MiB；仅同一个短 prompt，跨 QSA
budget、更多 prompt、恢复复用和实际服务未验收。
