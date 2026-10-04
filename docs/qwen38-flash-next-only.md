# Qwen3.8-Flash-Next GPU-only 产品宪章

**状态：已采纳，2026-10-03，经用户确认。**
本文取代 `v4-flash-only.md` 的主目标决策，不表示实现或性能验收已经完成。

## 目标与边界

- 唯一主目标：**Qwen3.8-Flash-Next，RadixArk NVFP4 权重**。
- 首个验收环境：**同一台 8×RTX 5090**；不以旧 Qwen 模型代替。
- 仅文本生成。不实现 vision、通用多模型 registry 或 speculative decoding。
- 专家、PLE 表、KV、GDN 状态及模型计算保持 GPU 驻留。禁止 CPU/SSD
  offload、unified-memory 迁移和静默 CPU 算子回退。
- CPU tokenizer、文件加载、控制与调度允许存在；GPU-only 不意味着没有 CPU 进程。
- `engine/` 独立自持执行，不 runtime import SGLang/vLLM。`scripts/` 中
  **外部基线**可以调用标准 SGLang，结果必须标明是基线而非 lite。
- Rust `control/` / `serving/` 边界不变；高级网关能力仍上移 UniGateway。

## 必须正确实现的模型图

检查点实际架构为 `qwen4_exp` / `Qwen4ExpForConditionalGeneration`。
实现需要覆盖 GDN 混合注意力、压缩 QSA、PLE n-gram 历史与 FP8 scale、
hyperconnection，以及 routed-expert NVFP4 的布局、scale 与分片。
不能把模型重命名或套入普通 Qwen-MoE 前向来代替。

KV 管理必须共同管理 QSA 页、GDN 状态和 PLE 历史的恢复、释放与复用。
先做正确的固定批执行，再依据实测决定 CUDA graph、融合算子及调度调整。

## 复用与性能纪律

允许 vendor 必要叶子代码，保留许可证头，记录准确 upstream commit。
Strata 仅提供 GPU GEMV、expert reduction、GPU token 选择、减少设备交接等
研究思路；不采用其 CPU 专家、SSD 分层或推测执行。

性能 KPI：**同机、同检查点、相同 GPU 资源预算、相同输入与输出长度的 warm tok/s
超过标准 SGLang**。用户已允许不同 GPU 并行结构，两边分别调优后公平对照；
必须披露实际 TP/EP/DP/PP，不得只调优 lite 而使用标准引擎的差配置。
相同 TP/EP 的结果仍作为受控对照保留。先确认生成正确性、权重/状态驻留和无静默 offload。
至少测单流 `1×128`、`1×256`，记录多次 warm 中位数、原始时长、软件版本、
配置和输出。端到端生成时长与纯 decode 时长须分别命名，不能混用。
合成图测量和 Strata 公布数字不是这一 KPI 的证据。

## 当前迁移状态

1. NVFP4 权重和 SGLang 0.5.20 基线已完成；TP8/EP8 的 256-token warm
   中位数为 114.45 tok/s，attention DP2 为 84.42 tok/s，暂不采用。
2. `engine/qwen38_runner/` 已有独立单序列文本原型，48 层真实权重加载与
   CUDA graph 捕获/释放已跑通。首轮八卡持久张量审计通过：rank0 为 1,650，
   其余各 1,458 个张量，全部 CUDA；不涵盖临时张量或 NCCL host staging。
   修正 BF16 GDN gate 及 packed-decode recurrent 输出顺序后，固定短 prompt
   的 64/128-token 自由生成 ID 与各自确定性参考完全一致；
   teacher-forcing top1 分别为 64/64 和 128/128，但最大所选 token
   logprob 误差分别仍约 0.144327 和 0.274001。
   再打包 QSA 的 K/V GEMV 后，单序列 runner 在两轮配对的
   256-token warm 约为 105.15/105.18 tok/s（相比紧邻版本
   的 104.78/104.76）；短输出提速不稳定。不含 serving
   scheduler，不能与 SGLang 的 Engine.generate
   直接当作等口径请求 KPI。**完整正确性、其他 prompt、长上下文/状态复用
   和服务端吞吐尚未验收**。来源见
   [QWEN38_SOURCES.md](vendor/QWEN38_SOURCES.md)。
3. 当前运行入口仍为 V4 默认。新路径验证前不翻转 gate 或宣称可部署。
4. 旧 V4 代码、技术文档与未提交修改保留为历史/兼容资产，不批量删除。
