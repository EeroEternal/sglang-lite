# DeepSeek V4-Flash 支持路线评审：先底座、后家族、最大化复用

**历史技术文档**：2026-10-03 当前主目标切换为
[Qwen3.8-Flash-Next GPU-only](./qwen38-flash-next-only.md)。下文保留 V4 技术与实测记录。

状态：设计评审稿（不改变现有 scope 结论，细化执行顺序与复用边界）。
关联文档：[scope.md](scope.md)、[standalone-inference-service-roadmap.md](standalone-inference-service-roadmap.md)、[architecture.md](architecture.md)。

## 1. 结论

**不把 V4-Flash 当成"补齐模型列表"的入口，而是当成逼出真实执行底座的压力测试目标。**

顺序：先补完通用 MoE 执行底座（真 paged attention、中央 engine loop、KV layout
抽象），再以 DeepSeek-V2-Lite 作为 MLA/非标准 KV layout 的先导模型，最后才把
`deepseek_v4` 作为一个受控家族注册进 `engine/models.py`。

同时明确**自研 vs 复用**边界：sglang-lite 只自持三个核心（RadixKVCache、
BatchingScheduler、MoEModelRunner 的组合与 engine loop），其余尽可能以小模块方式
复用 FlashInfer、sgl-kernel、Triton kernel、HF/官方模型定义等成熟组件。

## 2. 现状差距（基于当前代码）

V4 难的不是在 registry 加一行 `deepseek_v4`，而是它把「权重如何切到 8 卡」
「FP4/FP8 如何计算」「CSA/HCA 的 KV 形态」「Radix 如何复用这些 page」绑在一起。
当前执行面离这个要求还有距离：

1. **执行面仍是 HF 前向包装**。`engine/runner.py` 用
   `AutoModelForCausalLM + device_map=auto` 单进程加载；paged KV 是"影子存储"：
   每步 `radix.build_cache()` 把 page 重建成 HF `DynamicCache` 再喂给模型
   （`_past_for_seq`），真相仍是 HF `past_key_values`。284B/13B-active 级别下，
   重建 cache 的显存与拷贝开销本身就不可行。
2. **FlashInfer 未进入 attention hot path**。prefill/decode wrapper 已创建但
   forward 仍走 HF SDPA；FlashInfer 只用于 paged append。
3. **KV 写入是逐 token Python 循环**（`kv_cache.py: write_kv`），且 page 布局
   硬编码为全层同构 `[layers, blocks, block_size, kv_heads, head_dim]`。
   MLA compressed KV（DeepSeek-V2/V3 起）已不满足该假设——这笔 layout 债在
   V2-Lite 阶段就会被逼出，不必等 V4。
4. **P0 golden path 未闭环**：prefill 分组长度不一致时逐条串行回退；
   roadmap 中 M2/M3（真 per-layer KV + 中央 async loop）尚未完成。

因此第一步不是"接 V4"，而是把 M2/M3 真正做完。

## 3. 自研 vs 复用边界（本文档的核心补充）

原则：**核心组合权自持，叶子实现最大化复用**。sglang-lite 的差异化在于
「Radix 前缀复用 × continuous batching × MoE 执行」这条 hot path 的组合方式，
不在于任何单个 kernel 或 loader。凡与这条 hot path 的"驱动/调度核心"无关的内容，
一律优先引用外部小模块，不自研。

| 层 | 策略 | 复用来源 | 说明 |
| --- | --- | --- | --- |
| Attention kernel（paged prefill/decode、MLA） | **直接引用** | FlashInfer（含 MLA/paged wrapper）、sgl-kernel | 收口到 KernelBackend 接口后整体替换，不自写 kernel |
| MoE kernel（grouped GEMM、fused expert、topk routing） | **直接引用** | FlashInfer / sgl-kernel / Triton 现成 kernel | 同上；expert 路由计算留在模型图内 |
| 量化（FP8/FP4 GEMM、KV 量化） | **直接引用** | sgl-kernel / DeepSeek 官方 kernel（DeepGEMM 类） | 量化布局归 loader，量化计算归 KernelBackend，不散落 |
| 权重加载与分片 | Hybrid | HF safetensors 加载器 + 自持的分片映射 | 只自写 "哪个 shard 放哪张卡" 的映射，不自写文件格式 |
| Tokenizer / chat template | **直接引用** | HF tokenizers | 已是现状 |
| 模型图定义（V2-Lite / V4） | Hybrid | transformers remote code 或 DeepSeek 官方 `inference/` | 可整体引用模型定义，只要求其 attention/MoE 调用点可被 KernelBackend 接管 |
| TP 通信原语 | **直接引用** | torch.distributed / NCCL | 只自持 "TP=8 单机" 这一种受控拓扑的初始化与 shape 约定 |
| RadixKVCache（page 生命周期、COW、eviction） | **重构（自持）** | — | 核心差异化，唯一需要扩展的是 per-layer KV layout 描述符 |
| BatchingScheduler / engine loop | **重构（自持）** | — | 核心差异化 |
| Runner 的 batch 组装与采样 | **重构（自持）** | 采样可复用 flashinfer sampling kernel | 组合逻辑自持，叶子 kernel 复用 |

**不做的自研**：不自写 attention/MoE/量化 kernel，不自写分布式框架，不复制
SGLang 的完整 ModelRunner/EPLB/spec-decode 体系。若需要从 SGLang 借用，只借
**叶子级小模块**（如 sgl-kernel 的单个算子），不引入其调度或内存管理框架——
否则就失去了 lite 的意义。

对应到 AGENTS.md 的分层纪律，执行面重构建议只拆两层，避免 Phase 0 出现四层抽象：

- **KernelBackend**：attention + MoE + 量化计算的唯一收口，内部由外部 kernel 组成；
- **ModelLoader**：分片与量化布局；
- Prefill/Decode 执行逻辑暂留在 `ModelRunner` 内（保持 2 跳可追踪）。

### 3.0 可复用叶子组件清单（GPU 实测已回填，2026-08-04 / 8×5090）

以下均为独立可 pip 安装 / vendor 的叶子件，不携带外部调度或全局状态
（接入时在 KernelBackend 内固定版本并做 capability 探测）。实测环境：
torch `2.11.0+cu130`、CUDA 13.0、GPU RTX 5090（sm_120 / capability 12.0）。
测法见 3.0.1，逐项记录见 3.0.2；复现脚本 `scripts/leaf_component_probe.py`。

| 组件 | 实测版本 | 提供能力 | 接入点 | sm_120 结论 |
| --- | --- | --- | --- | --- |
| `flashinfer-python` | **0.6.12** | paged prefill/decode、`BatchMLAPagedAttentionWrapper`、`append_paged_kv_cache`；SM100：`trtllm_batch_decode_sparse_mla_dsv4`；SM120 叶子：`B12xMoEWrapper` / `Sm120B12xBlockScaledDenseGemmKernel` | KernelBackend（paged 已接入；MLA/sparse 按 arch 路由） | **标准 MLA/paged：可用**；**SM100 sparse MLA 在 sm_120 必挂**；须改走 SM120 专用 sparse MLA（见 3.0.3） |
| `sgl-kernel` | **0.4.4** | `topk_softmax` / MoE 辅助、`fp8_*_mm`、`dsv4_fused_*`、`cutlass_mla_decode`、norm/rope | KernelBackend（MoE/量化，S2+） | **可安装可 import**；topk 路由 id 与 torch 一致；**无 CSA/HCA 公开符号**，有 dsv4 融合算子 |
| `deep-gemm` / vendor | **2.5.0 SM120**（`engine/vendor/deep_gemm_sm120`，源自 vLLM 0.25 third_party） | `fp8_fp4_gemm_nt` / grouped FP4 | **已接入** Hybrid `kernel.fp4_gemm`（`v4_deep_gemm.py`） | pip `0.1.4` 仍 **sm_120 Unsupported**；运行时用 vendor，不 `import vllm` |
| DeepSeek 官方 `inference/` / transformers remote code | 本机有 `ds-v4-flash`；**无 V2-Lite 权重** | V4 模型图含 Indexer/Compressor/`sparse_attn`/HC(Sinkhorn) | ModelLoader / vendor（Hybrid） | V4 图可对照；**V2-Lite greedy/paged 验收 BLOCKED（缺权重）** |

**待实测确认（已实测 + 对照 vLLM SM120 路径后修正）**：

- flashinfer / sgl-kernel 对 V4-Flash 新 attention（官方代码表现为
  **Indexer + compressed KV + `sparse_attn`**，以及 **HC / `hc_split_sinkhorn`**；
  文档简称 CSA/HCA）：
  - 我们手测失败的是 **SM100 TRTLLM-GEN** 入口
    （`trtllm_batch_decode_sparse_mla_dsv4`）→ `Unsupported architecture`。
  - 这**不代表** SM120 不能跑 V4 sparse MLA，而是走错了架构族路径。
    vLLM 的做法是识别 `capability family 120` 后强制路由到
    FlashInfer **SM120 专用** sparse MLA（如 `FLASHINFER_MLA_SPARSE_SM120` /
    `sparse_mla_sm120`），**禁止**再走 SM100 TRTLLM。
  - 本机 flashinfer `0.6.12` 已暴露 SM120 叶子：`B12xMoEWrapper` /
    `b12x_fused_moe`、`gemm.Sm120B12xBlockScaledDenseGemmKernel`；
    标准 `BatchMLAPagedAttentionWrapper` 已在 sm_120 数值通过。
  - **SM120 sparse MLA 补测（2026-08-04）**：`0.6.12` 无
    `flashinfer.mla._sparse_mla_sm120`；隔离前缀安装
    `flashinfer-python==0.6.16.post1`（`FLASHINFER_DISABLE_VERSION_CHECK=1`，
    勿升级共享 venv——sglang 钉死 0.6.12 / cubin）后，公开 API
    `trtllm_batch_decode_sparse_mla_dsv4` **会路由到**
    `_trtllm_batch_decode_sparse_mla_dsv4_sm120`。冒烟成功条件：
    SWA/compressed KV 为 **packed uint8 last-dim 584**，并传
    `swa_topk_lens` + `extra_sparse_indices` / `extra_sparse_topk_lens`
    （bf16 512 会通过 Python 校验但内核仍要求 584）。复现：
    `FI_PREFIX=/tmp/fi1616 python scripts/try_flashinfer_016_sparse.py`。
  - sgl-kernel 提供 `dsv4_fused_*` 与 MLA 辅助，无 CSA/HCA 命名完整入口。
- deep-gemm 与 sm_120：pip **`deep_gemm==0.1.4` 仍 Unsupported**。  
  **已 vendor** vLLM 0.25 `third_party/deep_gemm` → `engine/vendor/deep_gemm_sm120`
 （2.5.0，含 py3.10–3.14 `_C*.so`）。Hybrid 默认
  `attach_v4_deep_gemm`：官方 MXFP4 **e8m0 1×32** →
  `fp8_fp4_gemm_nt(recipe_a=(1,128), recipe_b=(1,32))`；单核 vs TileLang
  **~2.3–2.5×**，e2e 仍受 ~72 `fp4_gemm`/tok launch 税限制。
- **B12xMoEWrapper**：construct 在 sm_120 可用，但 Hybrid 权是 **MXFP4 e8m0/sf32**
  且 TP+EP；B12x 要 **NVFP4 e4m3/sf16 + 无 EP** → `v4_b12x.py` **默认不 attach**。
  离线 repack + 非 EP 前不得焊进主路径。
- **不**把 pip `deep-gemm` 写入 `pyproject.toml`（用 vendor 树）；`sgl-kernel` 仍按需。

#### 3.0.1 叶子组件实测计划（GPU 环境，逐项验收）

统一前提：8×5090（sm_120）、CUDA/torch 版本与仓库固定的依赖矩阵一致；
每项先做"能装能跑"再做"数值对齐"；任何失败都记录具体版本与报错，
不带病接入 KernelBackend。

**T1 flashinfer-python（attention 主路径，已部分接入）**

- 测法：现有 `tests/test_gpu_paged_attention.py` 为基线；补充
  MLA wrapper 冒烟（`BatchMLAPagedAttentionWrapper` 或等价接口），用
  DeepSeek-V2-Lite 的 head 配置构造随机 KV，对比 torch 朴素实现的输出。
- 验收：paged prefill/decode 数值与参考实现 atol≤1e-2（bf16）；MLA wrapper
  在 sm_120 上可编译可运行。
- 回退：MLA 不可用则暂时用"解压回标准 KV + 普通 paged attention"的慢路径，
  并记 issue 跟踪上游。

**T2 sgl-kernel（MoE 与量化算子，S2+ 接入）**

- 测法：`pip install sgl-kernel` 后跑最小算子冒烟：`fused_moe`/topk 用
  Mixtral 小配置对比 HF 参考 MoE 层输出；FP8 GEMM 对比 bf16 GEMM 的
  相对误差。
- 验收：sm_120 上可安装（有预编译 wheel 或可源码编译）；fused_moe 输出与
  HF 参考 rel-err≤1e-2；单算子调用不依赖 SGLang runtime（import 仅
  sgl_kernel）。
- 回退：装不上或算子缺失时，MoE 先走 HF 模型图内的朴素实现，仅损失性能。

**T3 deep-gemm（FP8 grouped GEMM，V4 expert 候选）**

- 测法：安装后跑其自带 benchmark/测试；重点确认 sm_120 支持（该库最初面向
  Hopper sm_90，Blackwell 支持需实测）；FP8 grouped GEMM 数值对比 bf16 参考。
- 验收：sm_120 可运行且数值达标；不达标则明确记录"5090 不支持"。
- 回退：用 sgl-kernel 的 FP8 GEMM 或 cuBLASLt 路径替代。

**T4 官方模型图（V2-Lite 先导 → V4）**

- 测法：transformers remote code 加载 DeepSeek-V2-Lite，短 prompt greedy
  输出与 HF 参考一致（现有 reference correctness 测试的 GPU 版）；确认其
  attention/MoE 调用点可被 KernelBackend monkeypatch 接管（同 PR #4 手法）。
- 验收：V2-Lite 走 paged 路径 `paged_rebuild_count==0` 且逐 token 对齐；
  MLA 的 KV 进 RadixCache 需要的 layout 描述符明确（这是 S2 的输入）。
- 回退：remote code 不可控时 vendor 模型定义进仓库（标注来源与 license）。

产出要求：每项一份简短记录（版本、命令、结果、结论），汇总回填本文档
3.0 表格的"待实测确认"栏；全部通过后才把 sgl-kernel / deep-gemm 写进
`pyproject.toml` 可选依赖。

#### 3.0.2 实测记录（2026-08-04，5090 / sm_120）

**T1 flashinfer-python — PASS（标准路径）/ FAIL（V4 sparse）**

- 版本：flashinfer `0.6.12`；torch `2.11.0+cu130`
- 命令：`pytest tests/test_gpu_paged_attention.py`；
  `PYTHONPATH=. python scripts/leaf_component_probe.py`；
  另手测 `flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4`
- 结果：
  - 既有 GPU paged 基线 **3 passed**（`paged_rebuild_count==0`）
  - `BatchMLAPagedAttentionWrapper`（fa2）在 sm_120 可跑；vs torch 参考
    `max_abs≈0.0078`（bf16，≤1e-2）
  - `is_sm12x_supported(cuda)=True`
  - `trtllm_batch_decode_sparse_mla_dsv4`（64 heads / dim 512）→
    `TllmGenFmhaRunner ... Unsupported architecture`
- 结论：S2 可接标准 MLA wrapper；V4 sparse MLA **禁止**在 sm_120 上调用
  SM100 TRTLLM-GEN 入口，必须按 3.0.3 做 arch family 路由。

**T2 sgl-kernel — PARTIAL PASS（可装可跑，MoE 数值对齐未闭环）**

- 版本：sgl-kernel `0.4.4`（import 仅 `sgl_kernel`，无 SGLang runtime）
- 命令：probe 脚本 + `topk_softmax` / `fused_add_rmsnorm` 手测
- 结果：
  - 安装/import OK；暴露 `moe.*`、`fp8_*_mm`、`dsv4_fused_*`、`cutlass_mla_*`
  - `topk_softmax`：top-k **id 集合 8/8 与 torch.topk(softmax) 一致**
    （renormalize 后权重尺度不同，属预期）
  - `fused_add_rmsnorm` 在 sm_120 正常
  - 公开 API **无 CSA/HCA 命名**；有 V4 相关 `dsv4_fused_q_norm_rope` 等
- 结论：可作为 S2+ MoE/量化叶子候选；完整 `fused_moe`↔HF Mixtral rel-err
  验收仍待补；**暂不写入 pyproject**。

**T3 deep-gemm — FAIL（5090 不支持）**

- 版本：deep-gemm / `deep_gemm` `0.1.4`；`get_cuda_arch()==12.0`
- 命令：`deep_gemm.bf16_gemm_nt(a,b,d)`（256² bf16）
- 结果：`Assertion error (.../gemm.hpp:436): Unsupported architecture`
  （FP8 API 面存在但未再测——bf16 入口已失败）
- 结论：**明确记录「deep-gemm 0.1.4 在 sm_120 不可用」**（非“永远不能用
  DeepGEMM”，而是缺 SM120 内核/能力探测）。V4 expert GEMM 近期待
  SM120 版 DeepGEMM 或回退 FlashInfer B12x / sgl-kernel FP8 / cuBLASLt；
  **不写入 pyproject**。

**T4 官方模型图 — BLOCKED（V2-Lite）/ 可对照（V4-Flash）**

- 本机：`~/models/ds-v4-flash`（含 `inference/model.py`）；
  `deepseek-ai/DeepSeek-V4-Flash`；**无 DeepSeek-V2-Lite 权重/缓存**
- V4 图要点（S2/S5 输入）：`Attention` = MLA + sliding window +
  可选 `Compressor`/`Indexer` + `sparse_attn`；Block 侧 HC
 （`hc_mult` + `hc_split_sinkhorn`）；expert `fp4` + attention 侧 FP8 量化配置
- S2 layout 描述符需求（在缺 V2-Lite 权重时仍成立）：
  - 标准：`[layers, blocks, block_size, kv_heads, head_dim]`
  - MLA compressed：`ckv [blocks, page_size, kv_lora_rank]` +
    `kpe [blocks, page_size, qk_rope_head_dim]`（常见 `page_size=1`）
  - V4 另需 SWA 窗口池 + compressed 池 + sparse index（与当前单一
    isomorphic page 布局不兼容）
- 结论：V2-Lite greedy / `paged_rebuild_count==0` **未跑**（缺权重）；
  取得 V2-Lite 后再补 T4 闭环。

#### 3.0.3 SM120 架构族与 KernelBackend 路由（对照 vLLM，禁止抄调度）

vLLM 已证明：消费级 Blackwell（RTX 5090 / sm_120）与数据中心 Blackwell
（B200 / sm_100）**不是同一二进制兼容族**。sglang-lite 只借鉴其**叶子级
capability 探测 + 内核路由**，不引入 vLLM 调度/内存管理框架。

**硬件差异（为何不能复用 SM100 路径）**

| 特性 | SM100（B200） | SM120（5090） |
| --- | --- | --- |
| Shared Memory | ~228 KB | ~99 KB |
| Tensor Memory (TMEM) | 有 | 无 |
| 主要 MMA | tcgen05 / UMMA | mma.sync（偏 Ampere 风格） |
| 集群/多播 | 支持 | 基本不支持（1×1×1） |

依赖 TMEM / tcgen05 的内核（大量 SM100 FlashInfer TRTLLM、部分 DeepGEMM）
在 SM120 上会直接 `Unsupported architecture`——这与 §3.0.2 手测一致。

**vLLM 做法（参考，非目标 scope）**

1. 设备能力：把 `is_device_capability_family(120)` 纳入支持列表（相关改动如
   vLLM `#41062` / `#41028`）。
2. Attention：新增 `FLASHINFER_MLA_SPARSE_SM120`；DeepSeek-V4 sparse MLA 在
   SM12x **强制**走 SM120 专用内核，**禁止** SM100 TRTLLM；prefill/decode
   分治并缩小 workspace（适配 32GB）。
3. MoE/GEMM：**DeepGEMM SM120 vendor 已接**（单核 OK）；下一步
   `m_grouped_fp8_fp4_gemm_nt_contiguous` 降 launch 税。B12x 仅在 NVFP4
   repack + 非 EP 后可选。
4. 大 PR 参考：上游 `#43477`（DeepSeek V4 + SM120 基础）；社区更完整分支如
   jasl `#41834`（含更多 V4-Flash 细节）。**DSpark 投机解码仍为 sglang-lite
   scope 外**。

**本机 flashinfer SM120 叶子（2026-08-04 补测）**

- `0.6.12`：`B12xMoEWrapper` / `b12x_fused_moe`、
  `gemm.Sm120B12xBlockScaledDenseGemmKernel`、`mla.is_sm12x_supported==True`；
  **无** `_sparse_mla_sm120`（公开 dsv4 仍走 SM100 → 必挂）。
- `0.6.16.post1`（隔离 `/tmp/fi1616`）：存在
  `flashinfer.mla._sparse_mla_sm120`；公开
  `trtllm_batch_decode_sparse_mla_dsv4` **自动路由 SM120**；packed uint8 584
  + `swa_topk_lens` / `extra_sparse_*` 冒烟 **OK**（shape `(B,q,H,512)`，finite）。
- DeepGEMM：**vendor 2.5.0 SM120** 已通；pip `0.1.4` 仍 FAIL（勿用）。

**sglang-lite KernelBackend 必须遵守的规则**

1. capability 声明包含 `arch_family ∈ {sm90, sm100, sm120, ...}`，以及
   `sparse_mla_backend` / `moe_gemm_backend` 枚举；runner **不得**写
   `if major == 10`。
2. SM120 上 V4 sparse MLA：必须确认走 **SM120 后端**（FI≥0.6.16 的公开
   dsv4 可接受，因其内部 dispatch 到 `_…_sm120`）；在仅有 0.6.12 时**禁止**
   调用会落到 SM100 TRTLLM 的 dsv4，应改官方 `sparse_attn` 或升 FI。
3. MoE：SM120 **优先 vendor DeepGEMM**（`has_deep_gemm_sm120` probe）；B12x
   仅布局/EP 匹配时；否则 TileLang `fp4_gemm`。
4. 复用边界不变：只借叶子 kernel 与路由表，不借 vLLM ModelRunner/Scheduler。

**S5 前增量验收（补测）**

- [x] SM120 sparse MLA 入口可 import + 随机 KV 冒烟（FI 0.6.16.post1 隔离前缀；atol 另定）
- [~] `B12xMoEWrapper`：API 可 construct；**官方 Hybrid 布局不兼容**（见 `v4_b12x.py`）
- [x] DeepGEMM SM120 vendor：`fp8_fp4` 真实专家权 + e2e attach（2026-08-08 PRO6000）
- [x] KernelBackend 单测：伪造 sm100/sm120 capability 时路由表断言（`tests/test_capability_routing.py`）

### 3.1 硬件后端隔离（非 NVIDIA 卡的前置设计）

未来需要支持华为昇腾（Ascend/CANN）及其他非 NVIDIA GPU，因此硬件隔离边界必须在
KernelBackend 设计时一次划清，避免事后剥离 CUDA 假设：

- **KernelBackend 是唯一的硬件抽象点**。attention、MoE、量化、采样 kernel 与
  TP 通信原语全部收口在此；CUDA + FlashInfer/sgl-kernel 只是它的第一个实现
  （`CudaKernelBackend`），昇腾对应 `AscendKernelBackend`（torch_npu + CANN 算子/
  MindIE kernel），其他卡同理各自成模块。
- **KernelBackend 之外禁止硬件特有假设**：RadixKVCache、scheduler、loader、
  engine loop 只依赖 torch device 语义与 layout 描述符，不 import flashinfer、
  不写 `torch.cuda.*` 直调、不假定 NCCL（通信后端由 KernelBackend 声明，
  如 NCCL/HCCL）。现状中 runner 顶层 `import flashinfer` 与 CUDA-only 分支
  是要在 S1 重构中清掉的典型例子。
- **能力声明而非 if-else**：每个后端声明自己支持的 dtype（FP4/FP8 是否可用）、
  KV layout、CUDA graph 等价物等 capability；上层按声明降级（如昇腾初期
  BF16-only），不在核心路径散落设备判断。
- **优先级**：Phase 0/1 只实现和验证 CUDA 后端；昇腾等以"接口预留 + 不阻塞"的
  方式对待——即接口设计时用上述规则审查，但不提前实现，避免抽象空转。

## 4. 实施顺序

```text
S1  M2/M3 补完：paged KV 成为唯一真相（去掉 DynamicCache 重建），
    FlashInfer prefill/decode kernel 进入 hot path，中央 async engine loop。
    验证模型：fixture / Mixtral 小模型。
S2  KV layout 抽象（跳过 V2-Lite 先导，直接用 V4 形状）：
    RadixKVCache 增加 layout 描述符（标准 KV / MLA compressed / DSV4 packed），
    KernelBackend 收口 MLA + arch_family 路由（见 3.0.3）。
S3  单机 TP=8（受控拓扑，5090×8）：ModelLoader 分片 + 官方 convert 契约；
    EP 只做最小可用，不做 expert load-balancing 调度。
S4  数值路径：BF16 → FP8；量化计算全部走 KernelBackend（B12x / sgl-kernel）。
S5  注册 deepseek_v4 家族：tokenizer 直接引用；模型图 Hybrid
    （官方 inference/）；FP4 expert + FP8 attention 经 KernelBackend 门面；
    CSA/HCA 的 KV 形态以 S2 的 layout 描述符承接；
    KernelBackend 按 arch_family 路由 SM120 sparse MLA + B12x/MoE（见 3.0.3）。
    mHC、hash routing 等留在模型图内部，scheduler 不感知。
```

**决策（2026-08-05）**：跳过 DeepSeek-V2-Lite 先导（无权重且对 V4 终局价值低）。
叶子能力通过 FlashInfer / sgl-kernel / 官方 `inference/` **小包引用**接入，
禁止整段引入 vLLM/SGLang 调度。P0 脚本：`scripts/v4_official_smoke.sh`；
短生成：`scripts/v4_lite_short_gen.py`。

## 5. 明确不做或后置

与 scope.md 一致：

- DSpark/MTP 投机解码：scope 外，主模型稳定 decode 即可；
- 1M 上下文：后置，先保证常规长度正确性与吞吐；
- tool/reasoning parser：上移 gateway；
- 完整 EP、disagg、多节点：非目标。

## 6. 验收标准（窄口径）

8×5090 上：

1. 加载本地 `ds-v4-flash` 权重成功（TP=8）；
2. 短 prompt 生成正确，数值与官方参考实现 / vLLM 短输出对齐；
3. prefix 复用时 `cache_hit_tokens` 上升，且对应真实 prefill 计算减少；
4. standalone serving 能真实 SSE 流式输出。

任何一条达不到，说明底座还没好，不应继续堆 V4 特例。

### 6.1 P0 官方基线进度（2026-08-05 / 8×5090）

- **convert PASS**：`inference/convert.py --model-parallel 8` →
  `/tmp/ds-v4-mp8/model{0..7}-mp8.safetensors`。
- **generate PASS**：`torchrun --nproc-per-node=8 generate.py`，prompt
  `Hello`，greedy 输出：`Hello! How can I help you today`（max_new_tokens=8）。
- **环境要点**：
  1. `--config` 必须用 `inference/config.json`（ModelArgs 形），不是 HF config；
  2. TileLang JIT 只用系统 CUDA（`PATH=/usr/local/cuda/bin`，
     `CPATH=/usr/local/cuda/include`），禁止混入 pip `nvidia/cu13` 头；
  3. `fast_hadamard_transform`：需
     `pip install --no-build-isolation` + `TORCH_CUDA_ARCH_LIST=12.0` 从源码装。
- 脚本：`scripts/v4_official_smoke.sh`、`scripts/v4_lite_short_gen.py`。
- 引擎侧已落地：`arch_family` 路由、`KvLayout`、`deepseek_v4` 注册、
  `ModelLoader` TP=8 / Hybrid 入口（需 `WORLD_SIZE` +
  `SGLANG_LITE_DSV4_CONVERTED`）。

### 6.2 LiteEngine CB MVP（官方 forward 包装，2026-08-05）

- **范围**：`ModelRunner` 对 `_v4_hybrid` 调用官方
  `Transformer.forward(input_ids, start_pos)`；Scheduler continuous batching
  可出 token；KV 仍在官方 `Attention` 缓冲内；Radix prefix hit **禁用**。
- **加载顺序**：`ensure_tp_process_group()`（NCCL）必须在 `Transformer(...)`
  之前；入口脚本 `scripts/v4_lite_engine_gen.py`（`torchrun`，`start_loop=False`
  + `pump_until_idle`，全 rank 同步 forward）。
- **计时用例**（8×5090，`scripts/v4_lite_engine_gen.py`，prompt `Hello`，
  ignore_eos，warm tok/s）：

  | Case | LiteEngine warm | 官方基线 warm |
  | --- | --- | --- |
  | 1×128 | ~4.8–5.1 | ~5.0 |
  | 4×96 | ~18.3 | ~13.3 |
  | 1×256 | ~4.8 | ~4.9 |

  （仍走官方 `sparse_attn`；入口需在 import torch 前 remap
  `CUDA_VISIBLE_DEVICES=$LOCAL_RANK`，供 TileLang device_id=0。）
- **Prefix 复用（2026-08-05）**：`engine/v4_prefix_cache.py` 对官方
  `kv_cache` / `kv_state` / `score_state` 做 CPU 快照；admit 用最长精确前缀
  命中设 `cache_hit_tokens`/`cached_len`；exact hit 跳过 prefill forward，
  partial hit 只跑 suffix（`start_pos>0` 时按官方约束逐 token）。单测：
  `tests/test_v4_prefix_cache.py`。仍非 Radix 双池 COW。
- **Standalone SSE（2026-08-05）**：`python -m sglang_lite.process` 支持
  `torchrun` TP（rank0 HTTP NDJSON，其余 rank `broadcast_object_list` +
  `pump_until_idle`）；`sglang-lite-serving --tp 8` 或
  `scripts/v4_serve_sse.sh`；Rust SSE 输出 `chat.completion.chunk` + 最终
  `usage` + `data: [DONE]`。控制面单测：`control/tests/sse_stream_tests.rs`。
- **真机验收（2026-08-05 / 8×5090）**：
  - **Align**：`match_soft_top5=true`（`Hello`∈prefill top5；greedy 常翻成
    `你好`，logit 差≈0.09）。摘要 `/tmp/v4_align_summary.json`。
  - **SSE**：`torchrun -m sglang_lite.process` + `sglang-lite-serving
    --engine-url` → OpenAI `chat.completion.chunk` + `[DONE]`（需
    rustup stable≥1.85；系统 cargo 1.75 不够）。引擎侧需 **专用 CUDA
    线程**（asyncio executor 会触发 TileLang device mismatch）。
  - 脚本：`scripts/v4_remote_acceptance.sh`、`v4_align_tokens.py`、
    `v4_debug_first_token.py`。
- **KV 生命周期（Hybrid）**：`clear_v4_kv_slot` 在 cold prefill 前与请求
  `final` 时清官方 Attention/Compressor 行；prefix 仍靠 CPU 快照，非 Radix
  双池。
- **内核现状**：生产 decode 维持官方 `sparse_attn`；FI SM120 sparse MLA
  仍 blocked（§6.3），默认 `SGLANG_LITE_V4_DISABLE_FI_SPARSE=1`。
- **数值稳态 / logits 门禁（2026-08-05）**：`scripts/v4_logits_compare.py`
  （`torchrun` TP=8，`SGLANG_LITE_V4_DISABLE_FI_SPARSE=1`，seed 与 align
  同源）。8×5090 / prompt `Hello`（5 tok）Lite prefill 末步：argmax=`你好`
  (27.574) vs top2 `Hello` (27.483)，**top2 Δ≈0.09**；`gate_soft=true`
  （英文 lead 仍在 top5）。摘要 `/tmp/v4_logits_compare.json`，可选
  `--official-logits` 填 `max_abs` / `argmax_match` / `top5_overlap`。
  全层逐层 hook 仍不做（官方 Transformer 未暴露 per-layer API）。CVD
  remap 导致与官方 `set_device(local_rank)` 贪心首 token 可翻牌，属已知软门槛。
- **增量 decode UTF-8（2026-08-05）**：`ModelRunner.detokenize_delta` 收紧
  （不完整多字节空 delta；发散改写不回吐整段；lone `�` 抑制）；
  `SchedulerLoop._apply_stop_and_limits` 全分支走 delta；NDJSON
  `ensure_ascii=True`。单测 `tests/test_detokenize_delta.py`。
- **下一阶段**：见 **§8 通往健全引擎**（Phase 0c 自持 KV → Phase 1 换核 →
  Phase 2 生产硬化）。不再把 UTF-8 / logits 门禁当作未完成项。

### 6.3 SM120 sparse MLA decode 换核（2026-08-05）

- **入口**：`scripts/v4_lite_engine_gen.py` 在 import 前插入
  `SGLANG_LITE_FI_PREFIX`（默认 `/tmp/fi1616`）+
  `FLASHINFER_DISABLE_VERSION_CHECK=1`；共享 venv 仍可保留 FI **0.6.12**。
- **Pack**：`engine/dsv4_kv_pack.py` — bf16 KV → uint8 584
  （448 FP8 NoPE + 128 RoPE bf16 + 7 ue8m0 + pad）；优先用官方
  `kernel.act_quant`。
- **Hook**：`engine/v4_sparse_mla.py` `attach_v4_sparse_mla` 在 Hybrid load
  后替换 `model.sparse_attn` / `kernel.sparse_attn`；**仅 decode**
  （`q_len==1`）走 `KernelBackend.sparse_mla_decode_dsv4`；失败或 prefill
  回退官方 TileLang。
- **验收**：capability 为 `flashinfer_sparse_sm120`；hook 已 armed；同用例
  warm tok/s 对照 §6.2。
- **本机实况（2026-08-05）**：已在隔离前缀 `/tmp/fi1616_full` 对齐安装
  `flashinfer-python==0.6.16.post1` + `flashinfer-cubin==0.6.16.post1` +
  `flashinfer-jit-cache==0.6.16.post1+cu130`（含 `sparse_mla_sm120.so`）。
  probe（随机/真实 584 pack、H=8/64）仍 **absmean=0**（finite）。
  非版本错配问题；更像 SM120 dsv4 在本机 5090（sm_120）上的上游内核/
  调用约定未闭环。hook 零输出探测会回退官方 `sparse_attn`。
  远端直连 GitHub 超时，wheel 需本机下载后 `scp`。

### 6.4 PRO6000（8× RTX PRO 6000 / sm_120，2026-08-06）

Host：`ssh -p 2208 bodesi@39.183.171.3`（`pro6000`）。单卡 ~96GB，compute_cap 12.0。

| 项 | 结果 | 说明 |
| --- | --- | --- |
| FI 0.6.16 SM120 sparse MLA | **仍 blocked** | 与 5090 相同：probe **absmean=0**（finite）；换核继续默认关 |
| MP8 convert | **PASS** | `~/models/ds-v4-mp8`；缺失键仅 MTP（spec 外） |
| 官方 generate（**宿主** torch 2.11+cu130） | **PASS** | CVD remap + `set_device(0)`；`Hello` → `Hello! How can I help you today?`；prefill top1=`Hello`≈28.9 |
| 官方 / Hybrid（**Docker** torch 2.9.1+cu129） | **FAIL 数值** | prefill top5=`to`/`:`/`?`，logit~16；soft gate 失败。**非 Hybrid bug**，官方路径同坏 |
| Hybrid logits 门禁（宿主） | **PASS** | `gate_soft=true`；argmax=`Hello`≈28.27；top5 含 `Hello`/`Hi`/`你好` |
| Hybrid 吞吐门禁（宿主） | **PASS（基线）** | 见 §6.4.1；官方 `sparse_attn` + DISABLE_FI=1 |
| Phase 1 FI（实验） | **数值可对齐；e2e 未加速** | footer pack 后 vs 官方 maxdiff≈0.018；1×128 warm 仍慢于官方；**不默认开** |
| SGLang blackwell 镜像 | **SM120 内核不足** | `flash_mla sparse_decode_fwd: Unsupported architecture`；`latest` 亦无 sm_120 |

**关键环境结论（PRO6000）**

1. **必须在宿主 venv 跑数值路径**：`torch==2.11.0+cu130` + 系统 CUDA 13.0 + `tilelang==0.1.8` + `apache-tvm-ffi==0.1.9` + 自编译 `fast_hadamard_transform`（`TORCH_CUDA_ARCH_LIST=12.0`）。Docker `lmsysorg/sglang:deepseek-v4-blackwell`（cu129 / torch 2.9）在本机 **sm_120 上数值不可用**，不得作为 Hybrid 验收 runtime。
2. TileLang 仍需 **CVD=`LOCAL_RANK` → 每进程仅见 `cuda:0`**；官方 `generate.py` 的 `set_device(local_rank)` 在 CVD 后须改为 `set_device(0)`（见 `~/bench/host_gen_cvd.py`）。
3. 宿主 venv 建议：`source ~/venvs/sglang-lite/bin/activate`；`SGLANG_LITE_FI_PREFIX=`（空，避免 `/tmp/fi1616` 残缺 jit-cache 干扰 import）；`SGLANG_LITE_V4_DISABLE_FI_SPARSE=1`。
4. 装 `flashinfer-python` 时务必 **`--no-deps`**，否则会把 torch 降到 2.9.x / 混入 cu12 NCCL 导致 `undefined symbol: ncclDevCommDestroy`。

复现门禁：

```bash
source ~/venvs/sglang-lite/bin/activate
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda CPATH=/usr/local/cuda/include
export SGLANG_LITE_DSV4_HF=~/models/DeepSeek-V4-Flash-0731
export SGLANG_LITE_DSV4_CONVERTED=~/models/ds-v4-mp8
export SGLANG_LITE_V4_DISABLE_FI_SPARSE=1 SGLANG_LITE_FI_PREFIX=
torchrun --nproc-per-node=8 scripts/v4_logits_compare.py --prompt Hello --out ~/bench/v4_logits_host.json
# 期望 gate_soft=true，lite_argmax_text 含 Hello 或 你好
```

#### 6.4.1 PRO6000 Hybrid 吞吐基线（2026-08-06，宿主 torch 2.11+cu130）

脚本：`scripts/v4_lite_engine_gen.py`（TP=8，prompt `Hello`，ignore_eos，
`SGLANG_LITE_V4_DISABLE_FI_SPARSE=1`，官方 `sparse_attn`）。日志：
`~/bench/v4_thru_pro6000.log`。

| Case | cold tok/s | warm tok/s | load_s | sample 前缀 |
| --- | --- | --- | --- | --- |
| 1×128 | 6.90 | **7.59** | ~8.1 | `Hello! How can I help you today?` |
| 4×96 | 15.51 | **16.74** | ~7.6 | 同上 + 中文 DeepSeek 说明 |
| 1×256 | 7.12 | **7.51** | ~7.6 | 同上 + capital of France |

相对 8×5090 Hybrid 基线（§6.2：1×128 ~4.8–5.1 / 4×96 ~18.3 / 1×256 ~4.8）：
PRO6000 单请求 decode 更高，batch=4 接近同量级。本表为 **吞吐对照基线**——
FI SM120 默认未开；任何路径变更后 warm 应 ≥ 此表（允许小幅方差）。

#### 6.4.2 0c-4 page-stage 吞吐回归（2026-08-06，同宿主）

官方 `sparse_attn` + **page-primary stage**（`DISABLE_FI=1`）。日志：
`~/bench/v4_thru_0c4_{1x128,4x96,1x256}.log`；对照 JSON：
`~/bench/v4_thru_0c4_vs_baseline.json`。

| Case | 基线 warm | 0c-4 warm | Δ warm | 0c-4 cold | load_s |
| --- | --- | --- | --- | --- | --- |
| 1×128 | 7.59 | **7.43** | **−2.1%** | 6.74 | 7.35 |
| 4×96 | 16.74 | **16.59** | **−0.9%** | 14.67 | 8.28 |
| 1×256 | 7.51 | **7.33** | **−2.4%** | 6.95 | 7.61 |

样本前缀均 `Hello! How can I help you today`。门禁：**PASS**（|Δ warm| < 10%，
落在「小幅方差」内）。结论：0c-4 每步 stage 的 e2e 开销约 **1–2.5%**，可接受；
**不**据此关掉 page-primary。

## 7. 风险与开放问题

- V4-Flash 的 CSA/HCA、FP4 expert、mHC 细节以官方 `inference/` 参考实现为准，
  本文对其的描述属于需求输入，接入前需逐项核对；
- FlashInfer / DeepGEMM 的 **SM120 专用内核**进度是外部依赖：SM100 路径在
  5090 上不可用；KernelBackend 必须 capability 路由并允许回退官方
  `sparse_attn` / B12x / cuBLASLt（见 3.0.3）；
- 把 SM120 “当作 SM100” 是已验证的失败模式，禁止再出现；
- MLA / V4 sparse 的 Radix page 复用语义（compressed + SWA 双池、COW/fork）
  需要在 S2/S5 单独验证；
- vLLM `#43477` / 社区 `#41834` 仅作叶子路由参考，不扩大 sglang-lite 到
  DSpark/宽功能面。

## 8. 通往健全引擎（阶段定义与验收）

「健全」在本项目的窄口径：**不是** vLLM 功能面完整，而是 AGENTS /
[scope.md](scope.md) 的三类引擎能力在**自持路径**上闭环，外加可部署的
standalone SSE 与明确数值门禁。

1. **KV 生命周期 + prefix 复用**：Radix（或等价 BlockKV）真写/真读/COW/淘汰；
   `cache_hit_tokens` = 真实跳过的 prefill。
2. **Continuous scheduling**：多请求 CB、可取消/超时、TP 可部署。
3. **Model execution**：decode 热路径走自持 KernelBackend（capability 路由）；
   官方 `sparse_attn` 仅作回退，不是长期唯一路径。

当前：**Phase 0c 切片 1–3 + Phase 1 门禁骨架**——Hybrid + dual-pool page
restore + PRO6000 吞吐基线（§6.4.1）；decode 仍绑官方 `sparse_attn`（FI SM120
absmean=0，§6.3 / §8.3.1）。

```
Phase0 HybridMVP → Phase0b Stabilize → Phase0c OwnKV → Phase1 OwnKernels → Phase2 Production
```

### 8.1 Phase 0b — 稳态与门禁（已完成）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 增量 decode UTF-8 | **done** | `detokenize_delta` + loop 全分支；`tests/test_detokenize_delta.py` |
| Prefill logits 门禁 | **done** | `scripts/v4_logits_compare.py`；5090 soft top5 / top2 Δ≈0.09（§6.2）；**PRO6000 宿主** soft `Hello`/`你好`（§6.4） |
| TP SSE 硬化 | **done** | 专用 CUDA 线程；`v4_remote_acceptance.sh` 手工闸门 |
| Radix 双池 / FI 换核 | **不做**（本阶段） | 见 0c / 1 |

### 8.2 Phase 0c — 自持 KV（切片 1–4 + 真机门禁 **完成**）

目标：去掉「官方 Attention 缓冲 + CPU 整包快照」作为唯一 prefix 路径。

- Radix 扩展：compressed MLA + SWA / `dsv4_packed(584)` 双池（`KvLayout` 雏形已有）；
  page 分配/释放/COW。
- Hybrid 过渡：先双写或从官方 buffer 导出到 Radix page，再切「Radix 为源」。
- Scheduler：多 slot 与 `cached_len` 分组在 V4 上正确；结束必释放 page。

验收：二次同前缀 `cache_hit_tokens > 0` 且 prefill forward tokens 下降；取消/结束无泄漏。

#### 8.2.1 切片 1（已落地）— 双池 page API + Hybrid 双写

| 项 | 状态 | 说明 |
| --- | --- | --- |
| SWA + compressed packed 双池分配 | **done** | `RadixCache.packed_swa_cache` / `packed_comp_cache`；`write_packed_kv` / `read_packed_kv` |
| COW / release 覆盖双池 | **done** | `cow_block_if_shared` / `release_blocks` 同步拷贝与清零 |
| 导出 API | **done** | `engine/v4_dual_pool.py`：`dual_write_from_bf16` / `dual_write_from_model` |
| Hybrid prefill 后双写 | **done** | `_v4_maybe_save_prefix(..., radix=)` best-effort 导出；**restore 仍走 CPU snapshot** |
| 结束释放 dual pages | **done** | `v4_release_seq` → `release_dual_pool_pages` |
| 单测 | **done** | `tests/test_v4_dual_pool.py`、`test_kv_layout` 扩展 |

#### 8.2.2 切片 2（已落地）— 双池生命周期 + hit fork + decode append

| 项 | 状态 | 说明 |
| --- | --- | --- |
| Prefix cache 持有 dual page ref | **done** | `V4PrefixCache.bind_radix` + insert `fork_blocks`；evict/replace/clear 释放 |
| Seq 与 cache 所有权分离 | **done** | 写入 seq 保留 allocate-ref；cache 额外 fork；finish 只放 seq fork |
| Hit 路径 fork dual pages | **done** | admit → `v4_attach_dual_pool_from_entry` / `fork_dual_pool_for_hit`；`dual_hit_count` |
| Decode dual-append | **done** | `_v4_dual_append_decode` → `dual_append_from_model`（best-effort 扩页） |
| 可观测 | **done** | `get_stats()["dual_pool"]` + `v4_prefix` dual_* 计数 |
| 单测 | **done** | `tests/test_v4_dual_pool_lifecycle.py` |

#### 8.2.3 切片 3（已落地）— bf16 page restore（KV 以 page 为主源）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| bf16 restore 池 | **done** | `RadixCache.restore_bf16_cache` `[L,B,P,512]`；与 dual-write 同页 id |
| dual_write 写 restore 页 | **done** | `write_dual_pool_layer(..., write_restore_bf16=True)` + `layer_keys` |
| hit 优先 page restore | **done** | `_v4_ensure_restored` → `restore_dual_pool_to_model` 再 snapshot 补 state |
| slim snapshot | **done** | `dual_primary`：入库时丢掉已 page 化的 `*.kv_cache`，保留 state |
| 单测 | **done** | `tests/test_v4_dual_pool_restore.py` |

#### 8.2.4 PRO6000 真机 dual stats（2026-08-06，宿主 torch 2.11+cu130）

脚本：`~/bench/v4_dual_stats_probe.py`（cold + warm `Hello`，TP=8）。

| 指标 | cold | warm |
| --- | --- | --- |
| 文本 | `Hello! How can I help you today` | 同 |
| `cache_hit_tokens` | 0 | **5** |
| `dual_write_count` | 1（5 tokens） | 1 |
| `dual_hit_count` | 0 | **1** |
| `dual_restore_count` | 0 | **1** |
| `dual_append_count` | 7 | 14（decode 追加） |
| `prefix_dual_primary` | 1 | 1 |
| 门禁 | — | **PASS**（write/hit/restore/warm_hit 全 true） |

摘要：`~/bench/v4_dual_stats_pro6000.json`。退出时 NCCL destroy SIGABRT 仍有（与 align 相同，**不影响门禁结果**）。

#### 8.2.5 切片 4（已落地）— page-primary stage（官方核）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 页为 SoT 标记 | **done** | `seq._v4_page_primary`：dual-write slim 后 / hit restore 后置位 |
| decode 前 stage | **done** | `_v4_stage_pages_before_forward` → `stage_official_kv_from_pages` |
| 与 hit restore 分计数 | **done** | `dual_stage_count` vs `dual_restore_count` |
| attention 核 | **仍官方** | TileLang `sparse_attn` 读 stage 后的 module buffer；FI 非默认 |
| 单测 | **done** | `tests/test_v4_dual_pool_stage.py` |

契约：page-primary 时 **pages 是 KV 真源**；官方 buffer 只是
`sparse_attn` 的 staging。decode 后仍 `dual_append` 把新 token 写回 pages。

#### 8.2.6 PRO6000 0c-4 真机（2026-08-06，宿主 torch 2.11+cu130）

脚本：`scripts/v4_dual_stats_probe.py`（cold+warm `Hello`，max_new=8，官方核）。

| 指标 | 结果 |
| --- | --- |
| cold / warm 文本 | `Hello! How can I help you today` |
| `warm_cache_hit_tokens` | **5** |
| `dual_write_count` | 1（5 tokens） |
| `dual_hit_count` | **1** |
| `dual_restore_count` | **1** |
| `dual_stage_count` | **14**（cold 后 7，warm 累计 14） |
| `dual_append_count` | 14 |
| 门禁 | **PASS**（write/hit/restore/stage/warm_hit/text 全 true） |

摘要：`~/bench/v4_dual_stats_0c4.json`。退出时 NCCL destroy 仍可能挂
（与历史 dual probe 相同，**不影响门禁 JSON**）；探针已改为各 rank 对称
`shutdown`、无结果后 barrier。

吞吐回归见 **§6.4.2**（0c-4 vs §6.4.1：warm Δ ≈ −1%～−2.5%，**PASS**）。

**仍后置**：decode 直接读 packed 页做 FI leaf（不默认）；更细的 SWA/comp
分池语义与官方 ring 对齐。

### 8.3 Phase 1 — 自持 decode 内核 + MoE（**非当前主线**）

**产品策略（2026-08-06 冻结）**：

- **主路径 = 官方 `sparse_attn`**；生产默认 `SGLANG_LITE_V4_DISABLE_FI_SPARSE=1`。
- **FI 永不因 probe 自动成为默认**；仅 `SGLANG_LITE_V4_FORCE_FI_SPARSE=1` 或
  显式实验。sglang-lite 下一刀是 **0c-4 页为源**，不是默认换核。
- FI 仅 `KernelBackend` leaf；layout 知识留在 dual-pool / `dsv4_kv_pack`。
- 吞吐主门禁始终对照 **官方主路径** §6.4.1；FI 另表记录，不替代基线。

其余（MoE B12x / CUDA graph 等）有收益再开，不阻塞 0c-4。

#### 8.3.1 换核门禁骨架（2026-08-06 已落地）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 保守路由 | **done** | SM120 默认 `official_sparse_attn`；FI 仅 FORCE 或（实验）numerical_ok |
| 数值探针 API | **done** | footer pack 后探针可非零；**不等于**生产切 FI |
| 独立脚本 | **done** | `phase1_kernel_probe.py` / `phase1_fi_vs_official.py` |
| 单测 | **done** | `tests/test_capability_routing.py` |
| 零输出护栏 | **done** | hook 若 FI 空输出则进程内回退官方 |
| Hybrid 生产默认 | **官方** | `DISABLE_FI_SPARSE=1`；不计划在 0c-4 前改默认 |

#### 8.3.2 Path A：真实 tensor 对齐（2026-08-06 PRO6000）

脚本：`scripts/phase1_fi_vs_official.py`（hook 首个 decode 的官方
`sparse_attn` 张量 → pack → FI 多变体对照）。

**根因（已修）**：`to_paged_hnd` 原先按 token 交错存 584 B；FI SM120 DSV4
物理页是 **footer 布局**：

```
page 内: [0, page*576) = 每 token 的 [nope|rope]（576B）
         [page*576, page*584) = scale footer（每 token 8B）
```

见 FI `kv_cache_traits.cuh` / `kv_cache_io.cuh`。交错布局 → absmean=0 或
爆炸；footer 布局后与官方对齐。

| 变体 | FI absmean | max‖FI−ref‖ | mean‖diff‖ | 备注 |
| --- | --- | --- | --- | --- |
| official ref | 0.281 | — | — | TileLang |
| act_quant + footer | **0.281** | **0.0176** | 0.00284 | 推荐 |
| torch_fp8 + footer | 0.281 | 0.0176 | 0.00284 | 同量级 |
| no sinks | 0.292 | 0.69 | 0.012 | sinks 需保留 |
| 随机 uint8（旧探针） | 0.0 | — | — | 无效输入，不能当门禁 |

摘要：`~/bench/phase1_fi_vs_official_footer.json`。捕获层为纯 SWA
（`comp_cols=0`，`swa_lens=6`，prompt 短 decode）。

**结论**：FI **数值可用**（footer + 真实 tensor，max_abs_diff≈0.018），但 Hybrid
每步 pack 的 e2e **1×128 warm 慢于官方**（~6.13 vs ~7.47 tok/s，FORCE 试跑）。
**不**据此默认开 FI。主线回到 **0c-4 页为源**；FI 作页就绪后的可选 leaf。

#### 8.3.3 e2e 吞吐试跑（PRO6000，1×128，2026-08-06）

| 路径 | warm tok/s | 说明 |
| --- | --- | --- |
| 官方主路径 | **~7.47** | `DISABLE=1`，生产基线 |
| FI FORCE | ~6.13 | armed=True，fallback=0；仍慢（pack 税） |

复现：

```bash
source ~/venvs/sglang-lite/bin/activate
cd ~/project/sglang-lite
export PYTHONPATH=$PWD/engine:$PWD
CUDA_VISIBLE_DEVICES=0 python scripts/phase1_kernel_probe.py \
  --out ~/bench/phase1_kernel_probe_default.json
# 带 0.6.16 隔离前缀：
SGLANG_LITE_FI_PREFIX=/tmp/fi1616 FLASHINFER_DISABLE_VERSION_CHECK=1 \
  CUDA_VISIBLE_DEVICES=0 python scripts/phase1_kernel_probe.py \
  --out ~/bench/phase1_kernel_probe_fi1616.json
```

### 8.4 Phase 2 — 生产硬化

Prometheus（t/s、TTFT、cache_hit、batch、queue、TP 健康）、结构化日志 + request id、
优雅退出 / drain / 超时 / OOM 拒绝、Mixtral/Qwen-MoE 回归、`lite` 预设。
UniGateway 仅协议/metrics 联调，不引入业务逻辑进 engine。

#### 8.4.1 切片 1（已落地）— Prometheus dual / 0c 指标

| 项 | 状态 | 说明 |
| --- | --- | --- |
| `engine/metrics_prom.py` | **done** | `render_prometheus(stats)` 纯函数 |
| process `GET /metrics` | **done** | 使用 render；含 dual_write/hit/append/restore/**stage** |
| 既有 queue / cache / OOM | **done** | waiting/running/steps/hit/miss/blocks/oom |
| v4_hybrid + TP world | **done** | gauges |
| 单测 | **done** | `tests/test_metrics_prometheus.py` |

控制面 `control` 仍可代理 `engine-url/metrics`。

#### 8.4.2 切片 2（已落地）— TTFT / tok/s + graceful drain

| 项 | 状态 | 说明 |
| --- | --- | --- |
| TTFT | **done** | 首 completion token 时记 `first_token_ts`；sum/count/avg/last + 粗桶 |
| tok/s | **done** | 完成请求的 decode tokens / (finish−first_token)；avg/last |
| Prometheus | **done** | `ttft_seconds_*`、`tok_s_*`、`requests_completed_total`、`completion_tokens_total` |
| Drain | **done** | `EngineLoop.begin_drain` / `drain_status`；process `POST/GET /v1/drain` |
| Ready | **done** | drain 中 `GET /readyz` → 503 `draining`；`submit` 拒绝新请求 |
| 单测 | **done** | `tests/test_latency_drain.py` |

#### 8.4.3 切片 3（已落地）— request-id 结构化日志

| 项 | 状态 | 说明 |
| --- | --- | --- |
| `engine/reqlog.py` | **done** | JSON 单行：`ts/event/request_id/...` → logger `sglang_lite.req` |
| loop 事件 | **done** | submit / first_token / finish / cancel / drain |
| process HTTP | **done** | generate / reject / disconnect / cancel / drain |
| Rust control | **done** | `chat_completions_accept` 带 `request_id` |
| 开关 | **done** | `SGLANG_LITE_LOG_JSON=0` 关闭；默认开 |
| 单测 | **done** | `tests/test_reqlog.py` |

示例：

```json
{"ts":...,"event":"request_finish","request_id":"...","finish_reason":"length","completion_tokens":8,"ttft_s":0.12,"tok_s":7.5}
```

#### 8.4.4 切片 4（已落地）— soak / 多 MoE 回归 / lite 预设 / Runbook

| 项 | 状态 | 说明 |
| --- | --- | --- |
| Soak | **done** | `scripts/soak_stability.py`：多轮并发 + blocks/oom/error 门禁 |
| MoE 回归 | **done** | `scripts/moe_regression.py`：默认 tiny Mixtral fixture；可加多 `--model` |
| Fixture 构建 | **done** | `scripts/build_tiny_moe_fixture.py` |
| lite env | **done** | `scripts/env_lite.sh` + `Config.from_env("lite")` 默认 DISABLE_FI |
| Runbook | **done** | [runbook.md](./runbook.md) 起停 / metrics / drain / 门禁 / 故障 |

CPU 默认 soak/回归应 `overall: PASS`。

#### 8.4.5 PRO6000 真机 soak + 时长策略（2026-08-06）

| 跑次 | 配置 | 结果 |
| --- | --- | --- |
| V4 短 | TP=8，25 round，conc=2，max_new=8 | **PASS**：ok=50 err=0 blocks≡2 stage→350 ~46s |
| V4 **30 min 墙钟** | `--profile long --duration-s 1800` conc=2 | **PASS**：ok=**3010** err=0 rounds=**1505** blocks≡**2** stage→**21070** elapsed=**1801s**；tok_s_avg≈6.87；ttft_avg≈0.021s |

摘要：`~/bench/soak_v4_30min.json`。结论：page-primary stage 下长稳 **无 page 泄漏**（blocks 全程 2）、零错误。

时长策略见 [runbook.md](./runbook.md) §5 表（smoke/short/medium/long）。

**多 MoE 真机回归矩阵（PRO6000，≤300B，优先小模型）**

选型原则：总参 ≤300B、文本 MoE、可 `AutoModelForCausalLM` + `trust_remote_code`。
**MiniMax-M3 ~428B + 多模态** → 超出 300B/MVP 模态门禁，**不拉权重**；同系列用 **M2（~230B/10B act）** 代替。

| 模型 | 规模 | 权重源 | 真机结果 | 备注 |
| --- | --- | --- | --- | --- |
| Qwen1.5-MoE-A2.7B-Chat | ~14B / 2.7B act | ModelScope `qwen/...` | **PASS**（08-06 + 08-07 复测 + MLA 改后）：finish=length，load+gen≈9.7–12.5s | 标准 MHA + FI paged |
| DeepSeek-V2-Lite-Chat | 16B / 2.4B act | ModelScope `deepseek-ai/...` | **PASS**（2026-08-07）：finish=length，16 tok，load+gen≈9.5s | MLA：`use_paged_as_source=False`；`scripts/patch_deepseek_v2_tf5.py` |
| MiniMax-M2 | ~230B / 10B act FP8 | ModelScope `MiniMax/MiniMax-M2`（≈220G） | **PASS**（2026-08-07）：finish=length，8 tok，load+gen≈64s | 跳过 FI paged（GQA group=6）；`dequantize=true`；TF5 remote patches |
| MiniMax-M3 | ~428B + VL | — | **SKIP** | 超 300B + 多模态 out of MVP |
| DeepSeek-V4-Flash | Hybrid TP=8 | 已有 | 见 §8.4 soak / dual | 不在本表 `moe_regression` 路径 |

摘要：`~/bench/moe_reg_multi_summary.json`。

**Phase-A 吞吐探针（PRO6000，2026-08-07）— Qwen3-30B-A3B（文本 MoE）**

> 用户曾指定 [Qwen3.5-27B](https://www.modelscope.ai/models/Qwen/Qwen3.5-27B)：实测
> `model_type=qwen3_5` + vision，**Dense + 多模态 → scope 拒绝**。改测同量级文本 MoE
> `Qwen/Qwen3-30B-A3B-Instruct-2507`（`qwen3_moe`，128 experts / 8 active）。

| 项 | 结果 |
| --- | --- |
| 卡 | **1×** PRO 6000（`CUDA_VISIBLE_DEVICES=0`） |
| 门禁 | **PASS** `~/bench/moe_reg_qwen3_30b_a3b.json`（finish=length） |
| load_s | **~13.9–15 s** |
| 脚本 | `scripts/moe_thruput_probe.py` |

说明：tok/s 为 **加载后** wall-clock；作 Phase-A **sglang-lite 自基线**。

**同机对照 SGLang 0.5.16 Docker（2026-08-07，PRO6000 GPU0，同权重）**

| Engine | load_s | 1×64 warm | 1×128 warm | 备注 |
| --- | --- | --- | --- | --- |
| sglang-lite 首测 | 13.9 | 20.8 | 20.7 | `thru_qwen3_30b_a3b.json` |
| sglang-lite FI 静态 metadata | 15.2 | 22.8 | 22.7 | 每层免 `torch.tensor` 分配 |
| sglang-lite HF + inference_mode burst | ~14 | 21.6 | 21.9 | 修 burst 缺 `@torch.inference_mode` |
| sglang-lite HF + torch.compile(default) | ~14 | 38.7 | 38.8 | inductor；冷启动 ~150s |
| sglang-lite HF + experts=batched_mm eager | ~14 | **~46.6** | **~47.1** | bit-match `model.generate` |
| sglang-lite **batched_mm + torch.compile(default)** | 16.3→**111**† | **84.7** | **83.5** | `thru_qwen3_30b_warm_burst.json`；†load 含 compile warmup |
| HF `model.generate` 同权重 | — | ~45–46 | ~45–46 | eager 上限 |
| lab: StaticCache CUDA graph | — | — | ~77 | KV 一致但 attn 在 ~14 tok 漂移；默认关 |
| **SGLang 0.5.16 Docker** | 65.1 | **152.2** | **155.3** | 完整 CUDA graph capture |

比值（当前最优 warm）：**SGLang / lite ≈ 1.53×**（155/101，radix native+CG+fused）；  
FORCE_HF+compile 仍 ~84（≈1.87×）。相对首测 **+388%**（20.7→101）。  
已越过 HF.generate 与 0.5× SGLang 门槛（~77.5）。

**根因与开关（2026-08-07 → 08-08）**

1. **`run_decode_burst` 必须 `@torch.inference_mode()`**
2. **`SGLANG_LITE_FORCE_HF_CACHE=1`** + **`SGLANG_LITE_EXPERTS_IMPL=batched_mm`**  
   `grouped_mm` ~21；`batched_mm` ~47 且 bit-match generate。
3. **`torch.compile(mode=default)`**（FORCE_HF+batched_mm prefer；探针 `TORCH_COMPILE=1`）  
   暖路径 **~83–85 tok/s**。load 末 `_warmup_hf_compile_decode` 预热 decode 形，  
   把 inductor 冷启动并进 load_s（~110s），1×128 cold 可达 ~80。  
   greedy 与 eager 可在 ~11 tok 后分叉（fluent）；`TORCH_COMPILE=0` 关。
4. **紧凑 HF greedy burst**（`_decode_burst_hf_greedy`）：无 ones-mask、直连  
   `model()` + buffer；`DECODE_BURST` 默认 64（探针 128）。
5. **StaticCache + CUDA graph** 仍默认关：即使 `create_causal_mask` 4D 全长 mask，  
   仍在 ~14 tok 漂移；`reduce-overhead` 与 DynamicCache 仍 CUDAGraph 冲突。
6. **`logits_to_keep=1`**。
7. **FlashInfer cutlass fused MoE**（`engine/moe_hooks.py`，`SGLANG_LITE_FUSED_MOE=1`）  
   - 接线：Qwen3 `gate_up` **up||gate** pack 后 `cutlass_fused_moe`（层内 maxdiff ~0.016）。  
   - **B12x** 仅 FP4 权重，不适用于当前 bf16 检查点。  
   - **e2e PRO6000**：fused-only ~**44** tok/s < batched_mm ~47 < **compile ~84**。  
     fused+compile 更差（~30，dynamo 反复 graph-break）。默认 **关**。

脚本：`scripts/moe_thruput_probe.py`、`scripts/sglang_thru_docker.sh`。

8. **Radix-native paged path（2026-08-08）**  
   - **根因修复**：FI paged hook 漏了 Qwen3 **`q_norm`/`k_norm`** → 从第 1 个
     token 起崩溃式胡话；补上后 PAGE-eager 文本流畅（~44–52 tok/s）。  
   - 与 HF generate 仍可在 ~14 tok 处分叉（FI vs SDPA 数值，类似 StaticCache 现象）。  
   - **固定 max-pages + FI `use_cuda_graph`**：单次捕获贯穿 1×128。  
   - **Native layer loop**（`engine/native_decode.py`）：跳过 HF CausalLM.forward
     / `create_causal_mask` / DynamicCache；仍用 HF 权重与 FI attn hooks。  
   - **PRO6000 Qwen3-30B 1×128 warm**（同日）：  
     | 栈 | warm tok/s |  
     |----|------------|  
     | paged eager | ~45 |  
     | paged + CG + batched_mm | ~79 |  
     | **paged + native + CG + cutlass fused MoE** | **~101** |  
     | FORCE_HF + compile（探针默认 thruput） | ~84 |  
     | SGLang 0.5.16 | ~155 |  
   - 探针 `SGLANG_LITE_RADIX_NATIVE=1` 默认：`FORCE_HF=0`、`CUDA_GRAPH=1`、
     `FUSED_MOE=1`、`NATIVE_DECODE=1`。产品 thruput 默认仍可 FORCE_HF+compile。  
   - `SGLANG_LITE_PAGED_MAX_PAGES=256`（page_size=16 → 4k tok）。

**性能天花板（2026-08-08 续）**

| 栈 | warm tok/s | 备注 |
|----|------------|------|
| radix native+CG+**cutlass** MoE+QKV | **~103** | plan ~0.11ms / body ~9.5ms |
| body-only 理论 | ~106 | plan 占比 <2% |
| MoE leaf only (cutlass micro) | 0.057ms×48≈2.7ms | MoE 不是唯一大头 |
| SGLang 0.5.16 | **~155** | body 约需 ~6.5ms/tok |
| FORCE_HF+compile | ~84 | 默认 thruput |

**换 MoE 核实测（PRO6000，`scripts/moe_kernel_probe.py`）**

| Backend | 状态 | e2e warm |
|---------|------|----------|
| **cutlass** (FI) | OK | **~103** |
| **trtllm** bf16 | soft API 可见；**GEMM fail**（sm100f cubin on SM120） | fallback cutlass ~103 |
| **sgl-kernel** 0.3.21 | **import fail**（torch 2.11 ABI / 无 SM120 .so） | 无 fused → ~79 |

`SGLANG_LITE_MOE_BACKEND=auto|cutlass|trtllm|sgl`（auto=cutlass first）。

结论：宿主税已吃干；**本机可工作的 BF16 MoE 叶仍是 cutlass**。TRT-LLM /
sgl-kernel 在 PRO6000+torch2.11 上**尚未可用**，不是再剥 Python 能过 155。
要超过 SGLang 需要：SM120 可用的 sgl-kernel 构建、或 FP8/量化路径、或更短
全层 fused capture。

Registry：`engine/models.py` → `MINIMAX_MOE`。  
Runner：`runner.py` 对 MLA / MiniMax / 非 2^n GQA 跳过标准 FI paged，走 HF `use_cache`。  
权重侧 TF5 补丁（仅 PRO6000 权重目录，不进 engine）：  
- `scripts/patch_deepseek_v2_tf5.py`  
- `scripts/patch_minimax_m2_tf5.py` + `scripts/patch_minimax_m2_rope_init.py`  
- MiniMax `config.json`：`quantization_config.dequantize=true`（绕过 TF FP8Experts ModuleList 替换失败）

```bash
# 下载（HF 直连失败时用 ModelScope）
python -c "from modelscope import snapshot_download; print(snapshot_download('qwen/Qwen1.5-MoE-A2.7B-Chat', cache_dir='~/models/ms_cache'))"
python -c "from modelscope import snapshot_download; print(snapshot_download('deepseek-ai/DeepSeek-V2-Lite-Chat', cache_dir='~/models/ms_cache'))"
python -c "from modelscope import snapshot_download; print(snapshot_download('MiniMax/MiniMax-M2', cache_dir='~/models/ms_cache'))"
# 多模型回归
python scripts/moe_regression.py \
  --model ~/models/Qwen1.5-MoE-A2.7B-Chat \
  --model ~/models/DeepSeek-V2-Lite-Chat \
  --model ~/models/MiniMax-M2 \
  --device cuda --max-new 8 --out ~/bench/moe_reg_multi.json
```

Qwen 复测摘要：`~/bench/moe_reg_qwen_20260807_003156.json`（overall=PASS）。

### 8.5 明确永远不做

与 [scope.md](scope.md) 一致：structured output、投机解码、PD disagg、多模态、
动态多 LoRA、vLLM 宽 API、完整 EP 调度。健全 ≠ 功能面变宽。
