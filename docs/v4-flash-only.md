# sglang-lite → DeepSeek-V4-Flash 专用引擎（性能最大化）

**决策日期**：2026-08-08  
**状态**：历史宪章；2026-10-03 主目标已由用户切换为
[Qwen3.8-Flash-Next GPU-only](./qwen38-flash-next-only.md)。下文保留 V4 决策与实测记录，
不再代表当前 P0；现有 V4 代码仍保留。
**目标**：在 **DeepSeek-V4-Flash（含 0731 权重形态）** 上，单机/TP 吞吐与延迟 **超过同配置 SGLang**；不为其它模型保留通用热路径。

---

## 1. 一句话

> **只做 V4-Flash 的 Token Factory。**  
> 调度/KV 生命周期仍自持；模型图与 decode 核 **从 vLLM / SGLang / 官方 inference 搬代码进仓（vendor）**，不 `import sglang` / 不 `import vllm`，不依赖巨型可编辑安装。

---

## 2. 明确放弃

| 放弃 | 说明 |
|------|------|
| 多 MoE 家族 | Mixtral / Qwen3-MoE / MiniMax 等 **非默认、非门禁**；代码可删可挪 `legacy/` |
| HF thruput 默认栈 | FORCE_HF + torch.compile + Qwen thruput 探针 **退出 P0** |
| 通用 dense / 端侧 Qwen-FP8 | 本次 **不接**；若未来要加，另开产品线 |
| 宽协议 / 网关能力 | 仍上移 UniGateway；本仓只保最小 OpenAI 面 |
| 以「能跑很多模型」为 KPI | KPI 只有 **V4-Flash 相对 SGLang 的 tok/s 与 TTFT** |

---

## 3. 架构（高聚合、仍分层）

```text
control/ + serving/     Rust 薄控制面（保留）
        │  GenerationRequest / TokenDelta
        ▼
engine/
  loop + scheduler      自持 continuous batching（可按 V4 批形态简化）
  kv / dual-pool        自持 Radix 双池（V4 SWA + compressed）
  v4_runner/            ★ 唯一热路径：权加载、forward、graph、采样
  vendor/
    deepseek_infer/     官方 inference/ 必要子集（live graph）
    deep_gemm_sm120/    vLLM third_party DeepGEMM SM120（FP8×FP4，live）
    sglang_v4/          从 SGLang 拷贝的 V4 参考层（reference）
    vllm_v4/            从 vLLM 拷贝的 V4 参考骨架
  v4_deep_gemm.py       替换 kernel.fp4_gemm → DeepGEMM
  v4_moe_fast.py        激活专家 + fused act_quant
  v4_b12x.py            B12x 探针（默认不 attach）
```

**原则**：

1. **焊死模型**：启动时只接受 V4-Flash 配置/权重目录；其它 `assert` 拒绝。  
2. **搬代码不引大包**：允许依赖 **torch、transformers（tokenizer/config 最小）、flashinfer（或 vendor 的 .so）、cuda runtime**；**禁止** runtime `import sglang` / `import vllm`。  
3. **License**：vendor 目录保留原 Apache/许可头 + `NOTICE_VENDOR.md` 列出文件来源与 commit。  
4. **性能优先于抽象**：为 graph 可捕获，可破坏「干净 OOP」；一个文件一个 decode 热路径可以接受。

---

## 4. 从哪里搬什么（优先级）

| 优先级 | 来源 | 搬什么 | 不搬什么 |
|--------|------|--------|----------|
| P0 | DeepSeek 官方 `inference/` | `model.py` 注意力/MoE/Indexer 可运行子集、convert 权重布局约定 | 完整训练脚本、无关 demo |
| P0 | SGLang | V4 / DeepSeek V 系 **decode runner、CUDA graph 捕获、MoE/MLA 调用点** | 整个 server、多模型 registry、投机解码 |
| P0 | vLLM | V4 相关 **attention/MoE 执行与权预处理**（若比 SGLang 更贴 SM120） | engine 全家桶、多模态、调度器 |
| P1 | FlashInfer / sgl-kernel | **仅** V4 需要的 op 调用；能 vendor 的 cubin/封装进 `kernels/` | 整包当唯一架构 |
| P2 | 本仓已有 | `v4_dual_pool`、`dsv4_kv_pack`、`v4_sparse_mla`、TP SSE | Qwen native_decode / moe_hooks 通用 thruput |

**搬入规则**：

- 一次只 vendor **一个上游 commit 钉扎**（写在 `vendor/SOURCES.md`）。  
- 改名空间：`sglang.srt...` → `sglang_lite.vendor.sglang_v4...`。  
- 删掉对「其它模型」的分支；V4-only `if` 删掉，直接直线代码。  
- 能删的 Python 包装就删，热路径目标：**单文件 decode step 可读、可 CUDA graph**。

---

## 5. 性能目标（可验收）

在 **PRO6000 / 与现网一致的 TP** 上，固定：

- 权重：同一 DeepSeek-V4-Flash（0731）目录  
- 用例：至少 `1×128`、`1×256` decode warm；可选 prefill 长上下文 TTFT  
- 对照：同机 **SGLang 官方推荐启动参数**（版本钉死进 runbook）

| 指标 | 门槛 |
|------|------|
| warm decode tok/s | **> SGLang 同配置**（主 KPI） |
| cold start → ready | 记录即可，不牺牲 decode 换冷启动 |
| greedy 短 case | 与官方/HF golden **可复现对齐**（或声明允许的 first_diff） |
| 多并发 | 第二期；先赢单流/固定 batch |

---

## 6. 实施阶段

### Phase V0 — 立宪（本 PR 文档）

- [x] 本文 + scope / AGENTS 对齐  
- [x] `docs/vendor/SOURCES.md` + `engine/vendor/` 骨架 + `scripts/vendor_v4_slice.py`  
- [x] 拒绝非 V4 模型的 load 路径（`SGLANG_LITE_V4_ONLY=1` 默认开；fixture/stub 例外）

### Phase V1 — 最小可跑专用 runner

- [x] `engine/v4_runner/` 入口（identity + load + forward + encode + accel）  
- [x] 首次 vendor `deepseek_infer/`（`hf@60d8d707…`：model/kernel/encoding/convert）  
- [x] Hybrid load 走 `load_v4_flash`；forward 走 `V4DecodeAccelerator`  
- [x] 保留现有 Rust SSE / TokenDelta  
- [x] **PRO6000 真机**（`42e0ea0`，vendor graph，`DISABLE_FI=1`）`1×128` / `4×96` / `1×256` 跑通  

### PRO6000 验收（2026-08-08，宿主 torch 2.11+cu130，vendor `deepseek_infer`）

| Case | warm tok/s（vendor 初验） | warm **burst** | 旧基线 §6.4.1 |
|------|---------------------------|----------------|---------------|
| 1×128 | 7.44 | **9.34** (`7f6a99d`) | 7.59 |
| 4×96 | 16.48 | **26.54** | 16.74 |
| 1×256 | 7.48 | **9.08** | 7.51 |

**官方 ignore_eos 天花板**（同机同权，`scripts/v4_official_timed_ceiling.py`）：  
1×128 warm **≈9.08 tok/s**。Lite Hybrid **9.34** ≈ 官方天花板（host 税已吃光）。

| 路径 | 1×128 warm tok/s |
|------|------------------|
| 官方 vendor generate ignore_eos | 9.08 |
| Lite Hybrid burst + no dual re-stage | **9.34** |

日志：`~/bench/v4_official_ceiling.log`、`~/bench/v4_thru_nosync.log`。

**热路径修复**：

1. 禁止 decode 每步 dual-pool **re-stage**（只 prefix restore 后一次）  
2. V4 decode burst + `ignore_eos` 全程 GPU token（无逐步 `.item()`）  
3. thruput：`SGLANG_LITE_DECODE_BURST=128`、`SGLANG_LITE_V4_DUAL_APPEND=0`  

**结论（PRO6000 换核实验 2026-08-08）**：

| Decode sparse 后端 | 1×128 warm | 说明 |
|--------------------|------------|------|
| **TileLang official** | **~9.3** | 默认；单 op ~0.07ms |
| Torch gather（自研） | ~8.1 | 数值 maxdiff≈0.002；**更慢** |
| FI SM120（0.6.16） | ~7.2 | 数值可通，**pack 税**拖慢 e2e |

`SGLANG_LITE_V4_SPARSE=official|torch|fi`。默认 **official**。  

**时间占比（PRO6000，32 decode step profile）**

| 组件 | wall 占比 | 说明 |
|------|-----------|------|
| **MoE 整体** | **~52%** | 真正大头 |
| Linear/GEMM 合计 | ~25% | 含 Attention + MoE 内 |
| Expert 前向 | ~19% | 含在 MoE 内 |
| `sparse_attn` | ~2.5% | 再抠 sparse 收益上限很低 |

**MoE 加速实验（`engine/v4_moe_fast.py`，默认开）**

| 手段 | 结果 |
|------|------|
| 只跑 top-k **激活**专家（不扫 32 个 local idle） | 调度税↓ |
| SwiGLU **gate/up 共用一次 act_quant** | GEMM 前 quant 减半 |
| **DeepGEMM SM120** `fp8_fp4_gemm_nt`（`engine/v4_deep_gemm.py` + `vendor/deep_gemm_sm120`） | 官方 MXFP4 e8m0 **1×32** 布局已通；替换 `kernel.fp4_gemm`；**vs TileLang 单核 ~2.3–2.5×**，数值对齐（max err 0） |
| 纯 decode e2e（TP8，96 tok） | TileLang **~9.16** tok/s → DeepGEMM **~9.40** tok/s（~+3%）；每 token **~72.5** 次 `fp4_gemm` launch，GEMM 本体只占墙钟一小部分 |
| FI **B12x** MoE | 探针在 `v4_b12x.py`；**默认不开**（要 NVFP4 e4m3/sf16 + 无 EP；Hybrid 是 MXFP4 + TP/EP） |
| pip `deep_gemm` 0.1.4 bf16 | **SM120 Unsupported**（不可用；用 vendor SM120 版） |

开关：
- `SGLANG_LITE_V4_MOE_FAST=0` 关闭激活专家调度
- `SGLANG_LITE_V4_DEEP_GEMM=0` 回退 TileLang `fp4_gemm`
- `SGLANG_LITE_V4_B12X=1` 仅尝试 B12x（当前仍因布局/EP 拒绝 attach）

下一步：`m_grouped_fp8_fp4_gemm_nt_contiguous` 合并专家 launch（72→个位数/层）；CUDA graph decode；B12x 需离线 NVFP4 重打包 + 非 EP。

---

## 10. 同机对标 vLLM（PRO6000，2026-08-08）

**配置**：8× RTX PRO 6000，`DeepSeek-V4-Flash-0731`，vLLM **0.25.0** 官方 recipe  
（`vllm/vllm-openai:v0.25.0` + FI 0.6.14、`--tensor-parallel-size 8 --enable-expert-parallel --kv-cache-dtype fp8`）。

| Case（ignore_eos） | **vLLM warm tok/s** | **sglang-lite Hybrid warm** | 差距 |
|--------------------|---------------------|-----------------------------|------|
| 1×128 | **~72.3** | **~9.1–9.3** | **~7.8×** |
| 1×256 | **~72.4** | **~9.3** | **~7.8×** |
| 4 并发 ×96（合计） | **~256 agg** | **~27**（lite batch 内） | 数量级差距 |

- 启动：`scripts/vllm_v4_pro6000_start.sh`  
- 客户端：`scripts/vllm_v4_bench_client.py`  
- 日志：`~/bench/vllm_v4_kpi.json`（host）  
- vLLM 日志可见：`DeepGemmFP4Experts` + `sparse_mla_sm120_decode_dsv4`（完整生产核栈）。  
- lite 仍绑官方 TileLang expert loop + TileLang sparse → **对标已建立，差距主要在 MoE/核栈，不是 Python 调度零头。**

### Phase V2 — 焊核 + 满图

- [x] SGLang V4 切片 **reference** pin（`sglang@e732c0a9…`）— 不 runtime import  
- [x] 固定 start_pos CUDA graph 脚手架（`SGLANG_LITE_V4_CUDA_GRAPH`）  
- [ ] 按 reference 移植 dsv4 **叶子** op + 满位 decode graph（跨 start_pos）  
- sparse MLA：SM120 正确后端（禁止错走 SM100 TRTLLM）  
- MoE：FP8/分组 GEMM 以 V4 权格式为准  
- dual-pool page 为源，去掉不必要的 CPU snapshot 热路径

### Phase V3 — 对打 SGLang 与削脂

- [x] runbook + `scripts/v4_vs_sglang_bench.sh` 对照命令  
- [ ] 权重机验收 warm tok/s > SGLang  
- [ ] 删除 `legacy` 通用 MoE thruput 路径（或移出默认构建）  
- [ ] 包体/依赖清单最小化

---

## 7. 与旧文档关系

| 文档 | 关系 |
|------|------|
| [scope.md](./scope.md) | 使命改为 V4-Flash 专用；多 MoE 降为 legacy |
| [deepseek-v4-flash-plan.md](./deepseek-v4-flash-plan.md) | 技术细节仍有效；**目标从「家族之一」升级为「唯一产品」** |
| [pega-lessons.md](./pega-lessons.md) | 仍参考「热路径/门禁」；不参考多模型 feature 矩阵 |
| Qwen thruput / FORCE_HF | **归档**，不再作为主 KPI |

---

## 8. 非目标（再强调）

- 不做成「小 vLLM」  
- 不接 PD disagg / 投机 / 多模态（除非 V4 权重强制且为性能所必需）  
- 不通过 `pip install sglang` 当引擎  

---

## 9. 立即执行的工程口令

```text
1. 默认 SGLANG_LITE_V4_ONLY=1
2. vendor/ 钉上游 commit，拷代码改 import
3. 热路径只有 v4_runner
4. KPI = vs SGLang V4-Flash 同机 thruput
```
