# sglang-lite 运维 Runbook（稳部署）

当前主目标：**Qwen3.8-Flash-Next NVFP4、GPU-only**，见
[当前宪章](./qwen38-flash-next-only.md)。独立文本原型尚未完整验收，默认入口仍是 V4；
下方原有 V4 运维命令保留为历史/兼容路径，不可用于启动 Qwen3.8。
完整优化过程、原始 warm 时长与验收限制见
[qwen38-optimization-log.md](qwen38-optimization-log.md)。

## Qwen3.8 外部 SGLang 基线（已测量）

在独立 SGLang 环境中运行，不是 lite runner：

```bash
# venv 可执行文件（如 ninja）也必须在 PATH，不能只调用其 Python。
export PATH=/path/to/sglang-venv/bin:$HOME/.local/bin:$PATH
# 若继承环境带有旧 FlashInfer cubin，指定全新的当前版本缓存：
export FLASHINFER_CUBIN_DIR=/path/to/workspace/flashinfer-cubins-0.6.18
export FLASHINFER_WORKSPACE_BASE=/path/to/workspace
mkdir -p "$FLASHINFER_CUBIN_DIR"
/path/to/sglang-venv/bin/python scripts/qwen38_sglang_baseline.py \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 --tp 8 \
  --warm-runs 3 --lengths 64,128,256 --out /path/to/results/sglang.json
```

脚本显式禁用 PLE/权重 CPU offload、unified memory 和 speculative decoding，
采用纯文本 TP8/EP8、SM120 FlashInfer CUTLASS NVFP4 路径。
SGLang 的 Qwen4 override 默认启用 PLE CPU offload，因此必须显式设置
`ple_offload_embedding=False`，不能依赖默认参数。
这些是**请求配置**，并非运行后驻留证明；启动日志、实际 GPU 驻留和生成正确性仍须验收。
计时为单请求端到端生成时长（含 prefill/调度），不是纯 decode kernel 时长。
记录原始 warm 时长、中位数及输出样例；短输出不计入吞吐结果。
SGLang 0.5.20 的直接依赖已核对。该版本对 Qwen4Exp 不支持
`--language-model-only` 或 encoder disaggregation 的 `--language-only`；
因此基线只发送文本请求，但保留标准模型的 vision tower，不能强设这些参数。
lite 的目标仍只有文本执行。
继承环境含旧 FlashInfer cubin 时，使用空的隔离 `FLASHINFER_CUBIN_DIR`
和 `FLASHINFER_WORKSPACE_BASE`，让当前版本下载/编译所需算子，不加载旧 cubin，
也不设置 `FLASHINFER_DISABLE_VERSION_CHECK`。首个基线已跑通，重复性诊断见下节。

### 2026-10-03：首个真实基线与诊断

标准 SGLang 0.5.20 / Torch 2.13.0+cu130 / FlashInfer 0.6.18 /
sglang-kernel 0.4.7，8×5090，TP8/EP8，RadixArk NVFP4。三次 warm 中位数：

| 单请求输出 | 端到端时长 | 输出 tok/s |
|---|---:|---:|
| 64 | 0.7441 s | 86.01 |
| 128 | 1.2392 s | 103.29 |
| 256 | 2.2368 s | 114.45 |

这些是 **SGLang 基线**，不是 lite 成绩。输入为脚本默认的天空颜色问题，
`temperature=0`、`ignore_eos=True`；包含 prefill 和调度开销。
八卡逐 rank 的首次请求审计各遍历 2,024 个参数、buffer 和持久池张量，
没有发现非 CUDA 权重或浮点状态。此审计不枚举临时算子张量，
也不证明 NCCL 传输不使用 host staging。

`scripts/qwen38_sglang_diagnose.py` 是独立外部诊断：测试清缓存与缓存复用请求，
记录 token IDs、top-2 logprobs，支持 `--profile` 和 `--deterministic`。
标准路径的五次 64-token 请求在第 44 个 token 起分歧，
清缓存与未清缓存均如此，报告的 cached_tokens 都为 0，尚不能作为固定 golden。
确定性模式的默认 DeepGEMM 路由在 `M=8,N=2060,K=2560` 处因 TMA stride
对齐失败；诊断可用官方环境变量
`SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0` 选择 GPU Triton，
不得用 batch-variant fallback 或 CPU 路径掩盖问题。
该 Triton 确定性诊断已通过：五次 64-token 请求的 token IDs 完全一致，
八个 rank 的驻留审计仍无非 CUDA 权重或浮点状态。
这只验证一个短 prompt 的重复性，不等于整个模型图与 lite 的正确性已经验收，
也不能把确定性模式的速度当作上表标准 warm 速度。

```bash
SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0 \
  /path/to/sglang-venv/bin/python scripts/qwen38_sglang_diagnose.py \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 --tp 8 --deterministic \
  --out /path/to/results/deterministic
```

普通模式诊断和 profiling 去掉 `--deterministic` 并加 `--profile`。
诊断中的 worker hook 只枚举张量元数据，不替换算子、权重、状态或前向结果；
诊断与基准脚本都不属于 lite engine。

32-token profiling 的 TP0 累计 kernel 时间约 370.00 ms，其中
NCCL AllReduce 165.57 ms（44.75%），HC mix 41.98 ms（11.35%）。
这些是带 profiler 的 **累计 kernel duration**，不等于端到端墙钟占比，
也不能用来直接算 speedup。服务器的 CUDA peer-access 查询显示所有跨卡
配对均不支持 P2P；依赖 CUDA peer/IPC 直接读的优化不能未经验证移植。
attention DP2 保持全局 TP8/EP8，将 attention TP 降为 4。相同三次 warm
中位数为 67.37 / 77.48 / 84.42 tok/s（64 / 128 / 256）；比 DP1 慢，暂不采用。

## Qwen3.8 独立文本原型（仅验证，非服务入口）

`engine/qwen38_runner/` 不导入 SGLang/vLLM；使用 Torch、Triton 和 FlashInfer
叶子核，自持单序列 QSA KV、GDN 状态与 PLE 历史。已在真实权重上跑通 48 层
加载、graph 捕获/释放，并通过八 rank 的持久张量 GPU 审计。
rank0 的共享专家独占实现与标准引擎的共享专家 TP 分片不同，需披露。

```bash
# 复用前文隔离 Torch/FlashInfer 环境和 PATH/cache 设置。
/path/to/venv/bin/torchrun --standalone --nproc-per-node=8 \
  scripts/qwen38_lite_probe.py --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --tokens 64 --graph --golden /path/to/results/deterministic/diagnostic.json \
  --out /path/to/results/lite-probe.json
# 强制同一参考前缀，避免分歧后比较不同上下文的 logprob：
# 同一命令加 --teacher-force。
```

首次自由生成前 43 token 与确定性参考一致，第 44 个分歧；teacher forcing
的 top1 匹配 63/64，最大所选 token logprob 差为 0.279。
这仍是未完成的数值验收，不证明更长上下文、状态恢复/复用或可部署性。
探针计时包含逐 token 的 host 同步、排除 prefill，且实际只有 N-1 个 decode
步骤；**不能换算为上表的端到端 warm tok/s**。尚无验收后的 lite 吞吐成绩。

graph 必须在 `destroy_process_group()` 前 `reset()`，释放 NCCL graph 引用；
否则进程可能在退出阶段保留显存。一次未正确释放的旧探针导致后续启动 OOM，
已停止该自建进程并验证所有 GPU 释放，未更改显存预算或启用 offload。

## 1. 环境预设

```bash
cd /path/to/sglang-lite
source scripts/env_lite.sh

# DeepSeek-V4-Flash（历史/兼容路径）
export SGLANG_LITE_DSV4_HF=~/models/DeepSeek-V4-Flash-0731
export SGLANG_LITE_DSV4_CONVERTED=~/models/ds-v4-mp8
export SGLANG_LITE_MODEL="$SGLANG_LITE_DSV4_HF"
# 图默认走仓内 vendor（env_lite 已 setdefault）
# export SGLANG_LITE_DSV4_INFER=$PWD/engine/vendor/deepseek_infer
```

关键变量：

| 变量 | 默认 | 含义 |
|------|------|------|
| `SGLANG_LITE_V4_ONLY` | `1` | 拒绝非 V4-Flash 模型 |
| `SGLANG_LITE_DSV4_INFER` | `engine/vendor/deepseek_infer` | 官方/vendor 图路径 |
| `SGLANG_LITE_V4_DISABLE_FI_SPARSE` | `1` | 官方 `sparse_attn` 主路径 |
| `SGLANG_LITE_V4_DEEP_GEMM` | `1` | SM120 `vendor/deep_gemm_sm120` 替换 `fp4_gemm`；`0` 回退 TileLang |
| `SGLANG_LITE_V4_MOE_FAST` | `1` | 仅激活专家 + fused act_quant |
| `SGLANG_LITE_V4_B12X` | `0` | 实验：B12x（布局/EP 不兼容 Hybrid，默认不 attach） |
| `SGLANG_LITE_V4_CUDA_GRAPH` | 关 | 固定 start_pos 微基准 CUDA graph |
| `SGLANG_LITE_DECODE_BURST` | `64` | 单请求 decode 连打步数（thruput 可 `128`） |
| `SGLANG_LITE_V4_DUAL_APPEND` | `1` | decode 写 dual-pool；thruput 可 `0` |
| `SGLANG_LITE_LOG_JSON` | `1` | `sglang_lite.req` JSON 行日志 |
| `SGLANG_LITE_MAX_BATCH_SIZE` | `4` | continuous batch 上限 |
| `SGLANG_LITE_REQUEST_TIMEOUT` | `300` | 单请求超时秒 |

## 2. 启动

**DeepSeek-V4-Flash TP=8（PRO6000 宿主）— 原有路径，非 Qwen3.8 入口**

```bash
source scripts/env_lite.sh
export PYTHONPATH=$PWD/engine:$PWD
torchrun --nproc-per-node=8 -m sglang_lite.process \
  --model "$SGLANG_LITE_DSV4_HF" --device cuda --port 9001
```

**权重转换**（首次，官方 convert 已 vendor）：

```bash
export EXPERTS=256 MP=8
python engine/vendor/deepseek_infer/convert.py \
  --hf-ckpt-path "$SGLANG_LITE_DSV4_HF" \
  --save-path "$SGLANG_LITE_DSV4_CONVERTED" \
  --n-experts $EXPERTS --model-parallel $MP
```

**Legacy 多 MoE**（非 KPI，需显式）：`SGLANG_LITE_V4_ONLY=0`

可选控制面（Rust）：

```bash
# 另开终端
sglang-lite-serving --engine-url http://127.0.0.1:9001 --port 8000
```

## 2.1 KPI：对打 SGLang / vLLM

同机同权重 warm decode tok/s（见 v4-flash-only §5 / §10）：

```bash
bash scripts/v4_vs_sglang_bench.sh --mp 8 --max-new 128
# vLLM 0.25 baseline 客户端：scripts/vllm_v4_bench_client.py
```

记录 lite 与 SGLang / vLLM 的 warm tok/s；主 KPI 要求 **lite > SGLang**。

## 2.2 DeepGEMM FP4 微基准（PRO6000 SM120）

```bash
# 需已 convert 的 mp8 分片 + 空闲 8×GPU
export SGLANG_LITE_DSV4_HF=~/models/ds-v4-flash
export SGLANG_LITE_DSV4_CONVERTED=~/models/ds-v4-mp8
torchrun --nproc-per-node=8 scripts/v4_deep_gemm_thruput.py --max-new 96 --deep-gemm 1
torchrun --nproc-per-node=8 scripts/v4_deep_gemm_thruput.py --max-new 96 --deep-gemm 0
```

期望：attach 日志含 `v4 DeepGEMM armed`；单核 GEMM 相对 TileLang ~2.3×；纯 decode e2e 目前仅小幅提升（launch 税，见 v4-flash-only §9）。

跨位置 CUDA graph 实验使用 `scripts/v4_graph_thruput.py`，要求 DeepGEMM grouped
内核及转换后的 V4 权重。脚本在图内生成下一个 token 并更新位置，计时段不逐 token
同步 CPU；`CHECK match=true` 是使用结果前的数值门禁。它是独立吞吐探针，
**不是** standalone serving 路径或同机 SGLang KPI 的替代品。TileLang
对照请使用上面的 `v4_deep_gemm_thruput.py --deep-gemm 0`。

**Vendor 刷新**（`.so` 从 vLLM 镜像拷）：见 [vendor/SOURCES.md](./vendor/SOURCES.md) `deep_gemm_sm120`。

## 3. 健康与指标

```bash
curl -s localhost:9001/healthz
curl -s localhost:9001/readyz
curl -s localhost:9001/metrics | head -40
curl -s localhost:9001/stats | jq '.latency, .dual_pool, .cache'
```

关注：

- `sglang_lite_ready` / `sglang_lite_draining`
- `sglang_lite_kv_blocks_used`（soak 中不应无界爬升）
- `sglang_lite_oom_reject_count`
- `sglang_lite_dual_stage_count`（V4 page-primary）
- `sglang_lite_ttft_seconds_avg` / `sglang_lite_tok_s_avg`

结构化日志（request_id）：

```bash
# 进程 stdout / 日志里过滤
# {"event":"request_finish","request_id":"...","ttft_s":...,"tok_s":...}
```

## 4. 优雅排空

```bash
curl -s -X POST localhost:9001/v1/drain
# 轮询直到 idle
curl -s localhost:9001/v1/drain
# readyz 在 drain 中为 503
curl -s -o /dev/null -w '%{http_code}\n' localhost:9001/readyz
# 再停进程（Ctrl+C / kill）
```

## 5. 稳定性门禁（上线前）

**Soak 时长策略（`--profile`）**

| profile | 大致时长 | 默认 rounds | concurrency | max_new | 用途 |
|---------|----------|-------------|-------------|---------|------|
| `smoke` | 1–2 min | 10 | 4 | 4 | PR / 冒烟 |
| `short` | 5–10 min | 40 | 8 | 8 | 日常门禁 |
| `medium` | 20–30 min | 120 | 8 | 8 | 发版前 |
| `long` | **墙钟 1h**（`--duration-s 3600`） | 上限很大 | 4 | 8 | 稳部署 / 过夜前 |

也可用 `--duration-s N` 按秒截断（与 rounds 取先到者）。

```bash
# CPU fixture 短 soak
python scripts/soak_stability.py --profile short --out /tmp/soak.json
# overall PASS：errors=0、oom=0、blocks 稳定
```

**PRO6000 SSH**

```bash
ssh -p 2208 bodesi@39.183.171.3   # hostname=pro6000
# 代码：~/src/sglang-lite（git main）
# venv：source ~/venvs/sglang-lite/bin/activate
# 权重：~/models/DeepSeek-V4-Flash-0731  shards：~/models/ds-v4-mp8
```

**PRO6000 V4 Hybrid（真机）**

```bash
source scripts/env_lite.sh
export SGLANG_LITE_DSV4_HF=~/models/DeepSeek-V4-Flash-0731
export SGLANG_LITE_DSV4_CONVERTED=~/models/ds-v4-mp8
# 15–30 min 稳部署
torchrun --nproc-per-node=8 scripts/soak_stability.py \
  --model "$SGLANG_LITE_DSV4_HF" --device cuda \
  --profile long --duration-s 1800 --concurrency 2 --max-new 8 \
  --max-blocks-slack 256 \
  --out ~/bench/soak_v4_30min.json
```

判据：`errors=0`、`oom=0`、`blocks_used` 全程平坦（V4 实测恒为 2）、`dual_stage` 单调增。
**多 MoE 最小回归**

```bash
# CPU tiny Mixtral fixture
python scripts/moe_regression.py --out /tmp/moe_reg.json

# PRO6000 多 MoE 真机回归（≤300B；2026-08-07 已验 PASS×3）
# 矩阵：Qwen1.5-MoE-A2.7B + DeepSeek-V2-Lite + MiniMax-M2（~230B）
# MiniMax-M3 ~428B+多模态 → SKIP
source ~/venvs/sglang-lite/bin/activate
cd ~/project/sglang-lite
export PYTHONPATH=engine:.
export SGLANG_LITE_V4_DISABLE_FI_SPARSE=1

# 小：Qwen1.5-MoE（FI paged）
CUDA_VISIBLE_DEVICES=0 python scripts/moe_regression.py \
  --model ~/models/Qwen1.5-MoE-A2.7B-Chat --device cuda --max-new 16 \
  --out ~/bench/moe_reg_qwen.json

# Phase-A 吞吐探针（加载后 tok/s；非 vLLM/SGLang 对照）
# 注意：Qwen3.5-27B 是 Dense+多模态，**不在 scope**；用 Qwen3-30B-A3B（文本 MoE）。
# 探针默认：FORCE_HF_CACHE=1、EXPERTS_IMPL=batched_mm、TORCH_COMPILE=1。
# PRO6000 warm ≈ 83–85 tok/s（SGLang ≈155 → ~1.85×）。load 含 compile warmup ~110s。
CUDA_VISIBLE_DEVICES=0 python scripts/moe_thruput_probe.py \
  --model ~/models/Qwen3-30B-A3B-Instruct --device cuda \
  --cases 1x64,1x128 --out ~/bench/thru_qwen3_30b_a3b.json
# bit-exact eager（~47 tok/s）：SGLANG_LITE_TORCH_COMPILE=0 ...
# 实验 cutlass fused MoE（e2e ~44，默认关）：SGLANG_LITE_FUSED_MOE=1 SGLANG_LITE_TORCH_COMPILE=0 ...
# Radix-native（PRO6000 Qwen3-30B ~103 tok/s warm；距 SGLang~155 仍~1.5×）：
#   SGLANG_LITE_RADIX_NATIVE=1 python scripts/moe_thruput_probe.py ...
#   默认：CUDA_GRAPH=1 + FUSED_MOE=1 + NATIVE_DECODE=1 + FUSE_QKV=1
#   MoE 后端：SGLANG_LITE_MOE_BACKEND=cutlass|trtllm|sgl|auto（PRO6000 仅 cutlass 可用）
#   叶核探测：python scripts/moe_kernel_probe.py --e2e --model ~/models/Qwen3-30B-A3B-Instruct
#
# HF golden gate（见 docs/pega-lessons.md）：
#   # FORCE_HF exact（默认 require-exact；dump 单卡+batched_mm）
#   python scripts/hf_golden_gate.py gate \
#     --model ~/models/Qwen3-30B-A3B-Instruct --path force_hf --out-dir /tmp/golden
#   # radix-native 只报告 first-diff
#   python scripts/hf_golden_gate.py gate \
#     --model ~/models/Qwen3-30B-A3B-Instruct --path radix_native \
#     --out-dir /tmp/golden --no-require-exact

# 小：DeepSeek-V2-Lite（MLA → HF cache；首次需 TF5 patch）
python scripts/patch_deepseek_v2_tf5.py ~/models/DeepSeek-V2-Lite-Chat
CUDA_VISIBLE_DEVICES=0 python scripts/moe_regression.py \
  --model ~/models/DeepSeek-V2-Lite-Chat --device cuda --max-new 16 \
  --out ~/bench/moe_reg_ds_v2lite.json

# 中大：MiniMax-M2（GQA=6 跳过 FI paged；FP8 dequantize；多卡 device_map=auto）
python scripts/patch_minimax_m2_tf5.py ~/models/MiniMax-M2
python scripts/patch_minimax_m2_rope_init.py ~/models/MiniMax-M2
# config.quantization_config.dequantize=true（权重目录侧）
python scripts/moe_regression.py \
  --model ~/models/MiniMax-M2 --device cuda --max-new 8 \
  --out ~/bench/moe_reg_minimax_m2.json
```

下载权重（HF SSL 不稳时用 ModelScope）：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; print(snapshot_download('qwen/Qwen1.5-MoE-A2.7B-Chat', cache_dir='$HOME/models/ms_cache'))"
python -c "from modelscope import snapshot_download; print(snapshot_download('deepseek-ai/DeepSeek-V2-Lite-Chat', cache_dir='$HOME/models/ms_cache'))"
python -c "from modelscope import snapshot_download; print(snapshot_download('MiniMax/MiniMax-M2', cache_dir='$HOME/models/ms_cache'))"
# 软链到 ~/models/<Name>
```

**V4 dual + 吞吐（已有）**

```bash
torchrun --nproc-per-node=8 scripts/v4_dual_stats_probe.py --out ~/bench/v4_dual.json
torchrun --nproc-per-node=8 scripts/v4_lite_engine_gen.py --case 1x128
# 对照 docs/deepseek-v4-flash-plan.md §6.4.1 / §6.4.2
```

## 6. 常见故障

| 现象 | 处理 |
|------|------|
| soft gate 乱码 / top5 怪异 | 确认宿主 torch+cu130，勿用坏 Docker cu129 数值栈 |
| TileLang device mismatch | CVD=`LOCAL_RANK`，进程内 `cuda:0` |
| FI 更慢或全 0 | 保持 `DISABLE_FI=1`；FI 仅 FORCE 实验 |
| NCCL destroy 退出 SIGABRT | 已知；门禁看 JSON 结果，探针已避免结果后 barrier |
| blocks_used 只增不减 | soak FAIL；查 cancel/finish 是否 `v4_release_seq` / Radix release |
| `/readyz` 503 | 未 READY 或 **draining** |

## 7. 明确不做（部署范围）

structured output / tool 执行、投机解码、PD disagg、多模态、完整 EP、默认 FI 换核。  
业务与宽网关能力在 UniGateway，不进 `engine/`。
