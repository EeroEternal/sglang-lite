# agents.md — sglang-lite 项目代理协作规范

本文件用于指导 AI 代理（Grok、Claude、Copilot 等）与人类在此仓库中协作时的约定。

**请用中文与本项目的主要贡献者沟通。**

## 核心理念（不可违背）

sglang-lite 是 **Qwen3.8-Flash-Next 专用**、极致高内聚的 Token Factory（性能优先于通用性）。

- **唯一一等公民模型**：Qwen3.8-Flash-Next，当前目标权重为 RadixArk NVFP4。V4 和其它模型面为 legacy，不进 P0。
- 最核心且必须深度耦合的三个组件：
  1. KVCacheManager（QSA KV + GDN 状态 + PLE 历史，明确生命周期与 GPU 驻留）
  2. Scheduler（continuous batching；可按 Qwen3.8 批形态简化）
  3. **Qwen3.8 ModelRunner**（目标：CUDA graph decode；vendor 必要叶子实现，禁止 runtime `import sglang` / `import vllm`）
- 模型计算、专家、PLE 表及推理状态必须留在 GPU；禁止 CPU/SSD offload、unified-memory 回退。CPU tokenizer、加载和控制逻辑不在此限制内。
- **Rust 层是对外的控制点**（OpenAI 协议适配层）。所有请求验证、早期拒绝、streaming 控制、错误处理都必须在这里完成。
- 一切业务逻辑与高级网关能力 **必须上移** UniGateway。`engine/` 保持引擎库。
- KPI：**同机同权重相对 SGLang 的 warm tok/s**，不是「支持多少模型」。
- 遇到不确定时，**优先缩小 scope**；为 Qwen3.8 性能可牺牲抽象与通用性。

变更前必须阅读的文档：
- [docs/qwen38-flash-next-only.md](docs/qwen38-flash-next-only.md) —— **当前产品宪章（已采纳）**
- [docs/scope.md](docs/scope.md) —— Feature 取舍表
- [docs/deepseek-v4-flash-plan.md](docs/deepseek-v4-flash-plan.md) —— 历史 V4 技术细节（仅修改 V4 路径时必读）
- [docs/architecture.md](docs/architecture.md) —— 分层边界

## 严格禁止或需特别审查的变更

- 增加 API 层与 KV/Scheduler 内部的耦合
- 在 engine 里实现 structured output / JSON mode / grammar
- 添加多模态（vision）、speculative decoding、disagg prefill-decode 等非 MVP 特性
- 为追求 vLLM feature parity 把 Responses、多模态、LoRA、spec decode、disagg 等宽功能面引入 core
- 引入大量 feature flag 或实验性代码
- 对 Radix 树或 batching 核心逻辑进行大重构而没有清晰的性能/稳定性收益

## 语言与分层规则

| 层级         | 推荐语言          | 职责范围                                      | 说明 |
|--------------|-------------------|-----------------------------------------------|------|
| 控制面/API   | Rust (axum)       | OpenAI 协议、验证、streaming、metrics、early reject | 必须完全掌控 |
| 执行核心     | Python + Triton   | Radix KV Cache、Scheduler、ModelRunner + CUDA Graph | 保持简洁 |
| 通信边界     | HTTP（MVP）/ gRPC | 干净的 GenerationRequest / TokenDelta 协议     | 禁止 PyO3 / 同进程嵌入 |

## 编码规范

### Rust 部分
- 遵循项目中 unigateway / llm-connector 的风格：`cargo fmt`、`cargo clippy -- -D warnings`
- 公共 API 必须有文档注释
- 避免在热路径和错误路径使用 `.unwrap()`
- OpenAI 请求模型保持**极简**，scope 之外的字段要么明确拒绝，要么带 warning 忽略

### Python 部分
- 核心模块保持极小（每个主要概念一个文件）
- 显式优于隐式（这是 lite 的核心价值）
- 禁止在 engine 内部使用隐藏的全局状态
- 关键执行路径必须在 2 跳以内可追踪

### 通用
- 修改边界时同步更新 docs/scope.md 和 architecture.md
- 任何影响 OpenAI 契约或调度决策的变更必须补充测试
- 提交前运行格式化与检查

## 目录结构规范（按功能划分，保持根目录极简）

sglang-lite 强调高内聚与整洁，**根目录严禁堆放非核心内容**。

推荐采用**功能目录**组织（优于纯语言划分）：

**根目录仅允许：**
- 构建与元数据：pyproject.toml、Cargo.toml、README.md、AGENTS.md、.gitignore 等
- 文档：docs/
- 轻量示例：examples/

**核心功能目录：**
- `engine/` — **执行核心库**（功能目录）：RadixKVCache、Scheduler、ModelRunner、LiteEngine，以及极薄内部 process 入口（GenerationRequest/TokenDelta）。package 映射为 `import sglang_lite`。OpenAI 表面仍在 Rust `control/`。
- `control/` — **Rust 控制面库**（**纯 Rust** 实现）
  - sglang-lite 的薄控制面：OpenAI 协议最小解析/适配、请求验证、early reject、streaming 控制、内部 GenerationRequest / TokenDelta 协议。
  - 全部内容用 Rust 编写。
  - 极薄：只提供独立推理服务所需的最小控制点；高级网关能力由 Unigateway（外部 Rust）提供。
  - 严禁任何 Python 代码或重型 serving 逻辑。OpenAI 解析在此仅为薄适配。
- `serving/` — **Rust 可执行服务包装层**（**纯 Rust** 实现）
  - 组合 `control/` 库形成的可运行 standalone wrapper。
  - 这是官方独立服务入口，必须达到可部署的最小推理服务标准，但不得吸收路由、鉴权或业务逻辑。
- `scripts/` — 开发/基准/工具脚本
- `tests/` — 测试代码

**严禁放在根目录的内容必须归类：**
- 演示脚本 → `scripts/` 或 `examples/`
- 基准测试脚本 → `scripts/benchmark.py` 或 `serving/`
- 其他开发脚本 → `scripts/`
- 控制面库 → `control/`
- 完整服务 / 可执行 wrapper → `serving/`

**实践要求：**
- 优先使用功能目录（engine/、control/、serving/、scripts/）。
- 新增控制面相关代码必须放入 `control/`。
- 新增 serving 包装或可执行入口代码必须放入 `serving/`。
- 提交前检查：`find . -maxdepth 1 -name "*.py" | head -5`，确保根目录干净。
- 移动后同步更新引用和 AGENTS.md。
- `serving/` 内的代码必须保持薄：只负责组合，不实现核心逻辑。OpenAI 协议解析等最小适配可保留作为 sglang-lite 控制点，但完整实现应在 Unigateway。
- examples/ 用于演示；serving/ 用于生产级完整服务包装。
- 核心库使用功能名 `engine/`（执行核心，sglang_lite 包文件直接放在这里），避免用“python/”这类语言名目录。

## 模块边界定义（严格遵守）

| 模块               | 位置                  | 职责范围                                                                 | 严格禁止                                                                 |
|--------------------|-----------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
| **核心引擎库**    | `engine/`（功能目录，直接放 sglang_lite 包文件） | KVCache (Radix)、Scheduler、Batching、ModelRunner、核心执行。提供 LiteEngine / building blocks；可含极薄内部 process（`python -m sglang_lite.process`：GenerationRequest/TokenDelta NDJSON，非 OpenAI 表面）。 | 完整 OpenAI 服务器、跨后端 routing、auth、业务逻辑、宽协议面。 |
| **Rust 控制面库** | `control/` | 最小 OpenAI 协议适配、请求验证、真实 SSE streaming、内部 GenerationRequest / TokenDelta 协议、engine client、取消/超时传播。 | 核心模型执行、跨后端 routing、auth、业务逻辑、宽 OpenAI 表面。 |
| **Rust Serving 包装层** | `serving/` | 官方 standalone 可执行入口；组合 `control/`，负责启动、配置、健康与优雅退出。 | 核心引擎逻辑、多后端网关、业务逻辑。 |
| **示例**          | `examples/`          | 薄演示如何使用核心库或简单服务器。                                       | 作为生产服务代码。                                                       |
| **脚本**          | `scripts/`           | 开发工具、基准、demo 脚本。                                              | 生产服务入口。                                                           |
| **文档**          | `docs/`              | 架构、scope、边界说明。                                                  | 代码实现。                                                               |

**关键边界原则**：
- `engine/` 永远是**纯引擎库**（Token Factory）。
- sglang-lite 产品必须能独立提供最小、真实、可流式的推理服务，不依赖 SGLang、vLLM 或 UniGateway。
- `control/` 是薄 OpenAI 协议适配 + 内部协议 + 请求生命周期控制；仅实现单引擎服务闭环。
- `serving/` 是官方 standalone 可执行入口（组合 `control/`），不是 demo/legacy。
- UniGateway 是可选的高级网关，只负责跨后端 routing、auth、全局限流/策略和聚合能力。
- engine 主循环、continuous batching、KV 生命周期和 model execution 必须在 sglang-lite engine process 内部；UniGateway 只能通过 HTTP/gRPC 调用，不能直接组合内部类。
- 严禁在 sglang-lite 实现完整 OpenAI 表面或业务逻辑。
- Unigateway 核心使用 `PrefixCache`、`BlockKVCache`、`BackendCapabilities` 等通用抽象；Radix/MoE 特有逻辑只留在 sglang-lite backend 内。
- 任何跨越边界的变更必须先更新本文件 + docs/scope.md + docs/architecture.md。
- 新功能先问：它属于核心引擎、控制面、还是 serving 包装层？

## 模型支持策略

**只将 Qwen3.8-Flash-Next 作为主目标**（见 [docs/qwen38-flash-next-only.md](docs/qwen38-flash-next-only.md)）。

- 禁止扩展其它 Qwen 家族 / Mixtral / MiniMax 等通用热路径。
- 允许 **vendor** 上游（vLLM / SGLang / 官方 inference / Strata）中与目标图相关的叶子代码与核，保留许可证头并钉来源 commit。
- Strata 仅作为 GPU 算子参考；不引入其 CPU 专家、SSD 分层或推测执行架构。
- 禁止为「多模型 registry」增加抽象。

## 推荐工作流程

1. 先阅读 **qwen38-flash-next-only.md** + scope.md
2. 变更是否服务 **Qwen3.8 GPU-only 性能 KPI**？否则默认不做
3. 建立正确基线 → 搬必要叶子实现到 `vendor/` + SOURCES 钉 commit → 独立 runner
4. 小步验证；8×5090 上同机同权重对照 SGLang，先正确性后吞吐

## 当前阶段

**产品赌注经用户确认切换为 Qwen3.8-Flash-Next NVFP4、GPU-only**（2026-10-03）。

- 新宪章：`docs/qwen38-flash-next-only.md`。旧 V4 宪章与测量记录保留为历史资料。
- 代码现状仍默认 `SGLANG_LITE_V4_ONLY=1`；Qwen3.8 独立文本原型位于 `engine/qwen38_runner/`，完整数值/吞吐/服务验收未完成，不得仅关闭 gate 就声称支持。
- 标准 SGLang 基线已测量，256-token warm 为 114.45 tok/s；`scripts/qwen38_sglang_baseline.py` 是外部对照，不是 lite 实现。
- 优化过程、原始时长、失败实验与验收限制见 `docs/qwen38-optimization-log.md`。
- 先验证 GDN / QSA / PLE / hyperconnection / NVFP4 的完整图与 GPU 驻留，再迁移默认入口。
- 保留现有 V4 路径和未提交工作，不在产品切换时批量删除。
- 运维与历史命令见 [docs/runbook.md](docs/runbook.md)。

欢迎贡献，但请严格遵守高内聚与 scope 纪律。

## GitHub 多账号管理与推送（重要）

本地机器上同时登录了多个 GitHub 账号（例如 lipish 和 EeroEternal）时，推送前**必须**确认并切换到仓库所有者账号：

```bash
# 查看当前状态
gh auth status

# 切换到 EeroEternal（sglang-lite 仓库所属账号）
gh auth switch --user EeroEternal

# 确认切换成功后再推送
git push origin main
```

如果账号未登录：

```bash
gh auth login --hostname github.com --git-protocol https
gh auth switch --user EeroEternal
```

**注意**：
- 推送前务必 `gh auth status` 确认 Active account 是 EeroEternal。
- 使用 `gh auth switch` 比手动改 git config 更安全。
- 推送使用 https 协议时，gh 会自动处理凭证。

---

本文件为 agents.md（小写），与 AGENTS.md 内容精神保持一致，但以中文优先呈现。
