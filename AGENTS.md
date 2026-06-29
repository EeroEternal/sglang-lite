# agents.md — sglang-lite 项目代理协作规范

本文件用于指导 AI 代理（Grok、Claude、Copilot 等）与人类在此仓库中协作时的约定。

**请用中文与本项目的主要贡献者沟通。**

## 核心理念（不可违背）

sglang-lite 是一个**极致高内聚的 Token Factory（令牌工厂）**。

- 最核心且必须深度耦合的三个组件：
  1. KVCacheManager（默认使用 RadixAttention 前缀树）
  2. Scheduler（continuous batching 连续批处理）
  3. ModelRunner（重度使用 CUDA graph 的 decode 执行）
- **Rust 层是对外的控制点**（OpenAI 协议适配层）。所有请求验证、早期拒绝、streaming 控制、错误处理都必须在这里完成。
- 一切业务逻辑（agent loop、structured output、多模态、tool calling 执行等）、serving、配置、详细可观测性**必须上移**到 unigateway / gateway 层或独立薄层。sglang-lite 是纯引擎库。
- 遇到不确定时，**优先缩小 scope**，宁可删减功能，也不要增加复杂度。

变更前必须阅读的文档：
- [docs/scope.md](docs/scope.md) —— 权威的 Feature 取舍表
- [docs/architecture.md](docs/architecture.md) —— 架构边界
- [README.md](README.md) —— 项目使命与目标 workload

## 严格禁止或需特别审查的变更

- 增加 API 层与 KV/Scheduler 内部的耦合
- 在 engine 里实现 structured output / JSON mode / grammar
- 添加多模态（vision）、speculative decoding、disagg prefill-decode 等非 MVP 特性
- 引入大量 feature flag 或实验性代码
- 对 Radix 树或 batching 核心逻辑进行大重构而没有清晰的性能/稳定性收益

## 语言与分层规则

| 层级         | 推荐语言          | 职责范围                                      | 说明 |
|--------------|-------------------|-----------------------------------------------|------|
| 控制面/API   | Rust (axum)       | OpenAI 协议、验证、streaming、metrics、early reject | 必须完全掌控 |
| 执行核心     | Python + Triton   | Radix KV Cache、Scheduler、ModelRunner + CUDA Graph | 保持简洁 |
| 通信边界     | HTTP（MVP）/ gRPC / PyO3 | 干净的 GenerationRequest / TokenDelta 协议     | 初期用松耦合 |

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

## 模型支持策略

**只支持主流 MoE 模型**（DeepSeek、Qwen-MoE、Mixtral 等）。Dense 模型不在支持范围内。
新增模型必须满足：
1. 通过 tokenization + 短生成测试
2. 更新 scope.md 中的支持列表
3. 只支持 MoE 模型，不支持 dense 模型。MoE 支持以路由 + 高效 batching 为主，不引入过度复杂的专家并行。

## 推荐工作流程

1. 先阅读 scope.md 确认该功能是否属于核心
2. 小步提交，聚焦单一需求
3. 更新相应文档
4. 确保 Rust + Python 的集成测试能跑通

## 当前阶段

当前处于 **Phase 0**（核心验证阶段）。

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
