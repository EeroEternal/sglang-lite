# UniGateway 支持 sglang-lite 作为后端的需求描述

## 背景

- `sglang-lite` 定位为**极简的纯库**（MoE-focused Token Factory）。
- 其核心仅包含三个高内聚组件：
  1. Radix KVCacheManager（前缀共享）
  2. Continuous Batching Scheduler（MoE-aware）
  3. ModelRunner（MoE routing + 执行）
- 所有 serving、路由、鉴权、metrics、可观测性、配置管理、准入控制等都**明确剥离**到上层。
- `unigateway` 是面向嵌入的通用 LLM 网关库，当前已支持通过 `ProviderDriver` trait 接入不同后端。

目标：让 `unigateway` 能够将 `sglang-lite` 作为一种**本地 MoE 执行后端**使用，同时保持 `unigateway` 本身的通用性和可扩展性。

## 总体目标

1. 在 UniGateway 中支持 `sglang-lite` 作为一种可插拔的后端（backend/driver）。
2. 让使用 `sglang-lite` 的用户能够通过 UniGateway 统一的 OpenAI 兼容接口访问本地 MoE 模型。
3. 充分发挥 `sglang-lite` 的优势（Radix 前缀缓存 + 连续批处理），同时不破坏 UniGateway 的通用架构。
4. 所有与 `sglang-lite` 相关的特殊逻辑必须隔离在 driver 实现内部。

## 非目标

- 不要在 UniGateway 核心层（engine、protocol、pool 等）引入 MoE 或 sglang-lite 特有的概念。
- 不要把 sglang-lite 的实现细节（Radix 缓存结构、专家路由逻辑等）泄露到 UniGateway 的公共抽象中。
- 不要让 `unigateway` 对 `sglang-lite` 产生硬依赖（理想情况下应支持可选 feature 或独立 crate）。
- 目前不要求实现完整的专家并行（Expert Parallelism），先支持基础的本地 MoE 运行能力。

## 功能需求

### 1. Driver 支持
- 在 UniGateway 中实现一个名为 `sglang-lite`（或 `local-moe`）的 `ProviderDriver`。
- 支持通过 `driver_id` 识别和配置该驱动。
- 该驱动应能接收 `ProxyChatRequest`，并调用 `sglang-lite` 引擎执行。

### 2. 模型与端点配置
- 支持配置本地 MoE 模型（通过 model 名称 + 路径/配置）。
- 支持在 `DriverEndpointContext.metadata` 或专用配置中传入 sglang-lite 特有参数（如 Python 环境、模型路径、batch size 等）。

### 3. 能力声明
- 正确声明 `sglang-lite` 支持的能力（streaming、prefix caching 优势等）。
- 能够通过 UniGateway 的 capability 系统暴露给上层。

### 4. 执行路径
- 优先支持 **direct Python library 调用**（通过 PyO3 在 Rust 进程中直接嵌入 Python 引擎，或同进程调用）。
- 也应支持通过 gRPC / 子进程 / HTTP 的远程调用方式（作为备选，解耦更强但延迟更高）。
- 充分利用 `sglang-lite` 的 Radix 前缀缓存能力，实现高效的 prefix sharing。

### 5. 流式与协议兼容
- 完整支持 chat completions 的 streaming 和 non-streaming。
- 响应格式需与 UniGateway 现有的 `ProxySession` 等抽象兼容。
- 支持基本的 tool calling（占位或透传，具体执行可由上层处理）。

### 6. 错误与可观测性
- 错误应被正确映射为 UniGateway 的 `GatewayError`。
- 提供足够的 hook 点，让 unigateway 能够收集请求级指标（TTFT、tokens/s、cache hit rate 等）。
- 支持 request id 的透传和关联。

## 对 sglang-lite 库的期望（反向需求）

为了让 unigateway 能够方便、安全地驱动 sglang-lite，我们期望 sglang-lite 库满足以下特性：

- 提供清晰、稳定的 library API（推荐直接暴露 `LiteEngine`）。
- 支持通过简单构造器初始化：`LiteEngine::new(model_name, device)`。
- 提供同步或异步的生成接口，能够返回 token stream。
- 支持基本的 MoE 模型加载和执行。
- 提供可观测性 hook（可选），方便上层采集指标。
- **不包含**任何 HTTP server、配置中心、认证等网关逻辑。

（详细的 sglang-lite 库边界见 sglang-lite 仓库的 `docs/scope.md` 和 `docs/architecture.md`）

## 实现建议（保持通用性）

1. **使用现有抽象**：推荐实现 `ProviderDriver` trait，而不是新增专有接口。
2. **隔离实现**：sglang-lite 相关代码应放在独立的模块或子 crate 中（例如 `drivers/sglang_lite.rs` 或 `unigateway-sglang-lite` crate）。
3. **通过 metadata / config 传递参数**：利用 `DriverEndpointContext.metadata` 或专用配置结构体传递模型路径等信息。
4. **Driver ID 命名**：建议使用 `"sglang-lite"` 或 `"local-moe"` 作为 `driver_id`。
5. **可选依赖**：如果可能，通过 feature flag 控制是否编译 sglang-lite 驱动，避免强制依赖 Python 环境。

## 优先级

- **P0**：支持基本的 chat completions（streaming + non-streaming），通过 direct Python import 调用 sglang-lite。
- **P1**：支持通过配置指定 MoE 模型、基本的 prefix caching 效果验证。
- **P2**：支持 gRPC / 进程隔离模式；更完善的 metrics 透传；tool calling 占位支持。

## 验收标准

- 能够通过 UniGateway 配置一个 sglang-lite 后端，并用 OpenAI 客户端正常调用 MoE 模型。
- 前缀共享（Radix）效果能够在 UniGateway 层面被体现（cache hit 相关指标）。
- UniGateway 其他后端（OpenAI、Anthropic 等）的行为不受影响。
- 代码组织清晰，sglang-lite 相关逻辑未污染 UniGateway 核心抽象。

---

**联系人**：sglang-lite 项目组

**相关仓库**：
- sglang-lite: https://github.com/EeroEternal/sglang-lite
- unigateway: （当前仓库）

请在实现时优先保证 UniGateway 的通用性和可嵌入性。

---

## 附录：关于“unigateway 是 SDK 而不是二进制”的说明

unigateway 本身定位为**库 workspace**（SDK），这和把 sglang-lite driver 放在它里面是完全兼容的：

- 从 unigateway 自己的 AGENTS.md 可以看到：
  > “UniGateway 是面向嵌入方的本地优先 LLM **库 workspace** … HTTP、CLI、用户与租户管理由宿主应用自行实现。”

- 它的核心价值就是提供通用的 `ProviderDriver` 抽象、协议翻译、路由、引擎封装等。
- 添加一个 `SglangLiteDriver`（实现 `ProviderDriver`），只是多了一种 backend 支持，和支持 OpenAI、Anthropic 是一个性质。
- 真正的“二进制可执行文件”（监听端口、处理 HTTP 请求的服务器）仍然由**宿主应用**（使用 unigateway 的网关产品）来提供。
- sglang-lite 相关的特殊处理（本地 MoE、Radix 前缀缓存、direct import 等）应该隔离在 driver 实现里，不会污染 unigateway 核心抽象的通用性。

因此，把 sglang-lite 驱动放在 unigateway 里，不仅合适，而且是它作为通用 SDK 应该做的事情。

---

## 附：为什么提到 PyO3？

在“执行路径”一节中提到了“通过 PyO3 或同进程嵌入”，原因是：

- unigateway 主体是 Rust。
- sglang-lite 的核心引擎是 Python 库（`sglang_lite.engine.LiteEngine`）。
- 如果想在 unigateway（Rust 进程）里**直接、低开销地**调用 sglang-lite 引擎（而不是通过 HTTP/gRPC 连另一个 Python 进程），就需要 PyO3 作为 Rust ↔ Python 的 FFI 桥梁。

**使用 PyO3 的优势**：
- 极低的调用延迟（对 continuous batching 和 streaming 很重要）。
- 可以直接在同一个进程内操作 Python 对象，减少序列化开销。
- 便于 unigateway 直接管理 engine 的生命周期和资源。

**不使用 PyO3 的替代方案**：
- Rust driver 通过 gRPC/HTTP 调用一个独立的 sglang-lite Python 进程。
- 这种方式解耦更好、部署更灵活，但有额外的网络/序列化开销。

PyO3 不是强制要求，而是“高性能直连”路径的实现方式之一。需求里同时列出了两种选项，unigateway 团队可以根据实际情况选择。