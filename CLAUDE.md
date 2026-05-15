# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# 构建（默认目标为当前平台）
make build

# 运行全部单元测试
make test

# 竞态检测
make test-race

# 集成测试（需要 Docker 环境）
make test-integration

# 代码检查
make lint

# 构建并运行
make run

# 运行单个包的测试
GOOS=windows GOARCH=amd64 go test -v -cover ./internal/cleaner/...

# 交叉编译 Linux 二进制
GOOS=linux GOARCH=amd64 go build -o build/rag-flow ./cmd/RAG-Flow/
```

**注意**：`GOOS` 默认为 `linux`，在 Windows 上运行测试需设置 `GOOS=windows GOARCH=amd64`。uni-base 私有模块需设置 `GONOSUMDB=unitechs.com/unios-dice/uni-base GONOSUMCHECK=unitechs.com/unios-dice/uni-base`。

## Architecture

流式 RAG 数据处理管道，数据流：

```
Kafka → Consumer → Cleaner → Chunker → Embedder → QdrantWriter → Qdrant
                       │          │          │          │
                       └──────────┴──────────┴──────────┘
                            有界通道（背压控制）+ 批量累积
```

`cmd/RAG-Flow/main.go` 组装所有组件：从 TOML 加载配置，创建 Cleaner/Chunker/Embedder/Writer，用函数式选项构建 Pipeline，通过 uni-base Kafka 消费者接入数据，最后调用 `startup.Bootstrap()` 启动 HTTP 服务和服务注册。

### Pipeline Concurrency Model

`internal/pipeline/pipeline.go` 是核心编排器：
- 各阶段通过 **有界 channel** 连接，实现背压传播
- Cleaner/Chunker/Embedder 各有 N 个 worker goroutine
- Embedder 和 Writer 使用**双触发批量累积**：批次满或定时器到期时 flush
- 优雅停机：context 取消 → 排空通道 → WaitGroup 等待
- 所有错误汇聚到共享 `errCh`，由独立 goroutine 统一记录

### Key Interfaces

四个可插拔接口（定义在使用方，非实现方）：

- `cleaner.Cleaner` — `Clean(ctx, RawMessage) (CleanedDocument, error)`
- `chunker.Chunker` — `Chunk(ctx, CleanedDocument) ([]TextChunk, error)`
- `embedder.Embedder` — `Embed(ctx, []string) ([][]float32, error)` + `Dimensions() int`
- `writer.VectorWriter` — `Upsert(ctx, []EmbeddedChunk) error` + `Close() error`

### Data Types

`internal/models/document.go` 定义四个不可变数据类型：`RawMessage` → `CleanedDocument` → `TextChunk` → `EmbeddedChunk`。全部通过 `New*()` 构造函数创建，防御性复制 slice 和 map。

## Framework: uni-base

`unitechs.com/unios-dice/uni-base` 提供基础设施，**不要自行实现以下功能**：

| 能力 | 包路径 | 用法 |
|------|--------|------|
| 配置 | `core/config` | `conf.GetString()`、`conf.UnmarshalKey()` |
| 日志 | `core/log` | `log.Info()`、`log.Errorf()`、`log.NewLog("module")` |
| Kafka | `lib/mq/kafka` | `kafka.NewConsumer()`、`consumer.Consume(callback)` |
| HTTP | `core/route` | `route.RestfulAPIs` + `startup.Bootstrap()` |
| 启动 | `core/startup` | `startup.Bootstrap(&BootConfig{ServiceName, APIs})` |

配置文件在 `cmd/RAG-Flow/conf/configuration.toml`，使用 TOML 格式。新增配置段后用 `conf.UnmarshalKey("Section", &struct)` 读取。

## Patterns

- **Functional Options**：所有组件使用 `type Option func(*T)` 模式配置
- **Embedding Provider 切换**：`Embedding.Provider` 配置项选择 `"openai"` 或 `"tei"`，两者实现同一接口
- **OpenAI 客户端**：直接 HTTP 调用（无 SDK），内置指数退避+抖动重试，`retryableError` 标记可重试错误
- **Qdrant 点 ID**：UUID v5（namespace + docID:chunkIndex），保证幂等写入
- **测试**：表驱动测试 + `httptest.Server` mock HTTP API，无第三方测试库
