package main

import (
	"context"
	"net/http"
	"os/signal"
	"syscall"
	"time"

	"RAG-Flow/internal/chunker"
	"RAG-Flow/internal/cleaner"
	"RAG-Flow/internal/embedder"
	"RAG-Flow/internal/models"
	"RAG-Flow/internal/pipeline"
	"RAG-Flow/internal/writer"

	conf "unitechs.com/unios-dice/uni-base/core/config"
	"unitechs.com/unios-dice/uni-base/core/log"
	kafka "unitechs.com/unios-dice/uni-base/lib/mq/kafka"
	route "unitechs.com/unios-dice/uni-base/core/route"
	"unitechs.com/unios-dice/uni-base/core/startup"
)

// PipelineConfig 管道配置
type PipelineConfig struct {
	ChunkSize         int `toml:"ChunkSize"`
	ChunkOverlap      int `toml:"ChunkOverlap"`
	BatchSize         int `toml:"BatchSize"`
	FlushIntervalMs   int `toml:"FlushIntervalMs"`
	WorkerCount       int `toml:"WorkerCount"`
	ChannelBufferSize int `toml:"ChannelBufferSize"`
}

func main() {
	log.Info("RAG-Flow 启动中...")

	// 加载管道配置
	var pipelineCfg PipelineConfig
	if err := conf.UnmarshalKey("Pipeline", &pipelineCfg); err != nil {
		log.Fatalf("加载管道配置失败: %v", err)
	}
	setPipelineDefaults(&pipelineCfg)

	// 创建清洗器
	textCleaner := cleaner.New()

	// 创建分块器
	textChunker := chunker.New(pipelineCfg.ChunkSize, pipelineCfg.ChunkOverlap)

	// 创建向量化器
	emb := newEmbedder()

	// 创建 Qdrant 写入器
	qdrantWriter := newQdrantWriter(emb)

	// 创建管道
	pipe, err := pipeline.New(
		pipeline.WithCleaner(textCleaner),
		pipeline.WithChunker(textChunker),
		pipeline.WithEmbedder(emb),
		pipeline.WithWriter(qdrantWriter),
		pipeline.WithWorkerCount(pipelineCfg.WorkerCount),
		pipeline.WithBatchSize(pipelineCfg.BatchSize),
		pipeline.WithFlushInterval(time.Duration(pipelineCfg.FlushIntervalMs)*time.Millisecond),
		pipeline.WithChannelBufferSize(pipelineCfg.ChannelBufferSize),
	)
	if err != nil {
		log.Fatalf("创建管道失败: %v", err)
	}

	// 启动管道
	ctx := context.Background()
	pipe.Start(ctx)

	// 创建 Kafka 消费者
	consumer := newKafkaConsumer()

	// Kafka 消费 → 管道
	go func() {
		err := consumer.Consume(func(topic string, timestamp time.Time, value []byte) {
			msg := models.NewRawMessage(topic, 0, 0, nil, value, timestamp)
			if sendErr := pipe.Send(msg); sendErr != nil {
				log.Errorf("发送消息到管道失败: %v", sendErr)
			}
		})
		if err != nil {
			log.Errorf("Kafka 消费异常: %v", err)
		}
	}()

	log.Info("RAG-Flow 管道已启动，开始消费 Kafka 消息")

	// 注册 API
	apis := route.RestfulAPIs{
		{Path: "/status", Description: "管道状态", Handler: newStatusHandler()},
	}

	// 优雅停机：监听系统信号
	go func() {
		sigCh := make(chan struct{})
		_ = sigCh
		ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
		defer stop()
		<-ctx.Done()
		log.Info("收到停止信号，开始优雅停机...")
		pipe.Stop()
	}()

	// 通过 Bootstrap 启动 HTTP 服务、服务注册等
	startup.Bootstrap(&startup.BootConfig{
		ServiceName: "RAG-Flow",
		APIs:        apis,
	})
}

func setPipelineDefaults(cfg *PipelineConfig) {
	if cfg.ChunkSize <= 0 {
		cfg.ChunkSize = 512
	}
	if cfg.ChunkOverlap < 0 {
		cfg.ChunkOverlap = 0
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 64
	}
	if cfg.FlushIntervalMs <= 0 {
		cfg.FlushIntervalMs = 500
	}
	if cfg.WorkerCount <= 0 {
		cfg.WorkerCount = 4
	}
	if cfg.ChannelBufferSize <= 0 {
		cfg.ChannelBufferSize = 256
	}
}

func newEmbedder() embedder.Embedder {
	provider := conf.GetString("Embedding.Provider")
	switch provider {
	case "tei":
		return embedder.NewTEI(
			conf.GetString("Embedding.TeiUrl"),
		)
	default:
		return embedder.NewOpenAI(
			conf.GetString("Embedding.OpenAIApiKey"),
			conf.GetString("Embedding.OpenAIModel"),
		)
	}
}

func newQdrantWriter(emb embedder.Embedder) writer.VectorWriter {
	addr := conf.GetString("Qdrant.Url")
	host, port := writer.ParseQdrantHost(addr)
	collection := conf.GetString("Qdrant.Collection")
	apiKey := conf.GetString("Qdrant.ApiKey")
	vectorDim := emb.Dimensions()

	w, err := writer.NewQdrantWriter(host, port, collection, apiKey, vectorDim)
	if err != nil {
		log.Fatalf("创建 Qdrant 写入器失败: %v", err)
	}
	return w
}

func newKafkaConsumer() *kafka.Consumer {
	cfg := kafka.NewConsumerConfig(
		conf.GetString("Kafka.Brokers"),
		conf.GetString("Kafka.Topic"),
		conf.GetBool("Kafka.Oldest"),
	)
	cfg.Group = conf.GetString("Kafka.Group")
	cfg.UserName = conf.GetString("Kafka.UserName")
	cfg.Password = conf.GetString("Kafka.Password")
	cfg.Algorithm = conf.GetString("Kafka.Algorithm")

	c, err := kafka.NewConsumer(cfg)
	if err != nil {
		log.Fatalf("创建 Kafka 消费者失败: %v", err)
	}
	return c
}

func newStatusHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"running"}`))
	})
}
