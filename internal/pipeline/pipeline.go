package pipeline

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/dongxxg/RAG-flow/internal/chunker"
	"github.com/dongxxg/RAG-flow/internal/cleaner"
	"github.com/dongxxg/RAG-flow/internal/embedder"
	"github.com/dongxxg/RAG-flow/internal/metrics"
	"github.com/dongxxg/RAG-flow/internal/models"
	"github.com/dongxxg/RAG-flow/internal/writer"
)

// Pipeline RAG 数据处理管道
type Pipeline struct {
	cleaner           cleaner.Cleaner
	chunker           chunker.Chunker
	embedder          embedder.Embedder
	vectorWriter      writer.VectorWriter
	cleanerWorkers    int
	chunkerWorkers    int
	embedderWorkers   int
	writerWorkers     int
	batchSize         int
	flushInterval     time.Duration
	channelBufferSize int

	rawCh      chan models.RawMessage
	cleanCh    chan models.CleanedDocument
	chunkCh    chan models.TextChunk
	embeddedCh chan models.EmbeddedChunk
	errCh      chan error

	// 分阶段 WaitGroup，确保有序关闭
	cleanerWg  sync.WaitGroup
	chunkerWg  sync.WaitGroup
	embedderWg sync.WaitGroup
	writerWg   sync.WaitGroup

	cancel context.CancelFunc
	logger *slog.Logger

	// 统计计数器
	droppedChunks atomic.Int64
}

// New 创建管道实例
func New(opts ...Option) (*Pipeline, error) {
	p := &Pipeline{
		cleanerWorkers:    4,
		chunkerWorkers:    4,
		embedderWorkers:   4,
		writerWorkers:     1,
		batchSize:         64,
		flushInterval:     500 * time.Millisecond,
		channelBufferSize: 256,
		errCh:             make(chan error, 100),
		logger:            slog.Default().With("component", "pipeline"),
	}

	for _, opt := range opts {
		opt(p)
	}

	if p.cleaner == nil || p.chunker == nil || p.embedder == nil || p.vectorWriter == nil {
		return nil, fmt.Errorf("管道组件不完整：cleaner、chunker、embedder、writer 均不能为空")
	}

	p.rawCh = make(chan models.RawMessage, p.channelBufferSize)
	p.cleanCh = make(chan models.CleanedDocument, p.channelBufferSize)
	p.chunkCh = make(chan models.TextChunk, p.channelBufferSize)
	p.embeddedCh = make(chan models.EmbeddedChunk, p.channelBufferSize)

	return p, nil
}

// Send 将原始消息发送到管道入口
func (p *Pipeline) Send(msg models.RawMessage) error {
	select {
	case p.rawCh <- msg:
		metrics.MessagesReceived.WithLabelValues(msg.Topic).Inc()
		return nil
	default:
		metrics.MessagesReceived.WithLabelValues(msg.Topic).Inc()
		return fmt.Errorf("管道输入通道已满，丢弃消息 (topic=%s)", msg.Topic)
	}
}

// DroppedChunks 返回被丢弃的 chunk 总数
func (p *Pipeline) DroppedChunks() int64 {
	return p.droppedChunks.Load()
}

// Start 启动管道各阶段
func (p *Pipeline) Start(ctx context.Context) {
	ctx, p.cancel = context.WithCancel(ctx)

	for i := 0; i < p.cleanerWorkers; i++ {
		p.cleanerWg.Add(1)
		go p.runCleaner(ctx, i)
	}

	for i := 0; i < p.chunkerWorkers; i++ {
		p.chunkerWg.Add(1)
		go p.runChunker(ctx, i)
	}

	for i := 0; i < p.embedderWorkers; i++ {
		p.embedderWg.Add(1)
		go p.runEmbedder(ctx, i)
	}

	for i := 0; i < p.writerWorkers; i++ {
		p.writerWg.Add(1)
		go p.runWriter(ctx, i)
	}

	go p.runErrorCollector(ctx)
	go p.runChannelMonitor(ctx)

	p.logger.Info("管道已启动",
		"cleaner_workers", p.cleanerWorkers,
		"chunker_workers", p.chunkerWorkers,
		"embedder_workers", p.embedderWorkers,
		"writer_workers", p.writerWorkers,
	)
}

// Stop 停止管道并等待所有阶段完成
func (p *Pipeline) Stop() {
	p.logger.Info("管道停止中...")
	if p.cancel != nil {
		p.cancel()
	}

	// 阶段 1: 关闭入口，等待所有 cleaner 排空
	close(p.rawCh)
	p.cleanerWg.Wait()

	// 阶段 2: cleaner 已全部退出，关闭 cleanCh，等待所有 chunker 排空
	close(p.cleanCh)
	p.chunkerWg.Wait()

	// 阶段 3: chunker 已全部退出，关闭 chunkCh，等待所有 embedder 排空（含最终 flush）
	close(p.chunkCh)
	p.embedderWg.Wait()

	// 阶段 4: embedder 已全部退出，关闭 embeddedCh，等待 writer 排空（含最终 flush）
	close(p.embeddedCh)
	p.writerWg.Wait()

	// 关闭错误通道
	close(p.errCh)

	if dropped := p.droppedChunks.Load(); dropped > 0 {
		p.logger.Info("管道已停止", "dropped_chunks", dropped)
	} else {
		p.logger.Info("管道已停止")
	}
}

func (p *Pipeline) runCleaner(ctx context.Context, workerID int) {
	defer p.cleanerWg.Done()
	logger := p.logger.With("stage", "cleaner", "worker", workerID)

	for msg := range p.rawCh {
		start := time.Now()
		doc, err := p.cleaner.Clean(ctx, msg)
		metrics.StageDuration.WithLabelValues("cleaner").Observe(time.Since(start).Seconds())

		if err != nil {
			metrics.MessagesProcessed.WithLabelValues("cleaner", "error").Inc()
			p.errCh <- fmt.Errorf("清洗失败 (topic=%s): %w", msg.Topic, err)
			continue
		}
		metrics.MessagesProcessed.WithLabelValues("cleaner", "success").Inc()

		select {
		case p.cleanCh <- doc:
		case <-ctx.Done():
			logger.Info("worker 收到停止信号")
			return
		}
	}
	logger.Info("输入通道已关闭，worker 退出")
}

func (p *Pipeline) runChunker(ctx context.Context, workerID int) {
	defer p.chunkerWg.Done()
	logger := p.logger.With("stage", "chunker", "worker", workerID)

	for doc := range p.cleanCh {
		start := time.Now()
		chunks, err := p.chunker.Chunk(ctx, doc)
		metrics.StageDuration.WithLabelValues("chunker").Observe(time.Since(start).Seconds())

		if err != nil {
			metrics.MessagesProcessed.WithLabelValues("chunker", "error").Inc()
			p.errCh <- fmt.Errorf("分块失败 (docID=%s): %w", doc.DocID, err)
			continue
		}
		metrics.MessagesProcessed.WithLabelValues("chunker", "success").Inc()

		for _, chunk := range chunks {
			select {
			case p.chunkCh <- chunk:
			case <-ctx.Done():
				logger.Info("worker 收到停止信号")
				return
			}
		}
	}
	logger.Info("输入通道已关闭，worker 退出")
}

func (p *Pipeline) runEmbedder(ctx context.Context, workerID int) {
	defer p.embedderWg.Done()
	logger := p.logger.With("stage", "embedder", "worker", workerID)

	batch := make([]models.TextChunk, 0, p.batchSize)
	timer := time.NewTimer(p.flushInterval)
	defer timer.Stop()

	flush := func() {
		if len(batch) == 0 {
			return
		}
		batchLen := len(batch)
		metrics.BatchFlush.WithLabelValues("embedder").Inc()
		metrics.BatchSize.WithLabelValues("embedder").Observe(float64(batchLen))

		texts := make([]string, batchLen)
		for i, c := range batch {
			texts[i] = c.Content
		}

		start := time.Now()
		vectors, err := p.embedder.Embed(ctx, texts)
		metrics.StageDuration.WithLabelValues("embedder").Observe(time.Since(start).Seconds())

		if err != nil {
			p.droppedChunks.Add(int64(batchLen))
			metrics.ChunksDropped.WithLabelValues("embedder").Add(float64(batchLen))
			metrics.MessagesProcessed.WithLabelValues("embedder", "error").Inc()
			p.errCh <- fmt.Errorf("向量化失败 (batch=%d, dropped=%d): %w", batchLen, batchLen, err)
			batch = batch[:0]
			return
		}

		for i, chunk := range batch {
			if i >= len(vectors) {
				dropped := batchLen - i
				p.droppedChunks.Add(int64(dropped))
				metrics.ChunksDropped.WithLabelValues("embedder").Add(float64(dropped))
				p.errCh <- fmt.Errorf("向量数量不足 (expected=%d, got=%d, dropped=%d)", batchLen, len(vectors), dropped)
				break
			}
			embedded := models.NewEmbeddedChunk(
				chunk.DocID, chunk.Content, vectors[i], chunk.ChunkIndex, chunk.Metadata,
			)
			select {
			case p.embeddedCh <- embedded:
			case <-ctx.Done():
				p.droppedChunks.Add(int64(batchLen - i))
				return
			}
		}
		metrics.MessagesProcessed.WithLabelValues("embedder", "success").Inc()
		batch = batch[:0]
	}

	for {
		select {
		case chunk, ok := <-p.chunkCh:
			if !ok {
				flush()
				logger.Info("输入通道已关闭，worker 退出")
				return
			}
			batch = append(batch, chunk)
			if len(batch) >= p.batchSize {
				flush()
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}
				timer.Reset(p.flushInterval)
			}
		case <-timer.C:
			flush()
			timer.Reset(p.flushInterval)
		case <-ctx.Done():
			flush()
			return
		}
	}
}

func (p *Pipeline) runWriter(ctx context.Context, workerID int) {
	defer p.writerWg.Done()
	logger := p.logger.With("stage", "writer", "worker", workerID)

	batch := make([]models.EmbeddedChunk, 0, p.batchSize)
	timer := time.NewTimer(p.flushInterval)
	defer timer.Stop()

	flush := func() {
		if len(batch) == 0 {
			return
		}
		batchLen := len(batch)
		metrics.BatchFlush.WithLabelValues("writer").Inc()
		metrics.BatchSize.WithLabelValues("writer").Observe(float64(batchLen))

		start := time.Now()
		err := p.vectorWriter.Upsert(ctx, batch)
		metrics.StageDuration.WithLabelValues("writer").Observe(time.Since(start).Seconds())

		if err != nil {
			p.droppedChunks.Add(int64(batchLen))
			metrics.ChunksDropped.WithLabelValues("writer").Add(float64(batchLen))
			metrics.MessagesProcessed.WithLabelValues("writer", "error").Inc()
			p.errCh <- fmt.Errorf("写入向量库失败 (batch=%d, dropped=%d): %w", batchLen, batchLen, err)
		} else {
			metrics.MessagesProcessed.WithLabelValues("writer", "success").Inc()
		}
		batch = batch[:0]
	}

	for {
		select {
		case chunk, ok := <-p.embeddedCh:
			if !ok {
				flush()
				logger.Info("输入通道已关闭，writer 退出")
				return
			}
			batch = append(batch, chunk)
			if len(batch) >= p.batchSize {
				flush()
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}
				timer.Reset(p.flushInterval)
			}
		case <-timer.C:
			flush()
			timer.Reset(p.flushInterval)
		case <-ctx.Done():
			flush()
			return
		}
	}
}

func (p *Pipeline) runErrorCollector(ctx context.Context) {
	for {
		select {
		case err, ok := <-p.errCh:
			if !ok {
				return
			}
			p.logger.Error("管道错误", "error", err)
		case <-ctx.Done():
			return
		}
	}
}

// runChannelMonitor 定期采集通道利用率指标
func (p *Pipeline) runChannelMonitor(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	record := func(name string, l, c int) {
		if c > 0 {
			metrics.ChannelUtilization.WithLabelValues(name).Set(float64(l) / float64(c))
		}
	}

	for {
		select {
		case <-ticker.C:
			record("raw", len(p.rawCh), cap(p.rawCh))
			record("clean", len(p.cleanCh), cap(p.cleanCh))
			record("chunk", len(p.chunkCh), cap(p.chunkCh))
			record("embedded", len(p.embeddedCh), cap(p.embeddedCh))
		case <-ctx.Done():
			return
		}
	}
}
