package pipeline

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"RAG-Flow/internal/chunker"
	"RAG-Flow/internal/cleaner"
	"RAG-Flow/internal/embedder"
	"RAG-Flow/internal/models"
	"RAG-Flow/internal/writer"
)

// Pipeline RAG 数据处理管道
type Pipeline struct {
	cleaner           cleaner.Cleaner
	chunker           chunker.Chunker
	embedder          embedder.Embedder
	vectorWriter      writer.VectorWriter
	workerCount       int
	batchSize         int
	flushInterval     time.Duration
	channelBufferSize int

	rawCh      chan models.RawMessage
	cleanCh    chan models.CleanedDocument
	chunkCh    chan models.TextChunk
	embeddedCh chan models.EmbeddedChunk
	errCh      chan error

	wg     sync.WaitGroup
	cancel context.CancelFunc
	logger *slog.Logger
}

// New 创建管道实例
func New(opts ...Option) (*Pipeline, error) {
	p := &Pipeline{
		workerCount:       4,
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
		return nil
	default:
		return fmt.Errorf("管道输入通道已满，丢弃消息 (topic=%s)", msg.Topic)
	}
}

// Start 启动管道各阶段
func (p *Pipeline) Start(ctx context.Context) {
	ctx, p.cancel = context.WithCancel(ctx)

	for i := 0; i < p.workerCount; i++ {
		p.wg.Add(1)
		go p.runCleaner(ctx, i)
	}

	for i := 0; i < p.workerCount; i++ {
		p.wg.Add(1)
		go p.runChunker(ctx, i)
	}

	for i := 0; i < p.workerCount; i++ {
		p.wg.Add(1)
		go p.runEmbedder(ctx, i)
	}

	p.wg.Add(1)
	go p.runWriter(ctx)

	p.wg.Add(1)
	go p.runErrorCollector(ctx)

	p.logger.Info("管道已启动")
}

// Stop 停止管道并等待所有阶段完成
func (p *Pipeline) Stop() {
	p.logger.Info("管道停止中...")
	if p.cancel != nil {
		p.cancel()
	}
	close(p.rawCh)
	p.wg.Wait()
	close(p.errCh)
	p.logger.Info("管道已停止")
}

func (p *Pipeline) runCleaner(ctx context.Context, workerID int) {
	defer p.wg.Done()
	logger := p.logger.With("stage", "cleaner", "worker", workerID)

	for msg := range p.rawCh {
		doc, err := p.cleaner.Clean(ctx, msg)
		if err != nil {
			p.errCh <- fmt.Errorf("清洗失败 (topic=%s): %w", msg.Topic, err)
			continue
		}
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
	defer p.wg.Done()
	logger := p.logger.With("stage", "chunker", "worker", workerID)

	for doc := range p.cleanCh {
		chunks, err := p.chunker.Chunk(ctx, doc)
		if err != nil {
			p.errCh <- fmt.Errorf("分块失败 (docID=%s): %w", doc.DocID, err)
			continue
		}
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
	close(p.chunkCh)
}

func (p *Pipeline) runEmbedder(ctx context.Context, workerID int) {
	defer p.wg.Done()
	logger := p.logger.With("stage", "embedder", "worker", workerID)

	batch := make([]models.TextChunk, 0, p.batchSize)
	timer := time.NewTimer(p.flushInterval)
	defer timer.Stop()

	flush := func() {
		if len(batch) == 0 {
			return
		}
		texts := make([]string, len(batch))
		for i, c := range batch {
			texts[i] = c.Content
		}
		vectors, err := p.embedder.Embed(ctx, texts)
		if err != nil {
			p.errCh <- fmt.Errorf("向量化失败 (batch=%d): %w", len(batch), err)
			batch = batch[:0]
			return
		}
		for i, chunk := range batch {
			if i >= len(vectors) {
				break
			}
			embedded := models.NewEmbeddedChunk(
				chunk.DocID, chunk.Content, vectors[i], chunk.ChunkIndex, chunk.Metadata,
			)
			select {
			case p.embeddedCh <- embedded:
			case <-ctx.Done():
				return
			}
		}
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

func (p *Pipeline) runWriter(ctx context.Context) {
	defer p.wg.Done()
	logger := p.logger.With("stage", "writer")

	batch := make([]models.EmbeddedChunk, 0, p.batchSize)
	timer := time.NewTimer(p.flushInterval)
	defer timer.Stop()

	flush := func() {
		if len(batch) == 0 {
			return
		}
		if err := p.vectorWriter.Upsert(ctx, batch); err != nil {
			p.errCh <- fmt.Errorf("写入向量库失败 (batch=%d): %w", len(batch), err)
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
	defer p.wg.Done()
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
