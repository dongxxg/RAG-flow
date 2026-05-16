package pipeline

import (
	"time"

	"github.com/dongxxg/RAG-flow/internal/chunker"
	"github.com/dongxxg/RAG-flow/internal/cleaner"
	"github.com/dongxxg/RAG-flow/internal/embedder"
	"github.com/dongxxg/RAG-flow/internal/writer"
)

// Option 管道配置选项
type Option func(*Pipeline)

// WithWorkerCount 设置所有阶段的 worker 数（全局默认，可被阶段特定选项覆盖）
func WithWorkerCount(n int) Option {
	return func(p *Pipeline) {
		p.cleanerWorkers = n
		p.chunkerWorkers = n
		p.embedderWorkers = n
	}
}

// WithCleanerWorkers 设置 cleaner worker 数
func WithCleanerWorkers(n int) Option {
	return func(p *Pipeline) { p.cleanerWorkers = n }
}

// WithChunkerWorkers 设置 chunker worker 数
func WithChunkerWorkers(n int) Option {
	return func(p *Pipeline) { p.chunkerWorkers = n }
}

// WithEmbedderWorkers 设置 embedder worker 数
func WithEmbedderWorkers(n int) Option {
	return func(p *Pipeline) { p.embedderWorkers = n }
}

// WithWriterWorkers 设置 writer worker 数
func WithWriterWorkers(n int) Option {
	return func(p *Pipeline) { p.writerWorkers = n }
}

func WithBatchSize(n int) Option {
	return func(p *Pipeline) { p.batchSize = n }
}

func WithFlushInterval(d time.Duration) Option {
	return func(p *Pipeline) { p.flushInterval = d }
}

func WithChannelBufferSize(n int) Option {
	return func(p *Pipeline) { p.channelBufferSize = n }
}

func WithCleaner(c cleaner.Cleaner) Option {
	return func(p *Pipeline) { p.cleaner = c }
}

func WithChunker(ch chunker.Chunker) Option {
	return func(p *Pipeline) { p.chunker = ch }
}

func WithEmbedder(e embedder.Embedder) Option {
	return func(p *Pipeline) { p.embedder = e }
}

func WithWriter(w writer.VectorWriter) Option {
	return func(p *Pipeline) { p.vectorWriter = w }
}
