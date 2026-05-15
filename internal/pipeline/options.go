package pipeline

import (
	"time"

	"RAG-Flow/internal/chunker"
	"RAG-Flow/internal/cleaner"
	"RAG-Flow/internal/embedder"
	"RAG-Flow/internal/writer"
)

// Option 管道配置选项
type Option func(*Pipeline)

func WithWorkerCount(n int) Option {
	return func(p *Pipeline) { p.workerCount = n }
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
