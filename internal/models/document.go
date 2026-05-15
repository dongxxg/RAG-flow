package models

import "time"

// RawMessage 表示从 Kafka 接收的原始消息
type RawMessage struct {
	Topic     string
	Partition int32
	Offset    int64
	Key       []byte
	Value     []byte
	Timestamp time.Time
}

// NewRawMessage 创建 RawMessage 实例
func NewRawMessage(topic string, partition int32, offset int64, key, value []byte, timestamp time.Time) RawMessage {
	return RawMessage{
		Topic:     topic,
		Partition: partition,
		Offset:    offset,
		Key:       key,
		Value:     value,
		Timestamp: timestamp,
	}
}

// CleanedDocument 表示清洗后的文档
type CleanedDocument struct {
	Source   string
	DocID    string
	Title    string
	Content  string
	Metadata map[string]string
}

// NewCleanedDocument 创建 CleanedDocument 实例
func NewCleanedDocument(source, docID, title, content string, metadata map[string]string) CleanedDocument {
	meta := make(map[string]string, len(metadata))
	for k, v := range metadata {
		meta[k] = v
	}
	return CleanedDocument{
		Source:   source,
		DocID:    docID,
		Title:    title,
		Content:  content,
		Metadata: meta,
	}
}

// TextChunk 表示文档的一个文本分块
type TextChunk struct {
	DocID       string
	Content     string
	ChunkIndex  int
	StartOffset int
	EndOffset   int
	Metadata    map[string]string
}

// NewTextChunk 创建 TextChunk 实例
func NewTextChunk(docID, content string, chunkIndex, startOffset, endOffset int, metadata map[string]string) TextChunk {
	meta := make(map[string]string, len(metadata))
	for k, v := range metadata {
		meta[k] = v
	}
	return TextChunk{
		DocID:       docID,
		Content:     content,
		ChunkIndex:  chunkIndex,
		StartOffset: startOffset,
		EndOffset:   endOffset,
		Metadata:    meta,
	}
}

// EmbeddedChunk 表示含向量的文本分块
type EmbeddedChunk struct {
	DocID      string
	Content    string
	Vector     []float32
	ChunkIndex int
	Metadata   map[string]string
}

// NewEmbeddedChunk 创建 EmbeddedChunk 实例
func NewEmbeddedChunk(docID, content string, vector []float32, chunkIndex int, metadata map[string]string) EmbeddedChunk {
	meta := make(map[string]string, len(metadata))
	for k, v := range metadata {
		meta[k] = v
	}
	vec := make([]float32, len(vector))
	copy(vec, vector)
	return EmbeddedChunk{
		DocID:      docID,
		Content:    content,
		Vector:     vec,
		ChunkIndex: chunkIndex,
		Metadata:   meta,
	}
}
