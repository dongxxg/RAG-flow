package chunker

import (
	"context"
	"fmt"
	"strings"

	"RAG-Flow/internal/models"
)

// 默认分隔符优先级
var defaultSeparators = []string{"\n\n", "\n", ". ", " ", ""}

// Chunker 定义文本分块接口
type Chunker interface {
	Chunk(ctx context.Context, doc models.CleanedDocument) ([]models.TextChunk, error)
}

// RecursiveChunker 递归字符分块器
type RecursiveChunker struct {
	chunkSize  int
	overlap    int
	separators []string
}

// New 创建 RecursiveChunker 实例
func New(chunkSize, overlap int) *RecursiveChunker {
	return &RecursiveChunker{
		chunkSize:  chunkSize,
		overlap:    overlap,
		separators: defaultSeparators,
	}
}

// Chunk 将文档分割为多个文本分块
func (c *RecursiveChunker) Chunk(ctx context.Context, doc models.CleanedDocument) ([]models.TextChunk, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	if doc.Content == "" {
		return nil, fmt.Errorf("文档内容为空，docID: %s", doc.DocID)
	}

	if len(doc.Content) <= c.chunkSize {
		return []models.TextChunk{
			models.NewTextChunk(doc.DocID, doc.Content, 0, 0, len(doc.Content), doc.Metadata),
		}, nil
	}

	texts := c.splitText(doc.Content, c.separators)
	texts = c.mergeSmall(texts)
	texts = c.addOverlap(texts)

	chunks := make([]models.TextChunk, 0, len(texts))
	offset := 0
	for i, text := range texts {
		if strings.TrimSpace(text) == "" {
			continue
		}
		chunks = append(chunks, models.NewTextChunk(
			doc.DocID, text, i, offset, offset+len(text), doc.Metadata,
		))
		offset += len(text)
	}

	return chunks, nil
}

// splitText 递归分割文本
func (c *RecursiveChunker) splitText(text string, separators []string) []string {
	if len(text) <= c.chunkSize {
		return []string{text}
	}

	if len(separators) == 0 {
		return c.forceSplit(text)
	}

	sep := separators[0]
	remaining := separators[1:]

	if sep == "" {
		return c.forceSplit(text)
	}

	parts := strings.Split(text, sep)
	var result []string
	var current strings.Builder

	for _, part := range parts {
		if part == "" {
			continue
		}

		if current.Len() > 0 {
			candidate := current.String() + sep + part
			if len(candidate) <= c.chunkSize {
				current.Reset()
				current.WriteString(candidate)
				continue
			}
			// 当前累积已满，先保存
			result = append(result, current.String())
			current.Reset()
		}

		if len(part) <= c.chunkSize {
			current.WriteString(part)
		} else {
			// 单个 part 超过 chunkSize，需要递归拆分
			subResults := c.splitText(part, remaining)
			for _, sr := range subResults {
				if current.Len() > 0 {
					candidate := current.String() + sep + sr
					if len(candidate) <= c.chunkSize {
						current.Reset()
						current.WriteString(candidate)
						continue
					}
					result = append(result, current.String())
					current.Reset()
				}
				if len(sr) <= c.chunkSize {
					current.WriteString(sr)
				} else {
					result = append(result, sr)
				}
			}
		}
	}

	if current.Len() > 0 {
		result = append(result, current.String())
	}

	return result
}

// forceSplit 强制按字符数分割
func (c *RecursiveChunker) forceSplit(text string) []string {
	var result []string
	for i := 0; i < len(text); i += c.chunkSize {
		end := i + c.chunkSize
		if end > len(text) {
			end = len(text)
		}
		result = append(result, text[i:end])
	}
	return result
}

// mergeSmall 合并过小的分块
func (c *RecursiveChunker) mergeSmall(texts []string) []string {
	if len(texts) <= 1 {
		return texts
	}

	var merged []string
	current := texts[0]

	for i := 1; i < len(texts); i++ {
		candidate := current + texts[i]
		if len(candidate) <= c.chunkSize {
			current = candidate
		} else {
			merged = append(merged, current)
			current = texts[i]
		}
	}
	merged = append(merged, current)
	return merged
}

// addOverlap 为分块添加重叠区域
func (c *RecursiveChunker) addOverlap(texts []string) []string {
	if c.overlap <= 0 || len(texts) <= 1 {
		return texts
	}

	result := make([]string, len(texts))
	result[0] = texts[0]

	for i := 1; i < len(texts); i++ {
		prev := texts[i-1]
		overlapLen := c.overlap
		if overlapLen > len(prev) {
			overlapLen = len(prev)
		}
		overlap := prev[len(prev)-overlapLen:]
		result[i] = overlap + texts[i]
	}

	return result
}
