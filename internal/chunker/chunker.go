package chunker

import (
	"context"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/dongxxg/RAG-flow/internal/models"
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
// chunkSize 必须 > 0，overlap 必须 >= 0 且 < chunkSize
func New(chunkSize, overlap int) (*RecursiveChunker, error) {
	if chunkSize <= 0 {
		return nil, fmt.Errorf("chunkSize 必须 > 0，当前值: %d", chunkSize)
	}
	if overlap < 0 {
		return nil, fmt.Errorf("overlap 不能为负数，当前值: %d", overlap)
	}
	if overlap >= chunkSize {
		return nil, fmt.Errorf("overlap (%d) 必须 < chunkSize (%d)", overlap, chunkSize)
	}
	return &RecursiveChunker{
		chunkSize:  chunkSize,
		overlap:    overlap,
		separators: defaultSeparators,
	}, nil
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
	for _, text := range texts {
		if strings.TrimSpace(text) == "" {
			continue
		}
		chunks = append(chunks, models.NewTextChunk(
			doc.DocID, text, len(chunks), offset, offset+len(text), doc.Metadata,
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

// forceSplit 强制按字符数分割，保证 UTF-8 字符边界安全
func (c *RecursiveChunker) forceSplit(text string) []string {
	var result []string
	pos := 0
	for pos < len(text) {
		end := pos
		remaining := c.chunkSize
		// 按 rune 步进，确保 end 落在 UTF-8 字符边界上
		for remaining > 0 && end < len(text) {
			_, size := utf8.DecodeRuneInString(text[end:])
			if size == 0 {
				break
			}
			if size > remaining {
				break
			}
			end += size
			remaining -= size
		}
		if end == pos {
			// 单个 rune 超过 chunkSize，必须包含它以防无限循环
			_, size := utf8.DecodeRuneInString(text[pos:])
			end += size
		}
		result = append(result, text[pos:end])
		pos = end
	}
	return result
}

// mergeSmall 合并过小的分块
func (c *RecursiveChunker) mergeSmall(texts []string) []string {
	if len(texts) <= 1 {
		return texts
	}

	merged := make([]string, 0, len(texts))
	var b strings.Builder
	b.WriteString(texts[0])

	for i := 1; i < len(texts); i++ {
		candidateLen := b.Len() + len(texts[i])
		if candidateLen <= c.chunkSize {
			b.WriteString(texts[i])
		} else {
			merged = append(merged, b.String())
			b.Reset()
			b.WriteString(texts[i])
		}
	}
	merged = append(merged, b.String())
	return merged
}

// addOverlap 为分块添加重叠区域，保证 UTF-8 字符边界安全
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
		// 从 prev 尾部取 overlapLen 字节，但需要确保从有效的 UTF-8 字符边界开始
		start := len(prev) - overlapLen
		for start < len(prev) {
			if utf8.RuneStart(prev[start]) {
				break
			}
			start++
		}
		result[i] = prev[start:] + texts[i]
	}

	return result
}
