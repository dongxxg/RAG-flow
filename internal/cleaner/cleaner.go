package cleaner

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"

	"RAG-Flow/internal/models"
)

const minContentLength = 10

var (
	htmlTagRe     = regexp.MustCompile(`<[^>]*>`)
	controlCharRe = regexp.MustCompile(`[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]`)
	multiSpaceRe  = regexp.MustCompile(`[ \t]+`)
	multiNewlineRe = regexp.MustCompile(`\n{3,}`)
)

// ErrInvalidDocument 表示文档未通过校验
var ErrInvalidDocument = fmt.Errorf("invalid document")

// Cleaner 定义文本清洗接口
type Cleaner interface {
	Clean(ctx context.Context, msg models.RawMessage) (models.CleanedDocument, error)
}

// TextCleaner 文本清洗器实现
type TextCleaner struct{}

// New 创建 TextCleaner 实例
func New() *TextCleaner {
	return &TextCleaner{}
}

// incomingMessage 表示 Kafka 消息体的 JSON 结构
type incomingMessage struct {
	DocID    string            `json:"doc_id"`
	Source   string            `json:"source"`
	Title    string            `json:"title"`
	Content  string            `json:"content"`
	Metadata map[string]string `json:"metadata"`
}

// Clean 执行文本清洗流程
func (c *TextCleaner) Clean(ctx context.Context, msg models.RawMessage) (models.CleanedDocument, error) {
	select {
	case <-ctx.Done():
		return models.CleanedDocument{}, ctx.Err()
	default:
	}

	var incoming incomingMessage
	if err := json.Unmarshal(msg.Value, &incoming); err != nil {
		return models.CleanedDocument{}, fmt.Errorf("%w: JSON 解析失败: %v", ErrInvalidDocument, err)
	}

	if incoming.DocID == "" {
		return models.CleanedDocument{}, fmt.Errorf("%w: doc_id 不能为空", ErrInvalidDocument)
	}

	if incoming.Content == "" {
		return models.CleanedDocument{}, fmt.Errorf("%w: content 不能为空", ErrInvalidDocument)
	}

	cleaned := c.cleanText(incoming.Content)

	if utf8.RuneCountInString(cleaned) < minContentLength {
		return models.CleanedDocument{}, fmt.Errorf("%w: 清洗后内容过短（< %d 字符）", ErrInvalidDocument, minContentLength)
	}

	return models.NewCleanedDocument(
		incoming.Source,
		incoming.DocID,
		incoming.Title,
		cleaned,
		incoming.Metadata,
	), nil
}

// cleanText 执行文本清洗操作，返回新字符串
func (c *TextCleaner) cleanText(text string) string {
	text = ensureUTF8(text)
	text = htmlTagRe.ReplaceAllString(text, "")
	text = controlCharRe.ReplaceAllString(text, "")
	text = multiNewlineRe.ReplaceAllString(text, "\n\n")
	text = multiSpaceRe.ReplaceAllString(text, " ")
	text = strings.TrimSpace(text)
	return text
}

// ensureUTF8 确保文本为合法 UTF-8，替换非法字节
func ensureUTF8(text string) string {
	if utf8.ValidString(text) {
		return text
	}
	var buf bytes.Buffer
	for len(text) > 0 {
		r, size := utf8.DecodeRuneInString(text)
		if r == utf8.RuneError && size == 1 {
			buf.WriteRune('�')
		} else {
			buf.WriteRune(r)
		}
		text = text[size:]
	}
	return buf.String()
}
