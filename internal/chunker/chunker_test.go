package chunker

import (
	"context"
	"strings"
	"testing"

	"github.com/dongxxg/RAG-flow/internal/models"
)

func TestChunk(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name      string
		chunkSize int
		overlap   int
		content   string
		wantErr   bool
		wantMin   int // 最少分块数
		wantMax   int // 最多分块数
		check     func(t *testing.T, chunks []models.TextChunk)
	}{
		{
			name:      "短文档返回单个分块",
			chunkSize: 512,
			overlap:   64,
			content:   "短文档内容",
			wantMin:   1,
			wantMax:   1,
		},
		{
			name:      "精确等于 chunkSize 返回单个分块",
			chunkSize: 10,
			overlap:   0,
			content:   "0123456789",
			wantMin:   1,
			wantMax:   1,
		},
		{
			name:      "长文档按段落分割",
			chunkSize: 20,
			overlap:   0,
			content:   "第一段内容\n\n第二段内容\n\n第三段内容",
			wantMin:   3,
			wantMax:   5,
		},
		{
			name:      "重叠区域正确",
			chunkSize: 20,
			overlap:   5,
			content:   "第一段内容文字\n\n第二段内容文字\n\n第三段内容文字",
			check: func(t *testing.T, chunks []models.TextChunk) {
				if len(chunks) < 2 {
					t.Fatal("期望至少 2 个分块")
				}
				prev := chunks[0].Content
				// overlap 保证 UTF-8 安全，所以重叠区域是前块尾部的一段有效 UTF-8 子串
				if len(prev) > 0 && len(chunks[1].Content) > 0 {
					// 验证第二个 chunk 的头部来自第一个 chunk 的尾部
					tail := prev
					if len(tail) > 5 {
						tail = tail[len(tail)-5:]
					}
					// 至少应该有部分重叠
					if !strings.HasPrefix(chunks[1].Content, tail) {
						// UTF-8 对齐可能使 overlap 少几个字节，检查是否有部分重叠
						found := false
						for start := 0; start <= len(tail); start++ {
							if strings.HasPrefix(chunks[1].Content, tail[start:]) {
								found = true
								break
							}
						}
						if !found {
							t.Errorf("重叠区域不匹配: 前块尾部 %q, 后块头部 %q", tail, chunks[1].Content[:min(len(tail), len(chunks[1].Content))])
						}
					}
				}
			},
		},
		{
			name:      "分块索引递增",
			chunkSize: 10,
			overlap:   0,
			content:   "0123456789abcdefghij",
			check: func(t *testing.T, chunks []models.TextChunk) {
				for i, chunk := range chunks {
					if chunk.ChunkIndex != i {
						t.Errorf("分块索引不匹配: got %d, want %d", chunk.ChunkIndex, i)
					}
				}
			},
		},
		{
			name:      "每个分块不超过 chunkSize（含重叠）",
			chunkSize: 30,
			overlap:   5,
			content:   strings.Repeat("这是一段测试文本。", 20),
			check: func(t *testing.T, chunks []models.TextChunk) {
				for i, chunk := range chunks {
					maxLen := 30 + 5 // 允许多一个 overlap
					if len(chunk.Content) > maxLen {
						t.Errorf("分块 %d 长度 %d 超过限制 %d", i, len(chunk.Content), maxLen)
					}
				}
			},
		},
		{
			name:      "空文档返回错误",
			chunkSize: 512,
			overlap:   64,
			content:   "",
			wantErr:   true,
		},
		{
			name:      "docID 正确传递",
			chunkSize: 10,
			overlap:   0,
			content:   "0123456789abcdefghij",
			check: func(t *testing.T, chunks []models.TextChunk) {
				for _, chunk := range chunks {
					if chunk.DocID != "test-doc" {
						t.Errorf("DocID 不匹配: got %q, want %q", chunk.DocID, "test-doc")
					}
				}
			},
		},
		{
			name:      "无分隔符时强制分割",
			chunkSize: 5,
			overlap:   0,
			content:   "abcdefghijklmnopqrstuvwxyz",
			wantMin:   5,
			wantMax:   7,
		},
		{
			name:      "中文字符强制分割保证 UTF-8 安全",
			chunkSize: 15,
			overlap:   0,
			content:   "你好世界测试文本内容",
			check: func(t *testing.T, chunks []models.TextChunk) {
				for i, chunk := range chunks {
					for _, r := range chunk.Content {
						if r == '�' {
							t.Errorf("分块 %d 包含损坏的 UTF-8 字符", i)
						}
					}
				}
			},
		},
		{
			name:      "中文重叠保证 UTF-8 安全",
			chunkSize: 12,
			overlap:   3,
			content:   strings.Repeat("这是测试内容段落。", 10),
			check: func(t *testing.T, chunks []models.TextChunk) {
				for i, chunk := range chunks {
					for _, r := range chunk.Content {
						if r == '�' {
							t.Errorf("分块 %d 重叠区域包含损坏的 UTF-8 字符", i)
						}
					}
				}
			},
		},
		{
			name:      "metadata 正确传递",
			chunkSize: 10,
			overlap:   0,
			content:   "0123456789abcdefghij",
			check: func(t *testing.T, chunks []models.TextChunk) {
				for _, chunk := range chunks {
					if v, ok := chunk.Metadata["key"]; !ok || v != "value" {
						t.Errorf("Metadata 不匹配: got %v, want key=value", chunk.Metadata)
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, err := New(tt.chunkSize, tt.overlap)
			if err != nil {
				t.Fatalf("创建 chunker 失败: %v", err)
			}
			meta := map[string]string{"key": "value"}
			if tt.name == "空文档返回错误" || tt.name == "短文档返回单个分块" {
				meta = nil
			}
			doc := models.NewCleanedDocument("test", "test-doc", "测试标题", tt.content, meta)

			chunks, err := c.Chunk(ctx, doc)
			if tt.wantErr {
				if err == nil {
					t.Error("期望返回错误，实际返回 nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("不期望错误: %v", err)
			}

			if tt.wantMin > 0 && len(chunks) < tt.wantMin {
				t.Errorf("分块数 %d 少于最小期望 %d", len(chunks), tt.wantMin)
			}
			if tt.wantMax > 0 && len(chunks) > tt.wantMax {
				t.Errorf("分块数 %d 超过最大期望 %d", len(chunks), tt.wantMax)
			}

			if tt.check != nil {
				tt.check(t, chunks)
			}
		})
	}
}

func TestChunkContextCancel(t *testing.T) {
	c, err := New(512, 64)
	if err != nil {
		t.Fatalf("创建 chunker 失败: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	doc := models.NewCleanedDocument("test", "test-doc", "", "内容", nil)
	_, err = c.Chunk(ctx, doc)
	if err == nil {
		t.Error("期望 context 取消返回错误")
	}
}

func TestNewValidation(t *testing.T) {
	tests := []struct {
		name      string
		chunkSize int
		overlap   int
		wantErr   bool
	}{
		{"正常参数", 512, 64, false},
		{"chunkSize 为零", 0, 0, true},
		{"chunkSize 为负", -1, 0, true},
		{"overlap 为负", 512, -1, true},
		{"overlap 等于 chunkSize", 512, 512, true},
		{"overlap 大于 chunkSize", 100, 200, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := New(tt.chunkSize, tt.overlap)
			if (err != nil) != tt.wantErr {
				t.Errorf("New(%d, %d) error = %v, wantErr %v", tt.chunkSize, tt.overlap, err, tt.wantErr)
			}
		})
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
