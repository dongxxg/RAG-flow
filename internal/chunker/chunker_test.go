package chunker

import (
	"context"
	"strings"
	"testing"

	"RAG-Flow/internal/models"
)

func TestChunk(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name       string
		chunkSize  int
		overlap    int
		content    string
		wantErr    bool
		wantMin    int // 最少分块数
		wantMax    int // 最多分块数
		check      func(t *testing.T, chunks []models.TextChunk)
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
				prevTail := ""
				if len(prev) >= 5 {
					prevTail = prev[len(prev)-5:]
				}
				if !strings.HasPrefix(chunks[1].Content, prevTail) {
					t.Errorf("重叠区域不匹配: 前块尾部 %q, 后块头部 %q", prevTail, chunks[1].Content[:min(5, len(chunks[1].Content))])
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := New(tt.chunkSize, tt.overlap)
			doc := models.NewCleanedDocument("test", "test-doc", "测试标题", tt.content, nil)

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
	c := New(512, 64)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	doc := models.NewCleanedDocument("test", "test-doc", "", "内容", nil)
	_, err := c.Chunk(ctx, doc)
	if err == nil {
		t.Error("期望 context 取消返回错误")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
