package cleaner

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"RAG-Flow/internal/models"
)

func TestClean(t *testing.T) {
	c := New()

	tests := []struct {
		name    string
		msg     models.RawMessage
		wantErr bool
		check   func(t *testing.T, doc models.CleanedDocument)
	}{
		{
			name: "正常文本通过清洗",
			msg:  makeMsg(t, "doc1", "src", "标题", "这是一段正常的文本内容，长度足够。", nil),
			check: func(t *testing.T, doc models.CleanedDocument) {
				if doc.Content != "这是一段正常的文本内容，长度足够。" {
					t.Errorf("内容不匹配: got %q", doc.Content)
				}
				if doc.DocID != "doc1" {
					t.Errorf("DocID 不匹配: got %q", doc.DocID)
				}
			},
		},
		{
			name:    "空 JSON 返回错误",
			msg:     models.RawMessage{Value: []byte("{}"), Timestamp: time.Now()},
			wantErr: true,
		},
		{
			name:    "非法 JSON 返回错误",
			msg:     models.RawMessage{Value: []byte("not json"), Timestamp: time.Now()},
			wantErr: true,
		},
		{
			name:    "空 content 返回错误",
			msg:     makeMsg(t, "doc2", "src", "", "", nil),
			wantErr: true,
		},
		{
			name:    "空 doc_id 返回错误",
			msg:     makeMsg(t, "", "src", "", "有效内容足够长", nil),
			wantErr: true,
		},
		{
			name:    "内容过短返回错误",
			msg:     makeMsg(t, "doc3", "src", "", "短", nil),
			wantErr: true,
		},
		{
			name: "HTML 标签被剥离",
			msg: makeMsg(t, "doc4", "src", "", "<p>段落一</p><div><b>加粗</b></div><p>段落二继续</p>", nil),
			check: func(t *testing.T, doc models.CleanedDocument) {
				if contains(doc.Content, "<") || contains(doc.Content, ">") {
					t.Errorf("HTML 标签未清除: %q", doc.Content)
				}
			},
		},
		{
			name: "多余空白被规范化",
			msg: makeMsg(t, "doc5", "src", "", "  多余   空格   和\n\n\n\n换行  ", nil),
			check: func(t *testing.T, doc models.CleanedDocument) {
				if doc.Content != "多余 空格 和\n\n换行" {
					t.Errorf("空白未规范化: %q", doc.Content)
				}
			},
		},
		{
			name: "控制字符被移除",
			msg: makeMsg(t, "doc6", "src", "", "正常\x00\x01\x02文本\x07继续内容在这里", nil),
			check: func(t *testing.T, doc models.CleanedDocument) {
				for _, r := range doc.Content {
					if r < 0x20 && r != '\n' && r != '\t' {
						t.Errorf("控制字符未移除: %U", r)
					}
				}
			},
		},
		{
			name: "非法 UTF-8 被替换",
			msg: makeRawMsg("doc7", "src", "", "合法\xff\xfe文本内容继续补充", nil),
			check: func(t *testing.T, doc models.CleanedDocument) {
				for _, r := range doc.Content {
					if r == 0xFFFD {
						return
					}
				}
				t.Error("非法 UTF-8 未被替换")
			},
		},
		{
			name: "元数据正确传递",
			msg: makeMsg(t, "doc8", "src", "标题", "正常内容足够长在这里", map[string]string{"key": "value"}),
			check: func(t *testing.T, doc models.CleanedDocument) {
				if doc.Metadata["key"] != "value" {
					t.Errorf("元数据不匹配: got %v", doc.Metadata)
				}
			},
		},
		{
			name:    "context 取消立即返回",
			msg:     makeMsg(t, "doc9", "src", "", "正常内容足够长在这里", nil),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var ctx context.Context
			var cancel context.CancelFunc
			if tt.name == "context 取消立即返回" {
				ctx, cancel = context.WithCancel(context.Background())
				cancel()
			} else {
				ctx, cancel = context.WithCancel(context.Background())
			}
			defer cancel()

			doc, err := c.Clean(ctx, tt.msg)
			if tt.wantErr {
				if err == nil {
					t.Error("期望返回错误，实际返回 nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("不期望错误: %v", err)
			}
			if tt.check != nil {
				tt.check(t, doc)
			}
		})
	}
}

func makeMsg(t *testing.T, docID, source, title, content string, metadata map[string]string) models.RawMessage {
	t.Helper()
	body := struct {
		DocID    string            `json:"doc_id"`
		Source   string            `json:"source"`
		Title    string            `json:"title"`
		Content  string            `json:"content"`
		Metadata map[string]string `json:"metadata"`
	}{
		DocID: docID, Source: source, Title: title, Content: content, Metadata: metadata,
	}
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("构造测试消息失败: %v", err)
	}
	return models.RawMessage{Value: data, Timestamp: time.Now()}
}

// makeRawMsg 构造包含原始字节的测试消息（不经过 JSON 编码 content）
func makeRawMsg(docID, source, title, content string, metadata map[string]string) models.RawMessage {
	body := struct {
		DocID    string            `json:"doc_id"`
		Source   string            `json:"source"`
		Title    string            `json:"title"`
		Content  string            `json:"content"`
		Metadata map[string]string `json:"metadata"`
	}{
		DocID: docID, Source: source, Title: title, Content: content, Metadata: metadata,
	}
	data, _ := json.Marshal(body)
	return models.RawMessage{Value: data, Timestamp: time.Now()}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsSubstr(s, substr))
}

func containsSubstr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
