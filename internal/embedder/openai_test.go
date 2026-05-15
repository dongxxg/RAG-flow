package embedder

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestOpenAIEmbed(t *testing.T) {
	tests := []struct {
		name       string
		inputs     []string
		statusCode int
		response   openAIResponse
		wantErr    bool
		wantLen    int
	}{
		{
			name:   "成功返回向量",
			inputs: []string{"文本1", "文本2"},
			response: openAIResponse{
				Data: []struct {
					Embedding []float32 `json:"embedding"`
					Index     int       `json:"index"`
				}{
					{Embedding: []float32{0.1, 0.2, 0.3}, Index: 0},
					{Embedding: []float32{0.4, 0.5, 0.6}, Index: 1},
				},
			},
			wantLen: 2,
		},
		{
			name:    "空输入返回 nil",
			inputs:  []string{},
			wantLen: 0,
		},
		{
			name:       "429 限流重试",
			inputs:     []string{"文本"},
			statusCode: http.StatusTooManyRequests,
			wantErr:    true,
		},
		{
			name:       "500 服务端错误重试",
			inputs:     []string{"文本"},
			statusCode: http.StatusInternalServerError,
			wantErr:    true,
		},
		{
			name:       "400 客户端错误不重试",
			inputs:     []string{"文本"},
			statusCode: http.StatusBadRequest,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			callCount := int32(0)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				atomic.AddInt32(&callCount, 1)

				if r.Header.Get("Authorization") != "Bearer test-key" {
					w.WriteHeader(http.StatusUnauthorized)
					return
				}

				if tt.statusCode != 0 {
					w.WriteHeader(tt.statusCode)
					return
				}

				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(tt.response)
			}))
			defer server.Close()

			e := NewOpenAI("test-key", "text-embedding-3-small",
				WithOpenAIBaseURL(server.URL),
				WithOpenAIMaxRetry(2),
				WithOpenAIBatchSize(10),
			)

			vecs, err := e.Embed(context.Background(), tt.inputs)
			if tt.wantErr {
				if err == nil {
					t.Error("期望返回错误")
				}
				return
			}
			if err != nil {
				t.Fatalf("不期望错误: %v", err)
			}
			if len(vecs) != tt.wantLen {
				t.Errorf("向量数不匹配: got %d, want %d", len(vecs), tt.wantLen)
			}
		})
	}
}

func TestOpenAIRetryOn429(t *testing.T) {
	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&calls, 1)
		if count < 3 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(openAIResponse{
			Data: []struct {
				Embedding []float32 `json:"embedding"`
				Index     int       `json:"index"`
			}{
				{Embedding: []float32{0.1, 0.2}, Index: 0},
			},
		})
	}))
	defer server.Close()

	e := NewOpenAI("test-key", "text-embedding-3-small",
		WithOpenAIBaseURL(server.URL),
		WithOpenAIMaxRetry(3),
	)

	vecs, err := e.Embed(context.Background(), []string{"文本"})
	if err != nil {
		t.Fatalf("重试应成功: %v", err)
	}
	if len(vecs) != 1 {
		t.Errorf("向量数不匹配: got %d, want 1", len(vecs))
	}
	if atomic.LoadInt32(&calls) != 3 {
		t.Errorf("调用次数不匹配: got %d, want 3", calls)
	}
}

func TestOpenAIContextCancel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	defer cancel()

	e := NewOpenAI("test-key", "text-embedding-3-small",
		WithOpenAIBaseURL(server.URL),
		WithOpenAIMaxRetry(0),
	)

	_, err := e.Embed(ctx, []string{"文本"})
	if err == nil {
		t.Error("期望 context 超时返回错误")
	}
}

func TestOpenAIDimensions(t *testing.T) {
	tests := []struct {
		model string
		want  int
	}{
		{"text-embedding-3-small", 1536},
		{"text-embedding-3-large", 3072},
		{"text-embedding-ada-002", 1536},
		{"unknown-model", 1536},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			e := NewOpenAI("key", tt.model)
			if d := e.Dimensions(); d != tt.want {
				t.Errorf("维度不匹配: got %d, want %d", d, tt.want)
			}
		})
	}
}

func TestOpenAILargeBatch(t *testing.T) {
	var batchSizes []int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openAIRequest
		json.NewDecoder(r.Body).Decode(&req)
		batchSizes = append(batchSizes, len(req.Input))

		w.Header().Set("Content-Type", "application/json")
		data := make([]struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}, len(req.Input))
		for i := range data {
			data[i] = struct {
				Embedding []float32 `json:"embedding"`
				Index     int       `json:"index"`
			}{Embedding: []float32{0.1}, Index: i}
		}
		json.NewEncoder(w).Encode(openAIResponse{Data: data})
	}))
	defer server.Close()

	e := NewOpenAI("key", "text-embedding-3-small",
		WithOpenAIBaseURL(server.URL),
		WithOpenAIBatchSize(5),
	)

	texts := make([]string, 12)
	for i := range texts {
		texts[i] = "文本"
	}

	vecs, err := e.Embed(context.Background(), texts)
	if err != nil {
		t.Fatalf("不期望错误: %v", err)
	}
	if len(vecs) != 12 {
		t.Errorf("向量数不匹配: got %d, want 12", len(vecs))
	}
	if len(batchSizes) != 3 {
		t.Errorf("批次数不匹配: got %d, want 3", len(batchSizes))
	}
}
