package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// OpenAIEmbedder OpenAI 向量化实现
type OpenAIEmbedder struct {
	apiKey    string
	model     string
	baseURL   string
	batchSize int
	client    *http.Client
	maxRetry  int
}

// OpenAI 配置选项
type OpenAIOption func(*OpenAIEmbedder)

func WithOpenAIBatchSize(n int) OpenAIOption {
	return func(e *OpenAIEmbedder) { e.batchSize = n }
}

func WithOpenAIMaxRetry(n int) OpenAIOption {
	return func(e *OpenAIEmbedder) { e.maxRetry = n }
}

func WithOpenAIBaseURL(url string) OpenAIOption {
	return func(e *OpenAIEmbedder) { e.baseURL = url }
}

// NewOpenAI 创建 OpenAI 向量化客户端
func NewOpenAI(apiKey, model string, opts ...OpenAIOption) *OpenAIEmbedder {
	e := &OpenAIEmbedder{
		apiKey:    apiKey,
		model:     model,
		baseURL:   "https://api.openai.com/v1/embeddings",
		batchSize: 64,
		client: &http.Client{
			Timeout: 60 * time.Second,
		},
		maxRetry: 3,
	}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

// modelDimensions 各模型的默认向量维度
var modelDimensions = map[string]int{
	"text-embedding-3-small": 1536,
	"text-embedding-3-large": 3072,
	"text-embedding-ada-002": 1536,
}

// Dimensions 返回当前模型的向量维度
func (e *OpenAIEmbedder) Dimensions() int {
	if d, ok := modelDimensions[e.model]; ok {
		return d
	}
	return 1536
}

// openAIRequest OpenAI Embedding API 请求体
type openAIRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

// openAIResponse OpenAI Embedding API 响应体
type openAIResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

// Embed 将文本列表转为向量列表
func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	allVectors := make([][]float32, 0, len(texts))

	for i := 0; i < len(texts); i += e.batchSize {
		end := i + e.batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]

		vectors, err := e.embedBatch(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("批次 %d-%d 向量化失败: %w", i, end, err)
		}
		allVectors = append(allVectors, vectors...)
	}

	return allVectors, nil
}

// Close 关闭客户端
func (e *OpenAIEmbedder) Close() error {
	e.client.CloseIdleConnections()
	return nil
}

// embedBatch 处理单个批次的向量化请求
func (e *OpenAIEmbedder) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	return doWithRetry(retryConfig{maxRetry: e.maxRetry}, func(attempt int) ([][]float32, error) {
		if attempt > 0 {
			delay := backoffDuration(attempt)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}
		return e.doRequest(ctx, texts)
	})
}

// doRequest 执行单次 HTTP 请求
func (e *OpenAIEmbedder) doRequest(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody := openAIRequest{
		Input: texts,
		Model: e.model,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("序列化请求失败: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("创建请求失败: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, &retryableError{Err: fmt.Errorf("HTTP 请求失败: %w", err)}
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取响应失败: %w", err)
	}

	if resp.StatusCode == http.StatusTooManyRequests {
		return nil, &retryableError{Err: fmt.Errorf("速率限制 (429)")}
	}

	if resp.StatusCode >= 500 {
		return nil, &retryableError{Err: fmt.Errorf("服务端错误 (%d)", resp.StatusCode)}
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API 错误 (%d): %s", resp.StatusCode, string(body))
	}

	var result openAIResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("解析响应失败: %w", err)
	}

	if result.Error != nil {
		return nil, fmt.Errorf("API 返回错误: %s", result.Error.Message)
	}

	vectors := make([][]float32, len(result.Data))
	for _, d := range result.Data {
		vec := make([]float32, len(d.Embedding))
		copy(vec, d.Embedding)
		vectors[d.Index] = vec
	}

	return vectors, nil
}
