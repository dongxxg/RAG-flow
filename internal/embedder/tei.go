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

// TEIEmbedder HuggingFace Text-Embeddings-Inference 客户端
type TEIEmbedder struct {
	baseURL    string
	client     *http.Client
	batchSize  int
	dimensions int
	maxRetry   int
}

// TEIOption TEI 配置选项
type TEIOption func(*TEIEmbedder)

// WithTEIBatchSize 设置 TEI 批次大小
func WithTEIBatchSize(n int) TEIOption {
	return func(e *TEIEmbedder) { e.batchSize = n }
}

// WithTEIMaxRetry 设置 TEI 最大重试次数
func WithTEIMaxRetry(n int) TEIOption {
	return func(e *TEIEmbedder) { e.maxRetry = n }
}

// WithTEIDimensions 设置向量维度（覆盖自动检测）
func WithTEIDimensions(n int) TEIOption {
	return func(e *TEIEmbedder) { e.dimensions = n }
}

// NewTEI 创建 TEI 向量化客户端
func NewTEI(baseURL string, opts ...TEIOption) *TEIEmbedder {
	e := &TEIEmbedder{
		baseURL: baseURL,
		client: &http.Client{
			Timeout: 60 * time.Second,
		},
		batchSize: 64,
		maxRetry:  3,
	}
	for _, opt := range opts {
		opt(e)
	}

	// 如果没有通过选项指定维度，尝试自动检测
	if e.dimensions <= 0 {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		e.dimensions = e.fetchDimensions(ctx)
	}

	return e
}

// Dimensions 返回向量维度
func (e *TEIEmbedder) Dimensions() int {
	return e.dimensions
}

// Embed 将文本列表转为向量列表
func (e *TEIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
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
			return nil, fmt.Errorf("TEI 批次 %d-%d 向量化失败: %w", i, end, err)
		}
		allVectors = append(allVectors, vectors...)
	}

	return allVectors, nil
}

// Close 关闭客户端
func (e *TEIEmbedder) Close() error {
	e.client.CloseIdleConnections()
	return nil
}

func (e *TEIEmbedder) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
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

func (e *TEIEmbedder) doRequest(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody, err := json.Marshal(map[string]interface{}{
		"inputs": texts,
	})
	if err != nil {
		return nil, fmt.Errorf("序列化请求失败: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/embed", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("创建请求失败: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

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
		return nil, fmt.Errorf("TEI 错误 (%d): %s", resp.StatusCode, string(body))
	}

	var result [][]float32
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("解析响应失败: %w", err)
	}

	return result, nil
}

func (e *TEIEmbedder) fetchDimensions(ctx context.Context) int {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, e.baseURL+"/info", nil)
	if err != nil {
		return 768
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return 768
	}
	defer resp.Body.Close()

	var info struct {
		ModelID      string `json:"model_id"`
		ModelType    string `json:"model_type"`
		MaxSeqLength int    `json:"max_seq_length"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return 768
	}

	// TEI /info 不直接返回 embedding 维度
	// 通过一次 dummy embed 调用获取实际维度
	dim := e.probeDimensions(ctx)
	if dim > 0 {
		return dim
	}
	return 768
}

// probeDimensions 通过 embed 单条文本探测实际向量维度
func (e *TEIEmbedder) probeDimensions(ctx context.Context) int {
	probeCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	vectors, err := e.doRequest(probeCtx, []string{"probe"})
	if err != nil || len(vectors) == 0 || len(vectors[0]) == 0 {
		return 0
	}
	return len(vectors[0])
}
