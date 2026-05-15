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
}

// NewTEI 创建 TEI 向量化客户端
func NewTEI(baseURL string) *TEIEmbedder {
	e := &TEIEmbedder{
		baseURL: baseURL,
		client: &http.Client{
			Timeout: 60 * time.Second,
		},
		batchSize: 64,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	e.dimensions = e.fetchDimensions(ctx)

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

func (e *TEIEmbedder) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
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
		return nil, fmt.Errorf("HTTP 请求失败: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取响应失败: %w", err)
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
		return 768 // 默认维度
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return 768
	}
	defer resp.Body.Close()

	var info struct {
		ModelID string `json:"model_id"`
		MaxSeq  int    `json:"max_seq_length"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return 768
	}

	return 768
}
