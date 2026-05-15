package embedder

import "context"

// Embedder 定义文本向量化接口
type Embedder interface {
	// Embed 将文本列表转为向量列表
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	// Dimensions 返回向量的维度
	Dimensions() int
}
