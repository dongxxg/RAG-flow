package writer

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"RAG-Flow/internal/models"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"
)

// VectorWriter 定义向量数据库写入接口
type VectorWriter interface {
	Upsert(ctx context.Context, chunks []models.EmbeddedChunk) error
	Close() error
}

// QdrantWriter Qdrant 向量写入器
type QdrantWriter struct {
	client     *qdrant.Client
	collection string
	namespace  uuid.UUID
	batchSize  int
}

// QdrantOption Qdrant 写入器配置选项
type QdrantOption func(*QdrantWriter)

func WithQdrantBatchSize(n int) QdrantOption {
	return func(w *QdrantWriter) { w.batchSize = n }
}

// NewQdrantWriter 创建 Qdrant 写入器
func NewQdrantWriter(host string, port int, collection, apiKey string, vectorDim int, opts ...QdrantOption) (*QdrantWriter, error) {
	client, err := qdrant.NewClient(&qdrant.Config{
		Host:   host,
		Port:   port,
		APIKey: apiKey,
	})
	if err != nil {
		return nil, fmt.Errorf("创建 Qdrant 客户端失败: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := ensureCollection(ctx, client, collection, uint64(vectorDim)); err != nil {
		return nil, fmt.Errorf("初始化集合失败: %w", err)
	}

	w := &QdrantWriter{
		client:     client,
		collection: collection,
		namespace:  uuid.MustParse("6ba7b810-9dad-11d1-80b4-00c04fd430c8"),
		batchSize:  64,
	}
	for _, opt := range opts {
		opt(w)
	}
	return w, nil
}

// Upsert 批量写入向量
func (w *QdrantWriter) Upsert(ctx context.Context, chunks []models.EmbeddedChunk) error {
	if len(chunks) == 0 {
		return nil
	}

	for i := 0; i < len(chunks); i += w.batchSize {
		end := i + w.batchSize
		if end > len(chunks) {
			end = len(chunks)
		}
		batch := chunks[i:end]

		points := make([]*qdrant.PointStruct, 0, len(batch))
		for _, chunk := range batch {
			point, err := w.toPoint(chunk)
			if err != nil {
				return fmt.Errorf("构造点失败 (docID=%s): %w", chunk.DocID, err)
			}
			points = append(points, point)
		}

		_, err := w.client.Upsert(ctx, &qdrant.UpsertPoints{
			CollectionName: w.collection,
			Points:         points,
		})
		if err != nil {
			return fmt.Errorf("写入 Qdrant 失败 (批次 %d-%d): %w", i, end, err)
		}
	}

	return nil
}

// Close 关闭连接
func (w *QdrantWriter) Close() error {
	return w.client.Close()
}

// toPoint 将 EmbeddedChunk 转为 Qdrant 点结构
func (w *QdrantWriter) toPoint(chunk models.EmbeddedChunk) (*qdrant.PointStruct, error) {
	pointID := uuid.NewSHA1(w.namespace, []byte(chunk.DocID+":"+strconv.Itoa(chunk.ChunkIndex)))

	payload := map[string]*qdrant.Value{
		"doc_id":      qdrant.NewValueString(chunk.DocID),
		"chunk_index": qdrant.NewValueInt(int64(chunk.ChunkIndex)),
		"content":     qdrant.NewValueString(truncate(chunk.Content, 200)),
		"timestamp":   qdrant.NewValueString(time.Now().Format(time.RFC3339)),
	}

	for k, v := range chunk.Metadata {
		payload[k] = qdrant.NewValueString(v)
	}

	vectors := make([]float32, len(chunk.Vector))
	copy(vectors, chunk.Vector)

	return &qdrant.PointStruct{
		Id:      qdrant.NewIDUUID(pointID.String()),
		Vectors: qdrant.NewVectors(vectors...),
		Payload: payload,
	}, nil
}

// ensureCollection 确保集合存在，不存在则自动创建
func ensureCollection(ctx context.Context, client *qdrant.Client, collection string, vectorDim uint64) error {
	exists, err := client.CollectionExists(ctx, collection)
	if err != nil {
		return fmt.Errorf("检查集合是否存在失败: %w", err)
	}

	if exists {
		return nil
	}

	err = client.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: collection,
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			Size:     vectorDim,
			Distance: qdrant.Distance_Cosine,
		}),
	})
	if err != nil {
		return fmt.Errorf("创建集合失败: %w", err)
	}

	return nil
}

// truncate 截断字符串
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen]
}

// ParseQdrantHost 将 "host:port" 格式的 URL 解析为 host 和 port
func ParseQdrantHost(addr string) (string, int) {
	parts := strings.Split(addr, ":")
	if len(parts) != 2 {
		return addr, 6334
	}
	host := parts[0]
	port := 6334
	if p, err := strconv.Atoi(parts[1]); err == nil {
		port = p
	}
	return host, port
}
