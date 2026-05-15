.PHONY: build test test-race lint run clean docker-build

APP_NAME := rag-flow
BUILD_DIR := build

build:
	@echo "编译 $(APP_NAME)..."
	go build -o $(BUILD_DIR)/$(APP_NAME) ./cmd/RAG-Flow/

test:
	go test -cover ./...

test-race:
	go test -race -cover ./...

test-integration:
	go test -race -tags=integration -cover ./...

lint:
	golangci-lint run ./...

run: build
	$(BUILD_DIR)/$(APP_NAME)

clean:
	rm -rf $(BUILD_DIR)

docker-build:
	docker build -t $(APP_NAME):latest .
