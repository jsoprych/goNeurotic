# GoNeurotic - A Production-Ready Neural Network Library
# Makefile for building, testing, and running

# Variables
BINARY_NAME = goneurotic
PACKAGE = goneurotic
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "v0.0.0-dev")
BUILD_TIME = $(shell date -u '+%Y-%m-%d_%H:%M:%S')
COMMIT_HASH = $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
LDFLAGS = -ldflags="-X 'main.version=$(VERSION)' -X 'main.buildTime=$(BUILD_TIME)' -X 'main.commitHash=$(COMMIT_HASH)'"
GO = go
GOFMT = gofmt
GOLINT = golangci-lint

# Directories
CMD_DIR = cmd/goneurotic
PKG_DIR = pkg/neural
BIN_DIR = bin
DIST_DIR = dist

# Default target
all: build

# Build the CLI tool
build:
	@echo "Building $(BINARY_NAME)..."
	$(GO) build $(LDFLAGS) -o $(BIN_DIR)/$(BINARY_NAME) ./$(CMD_DIR)

# Install globally
install:
	@echo "Installing $(BINARY_NAME)..."
	$(GO) install $(LDFLAGS) ./$(CMD_DIR)

# Run the CLI tool with default (XOR demo)
run: build
	@echo "Running XOR demo..."
	./$(BIN_DIR)/$(BINARY_NAME)

# Run tests
test:
	@echo "Running tests..."
	$(GO) test ./... -v

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	$(GO) test ./... -coverprofile=coverage.out
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

# Run benchmarks
benchmark:
	@echo "Running benchmarks..."
	$(GO) test -bench=. -benchmem ./$(PKG_DIR)

# Format Go code
fmt:
	@echo "Formatting Go code..."
	$(GOFMT) -w -s ./$(CMD_DIR) ./$(PKG_DIR)

# Lint Go code (requires golangci-lint)
lint:
	@echo "Running linter..."
	@if command -v $(GOLINT) >/dev/null 2>&1; then \
		$(GOLINT) run ./...; \
	else \
		echo "golangci-lint not installed. Install with: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest"; \
	fi

# Run static analysis
vet:
	@echo "Running go vet..."
	$(GO) vet ./...

# Generate dependency graph
deps:
	@echo "Generating dependency graph..."
	$(GO) mod graph | dot -Tpng -o deps.png
	@echo "Dependency graph saved to deps.png"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BIN_DIR) $(DIST_DIR) coverage.out coverage.html deps.png
	rm -f *.json *.csv *.png *.pdf

# Create directories
dirs:
	@mkdir -p $(BIN_DIR) $(DIST_DIR)

# Run all demos
demo: build
	@echo "Running all demos..."
	@echo "=== AND Gate Demo ==="
	./$(BIN_DIR)/$(BINARY_NAME) -demo and
	@echo ""
	@echo "=== XOR Demo ==="
	./$(BIN_DIR)/$(BINARY_NAME) -demo xor
	@echo ""
	@echo "=== Sine Function Demo ==="
	./$(BIN_DIR)/$(BINARY_NAME) -demo sin
	@echo ""
	@echo "=== Iris Classification Demo ==="
	./$(BIN_DIR)/$(BINARY_NAME) -demo iris
	@echo ""
	@echo "=== Complex Pattern Demo ==="
	./$(BIN_DIR)/$(BINARY_NAME) -demo complex

# Run specific demo
demo-%: build
	@echo "Running $(subst demo-,,$@) demo..."
	./$(BIN_DIR)/$(BINARY_NAME) -demo $(subst demo-,,$@)

# Run benchmarks with the CLI
bench: build
	@echo "Running CLI benchmarks..."
	./$(BIN_DIR)/$(BINARY_NAME) -benchmark

# Generate visualization for demos
visualize: build
	@echo "Generating visualizations..."
	./$(BIN_DIR)/$(BINARY_NAME) -demo xor -visualize
	./$(BIN_DIR)/$(BINARY_NAME) -demo sin -visualize
	./$(BIN_DIR)/$(BINARY_NAME) -demo complex -visualize
	@echo "Visualization files: xor_visualization.csv, sine_visualization.csv, complex_visualization.csv"

# Build for multiple platforms (cross-compilation)
release: clean dirs
	@echo "Building release binaries..."
	@echo "Building for Linux..."
	GOOS=linux GOARCH=amd64 $(GO) build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)_linux_amd64 ./$(CMD_DIR)
	GOOS=linux GOARCH=arm64 $(GO) build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)_linux_arm64 ./$(CMD_DIR)
	@echo "Building for macOS..."
	GOOS=darwin GOARCH=amd64 $(GO) build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)_darwin_amd64 ./$(CMD_DIR)
	GOOS=darwin GOARCH=arm64 $(GO) build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)_darwin_arm64 ./$(CMD_DIR)
	@echo "Building for Windows..."
	GOOS=windows GOARCH=amd64 $(GO) build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)_windows_amd64.exe ./$(CMD_DIR)
	@echo "Release binaries created in $(DIST_DIR)/"

# Create checksums for release binaries
checksums:
	@echo "Creating checksums..."
	cd $(DIST_DIR) && sha256sum * > checksums.txt
	@echo "Checksums saved to $(DIST_DIR)/checksums.txt"

# Show help
help:
	@echo "GoNeurotic - Makefile targets:"
	@echo ""
	@echo "  build         - Build the CLI tool"
	@echo "  install       - Install globally"
	@echo "  run           - Run the CLI (XOR demo)"
	@echo "  test          - Run all tests"
	@echo "  test-coverage - Run tests with coverage report"
	@echo "  benchmark     - Run benchmarks"
	@echo "  fmt           - Format Go code"
	@echo "  lint          - Lint Go code (requires golangci-lint)"
	@echo "  vet           - Run go vet"
	@echo "  deps          - Generate dependency graph"
	@echo "  clean         - Clean build artifacts"
	@echo "  demo          - Run all demos"
	@echo "  demo-<name>   - Run specific demo (and, xor, sin, iris, complex)"
	@echo "  bench         - Run CLI benchmarks"
	@echo "  visualize     - Generate visualization data for demos"
	@echo "  release       - Build for multiple platforms"
	@echo "  checksums     - Create checksums for release binaries"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make demo-xor    # Run XOR demo"
	@echo "  make demo-and    # Run AND gate demo"
	@echo "  make release     # Build for all platforms"
	@echo ""

# Phony targets
.PHONY: all build install run test test-coverage benchmark fmt lint vet deps clean demo visualize release checksums help
