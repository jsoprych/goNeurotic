# GoNeurotic - Context Summary

## Project Overview
**GoNeurotic** is a production-ready neural network library written in Go, designed for both educational purposes and production use. The project has evolved from a simple 3-input AND gate demonstration to a full-featured neural network framework.

**Current Version**: v1.1.0 (Performance Optimization Release)

## Current State & Achievements

### ✅ Completed (v1.1.0)
1. **Performance Optimizations** (committed and tagged):
   - Pre-allocated buffer system for activations, derivatives, deltas, and weight updates
   - Derivative caching during forward pass
   - 50-70% reduction in memory allocations during training
   - 9-13% speedup for small networks

2. **Core Implementation** (`pkg/neural/network.go`):
   - Fully connected dense neural network
   - Multiple activation functions: Sigmoid, ReLU, Tanh, Linear
   - Multiple loss functions: Mean Squared Error, Binary Cross Entropy
   - Online, mini-batch, and full-batch training
   - Model serialization to JSON
   - Network cloning and learning rate management

3. **CLI Tool** (`cmd/goneurotic/main.go`):
   - Multiple built-in demos: XOR, AND gate, sine approximation, Iris classification, digit recognition
   - Command-line interface with training/testing options
   - Visualization support

4. **Comprehensive Documentation**:
   - Updated README.md with performance details and Mermaid diagrams
   - PERFORMANCE_REPORT.md with detailed analysis and visualizations
   - CHANGELOG.md following Keep a Changelog format
   - Makefile build system

### 🚧 In Progress (Blocked)
**BLAS Matrix Integration** (`pkg/neural/network_blas.go`):
- Goal: 10-50× performance improvement using BLAS libraries
- Status: **COMPILATION ERRORS** - BLAS API usage incorrect
- Issue: Using raw parameters instead of structured `blas64.General` and `blas64.Vector` types
- Files affected: `network_blas.go` (not committed due to errors)

## Technical Architecture

### Current Implementation (Working)
```go
// Network structure with optimization buffers
type Network struct {
    Weights        [][][]float64      // layer → neuron → input
    Biases         [][]float64        // layer → neuron
    activationBuffers  [][]float64    // pre-allocated activation storage
    derivativeBuffers  [][]float64    // pre-allocated derivative storage
    deltasBuffers      [][]float64    // pre-allocated delta storage
    weightUpdateBuffers [][][]float64 // batch training weight updates
    biasUpdateBuffers  [][]float64    // batch training bias updates
}
```

### BLAS Integration Goal
```go
// Target BLAS-optimized structure
type BLASOptimizer struct {
    weightFlatBuffers [][]float64  // layer → flat matrix (row-major)
    biasFlatBuffers   [][]float64  // layer → flat vector
    // BLAS operations: GEMV (matrix-vector), GER (rank-1 update), AXPY (vector add)
}
```

## Blocking Issues

### 1. BLAS API Misunderstanding
**Current (wrong):**
```go
blas64.Gemv(blas.NoTrans, fanOut, fanIn, 1.0,
    weightsFlat, fanIn, input, 1, 0.0, output, 1)
```

**Required (correct):**
```go
A := blas64.General{Rows: fanOut, Cols: fanIn, Data: weightsFlat, Stride: fanIn}
x := blas64.Vector{Data: input, Inc: 1}
y := blas64.Vector{Data: output, Inc: 1}
blas64.Gemv(blas.NoTrans, 1.0, A, x, 0.0, y)
```

### 2. Dependencies
- Go version: 1.24.0 (upgraded from 1.21)
- gonum.org/v1/gonum v0.17.0 added
- Build currently failing due to BLAS compilation errors

## Next Steps (Prioritized)

### 🥇 Immediate (Fix BLAS)
1. **Fix BLAS API calls** in `network_blas.go`:
   - Convert raw arrays to `blas64.General` and `blas64.Vector`
   - Fix all function calls: `Gemv`, `Ger`, `Axpy`
   - Test compilation

2. **Integration strategy** (choose one):
   - **Option A**: Replace core implementation with BLAS
   - **Option B**: Add BLAS as optional optimizer (`BLASNetwork` wrapper)
   - **Option C**: Use build tags for BLAS/non-BLAS versions

### 🥈 Medium Term (v1.2.0)
3. **Adam Optimizer Implementation**:
   - Add Optimizer interface
   - Implement Adam, SGD with momentum, RMSprop
   - 2-5× faster convergence expected

4. **Benchmark Comparison**:
   - Compare BLAS vs original performance
   - Document 10-50× speedup expectations

### 🥉 Long Term
5. **API Server** (v1.3.0):
   - REST API for model serving
   - Production deployment features
6. **Advanced Features**:
   - More activation functions (LeakyReLU, Softmax, GELU)
   - Regularization (L1/L2, dropout, batch normalization)
   - Convolutional/Recurrent network support

## Performance Expectations

| Operation | Current | With BLAS | Improvement |
|-----------|---------|-----------|-------------|
| FeedForward (medium) | ~27,000 ns/op | ~2,700 ns/op | **10×** |
| Train (medium) | ~46,000 ns/op | ~4,600 ns/op | **10×** |
| BatchTrain (64) | ~2.9M ns/op | ~290K ns/op | **10×** |
| Memory allocations | ~10 allocs/op | ~2 allocs/op | **5× reduction** |

## Git Status
- **Current branch**: `main`
- **Last commit**: `d6bd57a` (v1.1.0: performance optimizations)
- **Ahead of origin**: 1 commit
- **BLAS work**: Not committed (due to compilation errors)
- **Tags**: `v1.1.0` created

## Key Decisions Needed

### 1. BLAS Fix Approach
- Fix current `blas64` usage (recommended)
- OR switch to `gonum/mat` (higher-level, simpler)
- OR temporary workaround: comment out BLAS file

### 2. Integration Strategy
- Replace core vs wrapper vs build tags
- Backward compatibility requirements

### 3. Testing Strategy
- Keep original tests passing
- Add BLAS-specific tests
- Performance regression testing

## Restart Prompt Suggestions

When restarting, use prompts like:
- "Continue with BLAS integration fix"
- "Let's fix the BLAS API calls in network_blas.go"
- "What's the best approach to fix the BLAS compilation errors?"
- "Should we switch to gonum/mat instead of raw blas64?"
- "Let's implement the Adam optimizer next"

## File Structure
```
goNeurotic/
├── cmd/goneurotic/main.go          # CLI with demos
├── pkg/neural/
│   ├── network.go                  # Core implementation (working)
│   ├── network_blas.go             # BLAS integration (COMPILATION ERRORS)
│   ├── network_test.go             # Tests
│   └── network_benchmark_test.go   # Benchmarks
├── go.mod                          # Go 1.24 + gonum v0.17.0
├── README.md                       # Updated documentation
├── PERFORMANCE_REPORT.md           # Performance analysis
├── CHANGELOG.md                    # Version history
└── Makefile                        # Build system
```

## Quick Start Commands
```bash
# Test current implementation (exclude BLAS file)
go test ./pkg/neural -v

# Run XOR demo
./bin/goneurotic -demo xor

# Build project
make build

# Run benchmarks
make benchmark
```

---

**Last Updated**: BLAS integration started but blocked on API usage
**Next Priority**: Fix BLAS compilation errors for 10× performance improvement