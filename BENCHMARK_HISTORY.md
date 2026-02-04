# GoNeurotic Benchmark History

## Purpose

This document tracks performance benchmarks across different versions of the GoNeurotic neural network library. It serves as a historical record of optimization efforts and helps identify performance regressions or improvements.

## Benchmark Methodology

- **Hardware**: Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz (8 cores)
- **Go Version**: 1.24.0
- **Benchmark Tool**: Go's built-in testing framework with `-benchtime=1s`
- **Timing**: Nanoseconds per operation (lower is better)
- **Comparison**: All benchmarks compare operations with identical inputs and network configurations

## Version History

| Version | Date | Key Changes | Performance Focus |
|---------|------|-------------|-------------------|
| v1.1.0 | 2025-02-02 | Buffer reuse, derivative caching | Memory allocation reduction, 9-13% speedup |
| v1.2.0 | 2025-02-03 | BLAS acceleration (GEMV, GER, AXPY) | Matrix operation speed (5-9x improvement) |
| v1.3.0 | 2026-02-04 | BLAS batch training, optimizer system (SGD/Momentum/RMSprop/Adam), state serialization | Batch training speed (3-8x), faster convergence (2.8x) |

---

## v1.2.0 - BLAS Acceleration (2025-02-03)

### Core Matrix Operations

| Operation | Regular (ns/op) | BLAS (ns/op) | Speedup | Notes |
|-----------|-----------------|--------------|---------|-------|
| Matrix-Vector (100×200) | 26,971 | 4,304 | **6.3×** | Core GEMV operation |
| Rank-1 Update (100×200) | 32,801 | 5,756 | **5.7×** | Core GER operation |

### Forward Pass Performance

| Network Size | Regular (ns/op) | BLAS (ns/op) | Speedup |
|--------------|-----------------|--------------|---------|
| Small (10-20-10-5) | 1,994 | 1,118 | **1.8×** (44% faster) |
| Medium (50-100-50-10) | 45,171 | 5,069 | **8.9×** |
| Large (100-200-100-50-10) | 136,861 | 17,673 | **7.7×** |

### Training Performance (Forward + Backward Pass)

| Network Size | Regular (ns/op) | BLAS (ns/op) | Speedup |
|--------------|-----------------|--------------|---------|
| Small (10-20-5) | 3,132 | 1,732 | **1.8×** (45% faster) |
| Medium (50-100-20) | 56,908 | 8,227 | **6.9×** |

### Batch Training Performance

| Network Size | Batch Size | Regular (ns/op) | BLAS (ns/op) | Speedup |
|--------------|------------|-----------------|--------------|---------|
| Small (10-20-5) | 32 | 81,446 | 83,695 | **0.97×** (Note: BLAS batch training not optimized yet) |

### Network Creation Overhead

| Network Type | Time (ns/op) | Notes |
|--------------|--------------|-------|
| Regular | 295,554 | Base network creation |
| BLAS | 176,054 | **1.7× faster** creation |

---

## v1.3.0 - BLAS Batch Training & Optimizer System (2026-02-04)

### BLAS Batch Training Performance (Newly Optimized)

| Network Size | Batch Size | Regular (ns/op) | BLAS (ns/op) | Speedup | Notes |
|--------------|------------|-----------------|--------------|---------|-------|
| Small (10-20-5) | 32 | 116,878 | 39,888 | **2.9×** | BatchTrainBLAS implementation |
| Medium (50-100-20) | 64 | 3,483,555 | 448,494 | **7.8×** | Significant BLAS acceleration |
| Large (50-100-20) | 256 | 13,976,786 | 1,786,290 | **7.8×** | Scales well with batch size |

### Core BLAS Operations Performance

| Operation | Regular (ns/op) | BLAS (ns/op) | Speedup | Matrix Size |
|-----------|-----------------|--------------|---------|-------------|
| Matrix-Vector (GEMV) | 24,504 | 3,744 | **6.5×** | 100×200 |
| Rank-1 Update (GER) | 28,954 | 4,704 | **6.2×** | 100×200 |

### Optimizer System Performance

| Optimizer Type | Convergence (epochs to 95% XOR) | Relative Speed | Memory Overhead |
|----------------|---------------------------------|----------------|-----------------|
| SGD | ~5,000 | 1.0× (baseline) | Low |
| SGD+Momentum | ~3,500 | 1.4× faster | Medium |
| RMSprop | ~2,500 | 2.0× faster | Medium |
| Adam | ~1,800 | 2.8× faster | High |

### Training Performance with BLAS + Optimizers

| Network Size | Operation | Regular SGD (ns/op) | BLAS Adam (ns/op) | Combined Speedup |
|--------------|-----------|---------------------|-------------------|------------------|
| Medium | Single Train | 54,553 | 6,080 | **9.0×** |
| Medium | Batch Train (64) | 3,483,555 | 448,494 | **7.8×** |

### Memory Allocation Improvements

| Metric | v1.2.0 | v1.3.0 | Reduction |
|--------|--------|--------|-----------|
| Batch training allocations | ~10 allocs/op | ~6 allocs/op | **40% reduction** |
| Memory per batch op | ~500 B/op | ~350 B/op | **30% reduction** |
| Optimizer state memory | N/A | ~150 B/op | New feature |

---

## v1.1.0 - Buffer Optimization (2025-02-02)

### Small Network (2-3-1 layers)

| Operation | Before (ns/op) | After (ns/op) | Improvement |
|-----------|----------------|---------------|-------------|
| FeedForward | ~1,745 | 1,581 | **9.3% faster** |
| Train | ~2,770 | 2,504 | **9.6% faster** |
| Predict | ~1,203 | 1,048 | **12.9% faster** |
| BatchTrain (32 examples) | ~91,866 | 79,191 | **13.9% faster** |

### Memory Allocation Reduction

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Batch training memory | 1,536 B/op | ~500 B/op | **~67% less** |
| Allocation count | ~32 allocs/op | ~10 allocs/op | **~69% fewer allocations** |

---

## Performance Improvement Summary

### Cumulative Speedup from v1.0.0 to v1.3.0

| Operation | Network Size | v1.0.0 → v1.1.0 | v1.1.0 → v1.2.0 | v1.2.0 → v1.3.0 | Total Improvement |
|-----------|--------------|-----------------|-----------------|-----------------|-------------------|
| FeedForward | Medium | ~10% | 8.9× | 1.1× (BLAS refinement) | **~9.8× total** |
| Single Training | Medium | ~10% | 6.9× | 1.3× (optimizers) | **~9.9× total** |
| Batch Training | Medium (64) | ~14% | 1.0× (no BLAS) | 7.8× (BLAS batch) | **~8.9× total** |
| Matrix Operations | 100×200 | N/A | 6.3× | 1.0× (stable) | **6.3× total** |
| Convergence | XOR problem | N/A | N/A | 2.8× (Adam) | **2.8× total** |

### Key Insights (v1.3.0 Update)

1. **BLAS batch training optimization achieved**: Medium networks now see 7.8× speedup for batch training (64 examples), exceeding the 3-5× target
2. **Optimizer system delivers faster convergence**: Adam optimizer achieves 2.8× faster convergence on XOR compared to baseline SGD
3. **BLAS impact scales with both network and batch size**: Larger networks and batches benefit most from BLAS acceleration (7-9× improvement)
4. **Memory optimizations compound**: BLAS acceleration + buffer reuse + flat buffers reduce memory usage by 30% compared to v1.2.0
5. **State serialization enables training continuity**: Optimizer momentum/cache states can now be saved and restored, enabling experiment checkpointing
6. **Optimizer choice matters for convergence**: Adam > RMSprop > SGD+Momentum > SGD for convergence speed (2.8× → 2.0× → 1.4× → 1.0×)

---

## Future Benchmark Targets (Post-v1.3.0)

### ✅ Completed in v1.3.0:
- **Batch Training BLAS Optimization**: Achieved 7.8× speedup (exceeded 3-5× target)
- **Adam Optimizer System**: Implemented full optimizer interface with SGD, Momentum, RMSprop, Adam
- **Optimizer State Serialization**: Added state save/restore for training continuity

### 🎯 Priority 1: API Server & Production Features (v1.4.0)
- Goal: Add REST API for model serving and production deployment
- Target: <10ms inference latency, concurrent request support
- Method: HTTP server with model loading, prediction endpoints, monitoring

### 🎯 Priority 2: Advanced Network Architectures
- Goal: Support convolutional (CNN) and recurrent (RNN) networks
- Target: 5-10× faster training than naive implementations
- Method: BLAS-optimized convolution operations, LSTM/GRU cells

### 🎯 Priority 3: GPU Acceleration Exploration
- Goal: Investigate GPU acceleration using CUDA or OpenCL
- Target: 10-50× speedup for large networks on supported hardware
- Method: Integrate with GPU BLAS libraries (cuBLAS, clBLAS)

### 🎯 Priority 4: Advanced Optimizers & Regularization
- Goal: Add more optimizers (AdaGrad, AdaDelta, Nadam) and regularization (L1/L2, dropout)
- Target: 10-20% better generalization on complex datasets
- Method: Extend optimizer interface, add regularization layers

---

## How to Run Benchmarks

```bash
# Run all benchmarks
go test -bench=. -benchtime=1s ./pkg/neural

# Run only BLAS benchmarks
go test -bench="BLAS" -benchtime=1s ./pkg/neural

# Run with memory profiling
go test -bench=. -benchmem ./pkg/neural

# Compare two runs (requires benchstat)
go test -bench=. -count=5 ./pkg/neural > new.txt
go test -bench=. -count=5 ./pkg/neural > old.txt
benchstat old.txt new.txt
```

---

## Notes on Benchmark Consistency

1. **Warm-up cycles**: Go's benchmark framework includes warm-up, but for very consistent results, run with `-count=5`
2. **System variance**: Benchmarks run on same hardware but system load can affect results
3. **Statistical significance**: Differences <5% may not be statistically significant
4. **Memory benchmarks**: Use `-benchmem` flag for allocation statistics

---

*Last Updated: 2025-02-03*
*Next Planned Benchmark: After Adam optimizer implementation*