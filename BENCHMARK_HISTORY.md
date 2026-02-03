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

### Cumulative Speedup from v1.0.0 to v1.2.0

| Operation | Network Size | v1.0.0 → v1.1.0 | v1.1.0 → v1.2.0 | Total Improvement |
|-----------|--------------|-----------------|-----------------|-------------------|
| FeedForward | Medium | ~10% | 8.9× | **~9.8× total** |
| Training | Medium | ~10% | 6.9× | **~7.6× total** |
| Matrix Operations | 100×200 | N/A | 6.3× | **6.3× total** |

### Key Insights

1. **BLAS impact scales with network size**: Small networks see 1.8× improvement, medium networks see 6-9× improvement
2. **Memory optimization still valuable**: BLAS acceleration works synergistically with buffer reuse
3. **Batch training needs optimization**: Current BLAS implementation doesn't optimize batch operations yet
4. **Creation overhead reduced**: BLAS network creation is actually faster due to optimized buffer allocation

---

## Future Benchmark Targets

### Priority 1: Batch Training Optimization
- Goal: Apply BLAS optimization to batch training operations
- Target: 3-5× speedup for batch training
- Method: Implement BLAS-optimized `BatchTrainBLAS` method

### Priority 2: Adam Optimizer Integration  
- Goal: Implement Adam optimizer with BLAS acceleration
- Target: 2-5× faster convergence (iterations to target accuracy)
- Method: Add Optimizer interface with Adam, SGD+Momentum, RMSprop

### Priority 3: Memory Usage Optimization
- Goal: Further reduce memory allocations
- Target: <5 allocations per training iteration
- Method: Pool-based buffer management, zero-copy operations

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