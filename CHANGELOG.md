# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-02-02

### 🚀 Performance Optimization Release

This release introduces major performance optimizations to the neural network core, significantly reducing memory allocations and improving training speed through buffer reuse and derivative caching.

#### Added
- **Buffer reuse system**: Added pre-allocated buffers for activations, derivatives, deltas, and weight updates
- **Derivative caching**: Activation derivatives now computed during forward pass for efficiency
- **Performance benchmarking suite**: Comprehensive benchmarks for all core operations
- **Buffer management API**: `initializeBuffers()` and `ensureBuffers()` methods for internal buffer management
- **Performance report**: `PERFORMANCE_REPORT.md` with detailed analysis of optimization results

#### Changed
- **`Network` struct**: Added optimization buffer fields (not serialized):
  - `activationBuffers`: Pre-allocated activation storage
  - `derivativeBuffers`: Pre-allocated derivative storage  
  - `deltasBuffers`: Pre-allocated delta storage for backpropagation
  - `weightUpdateBuffers`: Batch training weight updates
  - `biasUpdateBuffers`: Batch training bias updates
  - `buffersInitialized`: Tracking flag for lazy initialization
- **`Train()` method**: Now uses optimized `feedForwardWithBuffers()` with derivative caching
- **`BatchTrain()` method**: Uses pre-allocated update buffers and optimized forward pass
- **`Predict()` method**: Uses optimized forward pass without derivative computation
- **`Load()` function**: Automatically initializes buffers when loading saved models
- **`Clone()` method**: Maintains buffer optimization in cloned networks

#### Performance Improvements

**Small Network (2-3-1 layers):**
- **FeedForward**: 1,581 ns/op (9.3% faster than previous version)
- **Train**: 2,504 ns/op (9.6% faster than previous version)
- **Predict**: 1,048 ns/op (12.9% faster than previous version)
- **BatchTrain (32 examples)**: 79,191 ns/op (13.9% faster than previous version)

**Memory Allocation Reduction:**
- Reduced allocations by 50-70% during training
- Batch training memory: From 1,536 B/op to ~500 B/op
- Allocation count: From 32 allocs/op to ~10 allocs/op

#### Technical Details

1. **Buffer Reuse Strategy**:
   - Activation buffers reused across forward passes
   - Delta buffers reused during backpropagation
   - Weight update buffers reset and reused across batch training
   - Derivative buffers filled during forward pass, consumed during backpropagation

2. **Derivative Computation Overlap**:
   - Activation derivatives computed when activation values are available
   - Eliminates redundant computation during backpropagation
   - Reduces overall computational overhead

3. **Memory Access Pattern Improvements**:
   - Better cache locality through buffer reuse
   - Reduced memory fragmentation
   - Lower garbage collection pressure

4. **API Compatibility**:
   - Public API remains unchanged
   - Serialization format unchanged (buffers excluded from JSON)
   - All existing code continues to work without modification

#### Fixed
- **Buffer initialization on load**: Buffers properly initialized when loading saved models
- **Memory safety**: Public `FeedForward()` method returns copies to prevent mutation issues
- **Edge cases**: Proper handling of variable-sized input buffers

#### Breaking Changes
- None. This release maintains full backward compatibility.

#### Migration Notes
- No migration required
- Existing saved models (`*.json`) remain compatible
- Performance improvements are automatic for all users
- Memory usage increase of 20-30% for buffer storage, offset by 50-70% reduction in dynamic allocations

#### Documentation Updates
- Updated README.md with performance optimization details and Mermaid diagrams
- Added `PERFORMANCE_REPORT.md` with comprehensive analysis and visualization
- Enhanced API documentation for new internal methods
- Added architecture diagrams illustrating neural network structure and buffer optimization system
- Added performance comparison visualizations showing before/after improvements

## [1.2.0] - 2025-02-03

### 🚀 BLAS Acceleration Release

This release introduces BLAS-accelerated matrix operations using the gonum BLAS64 library, providing dramatic performance improvements for medium to large networks while maintaining full compatibility.

#### Added
- **BLASOptimizer**: Core BLAS acceleration engine with flat buffer management
- **BLASNetwork**: BLAS-optimized network wrapper with automatic conversion
- **Global BLAS optimization**: Thread-safe global optimizer for consistent acceleration
- **BLAS integration tests**: Comprehensive test suite verifying numerical equivalence
- **BLAS performance benchmarks**: Specialized benchmarks for matrix operations
- **Benchmark history tracking**: `BENCHMARK_HISTORY.md` for performance tracking

#### Changed
- **Matrix operations**: Replaced nested loops with BLAS GEMV, GER, and AXPY calls
- **Buffer layout**: Added flat row-major buffers for BLAS compatibility
- **Performance characteristics**: Significantly faster matrix operations (5-9× speedup)
- **Network creation**: BLAS network creation is actually faster than regular creation

#### BLAS Performance Improvements

**Core Matrix Operations:**
- **Matrix-Vector (100×200)**: 26,971 ns/op → 4,304 ns/op (**6.3× faster**)
- **Rank-1 Update (100×200)**: 32,801 ns/op → 5,756 ns/op (**5.7× faster**)

**Forward Pass Performance:**
- **Small (10-20-10-5)**: 1,994 ns/op → 1,118 ns/op (**1.8× faster**)
- **Medium (50-100-50-10)**: 45,171 ns/op → 5,069 ns/op (**8.9× faster**)
- **Large (100-200-100-50-10)**: 136,861 ns/op → 17,673 ns/op (**7.7× faster**)

**Training Performance:**
- **Small (10-20-5)**: 3,132 ns/op → 1,732 ns/op (**1.8× faster**)
- **Medium (50-100-20)**: 56,908 ns/op → 8,227 ns/op (**6.9× faster**)

**Network Creation:**
- **Regular**: 295,554 ns/op → **BLAS**: 176,054 ns/op (**1.7× faster creation**)

#### Technical Details

1. **BLAS Integration Strategy**:
   - Uses gonum.org/v1/gonum/blas/blas64 for optimized BLAS operations
   - Maintains compatibility with existing Network API through BLASNetwork wrapper
   - Automatic conversion between jagged and flat buffer formats
   - Row-major layout with correct stride for BLAS compatibility

2. **Operation Optimizations**:
   - **GEMV**: Matrix-vector multiplication for forward and backward passes
   - **GER**: Rank-1 updates for weight optimization
   - **AXPY**: Vector addition for bias updates and accumulated gradients
   - **Transpose operations**: Efficient transposed matrix operations for backpropagation

3. **Memory Layout Improvements**:
   - Flat buffers reduce memory indirection
   - Row-major layout improves cache locality for BLAS operations
   - Conversion overhead minimized through caching and reuse

4. **API Compatibility**:
   - Existing Network API remains unchanged
   - BLAS acceleration optional through BLASNetwork type
   - Global BLAS optimization can be enabled for all networks
   - Full numerical equivalence with regular implementation

#### Fixed
- **BLAS API usage**: Corrected parameter ordering and structure usage for gonum BLAS64
- **Stride calculation**: Proper row-major stride for General matrices
- **Buffer synchronization**: Proper conversion between flat and jagged formats

#### Breaking Changes
- None. BLAS acceleration is opt-in through BLASNetwork type.

#### Migration Notes
- Existing code continues to work unchanged
- For BLAS acceleration, use `NewBLASNetwork()` instead of `NewNetwork()`
- Global BLAS optimization can be enabled with `EnableGlobalBLASOptimization()`
- Performance benefits increase with network size (minimal for very small networks)

#### Documentation Updates
- Updated `PERFORMANCE_REPORT.md` with BLAS acceleration results
- Added `BENCHMARK_HISTORY.md` for tracking performance across versions
- Enhanced API documentation for BLAS optimization features
- Added integration tests demonstrating BLAS vs regular equivalence

## [1.0.0] - Initial Release

### Features
- Complete neural network implementation with configurable architecture
- Multiple activation functions (Sigmoid, ReLU, Tanh, Linear)
- Multiple loss functions (Mean Squared Error, Binary Cross Entropy)
- Online, mini-batch, and full-batch training
- Model serialization to JSON format
- Network cloning and learning rate management
- Comprehensive test suite with 90%+ coverage
- CLI tool with multiple demos (XOR, AND, sine approximation, digit recognition, Iris classification)
- Production-ready error handling and validation

### Technical Foundation
- Clean Go architecture with separation of concerns
- Proper error handling and panic recovery
- Comprehensive test coverage
- Build system with cross-compilation support
- Code quality tools (linting, formatting, vetting)

---
*Changelog format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)*