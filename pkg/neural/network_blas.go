package neural

import (
	"fmt"
	"sync"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// BLASOptimizer provides BLAS-accelerated operations for neural networks
type BLASOptimizer struct {
	// Flat buffers for BLAS operations
	weightFlatBuffers [][]float64 // layer → flat matrix (row-major: rows = neurons, cols = inputs)
	biasFlatBuffers   [][]float64 // layer → flat vector

	// Temporary buffers to avoid allocations
	gemvYBuffers      [][]float64 // output buffers for GEMV
	gemvXTmpBuffers   [][]float64 // temporary input buffers
	gerTmpBuffers     [][]float64 // temporary buffers for GER

	// BLAS implementation (use blas64.Implementation())
}

// NewBLASOptimizer creates a new BLAS optimizer for the given network architecture
func NewBLASOptimizer(layerSizes []int) *BLASOptimizer {
	if len(layerSizes) < 2 {
		panic("network must have at least 2 layers")
	}

	optimizer := &BLASOptimizer{}

	numLayers := len(layerSizes)
	numWeightLayers := numLayers - 1

	// Initialize flat buffers
	optimizer.weightFlatBuffers = make([][]float64, numWeightLayers)
	optimizer.biasFlatBuffers = make([][]float64, numWeightLayers)
	optimizer.gemvYBuffers = make([][]float64, numWeightLayers)
	optimizer.gemvXTmpBuffers = make([][]float64, numWeightLayers)
	optimizer.gerTmpBuffers = make([][]float64, numWeightLayers)

	for i := 0; i < numWeightLayers; i++ {
		fanIn := layerSizes[i]
		fanOut := layerSizes[i+1]

		// Weight matrix: fanOut × fanIn (row-major)
		optimizer.weightFlatBuffers[i] = make([]float64, fanOut*fanIn)

		// Bias vector: fanOut
		optimizer.biasFlatBuffers[i] = make([]float64, fanOut)

		// Output buffer for GEMV: fanOut
		optimizer.gemvYBuffers[i] = make([]float64, fanOut)

		// Temporary input buffer (for transposed operations): fanIn
		optimizer.gemvXTmpBuffers[i] = make([]float64, fanIn)

		// Temporary buffer for GER updates: fanOut
		optimizer.gerTmpBuffers[i] = make([]float64, fanOut)
	}

	return optimizer
}

// ConvertWeightsToFlat converts jagged weight arrays to flat BLAS-friendly format
func (b *BLASOptimizer) ConvertWeightsToFlat(layer int, weights [][]float64) {
	fanOut := len(weights)
	if fanOut == 0 {
		return
	}
	fanIn := len(weights[0])
	flat := b.weightFlatBuffers[layer]

	// Convert to row-major format
	for i := 0; i < fanOut; i++ {
		rowOffset := i * fanIn
		for j := 0; j < fanIn; j++ {
			flat[rowOffset+j] = weights[i][j]
		}
	}
}

// ConvertWeightsFromFlat converts flat BLAS weights back to jagged format
func (b *BLASOptimizer) ConvertWeightsFromFlat(layer int, weights [][]float64) {
	fanOut := len(weights)
	if fanOut == 0 {
		return
	}
	fanIn := len(weights[0])
	flat := b.weightFlatBuffers[layer]

	// Convert from row-major format
	for i := 0; i < fanOut; i++ {
		rowOffset := i * fanIn
		for j := 0; j < fanIn; j++ {
			weights[i][j] = flat[rowOffset+j]
		}
	}
}

// ConvertBiasesToFlat copies biases to flat buffer
func (b *BLASOptimizer) ConvertBiasesToFlat(layer int, biases []float64) {
	copy(b.biasFlatBuffers[layer], biases)
}

// ConvertBiasesFromFlat copies biases from flat buffer
func (b *BLASOptimizer) ConvertBiasesFromFlat(layer int, biases []float64) {
	copy(biases, b.biasFlatBuffers[layer])
}

// ForwardPassBLAS performs a forward pass using BLAS-optimized operations
// Returns output activations for the layer
func (b *BLASOptimizer) ForwardPassBLAS(layer int, input []float64, output []float64,
	activationFunc func(float64) float64) []float64 {

	fanIn := len(b.gemvXTmpBuffers[layer])
	fanOut := len(output)

	if len(input) != fanIn {
		panic(fmt.Sprintf("input size mismatch: expected %d, got %d", fanIn, len(input)))
	}
	if len(output) != fanOut {
		panic(fmt.Sprintf("output size mismatch: expected %d, got %d", fanOut, len(output)))
	}

	// Get flat weight matrix
	weightsFlat := b.weightFlatBuffers[layer]

	// y = 1.0 * A * x + 0.0 * y (initialize output to zero)
	// First, set output to zero
	for i := range output {
		output[i] = 0.0
	}

	// Perform matrix-vector multiplication: y = A * x
	// Create BLAS structures
	weightsMat := blas64.General{
		Rows:   fanOut,
		Cols:   fanIn,
		Data:   weightsFlat,
		Stride: fanIn,
	}
	inputVec := blas64.Vector{
		N:    fanIn,
		Data: input,
		Inc:  1,
	}
	outputVec := blas64.Vector{
		N:    fanOut,
		Data: output,
		Inc:  1,
	}
	blas64.Gemv(blas.NoTrans, 1.0, weightsMat, inputVec, 0.0, outputVec)

	// Add bias: y = y + b
	biasVec := blas64.Vector{
		N:    fanOut,
		Data: b.biasFlatBuffers[layer],
		Inc:  1,
	}
	blas64.Axpy(1.0, biasVec, outputVec)

	// Apply activation function
	if activationFunc != nil {
		for i := range output {
			output[i] = activationFunc(output[i])
		}
	}

	return output
}

// BackwardPassBLAS computes deltas for backpropagation using BLAS
// deltaCurrent = f'(activationCurrent) ⊙ (W_nextᵀ * deltaNext)
func (b *BLASOptimizer) BackwardPassBLAS(currentLayer, nextLayer int,
	deltaNext []float64, activationCurrent []float64,
	derivativeFunc func(float64) float64, deltaCurrent []float64) []float64 {

	fanIn := len(activationCurrent)
	fanOut := len(deltaNext)

	if len(deltaCurrent) != fanIn {
		panic(fmt.Sprintf("deltaCurrent size mismatch: expected %d, got %d", fanIn, len(deltaCurrent)))
	}

	// Get weight matrix for next layer (W_next)
	weightsNextFlat := b.weightFlatBuffers[nextLayer]

	// Initialize deltaCurrent to zero
	for i := range deltaCurrent {
		deltaCurrent[i] = 0.0
	}

	// Compute: deltaCurrent = W_nextᵀ * deltaNext
	// Using transpose of weight matrix
	// Create BLAS structures
	weightsNextMat := blas64.General{
		Rows:   fanOut,
		Cols:   fanIn,
		Data:   weightsNextFlat,
		Stride: fanIn,
	}
	deltaNextVec := blas64.Vector{
		N:    fanOut,
		Data: deltaNext,
		Inc:  1,
	}
	deltaCurrentVec := blas64.Vector{
		N:    fanIn,
		Data: deltaCurrent,
		Inc:  1,
	}
	blas64.Gemv(blas.Trans, 1.0, weightsNextMat, deltaNextVec, 0.0, deltaCurrentVec)

	// Element-wise multiply with derivative: deltaCurrent = deltaCurrent ⊙ f'(activationCurrent)
	if derivativeFunc != nil {
		for i := range deltaCurrent {
			deltaCurrent[i] *= derivativeFunc(activationCurrent[i])
		}
	}

	return deltaCurrent
}

// UpdateWeightsBLAS performs weight updates using BLAS rank-1 update (GER)
// W = W - learningRate * (delta ⊗ activation_prev)
func (b *BLASOptimizer) UpdateWeightsBLAS(layer int, learningRate float64,
	delta []float64, activationPrev []float64) {

	fanOut := len(delta)
	fanIn := len(activationPrev)

	// Get flat weight matrix
	weightsFlat := b.weightFlatBuffers[layer]

	// Perform rank-1 update: A = A + alpha * x * yᵀ
	// where alpha = -learningRate, x = delta, y = activationPrev
	// This computes: W = W - learningRate * delta * activationPrevᵀ
	// Create BLAS structures
	weightsMat := blas64.General{
		Rows:   fanOut,
		Cols:   fanIn,
		Data:   weightsFlat,
		Stride: fanIn,
	}
	deltaVec := blas64.Vector{
		N:    fanOut,
		Data: delta,
		Inc:  1,
	}
	activationPrevVec := blas64.Vector{
		N:    fanIn,
		Data: activationPrev,
		Inc:  1,
	}
	blas64.Ger(-learningRate, deltaVec, activationPrevVec, weightsMat)
}

// UpdateBiasesBLAS performs bias updates using BLAS AXPY
// b = b - learningRate * delta
func (b *BLASOptimizer) UpdateBiasesBLAS(layer int, learningRate float64, delta []float64) {
	fanOut := len(delta)

	// b = b - learningRate * delta
	deltaVec := blas64.Vector{
		N:    fanOut,
		Data: delta,
		Inc:  1,
	}
	biasVec := blas64.Vector{
		N:    fanOut,
		Data: b.biasFlatBuffers[layer],
		Inc:  1,
	}
	blas64.Axpy(-learningRate, deltaVec, biasVec)
}

// BatchUpdateWeightsBLAS accumulates weight updates across a batch
// weightUpdates accumulates: Σ (delta ⊗ activation_prev) for each example
func (b *BLASOptimizer) BatchUpdateWeightsBLAS(layer int, learningRate float64,
	delta []float64, activationPrev []float64, weightUpdates []float64) {

	fanOut := len(delta)
	fanIn := len(activationPrev)

	// Accumulate: weightUpdates += delta * activationPrevᵀ
	// Create BLAS structures
	weightUpdatesMat := blas64.General{
		Rows:   fanOut,
		Cols:   fanIn,
		Data:   weightUpdates,
		Stride: fanIn,
	}
	deltaVec := blas64.Vector{
		N:    fanOut,
		Data: delta,
		Inc:  1,
	}
	activationPrevVec := blas64.Vector{
		N:    fanIn,
		Data: activationPrev,
		Inc:  1,
	}
	blas64.Ger(1.0, deltaVec, activationPrevVec, weightUpdatesMat)
}

// ApplyBatchWeightUpdatesBLAS applies accumulated weight updates
// W = W - (learningRate/batchSize) * weightUpdates
func (b *BLASOptimizer) ApplyBatchWeightUpdatesBLAS(layer int, learningRate, batchSize float64,
	weightUpdates []float64) {

	fanOut := len(b.gerTmpBuffers[layer])
	fanIn := len(weightUpdates) / fanOut
	if len(weightUpdates) != fanOut*fanIn {
		panic("weightUpdates size mismatch")
	}

	// Scale factor: -learningRate / batchSize
	scale := -learningRate / batchSize

	// Get flat weight matrix
	weightsFlat := b.weightFlatBuffers[layer]

	// W = W + scale * weightUpdates
	// Use AXPY on the flat arrays (treating matrices as vectors)
	totalElements := fanOut * fanIn
	weightUpdatesVec := blas64.Vector{
		N:    totalElements,
		Data: weightUpdates,
		Inc:  1,
	}
	weightsFlatVec := blas64.Vector{
		N:    totalElements,
		Data: weightsFlat,
		Inc:  1,
	}
	blas64.Axpy(scale, weightUpdatesVec, weightsFlatVec)

	// Reset weightUpdates to zero for next batch
	for i := range weightUpdates {
		weightUpdates[i] = 0.0
	}
}

// ApplyBatchBiasUpdatesBLAS applies accumulated bias updates
// b = b - (learningRate/batchSize) * biasUpdates
func (b *BLASOptimizer) ApplyBatchBiasUpdatesBLAS(layer int, learningRate, batchSize float64,
	biasUpdates []float64) {

	fanOut := len(biasUpdates)

	// b = b - (learningRate/batchSize) * biasUpdates
	scale := -learningRate / batchSize
	biasUpdatesVec := blas64.Vector{
		N:    fanOut,
		Data: biasUpdates,
		Inc:  1,
	}
	biasVec := blas64.Vector{
		N:    fanOut,
		Data: b.biasFlatBuffers[layer],
		Inc:  1,
	}
	blas64.Axpy(scale, biasUpdatesVec, biasVec)

	// Reset biasUpdates to zero for next batch
	for i := range biasUpdates {
		biasUpdates[i] = 0.0
	}
}

// BLASNetwork wraps a Network with BLAS optimization
type BLASNetwork struct {
	*Network
	optimizer *BLASOptimizer
	// Cache for converted data
	weightsConverted bool
}

// NewBLASNetwork creates a new BLAS-optimized network
func NewBLASNetwork(config NetworkConfig) *BLASNetwork {
	network := NewNetwork(config)
	blasNetwork := &BLASNetwork{
		Network:   network,
		optimizer: NewBLASOptimizer(config.LayerSizes),
	}

	// Convert initial weights and biases to flat format
	blasNetwork.ConvertToFlat()

	return blasNetwork
}

// ConvertToFlat converts all weights and biases to flat BLAS format
func (bn *BLASNetwork) ConvertToFlat() {
	for i := range bn.Weights {
		bn.optimizer.ConvertWeightsToFlat(i, bn.Weights[i])
		bn.optimizer.ConvertBiasesToFlat(i, bn.Biases[i])
	}
	bn.weightsConverted = true
}

// ConvertFromFlat converts all weights and biases from flat BLAS format back to jagged
func (bn *BLASNetwork) ConvertFromFlat() {
	for i := range bn.Weights {
		bn.optimizer.ConvertWeightsFromFlat(i, bn.Weights[i])
		bn.optimizer.ConvertBiasesFromFlat(i, bn.Biases[i])
	}
}

// FeedForwardBLAS performs a forward pass using BLAS optimization
func (bn *BLASNetwork) FeedForwardBLAS(input []float64) ([]float64, [][]float64) {
	if len(input) != bn.LayerSizes[0] {
		panic(fmt.Sprintf("Input size mismatch: expected %d, got %d", bn.LayerSizes[0], len(input)))
	}

	if !bn.weightsConverted {
		bn.ConvertToFlat()
	}

	// Create activation storage
	activations := make([][]float64, len(bn.LayerSizes))
	activations[0] = make([]float64, len(input))
	copy(activations[0], input)

	// Forward pass through each layer using BLAS
	for i := 0; i < len(bn.LayerSizes)-1; i++ {
		layerSize := bn.LayerSizes[i+1]
		layerActivations := make([]float64, layerSize)

		// Choose activation function
		var activationFunc func(float64) float64
		if i == len(bn.LayerSizes)-2 {
			activationFunc = bn.OutputActivation.Function
		} else {
			activationFunc = bn.Activation.Function
		}

		// Perform BLAS-optimized forward pass
		bn.optimizer.ForwardPassBLAS(i, activations[i], layerActivations, activationFunc)
		activations[i+1] = layerActivations
	}

	return activations[len(activations)-1], activations
}

// TrainBLAS trains the network on a single example using BLAS optimization
func (bn *BLASNetwork) TrainBLAS(input, target []float64) float64 {
	if !bn.weightsConverted {
		bn.ConvertToFlat()
	}

	// Forward pass
	output, activations := bn.FeedForwardBLAS(input)

	// Calculate loss
	loss := bn.LossFunction.Function(output, target)

	// Backward pass
	outputErrors := bn.LossFunction.Derivative(output, target)

	// Calculate deltas for each layer
	deltas := make([][]float64, len(bn.LayerSizes)-1)
	lastLayer := len(bn.LayerSizes) - 2

	// Output layer delta
	deltas[lastLayer] = make([]float64, bn.LayerSizes[lastLayer+1])
	for i := range deltas[lastLayer] {
		activation := activations[lastLayer+1][i]
		var derivative float64
		if lastLayer == len(bn.LayerSizes)-2 {
			derivative = bn.OutputActivation.Derivative(activation)
		} else {
			derivative = bn.Activation.Derivative(activation)
		}
		deltas[lastLayer][i] = outputErrors[i] * derivative
	}

	// Hidden layer deltas
	for layer := lastLayer - 1; layer >= 0; layer-- {
		deltas[layer] = make([]float64, bn.LayerSizes[layer+1])

		// Use BLAS for backward pass
		bn.optimizer.BackwardPassBLAS(layer, layer+1,
			deltas[layer+1], activations[layer+1],
			func(x float64) float64 {
				if layer == len(bn.LayerSizes)-2 {
					return bn.OutputActivation.Derivative(x)
				}
				return bn.Activation.Derivative(x)
			},
			deltas[layer])
	}

	// Update weights and biases using BLAS
	for layer := 0; layer < len(bn.LayerSizes)-1; layer++ {
		bn.optimizer.UpdateWeightsBLAS(layer, bn.LearningRate, deltas[layer], activations[layer])
		bn.optimizer.UpdateBiasesBLAS(layer, bn.LearningRate, deltas[layer])
	}

	return loss
}

// Global BLAS optimizer instance with thread safety
var (
	globalBLASOptimizer   *BLASOptimizer
	blasOptimizerOnce     sync.Once
	blasOptimizerLayerSizes []int
)

// EnableGlobalBLASOptimization enables BLAS optimization for all networks
// Call this once at program startup
func EnableGlobalBLASOptimization(layerSizes []int) {
	blasOptimizerOnce.Do(func() {
		globalBLASOptimizer = NewBLASOptimizer(layerSizes)
		blasOptimizerLayerSizes = make([]int, len(layerSizes))
		copy(blasOptimizerLayerSizes, layerSizes)
	})
}

// GetGlobalBLASOptimizer returns the global BLAS optimizer
// Returns nil if not enabled
func GetGlobalBLASOptimizer() *BLASOptimizer {
	return globalBLASOptimizer
}

// IsBLASAvailable checks if BLAS optimization is available
func IsBLASAvailable() bool {
	return globalBLASOptimizer != nil
}
