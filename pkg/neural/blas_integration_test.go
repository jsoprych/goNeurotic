package neural

import (
	"testing"
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// TestBLASGeneralStride verifies that we understand how to create General matrices
// with correct stride for row-major layout.
func TestBLASGeneralStride(t *testing.T) {
	// Create a 2x3 matrix in row-major order
	// [1, 2, 3]
	// [4, 5, 6]
	rows, cols := 2, 3
	data := []float64{1, 2, 3, 4, 5, 6} // row-major

	// For row-major layout, stride = number of columns
	stride := cols

	mat := blas64.General{
		Rows:   rows,
		Cols:   cols,
		Data:   data,
		Stride: stride,
	}

	// Verify dimensions
	if mat.Rows != rows {
		t.Errorf("mat.Rows = %d, want %d", mat.Rows, rows)
	}
	if mat.Cols != cols {
		t.Errorf("mat.Cols = %d, want %d", mat.Cols, cols)
	}
	if mat.Stride != stride {
		t.Errorf("mat.Stride = %d, want %d", mat.Stride, stride)
	}

	// Access element (i,j) = row*i + col*j
	// Element (1,1) = row 1, column 1 = data[1*stride + 1] = data[3 + 1] = data[4] = 5
	i, j := 1, 1
	expected := 5.0
	actual := mat.Data[i*mat.Stride+j]
	if actual != expected {
		t.Errorf("mat.Data[%d*%d + %d] = %f, want %f", i, mat.Stride, j, actual, expected)
	}
}

// TestBLASVector verifies Vector creation and usage.
func TestBLASVector(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5}
	inc := 1

	vec := blas64.Vector{
		N:    len(data),
		Data: data,
		Inc:  inc,
	}

	if vec.N != len(data) {
		t.Errorf("vec.N = %d, want %d", vec.N, len(data))
	}
	if vec.Inc != inc {
		t.Errorf("vec.Inc = %d, want %d", vec.Inc, inc)
	}
}

// TestBLASGemv verifies matrix-vector multiplication.
func TestBLASGemv(t *testing.T) {
	// 2x3 matrix in row-major
	// [1, 2, 3]
	// [4, 5, 6]
	mat := blas64.General{
		Rows:   2,
		Cols:   3,
		Data:   []float64{1, 2, 3, 4, 5, 6},
		Stride: 3, // row-major: stride = cols
	}

	// Vector x = [1, 2, 3]
	x := blas64.Vector{
		N:    3,
		Data: []float64{1, 2, 3},
		Inc:  1,
	}

	// Output vector y (initialize to [0, 0])
	y := blas64.Vector{
		N:    2,
		Data: []float64{0, 0},
		Inc:  1,
	}

	// Compute y = 1.0 * A * x + 0.0 * y
	blas64.Gemv(blas.NoTrans, 1.0, mat, x, 0.0, y)

	// Expected:
	// y[0] = 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
	// y[1] = 4*1 + 5*2 + 6*3 = 4 + 10 + 18 = 32
	expected := []float64{14, 32}
	for i := 0; i < len(expected); i++ {
		if y.Data[i] != expected[i] {
			t.Errorf("y[%d] = %f, want %f", i, y.Data[i], expected[i])
		}
	}

	// Test transpose: y = Aᵀ * x (now A is 2x3, Aᵀ is 3x2)
	// Need new vectors for transpose operation
	x2 := blas64.Vector{
		N:    2,
		Data: []float64{1, 2},
		Inc:  1,
	}

	y2 := blas64.Vector{
		N:    3,
		Data: []float64{0, 0, 0},
		Inc:  1,
	}

	// Compute y = 1.0 * Aᵀ * x + 0.0 * y
	blas64.Gemv(blas.Trans, 1.0, mat, x2, 0.0, y2)

	// Expected:
	// y[0] = 1*1 + 4*2 = 1 + 8 = 9
	// y[1] = 2*1 + 5*2 = 2 + 10 = 12
	// y[2] = 3*1 + 6*2 = 3 + 12 = 15
	expected2 := []float64{9, 12, 15}
	for i := 0; i < len(expected2); i++ {
		if y2.Data[i] != expected2[i] {
			t.Errorf("y2[%d] = %f, want %f", i, y2.Data[i], expected2[i])
		}
	}
}

// TestBLASAxpy verifies vector addition.
func TestBLASAxpy(t *testing.T) {
	x := blas64.Vector{
		N:    3,
		Data: []float64{1, 2, 3},
		Inc:  1,
	}

	y := blas64.Vector{
		N:    3,
		Data: []float64{4, 5, 6},
		Inc:  1,
	}

	// y = 2.0 * x + y
	// y[0] = 2*1 + 4 = 6
	// y[1] = 2*2 + 5 = 9
	// y[2] = 2*3 + 6 = 12
	blas64.Axpy(2.0, x, y)

	expected := []float64{6, 9, 12}
	for i := 0; i < len(expected); i++ {
		if y.Data[i] != expected[i] {
			t.Errorf("y[%d] = %f, want %f", i, y.Data[i], expected[i])
		}
	}
}

// TestBLASGer verifies rank-1 update.
func TestBLASGer(t *testing.T) {
	// Create 2x3 matrix initialized to zero
	mat := blas64.General{
		Rows:   2,
		Cols:   3,
		Data:   []float64{0, 0, 0, 0, 0, 0},
		Stride: 3,
	}

	x := blas64.Vector{
		N:    2,
		Data: []float64{1, 2},
		Inc:  1,
	}

	y := blas64.Vector{
		N:    3,
		Data: []float64{3, 4, 5},
		Inc:  1,
	}

	// A = 1.0 * x * yᵀ + A
	// This performs outer product: x (2x1) * yᵀ (1x3) = 2x3 matrix
	blas64.Ger(1.0, x, y, mat)

	// Expected matrix:
	// [1*3, 1*4, 1*5] = [3, 4, 5]
	// [2*3, 2*4, 2*5] = [6, 8, 10]
	expected := []float64{3, 4, 5, 6, 8, 10}
	for i := 0; i < len(expected); i++ {
		if mat.Data[i] != expected[i] {
			t.Errorf("mat.Data[%d] = %f, want %f", i, mat.Data[i], expected[i])
		}
	}

	// Test with negative alpha (used in weight updates: W = W - learningRate * delta * activationPrevᵀ)
	mat2 := blas64.General{
		Rows:   2,
		Cols:   3,
		Data:   []float64{10, 10, 10, 10, 10, 10},
		Stride: 3,
	}

	// A = -0.5 * x * yᵀ + A
	blas64.Ger(-0.5, x, y, mat2)

	// Expected: original + (-0.5 * outer product)
	// [10 - 0.5*3, 10 - 0.5*4, 10 - 0.5*5] = [10-1.5, 10-2, 10-2.5] = [8.5, 8, 7.5]
	// [10 - 0.5*6, 10 - 0.5*8, 10 - 0.5*10] = [10-3, 10-4, 10-5] = [7, 6, 5]
	expected2 := []float64{8.5, 8, 7.5, 7, 6, 5}
	for i := 0; i < len(expected2); i++ {
		if mat2.Data[i] != expected2[i] {
			t.Errorf("mat2.Data[%d] = %f, want %f", i, mat2.Data[i], expected2[i])
		}
	}
}

// TestBLASIntegration tests a complete forward pass similar to neural network.
func TestBLASIntegration(t *testing.T) {
	// Simulate a layer: 3 inputs, 2 neurons
	fanIn, fanOut := 3, 2

	// Weight matrix (2x3) in row-major
	weights := blas64.General{
		Rows:   fanOut,
		Cols:   fanIn,
		Data:   []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
		Stride: fanIn,
	}

	// Bias vector (2)
	biases := blas64.Vector{
		N:    fanOut,
		Data: []float64{0.1, 0.2},
		Inc:  1,
	}

	// Input vector (3)
	input := blas64.Vector{
		N:    fanIn,
		Data: []float64{1.0, 2.0, 3.0},
		Inc:  1,
	}

	// Output vector (2) initialized to zero
	output := blas64.Vector{
		N:    fanOut,
		Data: []float64{0, 0},
		Inc:  1,
	}

	// Step 1: output = weights * input
	blas64.Gemv(blas.NoTrans, 1.0, weights, input, 0.0, output)

	// Step 2: output = output + biases
	blas64.Axpy(1.0, biases, output)

	// Compute expected manually:
	// neuron1: 0.1*1 + 0.2*2 + 0.3*3 + 0.1 = 0.1 + 0.4 + 0.9 + 0.1 = 1.5
	// neuron2: 0.4*1 + 0.5*2 + 0.6*3 + 0.2 = 0.4 + 1.0 + 1.8 + 0.2 = 3.4
	expected := []float64{1.5, 3.4}
	for i := 0; i < len(expected); i++ {
		if output.Data[i] != expected[i] {
			t.Errorf("output[%d] = %f, want %f", i, output.Data[i], expected[i])
		}
	}
}

// TestBLASNeuralForwardPass tests a complete neural network forward pass using BLAS.
func TestBLASNeuralForwardPass(t *testing.T) {
	// Create a simple network: 2 inputs -> 3 hidden -> 1 output

	// Create BLAS-optimized buffers for each layer
	// Layer 0: weights (3x2), biases (3)
	weights0 := blas64.General{
		Rows:   3,
		Cols:   2,
		Data:   []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
		Stride: 2,
	}
	biases0 := blas64.Vector{
		N:    3,
		Data: []float64{0.1, 0.2, 0.3},
		Inc:  1,
	}

	// Layer 1: weights (1x3), biases (1)
	weights1 := blas64.General{
		Rows:   1,
		Cols:   3,
		Data:   []float64{0.7, 0.8, 0.9},
		Stride: 3,
	}
	biases1 := blas64.Vector{
		N:    1,
		Data: []float64{0.4},
		Inc:  1,
	}

	// Input vector
	input := blas64.Vector{
		N:    2,
		Data: []float64{1.0, 0.5},
		Inc:  1,
	}

	// Hidden layer activations
	hidden := blas64.Vector{
		N:    3,
		Data: []float64{0, 0, 0},
		Inc:  1,
	}

	// Output activation
	output := blas64.Vector{
		N:    1,
		Data: []float64{0},
		Inc:  1,
	}

	// Forward pass through first layer
	// hidden = weights0 * input + biases0
	blas64.Gemv(blas.NoTrans, 1.0, weights0, input, 0.0, hidden)
	blas64.Axpy(1.0, biases0, hidden)

	// Apply activation function (sigmoid) to hidden layer
	// In real code, this would be a function application
	for i := 0; i < hidden.N; i++ {
		// Simple sigmoid approximation for test
		if hidden.Data[i] < 0 {
			hidden.Data[i] = 0
		}
		hidden.Data[i] = 1.0 / (1.0 + hidden.Data[i]) // Not real sigmoid, just for test
	}

	// Forward pass through output layer
	// output = weights1 * hidden + biases1
	blas64.Gemv(blas.NoTrans, 1.0, weights1, hidden, 0.0, output)
	blas64.Axpy(1.0, biases1, output)

	// Apply output activation (sigmoid)
	for i := 0; i < output.N; i++ {
		if output.Data[i] < 0 {
			output.Data[i] = 0
		}
		output.Data[i] = 1.0 / (1.0 + output.Data[i])
	}

	// Verify we got a valid output
	if output.N != 1 {
		t.Errorf("output.N = %d, want 1", output.N)
	}
	if output.Data[0] < 0 || output.Data[0] > 1 {
		t.Errorf("output[0] = %f, should be between 0 and 1", output.Data[0])
	}
}

// TestBLASBackwardPass tests a backward pass similar to neural network backpropagation.
func TestBLASBackwardPass(t *testing.T) {
	// Simulate backward pass: deltaCurrent = f'(activationCurrent) ⊙ (W_nextᵀ * deltaNext)
	fanIn, fanOut := 3, 2

	// Weight matrix for next layer (W_next)
	weightsNext := blas64.General{
		Rows:   fanOut,
		Cols:   fanIn,
		Data:   []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
		Stride: fanIn,
	}

	// Delta from next layer
	deltaNext := blas64.Vector{
		N:    fanOut,
		Data: []float64{0.2, 0.3},
		Inc:  1,
	}

	// Activation current (for derivative calculation)
	activationCurrent := blas64.Vector{
		N:    fanIn,
		Data: []float64{0.4, 0.5, 0.6},
		Inc:  1,
	}

	// Delta current (output)
	deltaCurrent := blas64.Vector{
		N:    fanIn,
		Data: []float64{0, 0, 0},
		Inc:  1,
	}

	// Step 1: deltaCurrent = W_nextᵀ * deltaNext
	blas64.Gemv(blas.Trans, 1.0, weightsNext, deltaNext, 0.0, deltaCurrent)

	// Step 2: Apply derivative (simplified: f'(x) = 1 - x for sigmoid approximation)
	for i := 0; i < deltaCurrent.N; i++ {
		derivative := 1.0 - activationCurrent.Data[i] // Simplified sigmoid derivative
		deltaCurrent.Data[i] *= derivative
	}

	// Verify expected values
	// Compute manually: W_nextᵀ * deltaNext
	// [0.1*0.2 + 0.4*0.3, 0.2*0.2 + 0.5*0.3, 0.3*0.2 + 0.6*0.3] = [0.02+0.12, 0.04+0.15, 0.06+0.18] = [0.14, 0.19, 0.24]
	// Multiply by derivative: [0.14*(1-0.4), 0.19*(1-0.5), 0.24*(1-0.6)] = [0.14*0.6, 0.19*0.5, 0.24*0.4] = [0.084, 0.095, 0.096]
	expected := []float64{0.084, 0.095, 0.096}
	for i := 0; i < len(expected); i++ {
		if deltaCurrent.Data[i] != expected[i] {
			t.Errorf("deltaCurrent[%d] = %f, want %f", i, deltaCurrent.Data[i], expected[i])
		}
	}
}

// TestBLASTrainingStep compares a single training step between BLAS and regular implementations
func TestBLASTrainingStep(t *testing.T) {
	// Create a simple network: 2 inputs -> 3 hidden -> 1 output
	config := NetworkConfig{
		LayerSizes:       []int{2, 3, 1},
		LearningRate:     0.5,
		Activation:       Sigmoid,
		OutputActivation: Sigmoid,
		LossFunction:     MeanSquaredError,
	}

	// Create regular network
	regularNet := NewNetwork(config)

	// Create BLAS network
	blasNet := NewBLASNetwork(config)

	// Copy weights and biases from regular to BLAS to ensure same starting point
	for i := range regularNet.Weights {
		blasNet.blasOps.ConvertWeightsToFlat(i, regularNet.Weights[i])
		blasNet.blasOps.ConvertBiasesToFlat(i, regularNet.Biases[i])
	}
	blasNet.weightsConverted = true

	// Test input and target
	input := []float64{0.5, 0.3}
	target := []float64{0.8}

	// Perform a single training step with both networks
	regularLoss := regularNet.Train(input, target)
	blasLoss := blasNet.TrainBLAS(input, target)

	// Loss values should be identical (within tolerance)
	tol := 1e-10
	if math.Abs(regularLoss-blasLoss) > tol {
		t.Errorf("Loss mismatch: regular %f, BLAS %f", regularLoss, blasLoss)
	}

	// Convert BLAS weights back to jagged format for comparison
	blasNet.ConvertFromFlat()

	// Compare weights after training step
	for i := range regularNet.Weights {
		for j := range regularNet.Weights[i] {
			for k := range regularNet.Weights[i][j] {
				if math.Abs(regularNet.Weights[i][j][k] - blasNet.Weights[i][j][k]) > tol {
					t.Errorf("Weight[%d][%d][%d] mismatch: regular %f, BLAS %f",
						i, j, k, regularNet.Weights[i][j][k], blasNet.Weights[i][j][k])
				}
			}
		}
	}

	// Compare biases after training step
	for i := range regularNet.Biases {
		for j := range regularNet.Biases[i] {
			if math.Abs(regularNet.Biases[i][j] - blasNet.Biases[i][j]) > tol {
				t.Errorf("Bias[%d][%d] mismatch: regular %f, BLAS %f",
					i, j, regularNet.Biases[i][j], blasNet.Biases[i][j])
			}
		}
	}

	// Also compare forward pass outputs after training
	regularOutput := regularNet.Predict(input)
	blasOutput, _ := blasNet.FeedForwardBLAS(input)

	if len(regularOutput) != len(blasOutput) {
		t.Fatalf("Output length mismatch: regular %d, BLAS %d", len(regularOutput), len(blasOutput))
	}
	for i := range regularOutput {
		if math.Abs(regularOutput[i] - blasOutput[i]) > tol {
			t.Errorf("Output[%d] mismatch after training: regular %f, BLAS %f",
				i, regularOutput[i], blasOutput[i])
		}
	}
}
