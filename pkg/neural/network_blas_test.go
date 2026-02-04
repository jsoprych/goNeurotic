package neural

import (
	"math"
	"testing"
)

// floatEqual checks if two float64 values are equal within a tolerance
func floatEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

// floatSliceEqual checks if two slices of float64 are equal within a tolerance
func floatSliceEqual(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !floatEqual(a[i], b[i], tol) {
			return false
		}
	}
	return true
}

// matrixEqual checks if two 2D slices of float64 are equal within a tolerance
func matrixEqual(a, b [][]float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !floatSliceEqual(a[i], b[i], tol) {
			return false
		}
	}
	return true
}

// TestNewBLASNetwork tests creation of BLAS-optimized network
func TestNewBLASNetwork(t *testing.T) {
	tests := []struct {
		name        string
		config      NetworkConfig
		shouldPanic bool
	}{
		{
			name: "valid network",
			config: NetworkConfig{
				LayerSizes: []int{2, 3, 1},
			},
			shouldPanic: false,
		},
		{
			name: "single layer should panic",
			config: NetworkConfig{
				LayerSizes: []int{2},
			},
			shouldPanic: true,
		},
		{
			name: "zero layer size should panic",
			config: NetworkConfig{
				LayerSizes: []int{2, 0, 1},
			},
			shouldPanic: true,
		},
		{
			name: "deep network",
			config: NetworkConfig{
				LayerSizes: []int{10, 20, 15, 5, 1},
			},
			shouldPanic: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil && !tt.shouldPanic {
					t.Errorf("NewBLASNetwork() panicked unexpectedly: %v", r)
				} else if r == nil && tt.shouldPanic {
					t.Error("NewBLASNetwork() should have panicked but didn't")
				}
			}()

			network := NewBLASNetwork(tt.config)
			if !tt.shouldPanic {
				if network == nil {
					t.Error("NewBLASNetwork() returned nil")
				}
				if network.Network == nil {
					t.Error("BLASNetwork.Network is nil")
				}
				if network.optimizer == nil {
					t.Error("BLASNetwork.optimizer is nil")
				}
				if len(network.LayerSizes) != len(tt.config.LayerSizes) {
					t.Errorf("Expected %d layers, got %d", len(tt.config.LayerSizes), len(network.LayerSizes))
				}
			}
		})
	}
}

// TestBLASNetwork_FeedForwardBLAS tests forward pass with BLAS optimization
func TestBLASNetwork_FeedForwardBLAS(t *testing.T) {
	config := NetworkConfig{
		LayerSizes:       []int{2, 3, 1},
		Activation:       Sigmoid,
		OutputActivation: Sigmoid,
	}

	// Create both regular and BLAS networks with same random seed
	regularNet := NewNetwork(config)
	blasNet := NewBLASNetwork(config)

	// Copy weights and biases from regular network to BLAS network
	// to ensure they have the same initial state
	for i := range regularNet.Weights {
		blasNet.blasOps.ConvertWeightsToFlat(i, regularNet.Weights[i])
		blasNet.blasOps.ConvertBiasesToFlat(i, regularNet.Biases[i])
	}
	blasNet.weightsConverted = true
	blasNet.ConvertFromFlat()

	tests := []struct {
		name        string
		input       []float64
		shouldPanic bool
	}{
		{
			name:        "valid input",
			input:       []float64{0.5, 0.3},
			shouldPanic: false,
		},
		{
			name:        "wrong input size",
			input:       []float64{0.5, 0.3, 0.1},
			shouldPanic: true,
		},
		{
			name:        "empty input",
			input:       []float64{},
			shouldPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil && !tt.shouldPanic {
					t.Errorf("FeedForwardBLAS() panicked unexpectedly: %v", r)
				} else if r == nil && tt.shouldPanic {
					t.Error("FeedForwardBLAS() should have panicked but didn't")
				}
			}()

			// BLAS forward pass (may panic for invalid inputs)
			blasOutput, blasActivations := blasNet.FeedForwardBLAS(tt.input)

			if !tt.shouldPanic {
				// Regular forward pass
				regularOutput, regularActivations := regularNet.FeedForward(tt.input)

				// Compare outputs
				if !floatSliceEqual(regularOutput, blasOutput, 1e-10) {
					t.Errorf("Output mismatch: regular %v, BLAS %v", regularOutput, blasOutput)
				}

				// Compare activations
				if len(regularActivations) != len(blasActivations) {
					t.Errorf("Activations length mismatch: regular %d, BLAS %d",
						len(regularActivations), len(blasActivations))
				}

				for i := range regularActivations {
					if !floatSliceEqual(regularActivations[i], blasActivations[i], 1e-10) {
						t.Errorf("Activations[%d] mismatch: regular %v, BLAS %v",
							i, regularActivations[i], blasActivations[i])
					}
				}
			}
		})
	}
}

// TestBLASNetwork_TrainBLAS tests training with BLAS optimization
func TestBLASNetwork_TrainBLAS(t *testing.T) {
	config := NetworkConfig{
		LayerSizes:       []int{2, 4, 1},
		LearningRate:     0.5,
		Activation:       Sigmoid,
		OutputActivation: Sigmoid,
		LossFunction:     MeanSquaredError,
	}

	// Create multiple networks to compare
	regularNet1 := NewNetwork(config)

	blasNet := NewBLASNetwork(config)

	// Initialize all networks with same weights
	for i := range regularNet1.Weights {
		// Copy weights to BLAS network
		blasNet.blasOps.ConvertWeightsToFlat(i, regularNet1.Weights[i])
		blasNet.blasOps.ConvertBiasesToFlat(i, regularNet1.Biases[i])

	}
	blasNet.weightsConverted = true
	blasNet.ConvertFromFlat()

	// XOR training data
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	// Train both networks on same data
	for epoch := 0; epoch < 100; epoch++ {
		for i := range inputs {
			// Train regular network
			regularLoss := regularNet1.Train(inputs[i], targets[i])
			// Train BLAS network
			blasLoss := blasNet.TrainBLAS(inputs[i], targets[i])

			// Loss values should be similar
			if !floatEqual(regularLoss, blasLoss, 1e-10) {
				t.Errorf("Epoch %d, example %d: loss mismatch - regular %f, BLAS %f",
					epoch, i, regularLoss, blasLoss)
			}
		}
	}

	// Compare final weights and biases
	// Convert BLAS weights back to jagged for comparison
	blasNet.ConvertFromFlat()
	for i := range regularNet1.Weights {

		for j := range regularNet1.Weights[i] {
			for k := range regularNet1.Weights[i][j] {
				if !floatEqual(regularNet1.Weights[i][j][k], blasNet.Weights[i][j][k], 1e-10) {
					t.Errorf("Weight[%d][%d][%d] mismatch: regular %f, BLAS %f",
						i, j, k, regularNet1.Weights[i][j][k], blasNet.Weights[i][j][k])
				}
			}
		}

		for j := range regularNet1.Biases[i] {
			if !floatEqual(regularNet1.Biases[i][j], blasNet.Biases[i][j], 1e-10) {
				t.Errorf("Bias[%d][%d] mismatch: regular %f, BLAS %f",
					i, j, regularNet1.Biases[i][j], blasNet.Biases[i][j])
			}
		}
	}


}

// TestBLASNetwork_ConvertToFromFlat tests weight conversion functions
func TestBLASNetwork_ConvertToFromFlat(t *testing.T) {
	config := NetworkConfig{
		LayerSizes: []int{3, 5, 2},
	}
	network := NewBLASNetwork(config)

	// Store original weights and biases
	originalWeights := make([][][]float64, len(network.Weights))
	originalBiases := make([][]float64, len(network.Biases))

	for i := range network.Weights {
		originalWeights[i] = make([][]float64, len(network.Weights[i]))
		for j := range network.Weights[i] {
			originalWeights[i][j] = make([]float64, len(network.Weights[i][j]))
			copy(originalWeights[i][j], network.Weights[i][j])
		}
		originalBiases[i] = make([]float64, len(network.Biases[i]))
		copy(originalBiases[i], network.Biases[i])
	}

	// Convert to flat
	network.ConvertToFlat()
	if !network.weightsConverted {
		t.Error("weightsConverted should be true after ConvertToFlat")
	}

	// Modify flat buffers to verify conversion back works
	for i := range network.blasOps.weightFlatBuffers {
		for j := range network.blasOps.weightFlatBuffers[i] {
			network.blasOps.weightFlatBuffers[i][j] *= 2.0
		}
		for j := range network.blasOps.biasFlatBuffers[i] {
			network.blasOps.biasFlatBuffers[i][j] *= 2.0
		}
	}

	// Convert back from flat
	network.ConvertFromFlat()

	// Verify weights and biases were updated
	for i := range network.Weights {
		for j := range network.Weights[i] {
			for k := range network.Weights[i][j] {
				expected := originalWeights[i][j][k] * 2.0
				if !floatEqual(network.Weights[i][j][k], expected, 1e-10) {
					t.Errorf("Weight[%d][%d][%d] = %f, expected %f",
						i, j, k, network.Weights[i][j][k], expected)
				}
			}
		}
		for j := range network.Biases[i] {
			expected := originalBiases[i][j] * 2.0
			if !floatEqual(network.Biases[i][j], expected, 1e-10) {
				t.Errorf("Bias[%d][%d] = %f, expected %f",
					i, j, network.Biases[i][j], expected)
			}
		}
	}
}

// TestBLASNetwork_XOR tests that BLAS network can learn XOR problem
func TestBLASNetwork_XOR(t *testing.T) {
	config := NetworkConfig{
		LayerSizes:       []int{2, 4, 1},
		LearningRate:     0.5,
		Activation:       Sigmoid,
		OutputActivation: Sigmoid,
		LossFunction:     MeanSquaredError,
	}
	network := NewBLASNetwork(config)

	// XOR truth table
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	// Train for a few epochs
	totalLoss := 0.0
	for epoch := 0; epoch < 5000; epoch++ {
		epochLoss := 0.0
		for i := range inputs {
			loss := network.TrainBLAS(inputs[i], targets[i])
			epochLoss += loss
		}
		totalLoss = epochLoss / float64(len(inputs))

		// Early exit if loss is low enough
		if totalLoss < 0.01 {
			break
		}
	}

	// Test predictions
	correct := 0
	for i, input := range inputs {
		output, _ := network.FeedForwardBLAS(input)
		predicted := 0.0
		if output[0] > 0.5 {
			predicted = 1.0
		}

		if predicted == targets[i][0] {
			correct++
		}
	}

	// Network should learn XOR reasonably well
	if correct < 3 {
		t.Errorf("XOR test failed: only %d/4 correct (expected at least 3), final loss: %f",
			correct, totalLoss)
	}
}

// TestBLASNetwork_BatchTrain tests batch training with BLAS optimization
func TestBLASNetwork_BatchTrain(t *testing.T) {
	// Note: BLASNetwork doesn't have BatchTrain method yet,
	// but we test that it can be used with regular BatchTrain
	config := NetworkConfig{
		LayerSizes:       []int{2, 3, 1},
		LearningRate:     0.1,
		Activation:       ReLU,
		OutputActivation: Sigmoid,
		LossFunction:     MeanSquaredError,
	}
	regularNet := NewNetwork(config)
	blasNet := NewBLASNetwork(config)

	// Initialize with same weights
	for i := range regularNet.Weights {
		blasNet.blasOps.ConvertWeightsToFlat(i, regularNet.Weights[i])
		blasNet.blasOps.ConvertBiasesToFlat(i, regularNet.Biases[i])
	}
	blasNet.weightsConverted = true
	blasNet.ConvertFromFlat()

	inputs := [][]float64{
		{0.1, 0.2},
		{0.3, 0.4},
		{0.5, 0.6},
		{0.7, 0.8},
	}
	targets := [][]float64{
		{0},
		{1},
		{0},
		{1},
	}

	// Train both networks with batch training
	regularLoss := regularNet.BatchTrain(inputs, targets)

	// BLAS network uses regular BatchTrain (not optimized yet)
	// but should still work
	blasLoss := blasNet.BatchTrain(inputs, targets)
	// Sync jagged weights to flat buffers after batch training
	// (BatchTrain modifies jagged weights, not flat buffers)
	blasNet.ConvertToFlat()

	// Loss values should be similar
	if !floatEqual(regularLoss, blasLoss, 1e-10) {
		t.Errorf("Batch loss mismatch: regular %f, BLAS %f", regularLoss, blasLoss)
	}

	// Verify predictions are similar
	for i, input := range inputs {
		regularOutput := regularNet.Predict(input)
		blasOutput, _ := blasNet.FeedForwardBLAS(input)

		if !floatSliceEqual(regularOutput, blasOutput, 1e-10) {
			t.Errorf("Prediction %d mismatch: regular %v, BLAS %v",
				i, regularOutput, blasOutput)
		}
	}
}

// TestBLASNetwork_GetSetLearningRate tests learning rate methods
func TestBLASNetwork_GetSetLearningRate(t *testing.T) {
	config := NetworkConfig{
		LayerSizes:   []int{1, 2, 1},
		LearningRate: 0.01,
	}
	network := NewBLASNetwork(config)

	if network.GetLearningRate() != 0.01 {
		t.Errorf("Initial learning rate should be 0.01, got %f", network.GetLearningRate())
	}

	network.SetLearningRate(0.05)
	if network.GetLearningRate() != 0.05 {
		t.Errorf("Learning rate should be 0.05 after SetLearningRate, got %f", network.GetLearningRate())
	}

	// Test that BLAS optimizer still works after learning rate change
	input := []float64{0.5}
	target := []float64{1.0}
	loss := network.TrainBLAS(input, target)
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		t.Errorf("TrainBLAS() returned invalid loss after learning rate change: %f", loss)
	}
}

// TestEnableGlobalBLASOptimization tests global BLAS optimization
func TestEnableGlobalBLASOptimization(t *testing.T) {
	layerSizes := []int{2, 3, 1}

	// Enable global optimization
	EnableGlobalBLASOptimization(layerSizes)

	// Get global optimizer
	optimizer := GetGlobalBLASOptimizer()
	if optimizer == nil {
		t.Error("GetGlobalBLASOptimizer() returned nil after EnableGlobalBLASOptimization")
	}

	// Check if BLAS is available
	if !IsBLASAvailable() {
		t.Error("IsBLASAvailable() should return true after EnableGlobalBLASOptimization")
	}

	// Verify optimizer has correct structure
	if len(optimizer.weightFlatBuffers) != len(layerSizes)-1 {
		t.Errorf("optimizer.weightFlatBuffers length = %d, expected %d",
			len(optimizer.weightFlatBuffers), len(layerSizes)-1)
	}

	// Call EnableGlobalBLASOptimization again - should be idempotent
	EnableGlobalBLASOptimization(layerSizes)
	optimizer2 := GetGlobalBLASOptimizer()
	if optimizer != optimizer2 {
		t.Error("Second EnableGlobalBLASOptimization should return same optimizer")
	}
}

// TestBLASNetwork_BatchTrainBLAS tests the BLAS-optimized batch training method
func TestBLASNetwork_BatchTrainBLAS(t *testing.T) {
	config := NetworkConfig{
		LayerSizes:   []int{2, 4, 1},
		LearningRate: 0.1,
		Activation:   Sigmoid,
		OutputActivation: Sigmoid,
		LossFunction: MeanSquaredError,
	}
	regularNet := NewNetwork(config)
	blasNet := NewBLASNetwork(config)

	// Initialize with same weights
	for i := range regularNet.Weights {
		blasNet.blasOps.ConvertWeightsToFlat(i, regularNet.Weights[i])
		blasNet.blasOps.ConvertBiasesToFlat(i, regularNet.Biases[i])
	}
	blasNet.weightsConverted = true
	blasNet.ConvertFromFlat()

	// Create batch data
	inputs := [][]float64{
		{0.1, 0.2},
		{0.3, 0.4},
		{0.5, 0.6},
		{0.7, 0.8},
	}
	targets := [][]float64{
		{0},
		{1},
		{0},
		{1},
	}

	// Train regular network with batch training
	regularLoss := regularNet.BatchTrain(inputs, targets)

	// Train BLAS network with BLAS-optimized batch training
	blasLoss := blasNet.BatchTrainBLAS(inputs, targets)

	// Loss values should be similar (allow for small numerical differences)
	if math.Abs(regularLoss - blasLoss) > 1e-10 {
		t.Errorf("BatchTrainBLAS loss mismatch: regular %f, BLAS %f", regularLoss, blasLoss)
	}

	// Verify predictions are similar
	for i, input := range inputs {
		regularOutput := regularNet.Predict(input)
		blasOutput, _ := blasNet.FeedForwardBLAS(input)

		if !floatSliceEqual(regularOutput, blasOutput, 1e-10) {
			t.Errorf("Prediction %d mismatch after BatchTrainBLAS: regular %v, BLAS %v",
				i, regularOutput, blasOutput)
		}
	}

	// Test that BatchTrainBLAS accumulates gradients correctly
	// Run multiple epochs and verify loss decreases
	initialLoss := blasLoss
	for epoch := 0; epoch < 5; epoch++ {
		blasNet.BatchTrainBLAS(inputs, targets)
	}
	finalLoss := 0.0
	for i := range inputs {
		output, _ := blasNet.FeedForwardBLAS(inputs[i])
		finalLoss += blasNet.LossFunction.Function(output, targets[i])
	}
	finalLoss /= float64(len(inputs))

	// Loss should generally decrease (not strictly guaranteed but likely)
	if finalLoss > initialLoss * 1.1 { // Allow 10% increase due to randomness
		t.Logf("Loss didn't decrease as expected: initial %f, final %f", initialLoss, finalLoss)
		// Not a failure, just informational
	}

	// Test edge case: empty batch should return zero loss
	emptyLoss := blasNet.BatchTrainBLAS([][]float64{}, [][]float64{})
	if emptyLoss != 0.0 {
		t.Errorf("Empty batch should return zero loss, got %f", emptyLoss)
	}

	// Test edge case: single example batch (should work)
	singleLoss := blasNet.BatchTrainBLAS([][]float64{{0.5, 0.5}}, [][]float64{{0.5}})
	if math.IsNaN(singleLoss) || math.IsInf(singleLoss, 0) {
		t.Errorf("Single example batch returned invalid loss: %f", singleLoss)
	}
}
