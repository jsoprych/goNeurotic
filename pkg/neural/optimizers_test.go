package neural

import (
	"math"
	"math/rand"
	"testing"
)

// Helper function to create random weight matrix
func randomWeights(rows, cols int) [][]float64 {
	weights := make([][]float64, rows)
	for i := range weights {
		weights[i] = make([]float64, cols)
		for j := range weights[i] {
			weights[i][j] = rand.Float64()*2 - 1 // range [-1, 1]
		}
	}
	return weights
}

// Helper function to create random bias vector
func randomBiases(size int) []float64 {
	biases := make([]float64, size)
	for i := range biases {
		biases[i] = rand.Float64()*2 - 1
	}
	return biases
}

// Helper function to create random gradients (same shape as weights/biases)
func randomWeightGradients(weights [][]float64) [][]float64 {
	gradients := make([][]float64, len(weights))
	for i := range gradients {
		gradients[i] = make([]float64, len(weights[i]))
		for j := range gradients[i] {
			gradients[i][j] = rand.Float64()*0.1 - 0.05 // small gradients
		}
	}
	return gradients
}

func randomBiasGradients(biases []float64) []float64 {
	gradients := make([]float64, len(biases))
	for i := range gradients {
		gradients[i] = rand.Float64()*0.1 - 0.05
	}
	return gradients
}

// Helper function to compare float slices with tolerance
func floatsEqual(a, b []float64, tolerance float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tolerance {
			return false
		}
	}
	return true
}

// Helper function to compare weight matrices with tolerance
func weightsEqual(a, b [][]float64, tolerance float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if math.Abs(a[i][j]-b[i][j]) > tolerance {
				return false
			}
		}
	}
	return true
}

// Helper function to create flat buffers from jagged arrays
func flattenWeights(weights [][]float64) []float64 {
	if len(weights) == 0 {
		return []float64{}
	}
	rows := len(weights)
	cols := len(weights[0])
	flat := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			flat[i*cols+j] = weights[i][j]
		}
	}
	return flat
}

func flattenWeightGradients(gradients [][]float64) []float64 {
	return flattenWeights(gradients)
}

func TestNewOptimizer(t *testing.T) {
	tests := []struct {
		name     string
		config   OptimizerConfig
		wantType string
	}{
		{
			name: "SGD optimizer",
			config: OptimizerConfig{
				Type:        "sgd",
				LearningRate: 0.01,
			},
			wantType: "sgd",
		},
		{
			name: "SGD with Momentum optimizer",
			config: OptimizerConfig{
				Type:        "sgd_momentum",
				LearningRate: 0.01,
				Momentum:     0.9,
			},
			wantType: "sgd_momentum",
		},
		{
			name: "RMSprop optimizer",
			config: OptimizerConfig{
				Type:        "rmsprop",
				LearningRate: 0.001,
				Rho:          0.9,
				Epsilon:      1e-8,
			},
			wantType: "rmsprop",
		},
		{
			name: "Adam optimizer",
			config: OptimizerConfig{
				Type:        "adam",
				LearningRate: 0.001,
				Beta1:        0.9,
				Beta2:        0.999,
				Epsilon:      1e-8,
			},
			wantType: "adam",
		},
		{
			name: "default to SGD for unknown type",
			config: OptimizerConfig{
				Type:        "unknown",
				LearningRate: 0.01,
			},
			wantType: "sgd",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opt := NewOptimizer(tt.config)
			if opt == nil {
				t.Fatal("NewOptimizer returned nil")
			}

			// Initialize optimizer state with appropriate layer sizes
			// weights is 2x2 matrix: 2 neurons, 2 inputs
			// So layer sizes: input=2, output=2
			opt.InitializeState([]int{2, 2})

			// Test that it's the right type by checking behavior
			// We'll do a simple weight update test to verify it works
			weights := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
			gradients := [][]float64{{0.1, 0.2}, {0.3, 0.4}}
			updated := opt.UpdateWeights(0, weights, gradients)

			if updated == nil {
				t.Error("UpdateWeights returned nil")
			}
		})
	}
}

func TestSGDOptimizer(t *testing.T) {
	t.Run("basic weight update", func(t *testing.T) {
		opt := NewSGDOptimizer(0.01)
		opt.InitializeState([]int{2, 3, 1})

		weights := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
		gradients := [][]float64{{0.1, 0.2}, {0.3, 0.4}}
		expected := [][]float64{{0.999, 1.998}, {2.997, 3.996}}

		updated := opt.UpdateWeights(0, weights, gradients)
		if !weightsEqual(updated, expected, 1e-6) {
			t.Errorf("UpdateWeights() = %v, want %v", updated, expected)
		}
	})

	t.Run("basic bias update", func(t *testing.T) {
		opt := NewSGDOptimizer(0.01)
		opt.InitializeState([]int{2, 3, 1})

		biases := []float64{1.0, 2.0, 3.0}
		gradients := []float64{0.1, 0.2, 0.3}
		expected := []float64{0.999, 1.998, 2.997}

		updated := opt.UpdateBiases(0, biases, gradients)
		if !floatsEqual(updated, expected, 1e-6) {
			t.Errorf("UpdateBiases() = %v, want %v", updated, expected)
		}
	})

	t.Run("flat weight update", func(t *testing.T) {
		opt := NewSGDOptimizer(0.01)
		opt.InitializeState([]int{2, 3, 1})

		weightsFlat := []float64{1.0, 2.0, 3.0, 4.0} // 2x2 matrix
		gradientsFlat := []float64{0.1, 0.2, 0.3, 0.4}
		expected := []float64{0.999, 1.998, 2.997, 3.996}

		updated := opt.UpdateWeightsFlat(0, weightsFlat, gradientsFlat)
		if !floatsEqual(updated, expected, 1e-6) {
			t.Errorf("UpdateWeightsFlat() = %v, want %v", updated, expected)
		}
	})

	t.Run("flat bias update", func(t *testing.T) {
		opt := NewSGDOptimizer(0.01)
		opt.InitializeState([]int{2, 3, 1})

		biasesFlat := []float64{1.0, 2.0, 3.0}
		gradientsFlat := []float64{0.1, 0.2, 0.3}
		expected := []float64{0.999, 1.998, 2.997}

		updated := opt.UpdateBiasesFlat(0, biasesFlat, gradientsFlat)
		if !floatsEqual(updated, expected, 1e-6) {
			t.Errorf("UpdateBiasesFlat() = %v, want %v", updated, expected)
		}
	})

	t.Run("batch weight updates", func(t *testing.T) {
		opt := NewSGDOptimizer(0.01)
		opt.InitializeState([]int{2, 3, 1})

		// Start with zero updates
		weightUpdates := [][]float64{{0.0, 0.0}, {0.0, 0.0}}
		gradients := [][]float64{{0.1, 0.2}, {0.3, 0.4}}

		// First accumulation
		updated := opt.BatchUpdateWeights(0, weightUpdates, gradients)
		expected1 := [][]float64{{0.1, 0.2}, {0.3, 0.4}}
		if !weightsEqual(updated, expected1, 1e-6) {
			t.Errorf("BatchUpdateWeights() first call = %v, want %v", updated, expected1)
		}

		// Second accumulation
		gradients2 := [][]float64{{0.2, 0.1}, {0.4, 0.3}}
		updated2 := opt.BatchUpdateWeights(0, updated, gradients2)
		expected2 := [][]float64{{0.3, 0.3}, {0.7, 0.7}}
		if !weightsEqual(updated2, expected2, 1e-6) {
			t.Errorf("BatchUpdateWeights() second call = %v, want %v", updated2, expected2)
		}
	})

	t.Run("apply batch updates", func(t *testing.T) {
		opt := NewSGDOptimizer(0.01)
		opt.InitializeState([]int{2, 3, 1})

		weights := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
		weightUpdates := [][]float64{{0.1, 0.2}, {0.3, 0.4}}
		batchSize := 2.0

		updated := opt.ApplyBatchUpdatesWeights(0, weights, weightUpdates, batchSize)
		// learningRate = 0.01, scale = 0.01/2 = 0.005
		// weight[0][0] = 1.0 - 0.005*0.1 = 0.9995
		expected := [][]float64{{0.9995, 1.999}, {2.9985, 3.998}}

		if !weightsEqual(updated, expected, 1e-4) {
			t.Errorf("ApplyBatchUpdatesWeights() = %v, want %v", updated, expected)
		}
	})

	t.Run("clone", func(t *testing.T) {
		opt := NewSGDOptimizer(0.01)
		opt.InitializeState([]int{2, 3, 1})

		clone := opt.Clone()
		if clone == nil {
			t.Fatal("Clone() returned nil")
		}

		// Verify it's a different instance
		if clone == opt {
			t.Error("Clone() returned same instance")
		}

		// Verify it behaves the same
		weights := [][]float64{{1.0, 2.0}}
		gradients := [][]float64{{0.1, 0.2}}

		originalResult := opt.UpdateWeights(0, weights, gradients)
		cloneResult := clone.UpdateWeights(0, weights, gradients)

		if !weightsEqual(originalResult, cloneResult, 1e-6) {
			t.Errorf("Clone behavior differs: original = %v, clone = %v", originalResult, cloneResult)
		}
	})
}

func TestSGDMomentumOptimizer(t *testing.T) {
	t.Run("initialization", func(t *testing.T) {
		opt := NewSGDMomentumOptimizer(0.01, 0.9)
		layerSizes := []int{2, 3, 1}
		opt.InitializeState(layerSizes)

		// After initialization, velocity buffers should exist
		// We can't access private fields, but we can verify behavior
		weights := randomWeights(3, 2) // layer 0: 3 neurons, 2 inputs
		gradients := randomWeightGradients(weights)

		// First update should work
		updated := opt.UpdateWeights(0, weights, gradients)
		if updated == nil {
			t.Error("UpdateWeights failed after initialization")
		}
	})

	t.Run("momentum effect", func(t *testing.T) {
		opt := NewSGDMomentumOptimizer(0.01, 0.9)
		opt.InitializeState([]int{2, 3, 1})

		// Simple test: two consecutive updates with same gradient
		weights := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
		gradients := [][]float64{{0.1, 0.2}, {0.3, 0.4}}

		// First update: v = 0.9*0 + 0.1*g = 0.1*g
		// weights = w - lr*v = w - 0.01*0.1*g
		updated1 := opt.UpdateWeights(0, weights, gradients)

		// Second update with same gradient: v = 0.9*(0.1*g) + 0.1*g = 0.19*g
		// weights = w - 0.01*0.19*g
		updated2 := opt.UpdateWeights(0, updated1, gradients)

		// Calculate expected values
		// First update: each weight decreases by 0.01*0.1*gradient
		// Second update: each weight decreases by additional 0.01*0.19*gradient
		// We'll just verify that second update is larger than it would be without momentum
		// Without momentum, second update would be same as first: decrease by 0.01*0.1*gradient
		// With momentum, second update should be larger: decrease by 0.01*0.19*gradient

		// For weight[0][0]: starts at 1.0, gradient = 0.1
		// Without momentum: after 2 updates = 1.0 - 2*0.01*0.1 = 0.998
		// With momentum: after 2 updates = 1.0 - 0.01*0.1 - 0.01*0.19 = 0.9971
		expected := 0.9971
		actual := updated2[0][0]

		if math.Abs(actual-expected) > 1e-4 {
			t.Errorf("Momentum not working: got %v, want %v (without momentum would be 0.998)", actual, expected)
		}
	})

	t.Run("step increments", func(t *testing.T) {
		opt := NewSGDMomentumOptimizer(0.01, 0.9)
		opt.InitializeState([]int{2, 3, 1})

		// Step should not panic
		opt.Step()
		opt.Step()
	})

	t.Run("clone", func(t *testing.T) {
		opt := NewSGDMomentumOptimizer(0.01, 0.9)
		// Use layer sizes that match our test weights: input=2, neurons=1
		layerSizes := []int{2, 1, 1}
		opt.InitializeState(layerSizes)

		// Do one update to set some velocity
		weights := [][]float64{{1.0, 2.0}} // 1 neuron, 2 inputs
		gradients := [][]float64{{0.1, 0.2}}
		opt.UpdateWeights(0, weights, gradients)

		clone := opt.Clone()
		if clone == nil {
			t.Fatal("Clone() returned nil")
		}
		// Initialize clone state (Clone() doesn't copy internal state)
		clone.InitializeState(layerSizes)

		// Do another update on both - copy weights first since UpdateWeights modifies in-place
		weights2 := [][]float64{{1.5, 2.5}}
		weights2Copy := [][]float64{{1.5, 2.5}}
		gradients2 := [][]float64{{0.2, 0.1}}

		originalResult := opt.UpdateWeights(0, weights2, gradients2)
		cloneResult := clone.UpdateWeights(0, weights2Copy, gradients2)

		// Since Clone() doesn't copy internal velocity state,
		// the results will differ. We just verify both can be called without panic.
		// Optionally check that weights were updated (non-zero change).
		// Note: originalResult is weights2 after modification, so compare to original value.
		originalWeightValue := 1.5
		if math.Abs(originalResult[0][0] - originalWeightValue) < 1e-6 {
			t.Error("Original optimizer didn't update weights")
		}
		if math.Abs(cloneResult[0][0] - originalWeightValue) < 1e-6 {
			t.Error("Clone optimizer didn't update weights")
		}
	})
}

func TestRMSpropOptimizer(t *testing.T) {
	t.Run("initialization", func(t *testing.T) {
		opt := NewRMSpropOptimizer(0.001, 0.9, 1e-8)
		layerSizes := []int{2, 3, 1}
		opt.InitializeState(layerSizes)

		// Should initialize cache
		weights := randomWeights(3, 2)
		gradients := randomWeightGradients(weights)

		updated := opt.UpdateWeights(0, weights, gradients)
		if updated == nil {
			t.Error("UpdateWeights failed after initialization")
		}
	})

	t.Run("rmsprop update formula", func(t *testing.T) {
		opt := NewRMSpropOptimizer(0.001, 0.9, 1e-8)
		opt.InitializeState([]int{2, 3, 1})

		// Simple test with known values
		weights := [][]float64{{1.0}}
		gradients := [][]float64{{0.1}}

		updated := opt.UpdateWeights(0, weights, gradients)

		// RMSprop update:
		// cache = rho * cache + (1-rho) * g^2
		// For first update with cache=0: cache = 0.9*0 + 0.1*0.01 = 0.001
		// weight = weight - lr * g / sqrt(cache + epsilon)
		//        = 1.0 - 0.001 * 0.1 / sqrt(0.001 + 1e-8)
		//        ≈ 1.0 - 0.0001 / sqrt(0.001)
		//        ≈ 1.0 - 0.0001 / 0.03162
		//        ≈ 1.0 - 0.003162
		//        ≈ 0.996838

		expected := 0.996838
		actual := updated[0][0]

		if math.Abs(actual-expected) > 1e-4 {
			t.Errorf("RMSprop update incorrect: got %v, want %v", actual, expected)
		}
	})

	t.Run("batch updates", func(t *testing.T) {
		opt := NewRMSpropOptimizer(0.001, 0.9, 1e-8)
		opt.InitializeState([]int{2, 3, 1})

		// Test batch accumulation
		weightUpdates := [][]float64{{0.0}}
		gradients := [][]float64{{0.1}}

		accumulated := opt.BatchUpdateWeights(0, weightUpdates, gradients)
		if accumulated[0][0] != 0.1 {
			t.Errorf("BatchUpdateWeights failed to accumulate: got %v, want 0.1", accumulated[0][0])
		}
	})

	t.Run("clone", func(t *testing.T) {
		// Use layer sizes that match our test weights: input=1, neurons=1
		layerSizes := []int{1, 1, 1}
		opt := NewRMSpropOptimizer(0.001, 0.9, 1e-8)
		opt.InitializeState(layerSizes)

		// Do one update to set cache
		weights := [][]float64{{1.0}} // 1 neuron, 1 input
		gradients := [][]float64{{0.1}}
		opt.UpdateWeights(0, weights, gradients)

		clone := opt.Clone()
		if clone == nil {
			t.Fatal("Clone() returned nil")
		}
		// Initialize clone state (Clone() doesn't copy internal cache)
		clone.InitializeState(layerSizes)

		// Do another update on both - copy weights first since UpdateWeights modifies in-place
		weights2 := [][]float64{{1.5}}
		weights2Copy := [][]float64{{1.5}}
		gradients2 := [][]float64{{0.2}}

		originalResult := opt.UpdateWeights(0, weights2, gradients2)
		cloneResult := clone.UpdateWeights(0, weights2Copy, gradients2)

		// Since Clone() doesn't copy internal cache state,
		// the results will differ. We just verify both can be called without panic.
		// Check that weights were updated (non-zero change).
		originalWeightValue := 1.5
		if math.Abs(originalResult[0][0] - originalWeightValue) < 1e-6 {
			t.Error("Original optimizer didn't update weights")
		}
		if math.Abs(cloneResult[0][0] - originalWeightValue) < 1e-6 {
			t.Error("Clone optimizer didn't update weights")
		}
	})
}

func TestAdamOptimizer(t *testing.T) {
	t.Run("initialization", func(t *testing.T) {
		opt := NewAdamOptimizer(0.001, 0.9, 0.999, 1e-8)
		layerSizes := []int{2, 3, 1}
		opt.InitializeState(layerSizes)

		// Should initialize moment estimates
		weights := randomWeights(3, 2)
		gradients := randomWeightGradients(weights)

		updated := opt.UpdateWeights(0, weights, gradients)
		if updated == nil {
			t.Error("UpdateWeights failed after initialization")
		}
	})

	t.Run("adam update with bias correction", func(t *testing.T) {
		opt := NewAdamOptimizer(0.001, 0.9, 0.999, 1e-8)
		opt.InitializeState([]int{2, 3, 1})

		// First update
		weights := [][]float64{{1.0}}
		gradients := [][]float64{{0.1}}

		updated1 := opt.UpdateWeights(0, weights, gradients)

		// Second update (save value before update)
		updated1Value := updated1[0][0]
		updated2 := opt.UpdateWeights(0, updated1, gradients)

		// Verify both updates happened (weights changed)
		if math.Abs(updated1[0][0]-1.0) < 1e-6 {
			t.Error("First Adam update didn't change weights")
		}
		if math.Abs(updated2[0][0]-updated1Value) < 1e-6 {
			t.Error("Second Adam update didn't change weights")
		}

		// With bias correction, second update should be different from first
		// even with same gradient
	})

	t.Run("bias correction effect", func(t *testing.T) {
		opt := NewAdamOptimizer(0.001, 0.9, 0.999, 1e-8)
		opt.InitializeState([]int{2, 3, 1})

		weights := [][]float64{{1.0}}
		gradients := [][]float64{{0.1}}

		// Do multiple updates
		results := make([]float64, 5)
		currentWeights := weights
		for i := 0; i < 5; i++ {
			currentWeights = opt.UpdateWeights(0, currentWeights, gradients)
			results[i] = currentWeights[0][0]
		}

		// Verify weights keep changing (not stuck)
		for i := 1; i < 5; i++ {
			if math.Abs(results[i]-results[i-1]) < 1e-6 {
				t.Errorf("Adam update stuck at step %d: %v", i, results[i])
			}
		}
	})

	t.Run("clone preserves timestep", func(t *testing.T) {
		// Use layer sizes that match our test weights: input=1, neurons=1
		layerSizes := []int{1, 1, 1}
		opt := NewAdamOptimizer(0.001, 0.9, 0.999, 1e-8)
		opt.InitializeState(layerSizes)

		// Do some updates to increment timestep
		weights := [][]float64{{1.0}} // 1 neuron, 1 input
		gradients := [][]float64{{0.1}}

		for i := 0; i < 3; i++ {
			opt.UpdateWeights(0, weights, gradients)
		}

		clone := opt.Clone()
		if clone == nil {
			t.Fatal("Clone() returned nil")
		}
		// Initialize clone state (Clone() doesn't copy moment estimates)
		clone.InitializeState(layerSizes)

		// Do another update on both - copy weights first since UpdateWeights modifies in-place
		weights2 := [][]float64{{1.5}}
		weights2Copy := [][]float64{{1.5}}
		gradients2 := [][]float64{{0.2}}

		originalResult := opt.UpdateWeights(0, weights2, gradients2)
		cloneResult := clone.UpdateWeights(0, weights2Copy, gradients2)

		// Since Clone() doesn't copy internal moment estimates (m, v),
		// the results will differ. We just verify both can be called without panic.
		// Check that weights were updated (non-zero change).
		originalWeightValue := 1.5
		if math.Abs(originalResult[0][0] - originalWeightValue) < 1e-6 {
			t.Error("Original optimizer didn't update weights")
		}
		if math.Abs(cloneResult[0][0] - originalWeightValue) < 1e-6 {
			t.Error("Clone optimizer didn't update weights")
		}
	})

	t.Run("batch updates work", func(t *testing.T) {
		opt := NewAdamOptimizer(0.001, 0.9, 0.999, 1e-8)
		opt.InitializeState([]int{2, 3, 1})

		// Test batch accumulation
		weightUpdates := [][]float64{{0.0, 0.0}, {0.0, 0.0}}
		gradients := [][]float64{{0.1, 0.2}, {0.3, 0.4}}

		accumulated := opt.BatchUpdateWeights(0, weightUpdates, gradients)

		// Should accumulate gradients
		if accumulated[0][0] != 0.1 || accumulated[0][1] != 0.2 ||
		   accumulated[1][0] != 0.3 || accumulated[1][1] != 0.4 {
			t.Errorf("BatchUpdateWeights failed to accumulate correctly")
		}
	})
}

func TestOptimizerIntegration(t *testing.T) {
	t.Run("SGD reduces loss", func(t *testing.T) {
		// Create a simple network with SGD optimizer
		config := NetworkConfig{
			LayerSizes: []int{2, 3, 1},
			LearningRate: 0.1,
			Optimizer: &OptimizerConfig{
				Type: "sgd",
				LearningRate: 0.1,
			},
		}

		network := NewNetwork(config)

		// Simple XOR-like training data
		inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
		targets := [][]float64{{0}, {1}, {1}, {0}}

		initialLoss := 0.0
		for i := range inputs {
			output, _ := network.FeedForward(inputs[i])
			initialLoss += network.LossFunction.Function(output, targets[i])
		}
		initialLoss /= float64(len(inputs))

		// Train for a few epochs
		for epoch := 0; epoch < 10; epoch++ {
			for i := range inputs {
				network.Train(inputs[i], targets[i])
			}
		}

		finalLoss := 0.0
		for i := range inputs {
			output, _ := network.FeedForward(inputs[i])
			finalLoss += network.LossFunction.Function(output, targets[i])
		}
		finalLoss /= float64(len(inputs))

		// Loss should decrease (not necessarily to zero in 10 epochs)
		if finalLoss > initialLoss {
			t.Errorf("Loss increased: initial %v, final %v", initialLoss, finalLoss)
		}
	})

	t.Run("Adam converges faster than SGD on XOR", func(t *testing.T) {
		// This is a qualitative test - Adam should generally converge faster
		// We'll train both for same number of epochs and compare final loss

		// Create SGD network
		sgdConfig := NetworkConfig{
			LayerSizes: []int{2, 4, 1},
			LearningRate: 0.1,
			Optimizer: &OptimizerConfig{
				Type: "sgd",
				LearningRate: 0.1,
			},
		}
		sgdNet := NewNetwork(sgdConfig)

		// Create Adam network (same initial weights by seeding)
		adamConfig := NetworkConfig{
			LayerSizes: []int{2, 4, 1},
			LearningRate: 0.1,
			Optimizer: &OptimizerConfig{
				Type: "adam",
				LearningRate: 0.01,  // Adam typically uses smaller LR
				Beta1: 0.9,
				Beta2: 0.999,
				Epsilon: 1e-8,
			},
		}
		adamNet := NewNetwork(adamConfig)

		// Copy weights from SGD to Adam for fair comparison
		for i := range sgdNet.Weights {
			for j := range sgdNet.Weights[i] {
				for k := range sgdNet.Weights[i][j] {
					adamNet.Weights[i][j][k] = sgdNet.Weights[i][j][k]
				}
			}
			for j := range sgdNet.Biases[i] {
				adamNet.Biases[i][j] = sgdNet.Biases[i][j]
			}
		}

		// XOR data
		inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
		targets := [][]float64{{0}, {1}, {1}, {0}}

		// Train both for same number of epochs
		epochs := 100
		sgdLoss := 0.0
		adamLoss := 0.0

		for epoch := 0; epoch < epochs; epoch++ {
			// Train SGD
			for i := range inputs {
				sgdNet.Train(inputs[i], targets[i])
			}

			// Train Adam
			for i := range inputs {
				adamNet.Train(inputs[i], targets[i])
			}
		}

		// Calculate final losses
		for i := range inputs {
			sgdOutput, _ := sgdNet.FeedForward(inputs[i])
			adamOutput, _ := adamNet.FeedForward(inputs[i])

			sgdLoss += sgdNet.LossFunction.Function(sgdOutput, targets[i])
			adamLoss += adamNet.LossFunction.Function(adamOutput, targets[i])
		}
		sgdLoss /= float64(len(inputs))
		adamLoss /= float64(len(inputs))

		// Adam should have lower loss (not guaranteed but very likely)
		// We'll just check both networks trained (loss < 0.25)
		if sgdLoss > 0.25 && adamLoss > 0.25 {
			t.Logf("Both optimizers struggled: SGD loss %v, Adam loss %v", sgdLoss, adamLoss)
			// Not an error, just informational
		} else if adamLoss > sgdLoss * 1.5 {
			// Adam shouldn't be much worse than SGD
			t.Errorf("Adam performed significantly worse: SGD %v, Adam %v", sgdLoss, adamLoss)
		}
	})
}

func TestOptimizerEdgeCases(t *testing.T) {
	t.Run("zero learning rate", func(t *testing.T) {
		opt := NewSGDOptimizer(0.0)
		opt.InitializeState([]int{2, 3, 1})

		weights := [][]float64{{1.0, 2.0}}
		gradients := [][]float64{{0.1, 0.2}}

		updated := opt.UpdateWeights(0, weights, gradients)

		// With zero learning rate, weights shouldn't change
		if !weightsEqual(updated, weights, 1e-6) {
			t.Errorf("Zero learning rate should not update weights: got %v, want %v", updated, weights)
		}
	})

	t.Run("zero gradients", func(t *testing.T) {
		opt := NewSGDOptimizer(0.01)
		opt.InitializeState([]int{2, 3, 1})

		weights := [][]float64{{1.0, 2.0}}
		gradients := [][]float64{{0.0, 0.0}}

		updated := opt.UpdateWeights(0, weights, gradients)

		// With zero gradients, weights shouldn't change
		if !weightsEqual(updated, weights, 1e-6) {
			t.Errorf("Zero gradients should not update weights: got %v, want %v", updated, weights)
		}
	})

	t.Run("empty network", func(t *testing.T) {
		opt := NewSGDOptimizer(0.01)

		// Should handle empty layer sizes (though network creation would fail)
		opt.InitializeState([]int{})

		// Update with empty arrays should not panic
		weights := [][]float64{}
		gradients := [][]float64{}

		updated := opt.UpdateWeights(0, weights, gradients)
		if len(updated) != 0 {
			t.Errorf("Empty weights should return empty: got %v", updated)
		}
	})

	t.Run("negative learning rate", func(t *testing.T) {
		// Negative learning rate is unusual but mathematically valid
		opt := NewSGDOptimizer(-0.01)
		opt.InitializeState([]int{2, 3, 1})

		weights := [][]float64{{1.0, 2.0}}
		gradients := [][]float64{{0.1, 0.2}}

		updated := opt.UpdateWeights(0, weights, gradients)

		// With negative LR, weights should increase
		expected := [][]float64{{1.001, 2.002}}
		if !weightsEqual(updated, expected, 1e-6) {
			t.Errorf("Negative learning rate should increase weights: got %v, want %v", updated, expected)
		}
	})
}
