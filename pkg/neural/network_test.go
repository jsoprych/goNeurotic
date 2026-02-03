package neural

import (
	"math"
	"math/rand"
	"testing"
)

func TestNewNetwork(t *testing.T) {
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
			name: "negative layer size should panic",
			config: NetworkConfig{
				LayerSizes: []int{2, -1, 1},
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
					t.Errorf("NewNetwork() panicked unexpectedly: %v", r)
				} else if r == nil && tt.shouldPanic {
					t.Error("NewNetwork() should have panicked but didn't")
				}
			}()

			network := NewNetwork(tt.config)
			if !tt.shouldPanic {
				if network == nil {
					t.Error("NewNetwork() returned nil")
				}
				if len(network.LayerSizes) != len(tt.config.LayerSizes) {
					t.Errorf("Expected %d layers, got %d", len(tt.config.LayerSizes), len(network.LayerSizes))
				}
			}
		})
	}
}

func TestNetwork_FeedForward(t *testing.T) {
	config := NetworkConfig{
		LayerSizes: []int{2, 3, 1},
		Activation: Sigmoid,
		OutputActivation: Sigmoid,
	}
	network := NewNetwork(config)

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
					t.Errorf("FeedForward() panicked unexpectedly: %v", r)
				} else if r == nil && tt.shouldPanic {
					t.Error("FeedForward() should have panicked but didn't")
				}
			}()

			output, activations := network.FeedForward(tt.input)
			if !tt.shouldPanic {
				if output == nil {
					t.Error("FeedForward() returned nil output")
				}
				if activations == nil {
					t.Error("FeedForward() returned nil activations")
				}
				if len(output) != config.LayerSizes[len(config.LayerSizes)-1] {
					t.Errorf("Expected output size %d, got %d", config.LayerSizes[len(config.LayerSizes)-1], len(output))
				}
				if len(activations) != len(config.LayerSizes) {
					t.Errorf("Expected %d activation layers, got %d", len(config.LayerSizes), len(activations))
				}
			}
		})
	}
}

func TestNetwork_Predict(t *testing.T) {
	config := NetworkConfig{
		LayerSizes: []int{1, 2, 1},
	}
	network := NewNetwork(config)

	input := []float64{0.5}
	output := network.Predict(input)

	if output == nil {
		t.Fatal("Predict() returned nil")
	}
	if len(output) != 1 {
		t.Errorf("Expected output size 1, got %d", len(output))
	}
	// Output should be between 0 and 1 for sigmoid (default)
	if output[0] < 0 || output[0] > 1 {
		t.Errorf("Output %f should be between 0 and 1 for sigmoid activation", output[0])
	}
}

func TestNetwork_TrainXOR(t *testing.T) {
	// XOR problem is a classic test for neural networks
	config := NetworkConfig{
		LayerSizes:       []int{2, 4, 1},
		LearningRate:     0.5,
		Activation:       Sigmoid,
		OutputActivation: Sigmoid,
		LossFunction:     MeanSquaredError,
	}
	network := NewNetwork(config)

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
	for epoch := 0; epoch < 5000; epoch++ {
		for i := range inputs {
			network.Train(inputs[i], targets[i])
		}
	}

	// Test predictions
	correct := 0
	for i, input := range inputs {
		output := network.Predict(input)
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
		t.Errorf("XOR test failed: only %d/4 correct (expected at least 3)", correct)
	}
}

func TestNetwork_BatchTrain(t *testing.T) {
	config := NetworkConfig{
		LayerSizes:       []int{2, 3, 1},
		LearningRate:     0.1,
		Activation:       ReLU,
		OutputActivation: Sigmoid,
		LossFunction:     MeanSquaredError,
	}
	network := NewNetwork(config)

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

	loss := network.BatchTrain(inputs, targets)
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		t.Errorf("BatchTrain() returned invalid loss: %f", loss)
	}
	if loss < 0 {
		t.Errorf("BatchTrain() returned negative loss: %f", loss)
	}

	// Test with empty batch
	emptyLoss := network.BatchTrain([][]float64{}, [][]float64{})
	if emptyLoss != 0.0 {
		t.Errorf("Expected loss 0.0 for empty batch, got %f", emptyLoss)
	}

	// Test with mismatched input/target sizes
	defer func() {
		if r := recover(); r == nil {
			t.Error("BatchTrain() should panic with mismatched input/target sizes")
		}
	}()
	network.BatchTrain([][]float64{{1, 2}}, [][]float64{{0}, {1}})
}

func TestActivationFunctions(t *testing.T) {
	tests := []struct {
		name       string
		activation ActivationFunc
		input      float64
		expected   float64
	}{
		{
			name:       "sigmoid zero",
			activation: Sigmoid,
			input:      0.0,
			expected:   0.5,
		},
		{
			name:       "sigmoid large positive",
			activation: Sigmoid,
			input:      10.0,
			expected:   math.Exp(10) / (1 + math.Exp(10)),
		},
		{
			name:       "relu positive",
			activation: ReLU,
			input:      5.0,
			expected:   5.0,
		},
		{
			name:       "relu negative",
			activation: ReLU,
			input:      -5.0,
			expected:   0.0,
		},
		{
			name:       "tanh zero",
			activation: Tanh,
			input:      0.0,
			expected:   0.0,
		},
		{
			name:       "linear",
			activation: Linear,
			input:      3.14,
			expected:   3.14,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.activation.Function(tt.input)
			if math.Abs(result-tt.expected) > 1e-10 {
				t.Errorf("%s(%f) = %f, expected %f", tt.name, tt.input, result, tt.expected)
			}

			// Test derivative at this point
			derivative := tt.activation.Derivative(result)
			// Simple finite difference check
			h := 1e-6
			fxph := tt.activation.Function(tt.input + h)
			fxmh := tt.activation.Function(tt.input - h)
			numericalDerivative := (fxph - fxmh) / (2 * h)

			if math.Abs(derivative-numericalDerivative) > 1e-5 {
				t.Errorf("%s derivative at %f = %f, numerical derivative = %f", tt.name, tt.input, derivative, numericalDerivative)
			}
		})
	}
}

func TestLossFunctions(t *testing.T) {
	tests := []struct {
		name      string
		loss      LossFunc
		predicted []float64
		target    []float64
		expected  float64
	}{
		{
			name:      "MSE perfect prediction",
			loss:      MeanSquaredError,
			predicted: []float64{0.5, 0.7},
			target:    []float64{0.5, 0.7},
			expected:  0.0,
		},
		{
			name:      "MSE error",
			loss:      MeanSquaredError,
			predicted: []float64{0.5, 0.5},
			target:    []float64{1.0, 0.0},
			expected:  (0.25 + 0.25) / 2, // (0.5^2 + 0.5^2)/2
		},
		{
			name:      "binary crossentropy perfect prediction 0",
			loss:      BinaryCrossEntropy,
			predicted: []float64{0.0001},
			target:    []float64{0.0},
			expected:  -math.Log(1 - 0.0001),
		},
		{
			name:      "binary crossentropy perfect prediction 1",
			loss:      BinaryCrossEntropy,
			predicted: []float64{0.9999},
			target:    []float64{1.0},
			expected:  -math.Log(0.9999),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loss := tt.loss.Function(tt.predicted, tt.target)
			if math.Abs(loss-tt.expected) > 1e-6 {
				t.Errorf("%s loss = %f, expected %f", tt.name, loss, tt.expected)
			}

			// Test derivative
			derivative := tt.loss.Derivative(tt.predicted, tt.target)
			if len(derivative) != len(tt.predicted) {
				t.Errorf("%s derivative length = %d, expected %d", tt.name, len(derivative), len(tt.predicted))
			}
		})
	}
}

func TestNetwork_SaveLoad(t *testing.T) {
	config := NetworkConfig{
		LayerSizes:       []int{3, 5, 2},
		LearningRate:     0.1,
		Activation:       Tanh,
		OutputActivation: Sigmoid,
		LossFunction:     MeanSquaredError,
	}
	original := NewNetwork(config)

	// Train a bit on random data
	for i := 0; i < 100; i++ {
		input := []float64{rand.Float64(), rand.Float64(), rand.Float64()}
		target := []float64{rand.Float64(), rand.Float64()}
		original.Train(input, target)
	}

	// Save to temp file
	tempFile := t.TempDir() + "/test_network.json"
	if err := original.Save(tempFile); err != nil {
		t.Fatalf("Save() failed: %v", err)
	}

	// Load from file
	loaded, err := Load(tempFile)
	if err != nil {
		t.Fatalf("Load() failed: %v", err)
	}

	// Compare configurations
	if len(original.LayerSizes) != len(loaded.LayerSizes) {
		t.Fatalf("Layer sizes mismatch: original %v, loaded %v", original.LayerSizes, loaded.LayerSizes)
	}
	for i := range original.LayerSizes {
		if original.LayerSizes[i] != loaded.LayerSizes[i] {
			t.Errorf("Layer %d size mismatch: original %d, loaded %d", i, original.LayerSizes[i], loaded.LayerSizes[i])
		}
	}

	// Compare weights and biases (approximately)
	for i := range original.Weights {
		for j := range original.Weights[i] {
			for k := range original.Weights[i][j] {
				if math.Abs(original.Weights[i][j][k]-loaded.Weights[i][j][k]) > 1e-10 {
					t.Errorf("Weight[%d][%d][%d] mismatch: original %f, loaded %f", i, j, k, original.Weights[i][j][k], loaded.Weights[i][j][k])
				}
			}
		}
	}

	for i := range original.Biases {
		for j := range original.Biases[i] {
			if math.Abs(original.Biases[i][j]-loaded.Biases[i][j]) > 1e-10 {
				t.Errorf("Bias[%d][%d] mismatch: original %f, loaded %f", i, j, original.Biases[i][j], loaded.Biases[i][j])
			}
		}
	}

	// Test prediction consistency
	testInput := []float64{0.1, 0.2, 0.3}
	originalOutput := original.Predict(testInput)
	loadedOutput := loaded.Predict(testInput)

	if len(originalOutput) != len(loadedOutput) {
		t.Fatalf("Output length mismatch: original %d, loaded %d", len(originalOutput), len(loadedOutput))
	}

	for i := range originalOutput {
		if math.Abs(originalOutput[i]-loadedOutput[i]) > 1e-10 {
			t.Errorf("Output[%d] mismatch: original %f, loaded %f", i, originalOutput[i], loadedOutput[i])
		}
	}
}

func TestNetwork_Clone(t *testing.T) {
	config := NetworkConfig{
		LayerSizes:       []int{2, 4, 1},
		LearningRate:     0.2,
		Activation:       ReLU,
		OutputActivation: Sigmoid,
		LossFunction:     BinaryCrossEntropy,
	}
	original := NewNetwork(config)

	// Train a bit
	for i := 0; i < 50; i++ {
		input := []float64{rand.Float64(), rand.Float64()}
		target := []float64{rand.Float64()}
		original.Train(input, target)
	}

	clone := original.Clone()

	// Verify they have same structure
	if len(original.LayerSizes) != len(clone.LayerSizes) {
		t.Fatalf("Layer sizes mismatch: original %v, clone %v", original.LayerSizes, clone.LayerSizes)
	}

	// Verify they produce same outputs
	testInput := []float64{0.3, 0.7}
	originalOutput := original.Predict(testInput)
	cloneOutput := clone.Predict(testInput)

	for i := range originalOutput {
		if math.Abs(originalOutput[i]-cloneOutput[i]) > 1e-10 {
			t.Errorf("Output[%d] mismatch: original %f, clone %f", i, originalOutput[i], cloneOutput[i])
		}
	}

	// Verify they are independent - training one shouldn't affect the other
	original.SetLearningRate(0.5)
	clone.SetLearningRate(0.1)

	if original.GetLearningRate() != 0.5 {
		t.Errorf("Original learning rate should be 0.5, got %f", original.GetLearningRate())
	}
	if clone.GetLearningRate() != 0.1 {
		t.Errorf("Clone learning rate should be 0.1, got %f", clone.GetLearningRate())
	}
}

func TestNetwork_GetSetLearningRate(t *testing.T) {
	config := NetworkConfig{
		LayerSizes:   []int{1, 2, 1},
		LearningRate: 0.01,
	}
	network := NewNetwork(config)

	if network.GetLearningRate() != 0.01 {
		t.Errorf("Initial learning rate should be 0.01, got %f", network.GetLearningRate())
	}

	network.SetLearningRate(0.05)
	if network.GetLearningRate() != 0.05 {
		t.Errorf("Learning rate should be 0.05 after SetLearningRate, got %f", network.GetLearningRate())
	}

	// Test invalid learning rate
	defer func() {
		if r := recover(); r == nil {
			t.Error("SetLearningRate() should panic for non-positive learning rate")
		}
	}()
	network.SetLearningRate(0.0)
}

func TestWeightInitialization(t *testing.T) {
	// Test custom weight initializer
	customWeightInitializer := func(fanIn, fanOut int) float64 {
		return 0.1 * float64(fanIn+fanOut)
	}
	customBiasInitializer := func() float64 {
		return 0.5
	}

	config := NetworkConfig{
		LayerSizes:        []int{3, 4, 2},
		WeightInitializer: customWeightInitializer,
		BiasInitializer:   customBiasInitializer,
	}
	network := NewNetwork(config)

	// Check weights were initialized with custom function
	for i := range network.Weights {
		fanIn := config.LayerSizes[i]
		fanOut := config.LayerSizes[i+1]
		expectedWeight := 0.1 * float64(fanIn+fanOut)
		for j := range network.Weights[i] {
			for k := range network.Weights[i][j] {
				if network.Weights[i][j][k] != expectedWeight {
					t.Errorf("Weight[%d][%d][%d] = %f, expected %f", i, j, k, network.Weights[i][j][k], expectedWeight)
				}
			}
		}
	}

	// Check biases were initialized with custom function
	for i := range network.Biases {
		for j := range network.Biases[i] {
			if network.Biases[i][j] != 0.5 {
				t.Errorf("Bias[%d][%d] = %f, expected 0.5", i, j, network.Biases[i][j])
			}
		}
	}
}

func TestNetwork_GetLayerSizes(t *testing.T) {
	layerSizes := []int{5, 10, 7, 3}
	config := NetworkConfig{
		LayerSizes: layerSizes,
	}
	network := NewNetwork(config)

	retrieved := network.GetLayerSizes()
	if len(retrieved) != len(layerSizes) {
		t.Fatalf("GetLayerSizes() returned %d layers, expected %d", len(retrieved), len(layerSizes))
	}

	for i := range layerSizes {
		if retrieved[i] != layerSizes[i] {
			t.Errorf("Layer %d size = %d, expected %d", i, retrieved[i], layerSizes[i])
		}
	}
}
