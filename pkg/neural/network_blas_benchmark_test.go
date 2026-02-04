package neural

import (
	"math/rand"
	"testing"
)

func generateRandomDataBLAS(inputSize, outputSize, numSamples int) ([][]float64, [][]float64) {
	inputs := make([][]float64, numSamples)
	targets := make([][]float64, numSamples)
	for i := range inputs {
		inputs[i] = make([]float64, inputSize)
		targets[i] = make([]float64, outputSize)
		for j := range inputs[i] {
			inputs[i][j] = rand.Float64()
		}
		for j := range targets[i] {
			targets[i][j] = rand.Float64()
		}
	}
	return inputs, targets
}

// BenchmarkFeedForwardBLASSmall compares BLAS vs regular forward pass for small network
func BenchmarkFeedForwardBLASSmall(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{10, 20, 10, 5},
	}
	regularNet := NewNetwork(config)
	blasNet := NewBLASNetwork(config)

	// Ensure same weights for fair comparison
	for i := range regularNet.Weights {
		blasNet.blasOps.ConvertWeightsToFlat(i, regularNet.Weights[i])
		blasNet.blasOps.ConvertBiasesToFlat(i, regularNet.Biases[i])
	}
	blasNet.weightsConverted = true
	blasNet.ConvertFromFlat() // Sync back to jagged format

	input := make([]float64, 10)
	for i := range input {
		input[i] = rand.Float64()
	}

	b.Run("Regular", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			regularNet.FeedForward(input)
		}
	})

	b.Run("BLAS", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			blasNet.FeedForwardBLAS(input)
		}
	})
}

// BenchmarkFeedForwardBLASMedium compares BLAS vs regular forward pass for medium network
func BenchmarkFeedForwardBLASMedium(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{50, 100, 50, 10},
	}
	regularNet := NewNetwork(config)
	blasNet := NewBLASNetwork(config)

	for i := range regularNet.Weights {
		blasNet.blasOps.ConvertWeightsToFlat(i, regularNet.Weights[i])
		blasNet.blasOps.ConvertBiasesToFlat(i, regularNet.Biases[i])
	}
	blasNet.weightsConverted = true
	blasNet.ConvertFromFlat()

	input := make([]float64, 50)
	for i := range input {
		input[i] = rand.Float64()
	}

	b.Run("Regular", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			regularNet.FeedForward(input)
		}
	})

	b.Run("BLAS", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			blasNet.FeedForwardBLAS(input)
		}
	})
}

// BenchmarkFeedForwardBLASLarge compares BLAS vs regular forward pass for large network
func BenchmarkFeedForwardBLASLarge(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{100, 200, 100, 50, 10},
	}
	regularNet := NewNetwork(config)
	blasNet := NewBLASNetwork(config)

	for i := range regularNet.Weights {
		blasNet.blasOps.ConvertWeightsToFlat(i, regularNet.Weights[i])
		blasNet.blasOps.ConvertBiasesToFlat(i, regularNet.Biases[i])
	}
	blasNet.weightsConverted = true
	blasNet.ConvertFromFlat()

	input := make([]float64, 100)
	for i := range input {
		input[i] = rand.Float64()
	}

	b.Run("Regular", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			regularNet.FeedForward(input)
		}
	})

	b.Run("BLAS", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			blasNet.FeedForwardBLAS(input)
		}
	})
}

// BenchmarkTrainBLASSmall compares BLAS vs regular training for small network
func BenchmarkTrainBLASSmall(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{10, 20, 5},
	}
	regularNet := NewNetwork(config)
	blasNet := NewBLASNetwork(config)

	for i := range regularNet.Weights {
		blasNet.blasOps.ConvertWeightsToFlat(i, regularNet.Weights[i])
		blasNet.blasOps.ConvertBiasesToFlat(i, regularNet.Biases[i])
	}
	blasNet.weightsConverted = true
	blasNet.ConvertFromFlat()

	input := make([]float64, 10)
	target := make([]float64, 5)
	for i := range input {
		input[i] = rand.Float64()
	}
	for i := range target {
		target[i] = rand.Float64()
	}

	b.Run("Regular", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			regularNet.Train(input, target)
		}
	})

	b.Run("BLAS", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			blasNet.TrainBLAS(input, target)
		}
	})
}

// BenchmarkTrainBLASMedium compares BLAS vs regular training for medium network
func BenchmarkTrainBLASMedium(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{50, 100, 20},
	}
	regularNet := NewNetwork(config)
	blasNet := NewBLASNetwork(config)

	for i := range regularNet.Weights {
		blasNet.blasOps.ConvertWeightsToFlat(i, regularNet.Weights[i])
		blasNet.blasOps.ConvertBiasesToFlat(i, regularNet.Biases[i])
	}
	blasNet.weightsConverted = true
	blasNet.ConvertFromFlat()

	input := make([]float64, 50)
	target := make([]float64, 20)
	for i := range input {
		input[i] = rand.Float64()
	}
	for i := range target {
		target[i] = rand.Float64()
	}

	b.Run("Regular", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			regularNet.Train(input, target)
		}
	})

	b.Run("BLAS", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			blasNet.TrainBLAS(input, target)
		}
	})
}

// BenchmarkConversionOverhead measures the cost of converting weights to/from flat format
func BenchmarkConversionOverhead(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{50, 100, 20},
	}
	network := NewBLASNetwork(config)

	b.Run("ConvertToFlat", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			network.ConvertToFlat()
		}
	})

	b.Run("ConvertFromFlat", func(b *testing.B) {
		network.ConvertToFlat() // Ensure we have flat buffers first
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			network.ConvertFromFlat()
		}
	})
}

// BenchmarkBLASNetworkCreation measures the overhead of creating a BLAS-optimized network
func BenchmarkBLASNetworkCreation(b *testing.B) {
	for i := 0; i < b.N; i++ {
		config := NetworkConfig{
			LayerSizes: []int{50, 100, 20},
		}
		_ = NewBLASNetwork(config)
	}
}

// BenchmarkBatchTrainBLASSmall compares BLAS vs regular batch training for small batches
func BenchmarkBatchTrainBLASSmall(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{10, 20, 5},
	}
	regularNet := NewNetwork(config)
	blasNet := NewBLASNetwork(config)

	for i := range regularNet.Weights {
		blasNet.blasOps.ConvertWeightsToFlat(i, regularNet.Weights[i])
		blasNet.blasOps.ConvertBiasesToFlat(i, regularNet.Biases[i])
	}
	blasNet.weightsConverted = true
	blasNet.ConvertFromFlat()

	inputs, targets := generateRandomDataBLAS(10, 5, 32) // batch size 32

	b.Run("Regular", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			regularNet.BatchTrain(inputs, targets)
		}
	})

	b.Run("BLAS", func(b *testing.B) {
		// Use the new BLAS-optimized batch training
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			blasNet.BatchTrainBLAS(inputs, targets)
		}
	})
}

// BenchmarkBatchTrainBLASMedium compares BLAS vs regular batch training for medium batches
func BenchmarkBatchTrainBLASMedium(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{50, 100, 20},
	}
	regularNet := NewNetwork(config)
	blasNet := NewBLASNetwork(config)

	for i := range regularNet.Weights {
		blasNet.blasOps.ConvertWeightsToFlat(i, regularNet.Weights[i])
		blasNet.blasOps.ConvertBiasesToFlat(i, regularNet.Biases[i])
	}
	blasNet.weightsConverted = true
	blasNet.ConvertFromFlat()

	inputs, targets := generateRandomDataBLAS(50, 20, 64) // batch size 64

	b.Run("Regular", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			regularNet.BatchTrain(inputs, targets)
		}
	})

	b.Run("BLAS", func(b *testing.B) {
		// Use the new BLAS-optimized batch training
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			blasNet.BatchTrainBLAS(inputs, targets)
		}
	})
}

// BenchmarkBatchTrainBLASLarge compares BLAS vs regular batch training for large batches
func BenchmarkBatchTrainBLASLarge(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{50, 100, 20},
	}
	regularNet := NewNetwork(config)
	blasNet := NewBLASNetwork(config)

	for i := range regularNet.Weights {
		blasNet.blasOps.ConvertWeightsToFlat(i, regularNet.Weights[i])
		blasNet.blasOps.ConvertBiasesToFlat(i, regularNet.Biases[i])
	}
	blasNet.weightsConverted = true
	blasNet.ConvertFromFlat()

	inputs, targets := generateRandomDataBLAS(50, 20, 256) // batch size 256

	b.Run("Regular", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			regularNet.BatchTrain(inputs, targets)
		}
	})

	b.Run("BLAS", func(b *testing.B) {
		// Use the new BLAS-optimized batch training
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			blasNet.BatchTrainBLAS(inputs, targets)
		}
	})
}

// BenchmarkMatrixVectorMultiplication isolates the core BLAS operation
func BenchmarkMatrixVectorMultiplication(b *testing.B) {
	// Create a single layer to test GEMV performance
	fanIn, fanOut := 100, 200

	// Create weight matrix and vectors
	weights := make([][]float64, fanOut)
	for i := range weights {
		weights[i] = make([]float64, fanIn)
		for j := range weights[i] {
			weights[i][j] = rand.Float64()
		}
	}

	input := make([]float64, fanIn)
	for i := range input {
		input[i] = rand.Float64()
	}

	output := make([]float64, fanOut)

	// Create BLAS optimizer for this layer
	layerSizes := []int{fanIn, fanOut}
	optimizer := NewBLASOptimizer(layerSizes)
	optimizer.ConvertWeightsToFlat(0, weights)

	b.Run("ManualLoop", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Manual matrix-vector multiplication
			for j := 0; j < fanOut; j++ {
				sum := 0.0
				for k := 0; k < fanIn; k++ {
					sum += weights[j][k] * input[k]
				}
				output[j] = sum
			}
		}
	})

	b.Run("BLAS_GEMV", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// BLAS matrix-vector multiplication
			optimizer.ForwardPassBLAS(0, input, output, nil)
		}
	})
}

// BenchmarkRank1Update isolates the GER operation for weight updates
func BenchmarkRank1Update(b *testing.B) {
	fanIn, fanOut := 100, 200

	// Create weight matrix and vectors
	weights := make([][]float64, fanOut)
	for i := range weights {
		weights[i] = make([]float64, fanIn)
		for j := range weights[i] {
			weights[i][j] = rand.Float64()
		}
	}

	delta := make([]float64, fanOut)
	activationPrev := make([]float64, fanIn)
	for i := range delta {
		delta[i] = rand.Float64()
	}
	for i := range activationPrev {
		activationPrev[i] = rand.Float64()
	}

	learningRate := 0.01

	// Create BLAS optimizer
	layerSizes := []int{fanIn, fanOut}
	optimizer := NewBLASOptimizer(layerSizes)
	optimizer.ConvertWeightsToFlat(0, weights)

	b.Run("ManualUpdate", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Manual rank-1 update
			for j := 0; j < fanOut; j++ {
				for k := 0; k < fanIn; k++ {
					weights[j][k] -= learningRate * delta[j] * activationPrev[k]
				}
			}
		}
	})

	b.Run("BLAS_GER", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// BLAS rank-1 update
			optimizer.UpdateWeightsBLAS(0, learningRate, delta, activationPrev)
		}
	})
}
