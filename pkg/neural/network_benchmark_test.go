package neural

import (
	"math/rand"
	"testing"
)

func generateRandomData(inputSize, outputSize, numSamples int) ([][]float64, [][]float64) {
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

func BenchmarkFeedForwardSmall(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{10, 20, 10, 5},
	}
	network := NewNetwork(config)
	input := make([]float64, 10)
	for i := range input {
		input[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network.FeedForward(input)
	}
}

func BenchmarkFeedForwardMedium(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{50, 100, 50, 10},
	}
	network := NewNetwork(config)
	input := make([]float64, 50)
	for i := range input {
		input[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network.FeedForward(input)
	}
}

func BenchmarkFeedForwardLarge(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{100, 200, 100, 50, 10},
	}
	network := NewNetwork(config)
	input := make([]float64, 100)
	for i := range input {
		input[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network.FeedForward(input)
	}
}

func BenchmarkTrainSmall(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{10, 20, 5},
	}
	network := NewNetwork(config)
	input := make([]float64, 10)
	target := make([]float64, 5)
	for i := range input {
		input[i] = rand.Float64()
	}
	for i := range target {
		target[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network.Train(input, target)
	}
}

func BenchmarkTrainMedium(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{50, 100, 20},
	}
	network := NewNetwork(config)
	input := make([]float64, 50)
	target := make([]float64, 20)
	for i := range input {
		input[i] = rand.Float64()
	}
	for i := range target {
		target[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network.Train(input, target)
	}
}

func BenchmarkBatchTrainSmallBatch(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{10, 20, 5},
	}
	network := NewNetwork(config)
	inputs, targets := generateRandomData(10, 5, 32) // batch size 32

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network.BatchTrain(inputs, targets)
	}
}

func BenchmarkBatchTrainMediumBatch(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{50, 100, 20},
	}
	network := NewNetwork(config)
	inputs, targets := generateRandomData(50, 20, 64) // batch size 64

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network.BatchTrain(inputs, targets)
	}
}

func BenchmarkBatchTrainLargeBatch(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{50, 100, 20},
	}
	network := NewNetwork(config)
	inputs, targets := generateRandomData(50, 20, 256) // batch size 256

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network.BatchTrain(inputs, targets)
	}
}

func BenchmarkNetworkCreation(b *testing.B) {
	for i := 0; i < b.N; i++ {
		config := NetworkConfig{
			LayerSizes: []int{50, 100, 50, 10},
		}
		_ = NewNetwork(config)
	}
}

func BenchmarkPredictSmall(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{10, 20, 5},
	}
	network := NewNetwork(config)
	input := make([]float64, 10)
	for i := range input {
		input[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network.Predict(input)
	}
}

func BenchmarkPredictMedium(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{50, 100, 20},
	}
	network := NewNetwork(config)
	input := make([]float64, 50)
	for i := range input {
		input[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network.Predict(input)
	}
}

// BenchmarkFeedForwardWithCache simulates the performance improvement
// from caching activation derivatives (quick win #2)
func BenchmarkFeedForwardWithCache(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{50, 100, 50, 10},
	}
	network := NewNetwork(config)
	input := make([]float64, 50)
	for i := range input {
		input[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Simulate cached derivative by computing once
		output, activations := network.FeedForward(input)
		_ = output
		_ = activations
		// In optimized version, derivatives would be cached during forward pass
	}
}

// BenchmarkBatchTrainParallel simulates parallel processing improvement
func BenchmarkBatchTrainParallel(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{50, 100, 20},
	}
	network := NewNetwork(config)
	inputs, targets := generateRandomData(50, 20, 256)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Current implementation (serial)
		network.BatchTrain(inputs, targets)
		// Parallel version would use goroutines
	}
}

// BenchmarkMemoryAllocation measures allocation overhead
func BenchmarkMemoryAllocation(b *testing.B) {
	config := NetworkConfig{
		LayerSizes: []int{100, 200, 100, 50, 10},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		network := NewNetwork(config)
		_ = network
	}
}
