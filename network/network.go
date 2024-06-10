package network

import (
	"math"
	"math/rand"
	"time"
)

// Network struct represents a simple neural network with multiple layers
type Network struct {
	Layers       []int
	Weights      [][][]float64
	Biases       [][]float64
	LearningRate float64
}

// NewNetwork creates a new neural network with the specified layers and learning rate
func NewNetwork(layers []int, learningRate float64) *Network {
	network := &Network{
		Layers:       layers,
		Weights:      initializeWeights(layers),
		Biases:       initializeBiases(layers),
		LearningRate: learningRate,
	}
	return network
}

func initializeWeights(layers []int) [][][]float64 {
	rand.Seed(time.Now().UnixNano())
	weights := make([][][]float64, len(layers)-1)
	for i := 0; i < len(layers)-1; i++ {
		weights[i] = make([][]float64, layers[i+1])
		for j := range weights[i] {
			weights[i][j] = make([]float64, layers[i])
			for k := range weights[i][j] {
				weights[i][j][k] = rand.Float64()*0.2 - 0.1 // Initialize weights to small random values
			}
		}
	}
	return weights
}

func initializeBiases(layers []int) [][]float64 {
	biases := make([][]float64, len(layers)-1)
	for i := 1; i < len(layers); i++ {
		biases[i-1] = make([]float64, layers[i])
		for j := range biases[i-1] {
			biases[i-1][j] = rand.Float64()*0.2 - 0.1 // Initialize biases to small random values
		}
	}
	return biases
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dsigmoid(y float64) float64 {
	return y * (1 - y)
}

func dotProduct(weights [][]float64, inputs []float64) []float64 {
	outputs := make([]float64, len(weights))
	for i := range weights {
		sum := 0.0
		for j := range weights[i] {
			sum += weights[i][j] * inputs[j]
		}
		outputs[i] = sum
	}
	return outputs
}

func addBias(inputs, bias []float64) {
	for i := range inputs {
		inputs[i] += bias[i]
	}
}

func applyActivation(inputs []float64) []float64 {
	outputs := make([]float64, len(inputs))
	for i, input := range inputs {
		outputs[i] = sigmoid(input)
	}
	return outputs
}

func (n *Network) FeedForward(input []float64) [][]float64 {
	activations := make([][]float64, len(n.Layers))
	activations[0] = input
	for i := 0; i < len(n.Layers)-1; i++ {
		inputs := dotProduct(n.Weights[i], activations[i])
		addBias(inputs, n.Biases[i])
		activations[i+1] = applyActivation(inputs)
	}
	return activations
}

func (n *Network) Train(input, target []float64) {
	// Forward pass
	activations := n.FeedForward(input)

	// Calculate output error
	outputErrors := make([]float64, len(target))
	for i := range target {
		outputErrors[i] = target[i] - activations[len(n.Layers)-1][i]
	}

	// Backpropagate the error
	errors := outputErrors
	for i := len(n.Layers) - 2; i >= 0; i-- {
		layerErrors := make([]float64, len(n.Weights[i][0]))
		for j := range n.Weights[i] {
			for k := range n.Weights[i][j] {
				gradient := dsigmoid(activations[i+1][j]) * errors[j] * activations[i][k]
				n.Weights[i][j][k] += n.LearningRate * gradient
				layerErrors[k] += errors[j] * n.Weights[i][j][k]
			}
			n.Biases[i][j] += n.LearningRate * dsigmoid(activations[i+1][j]) * errors[j]
		}
		errors = layerErrors
	}
}
