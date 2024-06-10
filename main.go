package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Network struct represents a simple neural network with multiple layers
type Network struct {
	layers       []int
	weights      [][][]float64
	biases       [][]float64
	learningRate float64
}

// NewNetwork creates a new neural network with the specified layers and learning rate
func NewNetwork(layers []int, learningRate float64) *Network {
	network := &Network{
		layers:       layers,
		weights:      initializeWeights(layers),
		biases:       initializeBiases(layers),
		learningRate: learningRate,
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

func (n *Network) feedForward(input []float64) [][]float64 {
	activations := make([][]float64, len(n.layers))
	activations[0] = input
	for i := 0; i < len(n.layers)-1; i++ {
		inputs := dotProduct(n.weights[i], activations[i])
		addBias(inputs, n.biases[i])
		activations[i+1] = applyActivation(inputs)
	}
	return activations
}

func (n *Network) train(input, target []float64) {
	// Forward pass
	activations := n.feedForward(input)

	// Calculate output error
	outputErrors := make([]float64, len(target))
	for i := range target {
		outputErrors[i] = target[i] - activations[len(n.layers)-1][i]
	}

	// Backpropagate the error
	errors := outputErrors
	for i := len(n.layers) - 2; i >= 0; i-- {
		layerErrors := make([]float64, len(n.weights[i][0]))
		for j := range n.weights[i] {
			for k := range n.weights[i][j] {
				gradient := dsigmoid(activations[i+1][j]) * errors[j] * activations[i][k]
				n.weights[i][j][k] += n.learningRate * gradient
				layerErrors[k] += errors[j] * n.weights[i][j][k]
			}
			n.biases[i][j] += n.learningRate * dsigmoid(activations[i+1][j]) * errors[j]
		}
		errors = layerErrors
	}
}

func main() {
	layers := []int{3, 5, 1}           // Simplified network: 3 input nodes, 5 hidden nodes, 1 output node
	network := NewNetwork(layers, 0.1) // Adjusted learning rate for better stability

	// Training data for 3-input AND gate
	inputs := [][]float64{{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}}
	targets := [][]float64{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {1}}

	// Train the network
	for epoch := 0; epoch < 10000; epoch++ {
		totalError := 0.0
		for j := range inputs {
			network.train(inputs[j], targets[j])
			output := network.feedForward(inputs[j])[len(layers)-1]
			totalError += math.Pow(targets[j][0]-output[0], 2)
		}
		if epoch%1000 == 0 {
			fmt.Printf("Epoch: %d, Total Error: %f\n", epoch, totalError)
		}
	}

	// Test the network
	for _, input := range inputs {
		output := network.feedForward(input)[len(layers)-1]
		fmt.Printf("Input: %v, Output: %v\n", input, output)
	}
}
