package main

import (
	"fmt"
	"math"

	"github.com/jsoprych/goneurotic/network" // Adjust the import path based on your actual repository structure
)

func main() {
	layers := []int{3, 5, 1}                   // Simplified network: 3 input nodes, 5 hidden nodes, 1 output node
	network := network.NewNetwork(layers, 0.1) // Adjusted learning rate for better stability

	// Training data for 3-input AND gate
	inputs := [][]float64{{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}}
	targets := [][]float64{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {1}}

	// Train the network
	for epoch := 0; epoch < 10000; epoch++ {
		totalError := 0.0
		for j := range inputs {
			network.Train(inputs[j], targets[j])
			output := network.FeedForward(inputs[j])[len(layers)-1]
			totalError += math.Pow(targets[j][0]-output[0], 2)
		}
		if epoch%1000 == 0 {
			fmt.Printf("Epoch: %d, Total Error: %f\n", epoch, totalError)
		}
	}

	// Test the network
	for _, input := range inputs {
		output := network.FeedForward(input)[len(layers)-1]
		fmt.Printf("Input: %v, Output: %v\n", input, output)
	}
}
