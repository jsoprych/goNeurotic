package network

import (
	"math"
	"testing"
)

func TestNetwork(t *testing.T) {
	layers := []int{3, 5, 1}
	n := NewNetwork(layers, 0.1)

	// Training data for 3-input AND gate
	inputs := [][]float64{{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}}
	targets := [][]float64{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {1}}

	// Train the network
	for epoch := 0; epoch < 10000; epoch++ {
		for j := range inputs {
			n.Train(inputs[j], targets[j])
		}
	}

	// Test the network
	for i, input := range inputs {
		output := n.FeedForward(input)[len(layers)-1][0]
		expected := targets[i][0]
		if math.Abs(output-expected) > 0.1 {
			t.Errorf("Input: %v, Expected: %v, Output: %v", input, expected, output)
		}
	}
}
