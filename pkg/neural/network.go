package neural

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"
)

// ActivationFunc represents an activation function and its derivative
type ActivationFunc struct {
	Function     func(float64) float64 `json:"-"`
	Derivative   func(float64) float64 `json:"-"`
	Name         string               `json:"name"`
}



func (a *ActivationFunc) UnmarshalJSON(data []byte) error {
	type Alias ActivationFunc
	aux := &struct {
		*Alias
	}{
		Alias: (*Alias)(a),
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	initMaps()
	activation, ok := activationsByName[a.Name]
	if !ok {
		return fmt.Errorf("unknown activation function: %s", a.Name)
	}
	a.Function = activation.Function
	a.Derivative = activation.Derivative
	return nil
}

// Predefined activation functions
var (
	Sigmoid = ActivationFunc{
		Name: "sigmoid",
		Function: func(x float64) float64 {
			return 1.0 / (1.0 + math.Exp(-x))
		},
		Derivative: func(y float64) float64 {
			return y * (1.0 - y)
		},
	}

	ReLU = ActivationFunc{
		Name: "relu",
		Function: func(x float64) float64 {
			if x > 0 {
				return x
			}
			return 0
		},
		Derivative: func(y float64) float64 {
			if y > 0 {
				return 1.0
			}
			return 0
		},
	}

	Tanh = ActivationFunc{
		Name: "tanh",
		Function: func(x float64) float64 {
			return math.Tanh(x)
		},
		Derivative: func(y float64) float64 {
			return 1.0 - y*y
		},
	}

	Linear = ActivationFunc{
		Name: "linear",
		Function: func(x float64) float64 {
			return x
		},
		Derivative: func(y float64) float64 {
			return 1.0
		},
	}
)

// LossFunc represents a loss function and its derivative
type LossFunc struct {
	Function   func(predicted, target []float64) float64 `json:"-"`
	Derivative func(predicted, target []float64) []float64 `json:"-"`
	Name       string `json:"name"`
}



func (l *LossFunc) UnmarshalJSON(data []byte) error {
	type Alias LossFunc
	aux := &struct {
		*Alias
	}{
		Alias: (*Alias)(l),
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	initMaps()
	loss, ok := lossesByName[l.Name]
	if !ok {
		return fmt.Errorf("unknown loss function: %s", l.Name)
	}
	l.Function = loss.Function
	l.Derivative = loss.Derivative
	return nil
}

// Predefined loss functions
var (
	MeanSquaredError = LossFunc{
		Name: "mse",
		Function: func(predicted, target []float64) float64 {
			sum := 0.0
			for i := range predicted {
				diff := predicted[i] - target[i]
				sum += diff * diff
			}
			return sum / float64(len(predicted))
		},
		Derivative: func(predicted, target []float64) []float64 {
			derivatives := make([]float64, len(predicted))
			for i := range predicted {
				derivatives[i] = 2.0 * (predicted[i] - target[i]) / float64(len(predicted))
			}
			return derivatives
		},
	}

	BinaryCrossEntropy = LossFunc{
		Name: "binary_crossentropy",
		Function: func(predicted, target []float64) float64 {
			sum := 0.0
			for i := range predicted {
				// Avoid log(0) issues
				p := math.Max(1e-15, math.Min(1-1e-15, predicted[i]))
				sum += target[i]*math.Log(p) + (1-target[i])*math.Log(1-p)
			}
			return -sum / float64(len(predicted))
		},
		Derivative: func(predicted, target []float64) []float64 {
			derivatives := make([]float64, len(predicted))
			for i := range predicted {
				// Avoid division by zero
				p := math.Max(1e-15, math.Min(1-1e-15, predicted[i]))
				derivatives[i] = (p - target[i]) / (p * (1 - p)) / float64(len(predicted))
			}
			return derivatives
		},
	}
)

var (
	activationsByName map[string]ActivationFunc
	lossesByName      map[string]LossFunc
	once              sync.Once
)

func initMaps() {
	once.Do(func() {
		activationsByName = map[string]ActivationFunc{
			"sigmoid": Sigmoid,
			"relu":    ReLU,
			"tanh":    Tanh,
			"linear":  Linear,
		}

		lossesByName = map[string]LossFunc{
			"mse":                  MeanSquaredError,
			"binary_crossentropy":  BinaryCrossEntropy,
		}
	})
}

// GetActivationByName returns an activation function by name
func GetActivationByName(name string) (ActivationFunc, error) {
	initMaps()
	activation, ok := activationsByName[name]
	if !ok {
		return ActivationFunc{}, fmt.Errorf("unknown activation function: %s", name)
	}
	return activation, nil
}

// GetLossByName returns a loss function by name
func GetLossByName(name string) (LossFunc, error) {
	initMaps()
	loss, ok := lossesByName[name]
	if !ok {
		return LossFunc{}, fmt.Errorf("unknown loss function: %s", name)
	}
	return loss, nil
}

// NetworkConfig holds configuration for creating a neural network
// NetworkConfig holds configuration parameters for creating a neural network.
// It specifies the architecture, learning hyperparameters, activation functions,
// loss function, and optional custom weight/bias initializers.
type NetworkConfig struct {
	LayerSizes        []int           `json:"layer_sizes"`        // Number of neurons in each layer
	LearningRate      float64         `json:"learning_rate"`      // Learning rate for gradient descent
	Activation        ActivationFunc  `json:"activation"`        // Activation function for hidden layers
	OutputActivation  ActivationFunc  `json:"output_activation"`  // Activation function for output layer
	LossFunction      LossFunc        `json:"loss_function"`      // Loss function for training
	WeightInitializer func(int, int) float64 `json:"-"` // Function to initialize weights (not serialized)
	BiasInitializer   func() float64         `json:"-"` // Function to initialize biases (not serialized)
}

// Network represents a neural network
// Network represents a neural network with configurable architecture, activation functions,
// and loss functions. It stores weights and biases for each layer and provides methods
// for forward propagation, training, and serialization.
type Network struct {
	Config         NetworkConfig      `json:"config"`
	Weights        [][][]float64      `json:"weights"` // weights[layer][neuron][input]
	Biases         [][]float64        `json:"biases"`   // biases[layer][neuron]
	LayerSizes     []int              `json:"layer_sizes"`
	LearningRate   float64            `json:"learning_rate"`
	Activation     ActivationFunc     `json:"activation"`
	OutputActivation ActivationFunc   `json:"output_activation"`
	LossFunction   LossFunc           `json:"loss_function"`
	// Optimization buffers (not serialized)
	activationBuffers  [][]float64    `json:"-"` // pre-allocated activation storage
	derivativeBuffers  [][]float64    `json:"-"` // pre-allocated derivative storage
	deltasBuffers      [][]float64    `json:"-"` // pre-allocated delta storage for backprop
	weightUpdateBuffers [][][]float64 `json:"-"` // batch training weight updates
	biasUpdateBuffers  [][]float64    `json:"-"` // batch training bias updates
	buffersInitialized bool           `json:"-"` // track if buffers are initialized
}

// NewNetwork creates a new neural network with the given configuration
func NewNetwork(config NetworkConfig) *Network {
	// Set defaults if not provided
	if config.LearningRate <= 0 {
		config.LearningRate = 0.01
	}

	if config.Activation.Name == "" {
		config.Activation = ReLU
	}

	if config.OutputActivation.Name == "" {
		config.OutputActivation = Sigmoid
	}

	if config.LossFunction.Name == "" {
		config.LossFunction = MeanSquaredError
	}

	if config.WeightInitializer == nil {
		config.WeightInitializer = func(fanIn, fanOut int) float64 {
			// Xavier/Glorot initialization
			limit := math.Sqrt(6.0 / float64(fanIn+fanOut))
			return rand.Float64()*2*limit - limit
		}
	}

	if config.BiasInitializer == nil {
		config.BiasInitializer = func() float64 {
			return 0.0
		}
	}

	// Validate layer sizes
	if len(config.LayerSizes) < 2 {
		panic("Network must have at least 2 layers (input and output)")
	}

	for i, size := range config.LayerSizes {
		if size <= 0 {
			panic(fmt.Sprintf("Layer %d must have positive size, got %d", i, size))
		}
	}

	// Initialize weights and biases
	numLayers := len(config.LayerSizes)
	weights := make([][][]float64, numLayers-1)
	biases := make([][]float64, numLayers-1)

	for i := 0; i < numLayers-1; i++ {
		fanIn := config.LayerSizes[i]
		fanOut := config.LayerSizes[i+1]

		weights[i] = make([][]float64, fanOut)
		for j := range weights[i] {
			weights[i][j] = make([]float64, fanIn)
			for k := range weights[i][j] {
				weights[i][j][k] = config.WeightInitializer(fanIn, fanOut)
			}
		}

		biases[i] = make([]float64, fanOut)
		for j := range biases[i] {
			biases[i][j] = config.BiasInitializer()
		}
	}

	// Initialize optimization buffers
	activationBuffers := make([][]float64, len(config.LayerSizes))
	derivativeBuffers := make([][]float64, len(config.LayerSizes)-1)
	deltasBuffers := make([][]float64, len(config.LayerSizes)-1)
	weightUpdateBuffers := make([][][]float64, len(weights))
	biasUpdateBuffers := make([][]float64, len(biases))

	// Initialize derivative and delta buffers for each layer
	for i := 0; i < len(config.LayerSizes)-1; i++ {
		derivativeBuffers[i] = make([]float64, config.LayerSizes[i+1])
		deltasBuffers[i] = make([]float64, config.LayerSizes[i+1])
	}

	// Initialize weight update buffers (same shape as weights)
	for i := range weights {
		weightUpdateBuffers[i] = make([][]float64, len(weights[i]))
		for j := range weightUpdateBuffers[i] {
			weightUpdateBuffers[i][j] = make([]float64, len(weights[i][j]))
		}
		biasUpdateBuffers[i] = make([]float64, len(biases[i]))
	}

	return &Network{
		Config:               config,
		Weights:              weights,
		Biases:               biases,
		LayerSizes:           config.LayerSizes,
		LearningRate:         config.LearningRate,
		Activation:           config.Activation,
		OutputActivation:     config.OutputActivation,
		LossFunction:         config.LossFunction,
		activationBuffers:    activationBuffers,
		derivativeBuffers:    derivativeBuffers,
		deltasBuffers:        deltasBuffers,
		weightUpdateBuffers:  weightUpdateBuffers,
		biasUpdateBuffers:    biasUpdateBuffers,
		buffersInitialized:   true,
	}
}

// FeedForward performs a forward pass through the network
func (n *Network) FeedForward(input []float64) ([]float64, [][]float64) {
	if len(input) != n.LayerSizes[0] {
		panic(fmt.Sprintf("Input size mismatch: expected %d, got %d", n.LayerSizes[0], len(input)))
	}

	// Store activations for each layer (including input layer)
	activations := make([][]float64, len(n.LayerSizes))
	activations[0] = make([]float64, len(input))
	copy(activations[0], input)

	// Forward pass through each layer
	for i := 0; i < len(n.LayerSizes)-1; i++ {
		layerSize := n.LayerSizes[i+1]
		layerActivations := make([]float64, layerSize)

		for j := 0; j < layerSize; j++ {
			// Compute weighted sum
			sum := n.Biases[i][j]
			for k := 0; k < n.LayerSizes[i]; k++ {
				sum += n.Weights[i][j][k] * activations[i][k]
			}

			// Apply activation function
			if i == len(n.LayerSizes)-2 {
				// Output layer
				layerActivations[j] = n.OutputActivation.Function(sum)
			} else {
				// Hidden layers
				layerActivations[j] = n.Activation.Function(sum)
			}
		}

		activations[i+1] = layerActivations
	}

	return activations[len(activations)-1], activations
}

// feedForwardWithBuffers performs a forward pass using pre-allocated buffers
func (n *Network) feedForwardWithBuffers(input []float64, storeDerivatives bool) ([]float64, [][]float64, [][]float64) {
	if len(input) != n.LayerSizes[0] {
		panic(fmt.Sprintf("Input size mismatch: expected %d, got %d", n.LayerSizes[0], len(input)))
	}

	// Ensure buffers are initialized
	n.ensureBuffers()

	// Use pre-allocated activation buffers
	activations := n.activationBuffers
	if activations[0] == nil || len(activations[0]) != len(input) {
		// Initialize buffers if not already properly sized
		activations[0] = make([]float64, len(input))
	}
	copy(activations[0], input)

	// Forward pass through each layer
	for i := 0; i < len(n.LayerSizes)-1; i++ {
		layerSize := n.LayerSizes[i+1]
		layerActivations := activations[i+1]
		if layerActivations == nil || len(layerActivations) != layerSize {
			layerActivations = make([]float64, layerSize)
			activations[i+1] = layerActivations
		}

		for j := 0; j < layerSize; j++ {
			// Compute weighted sum
			sum := n.Biases[i][j]
			for k := 0; k < n.LayerSizes[i]; k++ {
				sum += n.Weights[i][j][k] * activations[i][k]
			}

			// Apply activation function and optionally store derivative
			if i == len(n.LayerSizes)-2 {
				// Output layer
				layerActivations[j] = n.OutputActivation.Function(sum)
				if storeDerivatives {
					n.derivativeBuffers[i][j] = n.OutputActivation.Derivative(layerActivations[j])
				}
			} else {
				// Hidden layers
				layerActivations[j] = n.Activation.Function(sum)
				if storeDerivatives {
					n.derivativeBuffers[i][j] = n.Activation.Derivative(layerActivations[j])
				}
			}
		}
	}

	if storeDerivatives {
		return activations[len(activations)-1], activations, n.derivativeBuffers
	}
	return activations[len(activations)-1], activations, nil
}

// Predict returns the output of the network for the given input
func (n *Network) Predict(input []float64) []float64 {
	output, _, _ := n.feedForwardWithBuffers(input, false)
	return output
}

// Train trains the network on a single example
func (n *Network) Train(input, target []float64) float64 {
	// Forward pass with derivative caching
	output, activations, derivatives := n.feedForwardWithBuffers(input, true)

	// Calculate loss
	loss := n.LossFunction.Function(output, target)

	// Backward pass (backpropagation)
	outputErrors := n.LossFunction.Derivative(output, target)

	// Use pre-allocated deltas buffers
	deltas := n.deltasBuffers

	// Calculate output layer deltas
	lastLayer := len(n.LayerSizes) - 2

	for i := range deltas[lastLayer] {
		deltas[lastLayer][i] = outputErrors[i] * derivatives[lastLayer][i]
	}

	// Backpropagate through hidden layers
	for layer := lastLayer - 1; layer >= 0; layer-- {
		// deltas[layer] is already allocated in n.deltasBuffers

		for i := 0; i < n.LayerSizes[layer+1]; i++ {
			// Calculate error from next layer
			errorSum := 0.0
			for j := 0; j < n.LayerSizes[layer+2]; j++ {
				errorSum += n.Weights[layer+1][j][i] * deltas[layer+1][j]
			}

			deltas[layer][i] = derivatives[layer][i] * errorSum
		}
	}

	// Update weights and biases
	for layer := 0; layer < len(n.LayerSizes)-1; layer++ {
		for i := 0; i < n.LayerSizes[layer+1]; i++ {
			for j := 0; j < n.LayerSizes[layer]; j++ {
				n.Weights[layer][i][j] -= n.LearningRate * deltas[layer][i] * activations[layer][j]
			}
			n.Biases[layer][i] -= n.LearningRate * deltas[layer][i]
		}
	}

	return loss
}

// BatchTrain trains the network on a batch of examples
func (n *Network) BatchTrain(inputs, targets [][]float64) float64 {
	if len(inputs) != len(targets) {
		panic("Number of inputs must match number of targets")
	}

	if len(inputs) == 0 {
		return 0.0
	}

	// Ensure optimization buffers are initialized
	n.ensureBuffers()

	// Use pre-allocated update buffers
	weightUpdates := n.weightUpdateBuffers
	biasUpdates := n.biasUpdateBuffers

	// Reset update buffers to zero
	for i := range weightUpdates {
		for j := range weightUpdates[i] {
			for k := range weightUpdates[i][j] {
				weightUpdates[i][j][k] = 0.0
			}
		}
		for j := range biasUpdates[i] {
			biasUpdates[i][j] = 0.0
		}
	}

	totalLoss := 0.0

	// Use pre-allocated deltas buffer
	deltas := n.deltasBuffers

	// Accumulate gradients from all examples
	for exampleIdx := range inputs {
		// Forward pass with derivative caching
		output, activations, derivatives := n.feedForwardWithBuffers(inputs[exampleIdx], true)

		// Calculate loss
		totalLoss += n.LossFunction.Function(output, targets[exampleIdx])

		// Backpropagation
		outputErrors := n.LossFunction.Derivative(output, targets[exampleIdx])

		// Calculate output layer deltas
		lastLayer := len(n.LayerSizes) - 2
		for i := range deltas[lastLayer] {
			deltas[lastLayer][i] = outputErrors[i] * derivatives[lastLayer][i]
		}

		// Backpropagate through hidden layers
		for layer := lastLayer - 1; layer >= 0; layer-- {
			for i := 0; i < n.LayerSizes[layer+1]; i++ {
				// Calculate error from next layer
				errorSum := 0.0
				for j := 0; j < n.LayerSizes[layer+2]; j++ {
					errorSum += n.Weights[layer+1][j][i] * deltas[layer+1][j]
				}

				deltas[layer][i] = derivatives[layer][i] * errorSum
			}
		}

		// Accumulate updates
		for layer := 0; layer < len(n.LayerSizes)-1; layer++ {
			for i := 0; i < n.LayerSizes[layer+1]; i++ {
				for j := 0; j < n.LayerSizes[layer]; j++ {
					weightUpdates[layer][i][j] += deltas[layer][i] * activations[layer][j]
				}
				biasUpdates[layer][i] += deltas[layer][i]
			}
		}
	}

	// Apply averaged updates
	batchSize := float64(len(inputs))
	for layer := 0; layer < len(n.LayerSizes)-1; layer++ {
		for i := 0; i < n.LayerSizes[layer+1]; i++ {
			for j := 0; j < n.LayerSizes[layer]; j++ {
				n.Weights[layer][i][j] -= n.LearningRate * weightUpdates[layer][i][j] / batchSize
			}
			n.Biases[layer][i] -= n.LearningRate * biasUpdates[layer][i] / batchSize
		}
	}

	return totalLoss / batchSize
}

// Save saves the network to a file
func (n *Network) Save(filename string) error {
	data, err := json.MarshalIndent(n, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filename, data, 0644)
}

// Load loads a network from a file
func Load(filename string) (*Network, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var network Network
	if err := json.Unmarshal(data, &network); err != nil {
		return nil, err
	}

	// Initialize buffers that aren't serialized
	network.initializeBuffers()

	return &network, nil
}

// GetLayerSizes returns the sizes of each layer in the network
func (n *Network) GetLayerSizes() []int {
	return n.LayerSizes
}

// GetLearningRate returns the learning rate
func (n *Network) GetLearningRate() float64 {
	return n.LearningRate
}

// SetLearningRate sets a new learning rate
func (n *Network) SetLearningRate(lr float64) {
	if lr <= 0 {
		panic("Learning rate must be positive")
	}
	n.LearningRate = lr
}

// Clone creates a deep copy of the network
func (n *Network) Clone() *Network {
	// Create a new network with the same config
	config := n.Config
	config.LayerSizes = make([]int, len(n.LayerSizes))
	copy(config.LayerSizes, n.LayerSizes)

	newNetwork := NewNetwork(config)

	// Copy weights and biases
	for i := range n.Weights {
		for j := range n.Weights[i] {
			copy(newNetwork.Weights[i][j], n.Weights[i][j])
		}
	}

	for i := range n.Biases {
		copy(newNetwork.Biases[i], n.Biases[i])
	}

	return newNetwork
}

// initializeBuffers initializes the optimization buffers
func (n *Network) initializeBuffers() {
	// Initialize buffers if they're nil
	if n.activationBuffers == nil {
		n.activationBuffers = make([][]float64, len(n.LayerSizes))
	}
	if n.derivativeBuffers == nil {
		n.derivativeBuffers = make([][]float64, len(n.LayerSizes)-1)
	}
	if n.deltasBuffers == nil {
		n.deltasBuffers = make([][]float64, len(n.LayerSizes)-1)
	}
	if n.weightUpdateBuffers == nil {
		n.weightUpdateBuffers = make([][][]float64, len(n.Weights))
	}
	if n.biasUpdateBuffers == nil {
		n.biasUpdateBuffers = make([][]float64, len(n.Biases))
	}

	// Initialize derivative and delta buffers for each layer
	for i := 0; i < len(n.LayerSizes)-1; i++ {
		if n.derivativeBuffers[i] == nil || len(n.derivativeBuffers[i]) != n.LayerSizes[i+1] {
			n.derivativeBuffers[i] = make([]float64, n.LayerSizes[i+1])
		}
		if n.deltasBuffers[i] == nil || len(n.deltasBuffers[i]) != n.LayerSizes[i+1] {
			n.deltasBuffers[i] = make([]float64, n.LayerSizes[i+1])
		}
	}

	// Initialize weight update buffers (same shape as weights)
	for i := range n.Weights {
		if n.weightUpdateBuffers[i] == nil || len(n.weightUpdateBuffers[i]) != len(n.Weights[i]) {
			n.weightUpdateBuffers[i] = make([][]float64, len(n.Weights[i]))
		}
		for j := range n.Weights[i] {
			if n.weightUpdateBuffers[i][j] == nil || len(n.weightUpdateBuffers[i][j]) != len(n.Weights[i][j]) {
				n.weightUpdateBuffers[i][j] = make([]float64, len(n.Weights[i][j]))
			}
		}
		if n.biasUpdateBuffers[i] == nil || len(n.biasUpdateBuffers[i]) != len(n.Biases[i]) {
			n.biasUpdateBuffers[i] = make([]float64, len(n.Biases[i]))
		}
	}
	n.buffersInitialized = true
}

// ensureBuffers ensures that all optimization buffers are properly initialized
func (n *Network) ensureBuffers() {
	if !n.buffersInitialized || n.activationBuffers == nil || n.derivativeBuffers == nil || n.deltasBuffers == nil ||
		n.weightUpdateBuffers == nil || n.biasUpdateBuffers == nil {
		n.initializeBuffers()
		n.buffersInitialized = true
	}
}

// init initializes the random seed
func init() {
	rand.Seed(time.Now().UnixNano())
}
