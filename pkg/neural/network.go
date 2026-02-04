package neural

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
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
				// Avoid division by zero issues
				p := math.Max(1e-15, math.Min(1-1e-15, predicted[i]))
				derivatives[i] = (p - target[i]) / (p * (1 - p)) / float64(len(predicted))
			}
			return derivatives
		},
	}
)

// NetworkConfig holds configuration for creating a neural network.
// It specifies the architecture, learning hyperparameters, activation functions,
// loss function, and optional custom weight/bias initializers.
type NetworkConfig struct {
	LayerSizes        []int           `json:"layer_sizes"`        // Number of neurons in each layer
	LearningRate      float64         `json:"learning_rate"`      // Learning rate for gradient descent
	Activation        ActivationFunc  `json:"activation"`        // Activation function for hidden layers
	OutputActivation  ActivationFunc  `json:"output_activation"`  // Activation function for output layer
	LossFunction      LossFunc        `json:"loss_function"`      // Loss function for training
	Optimizer         *OptimizerConfig `json:"optimizer,omitempty"` // Optional optimizer configuration
	WeightInitializer func(int, int) float64 `json:"-"` // Function to initialize weights (not serialized)
	BiasInitializer   func() float64         `json:"-"` // Function to initialize biases (not serialized)
}

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
	optimizer      Optimizer          `json:"-"` // Optimizer for parameter updates
	OptimizerState map[string]interface{} `json:"optimizer_state,omitempty"` // Serialized optimizer state
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

	// Set default optimizer if not provided
	if config.Optimizer == nil {
		config.Optimizer = &OptimizerConfig{
			Type:         "sgd",
			LearningRate: config.LearningRate,
		}
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

	// Create optimizer
	optimizer := NewOptimizer(*config.Optimizer)

	// Create network
	network := &Network{
		Config:               config,
		Weights:              weights,
		Biases:               biases,
		LayerSizes:           config.LayerSizes,
		LearningRate:         config.LearningRate,
		Activation:           config.Activation,
		OutputActivation:     config.OutputActivation,
		LossFunction:         config.LossFunction,
		optimizer:            optimizer,
		activationBuffers:    activationBuffers,
		derivativeBuffers:    derivativeBuffers,
		deltasBuffers:        deltasBuffers,
		weightUpdateBuffers:  weightUpdateBuffers,
		biasUpdateBuffers:    biasUpdateBuffers,
		buffersInitialized:   true,
	}

	// Initialize optimizer state
	optimizer.InitializeState(config.LayerSizes)

	return network
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

	// Update weights and biases using optimizer
	for layer := 0; layer < len(n.LayerSizes)-1; layer++ {
		// Create weight gradients
		weightGradients := CreateWeightGradients(layer, deltas[layer], activations[layer])

		// Update weights using optimizer
		n.Weights[layer] = n.optimizer.UpdateWeights(layer, n.Weights[layer], weightGradients)

		// Update biases using optimizer
		n.Biases[layer] = n.optimizer.UpdateBiases(layer, n.Biases[layer], deltas[layer])
	}

	// Step optimizer
	n.optimizer.Step()

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
				errorSum := 0.0
				for j := 0; j < n.LayerSizes[layer+2]; j++ {
					errorSum += n.Weights[layer+1][j][i] * deltas[layer+1][j]
				}
				deltas[layer][i] = derivatives[layer][i] * errorSum
			}
		}

		// Accumulate weight and bias updates
		for layer := 0; layer < len(n.LayerSizes)-1; layer++ {
			// Create weight gradients
			weightGradients := CreateWeightGradients(layer, deltas[layer], activations[layer])

			// Accumulate weight updates
			weightUpdates[layer] = n.optimizer.BatchUpdateWeights(layer, weightUpdates[layer], weightGradients)

			// Accumulate bias updates
			biasUpdates[layer] = n.optimizer.BatchUpdateBiases(layer, biasUpdates[layer], deltas[layer])
		}
	}

	// Apply batch updates using optimizer
	batchSize := float64(len(inputs))
	for layer := 0; layer < len(n.LayerSizes)-1; layer++ {
		// Apply weight updates
		n.Weights[layer] = n.optimizer.ApplyBatchUpdatesWeights(layer, n.Weights[layer], weightUpdates[layer], batchSize)

		// Apply bias updates
		n.Biases[layer] = n.optimizer.ApplyBatchUpdatesBiases(layer, n.Biases[layer], biasUpdates[layer], batchSize)
	}

	// Step optimizer
	n.optimizer.Step()

	return totalLoss / float64(len(inputs))
}

// ensureBuffers initializes optimization buffers if not already initialized
func (n *Network) ensureBuffers() {
	if n.buffersInitialized {
		return
	}

	// Initialize buffers
	n.activationBuffers = make([][]float64, len(n.LayerSizes))
	n.derivativeBuffers = make([][]float64, len(n.LayerSizes)-1)
	n.deltasBuffers = make([][]float64, len(n.LayerSizes)-1)
	n.weightUpdateBuffers = make([][][]float64, len(n.Weights))
	n.biasUpdateBuffers = make([][]float64, len(n.Biases))

	// Initialize derivative and delta buffers for each layer
	for i := 0; i < len(n.LayerSizes)-1; i++ {
		n.derivativeBuffers[i] = make([]float64, n.LayerSizes[i+1])
		n.deltasBuffers[i] = make([]float64, n.LayerSizes[i+1])
	}

	// Initialize weight update buffers (same shape as weights)
	for i := range n.Weights {
		n.weightUpdateBuffers[i] = make([][]float64, len(n.Weights[i]))
		for j := range n.weightUpdateBuffers[i] {
			n.weightUpdateBuffers[i][j] = make([]float64, len(n.Weights[i][j]))
		}
		n.biasUpdateBuffers[i] = make([]float64, len(n.Biases[i]))
	}

	n.buffersInitialized = true
}

// GetLearningRate returns the current learning rate
func (n *Network) GetLearningRate() float64 {
	return n.LearningRate
}

// SetLearningRate sets the learning rate
func (n *Network) SetLearningRate(rate float64) {
	if rate <= 0 {
		panic("Learning rate must be positive")
	}
	n.LearningRate = rate
}

// GetLayerSizes returns the network's layer sizes
func (n *Network) GetLayerSizes() []int {
	return n.LayerSizes
}

// Clone creates a deep copy of the network
func (n *Network) Clone() *Network {
	// Clone config
	configCopy := n.Config
	configCopy.LayerSizes = make([]int, len(n.LayerSizes))
	copy(configCopy.LayerSizes, n.LayerSizes)

	// Clone weights and biases
	weightsCopy := make([][][]float64, len(n.Weights))
	for i := range n.Weights {
		weightsCopy[i] = make([][]float64, len(n.Weights[i]))
		for j := range n.Weights[i] {
			weightsCopy[i][j] = make([]float64, len(n.Weights[i][j]))
			copy(weightsCopy[i][j], n.Weights[i][j])
		}
	}

	biasesCopy := make([][]float64, len(n.Biases))
	for i := range n.Biases {
		biasesCopy[i] = make([]float64, len(n.Biases[i]))
		copy(biasesCopy[i], n.Biases[i])
	}

	// Clone optimizer
	optimizerCopy := n.optimizer.Clone()

	// Create new network
	network := &Network{
		Config:               configCopy,
		Weights:              weightsCopy,
		Biases:               biasesCopy,
		LayerSizes:           configCopy.LayerSizes,
		LearningRate:         n.LearningRate,
		Activation:           n.Activation,
		OutputActivation:     n.OutputActivation,
		LossFunction:         n.LossFunction,
		optimizer:            optimizerCopy,
		buffersInitialized:   false, // Buffers will be initialized lazily
	}

	// Initialize optimizer state
	optimizerCopy.InitializeState(configCopy.LayerSizes)

	return network
}

// Save saves the network to a file
func (n *Network) Save(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	// Save optimizer state before encoding
	n.OptimizerState = n.optimizer.State()
	return encoder.Encode(n)
}

// Load loads a network from a file
func Load(filename string) (*Network, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var network Network
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&network); err != nil {
		return nil, err
	}

	// Initialize optimizer based on config
	if network.Config.Optimizer == nil {
		network.Config.Optimizer = &OptimizerConfig{
			Type:         "sgd",
			LearningRate: network.Config.LearningRate,
		}
	}
	network.optimizer = NewOptimizer(*network.Config.Optimizer)
	network.optimizer.InitializeState(network.Config.LayerSizes)

	// Restore optimizer state if available
	if network.OptimizerState != nil {
		if err := network.optimizer.SetState(network.OptimizerState); err != nil {
			return nil, fmt.Errorf("failed to restore optimizer state: %w", err)
		}
	}

	// Buffers will be initialized lazily when needed
	network.buffersInitialized = false

	return &network, nil
}

// Initialize maps for activation and loss functions
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
