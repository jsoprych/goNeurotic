package neural

import (
	"math"
)

// Optimizer is the interface for neural network optimization algorithms.
type Optimizer interface {
	// InitializeState initializes optimizer state for a network.
	// This should be called before training begins.
	InitializeState(layerSizes []int)

	// UpdateWeights updates weights for a specific layer.
	// weights: current weight matrix (jagged [neuron][input])
	// gradients: weight gradients (deltas[i] * activations_prev[j])
	// Returns updated weights.
	UpdateWeights(layer int, weights [][]float64, gradients [][]float64) [][]float64

	// UpdateBiases updates biases for a specific layer.
	// biases: current bias vector
	// gradients: bias gradients (deltas)
	// Returns updated biases.
	UpdateBiases(layer int, biases []float64, gradients []float64) []float64

	// UpdateWeightsFlat updates weights using flat buffers (for BLAS optimization).
	// weightsFlat: flat weight buffer (row-major: rows=neurons, cols=inputs)
	// gradientsFlat: flat gradient buffer (same layout as weightsFlat)
	// Returns updated flat weights (in-place modification).
	UpdateWeightsFlat(layer int, weightsFlat []float64, gradientsFlat []float64) []float64

	// UpdateBiasesFlat updates biases using flat buffers (for BLAS optimization).
	// biasesFlat: flat bias buffer
	// gradientsFlat: flat gradient buffer
	// Returns updated flat biases (in-place modification).
	UpdateBiasesFlat(layer int, biasesFlat []float64, gradientsFlat []float64) []float64

	// BatchUpdateWeights accumulates weight updates for batch training.
	// weightUpdates: accumulated weight updates matrix
	// gradients: weight gradients for current example
	// Returns updated weightUpdates.
	BatchUpdateWeights(layer int, weightUpdates [][]float64, gradients [][]float64) [][]float64

	// BatchUpdateBiases accumulates bias updates for batch training.
	// biasUpdates: accumulated bias updates vector
	// gradients: bias gradients for current example
	// Returns updated biasUpdates.
	BatchUpdateBiases(layer int, biasUpdates []float64, gradients []float64) []float64

	// ApplyBatchUpdates applies accumulated batch updates to weights and biases.
	// weights/biases: current parameters
	// weightUpdates/biasUpdates: accumulated updates
	// batchSize: number of examples in batch
	// Returns updated parameters.
	ApplyBatchUpdatesWeights(layer int, weights [][]float64, weightUpdates [][]float64, batchSize float64) [][]float64
	ApplyBatchUpdatesBiases(layer int, biases []float64, biasUpdates []float64, batchSize float64) []float64

	// Step increments the optimizer's timestep (for optimizers like Adam that need it).
	Step()

	// Clone creates a deep copy of the optimizer.
	    Clone() Optimizer

	    // State returns the optimizer's internal state for serialization.
	    // Returns a map that can be JSON marshaled.
	    State() map[string]interface{}

	    // SetState restores the optimizer's internal state from serialized data.
	    SetState(state map[string]interface{}) error
	}

// OptimizerConfig holds configuration for optimizers.
type OptimizerConfig struct {
	Type       string  `json:"type"`
	LearningRate float64 `json:"learning_rate"`
	// Momentum parameters
	Momentum     float64 `json:"momentum,omitempty"`
	// RMSprop parameters
	Rho          float64 `json:"rho,omitempty"`
	Epsilon      float64 `json:"epsilon,omitempty"`
	// Adam parameters
	Beta1        float64 `json:"beta1,omitempty"`
	Beta2        float64 `json:"beta2,omitempty"`
}

// Default optimizer configurations
var (
	DefaultSGDConfig = OptimizerConfig{
		Type:        "sgd",
		LearningRate: 0.01,
	}

	DefaultSGDMomentumConfig = OptimizerConfig{
		Type:        "sgd_momentum",
		LearningRate: 0.01,
		Momentum:     0.9,
	}

	DefaultRMSpropConfig = OptimizerConfig{
		Type:        "rmsprop",
		LearningRate: 0.001,
		Rho:          0.9,
		Epsilon:      1e-8,
	}

	DefaultAdamConfig = OptimizerConfig{
		Type:        "adam",
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
	}
)

// NewOptimizer creates a new optimizer from configuration.
func NewOptimizer(config OptimizerConfig) Optimizer {
	switch config.Type {
	case "sgd":
		return NewSGDOptimizer(config.LearningRate)
	case "sgd_momentum":
		return NewSGDMomentumOptimizer(config.LearningRate, config.Momentum)
	case "rmsprop":
		return NewRMSpropOptimizer(config.LearningRate, config.Rho, config.Epsilon)
	case "adam":
		return NewAdamOptimizer(config.LearningRate, config.Beta1, config.Beta2, config.Epsilon)
	default:
		// Default to SGD
		return NewSGDOptimizer(config.LearningRate)
	}
}

// ============================================================================
// SGD Optimizer (Stochastic Gradient Descent)
// ============================================================================

// SGDOptimizer implements basic stochastic gradient descent.
type SGDOptimizer struct {
	learningRate float64
}

// NewSGDOptimizer creates a new SGD optimizer.
func NewSGDOptimizer(learningRate float64) *SGDOptimizer {
	return &SGDOptimizer{
		learningRate: learningRate,
	}
}

func (o *SGDOptimizer) InitializeState(layerSizes []int) {
	// SGD has no state to initialize
}

func (o *SGDOptimizer) UpdateWeights(layer int, weights [][]float64, gradients [][]float64) [][]float64 {
	for i := range weights {
		for j := range weights[i] {
			weights[i][j] -= o.learningRate * gradients[i][j]
		}
	}
	return weights
}

func (o *SGDOptimizer) UpdateBiases(layer int, biases []float64, gradients []float64) []float64 {
	for i := range biases {
		biases[i] -= o.learningRate * gradients[i]
	}
	return biases
}

func (o *SGDOptimizer) UpdateWeightsFlat(layer int, weightsFlat []float64, gradientsFlat []float64) []float64 {
	for i := range weightsFlat {
		weightsFlat[i] -= o.learningRate * gradientsFlat[i]
	}
	return weightsFlat
}

func (o *SGDOptimizer) UpdateBiasesFlat(layer int, biasesFlat []float64, gradientsFlat []float64) []float64 {
	for i := range biasesFlat {
		biasesFlat[i] -= o.learningRate * gradientsFlat[i]
	}
	return biasesFlat
}

func (o *SGDOptimizer) BatchUpdateWeights(layer int, weightUpdates [][]float64, gradients [][]float64) [][]float64 {
	for i := range gradients {
		for j := range gradients[i] {
			weightUpdates[i][j] += gradients[i][j]
		}
	}
	return weightUpdates
}

func (o *SGDOptimizer) BatchUpdateBiases(layer int, biasUpdates []float64, gradients []float64) []float64 {
	for i := range gradients {
		biasUpdates[i] += gradients[i]
	}
	return biasUpdates
}

func (o *SGDOptimizer) ApplyBatchUpdatesWeights(layer int, weights [][]float64, weightUpdates [][]float64, batchSize float64) [][]float64 {
	scale := o.learningRate / batchSize
	for i := range weights {
		for j := range weights[i] {
			weights[i][j] -= scale * weightUpdates[i][j]
		}
	}
	return weights
}

func (o *SGDOptimizer) ApplyBatchUpdatesBiases(layer int, biases []float64, biasUpdates []float64, batchSize float64) []float64 {
	scale := o.learningRate / batchSize
	for i := range biases {
		biases[i] -= scale * biasUpdates[i]
	}
	return biases
}

func (o *SGDOptimizer) Step() {
	// SGD doesn't need timestep tracking
}

func (o *SGDOptimizer) Clone() Optimizer {
	return NewSGDOptimizer(o.learningRate)
}

func (o *SGDOptimizer) State() map[string]interface{} {
	return map[string]interface{}{
		"type":          "sgd",
		"learning_rate": o.learningRate,
	}
}

func (o *SGDOptimizer) SetState(state map[string]interface{}) error {
	// SGD has no internal state beyond learning rate, which is set at creation
	// We could update learningRate if provided, but for consistency we ignore
	return nil
}

// ============================================================================
// SGD with Momentum Optimizer
// ============================================================================

// SGDMomentumOptimizer implements SGD with momentum.
type SGDMomentumOptimizer struct {
	learningRate float64
	momentum     float64
	// State: velocity for each parameter
	weightVelocities [][][]float64 // [layer][neuron][input]
	biasVelocities   [][]float64   // [layer][neuron]
}

// NewSGDMomentumOptimizer creates a new SGD with momentum optimizer.
func NewSGDMomentumOptimizer(learningRate, momentum float64) *SGDMomentumOptimizer {
	return &SGDMomentumOptimizer{
		learningRate: learningRate,
		momentum:     momentum,
	}
}

func (o *SGDMomentumOptimizer) InitializeState(layerSizes []int) {
	numLayers := len(layerSizes) - 1
	o.weightVelocities = make([][][]float64, numLayers)
	o.biasVelocities = make([][]float64, numLayers)

	for i := 0; i < numLayers; i++ {
		fanIn := layerSizes[i]
		fanOut := layerSizes[i+1]

		o.weightVelocities[i] = make([][]float64, fanOut)
		for j := range o.weightVelocities[i] {
			o.weightVelocities[i][j] = make([]float64, fanIn)
		}

		o.biasVelocities[i] = make([]float64, fanOut)
	}
}

func (o *SGDMomentumOptimizer) UpdateWeights(layer int, weights [][]float64, gradients [][]float64) [][]float64 {
	velocity := o.weightVelocities[layer]
	for i := range weights {
		for j := range weights[i] {
			// Update velocity: v = momentum * v - learning_rate * g
			velocity[i][j] = o.momentum * velocity[i][j] - o.learningRate * gradients[i][j]
			// Update weights: w = w + v
			weights[i][j] += velocity[i][j]
		}
	}
	return weights
}

func (o *SGDMomentumOptimizer) UpdateBiases(layer int, biases []float64, gradients []float64) []float64 {
	velocity := o.biasVelocities[layer]
	for i := range biases {
		velocity[i] = o.momentum * velocity[i] - o.learningRate * gradients[i]
		biases[i] += velocity[i]
	}
	return biases
}

func (o *SGDMomentumOptimizer) UpdateWeightsFlat(layer int, weightsFlat []float64, gradientsFlat []float64) []float64 {
	// For flat buffers, we need to convert to jagged or maintain flat velocities
	// For simplicity, we'll use jagged operations for now
	// In practice, we would maintain flat velocity buffers
	panic("SGDMomentumOptimizer.UpdateWeightsFlat not implemented for flat buffers")
}

func (o *SGDMomentumOptimizer) UpdateBiasesFlat(layer int, biasesFlat []float64, gradientsFlat []float64) []float64 {
	panic("SGDMomentumOptimizer.UpdateBiasesFlat not implemented for flat buffers")
}

func (o *SGDMomentumOptimizer) BatchUpdateWeights(layer int, weightUpdates [][]float64, gradients [][]float64) [][]float64 {
	for i := range gradients {
		for j := range gradients[i] {
			weightUpdates[i][j] += gradients[i][j]
		}
	}
	return weightUpdates
}

func (o *SGDMomentumOptimizer) BatchUpdateBiases(layer int, biasUpdates []float64, gradients []float64) []float64 {
	for i := range gradients {
		biasUpdates[i] += gradients[i]
	}
	return biasUpdates
}

func (o *SGDMomentumOptimizer) ApplyBatchUpdatesWeights(layer int, weights [][]float64, weightUpdates [][]float64, batchSize float64) [][]float64 {
	scale := 1.0 / batchSize
	velocity := o.weightVelocities[layer]
	for i := range weights {
		for j := range weights[i] {
			// Average gradient
			avgGrad := weightUpdates[i][j] * scale
			// Update velocity
			velocity[i][j] = o.momentum * velocity[i][j] - o.learningRate * avgGrad
			// Update weights
			weights[i][j] += velocity[i][j]
		}
	}
	return weights
}

func (o *SGDMomentumOptimizer) ApplyBatchUpdatesBiases(layer int, biases []float64, biasUpdates []float64, batchSize float64) []float64 {
	scale := 1.0 / batchSize
	velocity := o.biasVelocities[layer]
	for i := range biases {
		avgGrad := biasUpdates[i] * scale
		velocity[i] = o.momentum * velocity[i] - o.learningRate * avgGrad
		biases[i] += velocity[i]
	}
	return biases
}

func (o *SGDMomentumOptimizer) Step() {
	// No timestep needed for momentum
}

func (o *SGDMomentumOptimizer) Clone() Optimizer {
	clone := NewSGDMomentumOptimizer(o.learningRate, o.momentum)
	// Note: state is not cloned, should be reinitialized
	return clone
}

func (o *SGDMomentumOptimizer) State() map[string]interface{} {
	return map[string]interface{}{
		"type":           "sgd_momentum",
		"learning_rate":  o.learningRate,
		"momentum":       o.momentum,
		"weight_velocities": o.weightVelocities,
		"bias_velocities":   o.biasVelocities,
	}
}

func (o *SGDMomentumOptimizer) SetState(state map[string]interface{}) error {
	// Restore learning rate and momentum if provided
	if lr, ok := state["learning_rate"].(float64); ok {
		o.learningRate = lr
	}
	if mom, ok := state["momentum"].(float64); ok {
		o.momentum = mom
	}

	// Restore velocities if provided
	if wv, ok := state["weight_velocities"].([][][]float64); ok {
		o.weightVelocities = wv
	}
	if bv, ok := state["bias_velocities"].([][]float64); ok {
		o.biasVelocities = bv
	}

	return nil
}

// ============================================================================
// RMSprop Optimizer
// ============================================================================

// RMSpropOptimizer implements RMSprop optimization.
type RMSpropOptimizer struct {
	learningRate float64
	rho          float64
	epsilon      float64
	// State: squared gradient accumulator
	weightCache [][][]float64 // [layer][neuron][input]
	biasCache   [][]float64   // [layer][neuron]
}

// NewRMSpropOptimizer creates a new RMSprop optimizer.
func NewRMSpropOptimizer(learningRate, rho, epsilon float64) *RMSpropOptimizer {
	return &RMSpropOptimizer{
		learningRate: learningRate,
		rho:          rho,
		epsilon:      epsilon,
	}
}

func (o *RMSpropOptimizer) InitializeState(layerSizes []int) {
	numLayers := len(layerSizes) - 1
	o.weightCache = make([][][]float64, numLayers)
	o.biasCache = make([][]float64, numLayers)

	for i := 0; i < numLayers; i++ {
		fanIn := layerSizes[i]
		fanOut := layerSizes[i+1]

		o.weightCache[i] = make([][]float64, fanOut)
		for j := range o.weightCache[i] {
			o.weightCache[i][j] = make([]float64, fanIn)
		}

		o.biasCache[i] = make([]float64, fanOut)
	}
}

func (o *RMSpropOptimizer) UpdateWeights(layer int, weights [][]float64, gradients [][]float64) [][]float64 {
	cache := o.weightCache[layer]
	for i := range weights {
		for j := range weights[i] {
			// Update cache: cache = rho * cache + (1 - rho) * g^2
			cache[i][j] = o.rho*cache[i][j] + (1-o.rho)*gradients[i][j]*gradients[i][j]
			// Update weights: w = w - learning_rate * g / (sqrt(cache) + epsilon)
			weights[i][j] -= o.learningRate * gradients[i][j] / (math.Sqrt(cache[i][j]) + o.epsilon)
		}
	}
	return weights
}

func (o *RMSpropOptimizer) UpdateBiases(layer int, biases []float64, gradients []float64) []float64 {
	cache := o.biasCache[layer]
	for i := range biases {
		cache[i] = o.rho*cache[i] + (1-o.rho)*gradients[i]*gradients[i]
		biases[i] -= o.learningRate * gradients[i] / (math.Sqrt(cache[i]) + o.epsilon)
	}
	return biases
}

func (o *RMSpropOptimizer) UpdateWeightsFlat(layer int, weightsFlat []float64, gradientsFlat []float64) []float64 {
	panic("RMSpropOptimizer.UpdateWeightsFlat not implemented for flat buffers")
}

func (o *RMSpropOptimizer) UpdateBiasesFlat(layer int, biasesFlat []float64, gradientsFlat []float64) []float64 {
	panic("RMSpropOptimizer.UpdateBiasesFlat not implemented for flat buffers")
}

func (o *RMSpropOptimizer) BatchUpdateWeights(layer int, weightUpdates [][]float64, gradients [][]float64) [][]float64 {
	for i := range gradients {
		for j := range gradients[i] {
			weightUpdates[i][j] += gradients[i][j]
		}
	}
	return weightUpdates
}

func (o *RMSpropOptimizer) BatchUpdateBiases(layer int, biasUpdates []float64, gradients []float64) []float64 {
	for i := range gradients {
		biasUpdates[i] += gradients[i]
	}
	return biasUpdates
}

func (o *RMSpropOptimizer) ApplyBatchUpdatesWeights(layer int, weights [][]float64, weightUpdates [][]float64, batchSize float64) [][]float64 {
	scale := 1.0 / batchSize
	cache := o.weightCache[layer]
	for i := range weights {
		for j := range weights[i] {
			// Average gradient
			avgGrad := weightUpdates[i][j] * scale
			// Update cache with average gradient
			cache[i][j] = o.rho*cache[i][j] + (1-o.rho)*avgGrad*avgGrad
			// Update weights
			weights[i][j] -= o.learningRate * avgGrad / (math.Sqrt(cache[i][j]) + o.epsilon)
		}
	}
	return weights
}

func (o *RMSpropOptimizer) ApplyBatchUpdatesBiases(layer int, biases []float64, biasUpdates []float64, batchSize float64) []float64 {
	scale := 1.0 / batchSize
	cache := o.biasCache[layer]
	for i := range biases {
		avgGrad := biasUpdates[i] * scale
		cache[i] = o.rho*cache[i] + (1-o.rho)*avgGrad*avgGrad
		biases[i] -= o.learningRate * avgGrad / (math.Sqrt(cache[i]) + o.epsilon)
	}
	return biases
}

func (o *RMSpropOptimizer) Step() {
	// No timestep needed for RMSprop
}

func (o *RMSpropOptimizer) Clone() Optimizer {
	return NewRMSpropOptimizer(o.learningRate, o.rho, o.epsilon)
}

func (o *RMSpropOptimizer) State() map[string]interface{} {
	return map[string]interface{}{
		"type":           "rmsprop",
		"learning_rate":  o.learningRate,
		"rho":            o.rho,
		"epsilon":        o.epsilon,
		"weight_cache":   o.weightCache,
		"bias_cache":     o.biasCache,
	}
}

func (o *RMSpropOptimizer) SetState(state map[string]interface{}) error {
	// Restore parameters if provided
	if lr, ok := state["learning_rate"].(float64); ok {
		o.learningRate = lr
	}
	if rho, ok := state["rho"].(float64); ok {
		o.rho = rho
	}
	if eps, ok := state["epsilon"].(float64); ok {
		o.epsilon = eps
	}

	// Restore cache if provided
	if wc, ok := state["weight_cache"].([][][]float64); ok {
		o.weightCache = wc
	}
	if bc, ok := state["bias_cache"].([][]float64); ok {
		o.biasCache = bc
	}

	return nil
}

// ============================================================================
// Adam Optimizer
// ============================================================================

// AdamOptimizer implements the Adam optimization algorithm.
type AdamOptimizer struct {
	learningRate float64
	beta1        float64
	beta2        float64
	epsilon      float64
	timestep     int
	// State: first and second moment estimates
	weightM [][][]float64 // First moment (mean)
	weightV [][][]float64 // Second moment (uncentered variance)
	biasM   [][]float64   // First moment for biases
	biasV   [][]float64   // Second moment for biases
}

// NewAdamOptimizer creates a new Adam optimizer.
func NewAdamOptimizer(learningRate, beta1, beta2, epsilon float64) *AdamOptimizer {
	return &AdamOptimizer{
		learningRate: learningRate,
		beta1:        beta1,
		beta2:        beta2,
		epsilon:      epsilon,
		timestep:     0,
	}
}

func (o *AdamOptimizer) InitializeState(layerSizes []int) {
	numLayers := len(layerSizes) - 1
	o.weightM = make([][][]float64, numLayers)
	o.weightV = make([][][]float64, numLayers)
	o.biasM = make([][]float64, numLayers)
	o.biasV = make([][]float64, numLayers)

	for i := 0; i < numLayers; i++ {
		fanIn := layerSizes[i]
		fanOut := layerSizes[i+1]

		o.weightM[i] = make([][]float64, fanOut)
		o.weightV[i] = make([][]float64, fanOut)
		for j := range o.weightM[i] {
			o.weightM[i][j] = make([]float64, fanIn)
			o.weightV[i][j] = make([]float64, fanIn)
		}

		o.biasM[i] = make([]float64, fanOut)
		o.biasV[i] = make([]float64, fanOut)
	}
}

func (o *AdamOptimizer) UpdateWeights(layer int, weights [][]float64, gradients [][]float64) [][]float64 {
	o.timestep++
	m := o.weightM[layer]
	v := o.weightV[layer]

	// Bias correction terms
	beta1_t := math.Pow(o.beta1, float64(o.timestep))
	beta2_t := math.Pow(o.beta2, float64(o.timestep))
	learningRate_t := o.learningRate * math.Sqrt(1-beta2_t) / (1 - beta1_t)

	for i := range weights {
		for j := range weights[i] {
			g := gradients[i][j]

			// Update biased first moment estimate
			m[i][j] = o.beta1*m[i][j] + (1-o.beta1)*g
			// Update biased second raw moment estimate
			v[i][j] = o.beta2*v[i][j] + (1-o.beta2)*g*g

			// Compute bias-corrected first moment estimate
			m_hat := m[i][j] / (1 - beta1_t)
			// Compute bias-corrected second raw moment estimate
			v_hat := v[i][j] / (1 - beta2_t)

			// Update parameters
			weights[i][j] -= learningRate_t * m_hat / (math.Sqrt(v_hat) + o.epsilon)
		}
	}
	return weights
}

func (o *AdamOptimizer) UpdateBiases(layer int, biases []float64, gradients []float64) []float64 {
	o.timestep++
	m := o.biasM[layer]
	v := o.biasV[layer]

	beta1_t := math.Pow(o.beta1, float64(o.timestep))
	beta2_t := math.Pow(o.beta2, float64(o.timestep))
	learningRate_t := o.learningRate * math.Sqrt(1-beta2_t) / (1 - beta1_t)

	for i := range biases {
		g := gradients[i]

		m[i] = o.beta1*m[i] + (1-o.beta1)*g
		v[i] = o.beta2*v[i] + (1-o.beta2)*g*g

		m_hat := m[i] / (1 - beta1_t)
		v_hat := v[i] / (1 - beta2_t)

		biases[i] -= learningRate_t * m_hat / (math.Sqrt(v_hat) + o.epsilon)
	}
	return biases
}

func (o *AdamOptimizer) UpdateWeightsFlat(layer int, weightsFlat []float64, gradientsFlat []float64) []float64 {
	panic("AdamOptimizer.UpdateWeightsFlat not implemented for flat buffers")
}

func (o *AdamOptimizer) UpdateBiasesFlat(layer int, biasesFlat []float64, gradientsFlat []float64) []float64 {
	panic("AdamOptimizer.UpdateBiasesFlat not implemented for flat buffers")
}

func (o *AdamOptimizer) BatchUpdateWeights(layer int, weightUpdates [][]float64, gradients [][]float64) [][]float64 {
	for i := range gradients {
		for j := range gradients[i] {
			weightUpdates[i][j] += gradients[i][j]
		}
	}
	return weightUpdates
}

func (o *AdamOptimizer) BatchUpdateBiases(layer int, biasUpdates []float64, gradients []float64) []float64 {
	for i := range gradients {
		biasUpdates[i] += gradients[i]
	}
	return biasUpdates
}

func (o *AdamOptimizer) ApplyBatchUpdatesWeights(layer int, weights [][]float64, weightUpdates [][]float64, batchSize float64) [][]float64 {
	o.timestep++
	m := o.weightM[layer]
	v := o.weightV[layer]

	beta1_t := math.Pow(o.beta1, float64(o.timestep))
	beta2_t := math.Pow(o.beta2, float64(o.timestep))
	learningRate_t := o.learningRate * math.Sqrt(1-beta2_t) / (1 - beta1_t)

	scale := 1.0 / batchSize

	for i := range weights {
		for j := range weights[i] {
			// Average gradient
			g := weightUpdates[i][j] * scale

			m[i][j] = o.beta1*m[i][j] + (1-o.beta1)*g
			v[i][j] = o.beta2*v[i][j] + (1-o.beta2)*g*g

			m_hat := m[i][j] / (1 - beta1_t)
			v_hat := v[i][j] / (1 - beta2_t)

			weights[i][j] -= learningRate_t * m_hat / (math.Sqrt(v_hat) + o.epsilon)
		}
	}
	return weights
}

func (o *AdamOptimizer) ApplyBatchUpdatesBiases(layer int, biases []float64, biasUpdates []float64, batchSize float64) []float64 {
	o.timestep++
	m := o.biasM[layer]
	v := o.biasV[layer]

	beta1_t := math.Pow(o.beta1, float64(o.timestep))
	beta2_t := math.Pow(o.beta2, float64(o.timestep))
	learningRate_t := o.learningRate * math.Sqrt(1-beta2_t) / (1 - beta1_t)

	scale := 1.0 / batchSize

	for i := range biases {
		g := biasUpdates[i] * scale

		m[i] = o.beta1*m[i] + (1-o.beta1)*g
		v[i] = o.beta2*v[i] + (1-o.beta2)*g*g

		m_hat := m[i] / (1 - beta1_t)
		v_hat := v[i] / (1 - beta2_t)

		biases[i] -= learningRate_t * m_hat / (math.Sqrt(v_hat) + o.epsilon)
	}
	return biases
}

func (o *AdamOptimizer) Step() {
	// Timestep is incremented in Update methods
}

func (o *AdamOptimizer) Clone() Optimizer {
	clone := NewAdamOptimizer(o.learningRate, o.beta1, o.beta2, o.epsilon)
	clone.timestep = o.timestep
	// Note: moment estimates are not cloned, should be reinitialized
	return clone
}

func (o *AdamOptimizer) State() map[string]interface{} {
	return map[string]interface{}{
		"type":           "adam",
		"learning_rate":  o.learningRate,
		"beta1":          o.beta1,
		"beta2":          o.beta2,
		"epsilon":        o.epsilon,
		"timestep":       o.timestep,
		"weight_m":       o.weightM,
		"weight_v":       o.weightV,
		"bias_m":         o.biasM,
		"bias_v":         o.biasV,
	}
}

func (o *AdamOptimizer) SetState(state map[string]interface{}) error {
	// Restore parameters if provided
	if lr, ok := state["learning_rate"].(float64); ok {
		o.learningRate = lr
	}
	if b1, ok := state["beta1"].(float64); ok {
		o.beta1 = b1
	}
	if b2, ok := state["beta2"].(float64); ok {
		o.beta2 = b2
	}
	if eps, ok := state["epsilon"].(float64); ok {
		o.epsilon = eps
	}
	if ts, ok := state["timestep"].(int); ok {
		o.timestep = ts
	}

	// Restore moment estimates if provided
	if wm, ok := state["weight_m"].([][][]float64); ok {
		o.weightM = wm
	}
	if wv, ok := state["weight_v"].([][][]float64); ok {
		o.weightV = wv
	}
	if bm, ok := state["bias_m"].([][]float64); ok {
		o.biasM = bm
	}
	if bv, ok := state["bias_v"].([][]float64); ok {
		o.biasV = bv
	}

	return nil
}

// ============================================================================
// Utility functions
// ============================================================================

// CreateWeightGradients creates weight gradients from deltas and activations.
// This is a helper function used by training algorithms.
func CreateWeightGradients(layer int, deltas []float64, activationsPrev []float64) [][]float64 {
	fanOut := len(deltas)
	fanIn := len(activationsPrev)

	gradients := make([][]float64, fanOut)
	for i := range gradients {
		gradients[i] = make([]float64, fanIn)
		for j := range gradients[i] {
			gradients[i][j] = deltas[i] * activationsPrev[j]
		}
	}
	return gradients
}

// CreateWeightGradientsFlat creates flat weight gradients for BLAS operations.
func CreateWeightGradientsFlat(layer int, deltas []float64, activationsPrev []float64) []float64 {
	fanOut := len(deltas)
	fanIn := len(activationsPrev)

	gradientsFlat := make([]float64, fanOut*fanIn)
	for i := 0; i < fanOut; i++ {
		offset := i * fanIn
		for j := 0; j < fanIn; j++ {
			gradientsFlat[offset+j] = deltas[i] * activationsPrev[j]
		}
	}
	return gradientsFlat
}
