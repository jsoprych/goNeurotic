package timeseries

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"goneurotic/pkg/neural"
)

// PipelineConfig holds configuration for the forecasting pipeline
type PipelineConfig struct {
	// Data configuration
	WindowSize       int     // Input window size
	ForecastHorizon  int     // Number of steps to forecast
	StepSize         int     // Step between windows (default 1)
	TestSize         int     // Number of observations for testing
	ValidationMethod string  // "walk_forward", "holdout", "expanding_window"

	// Model configuration
	ModelType        string  // "neural_network", "baseline", "ensemble"
	BaselineMethod   string  // Baseline method if using baselines
	NeuralConfig     NeuralConfig // Neural network configuration

	// Feature engineering
	IncludeDateFeatures bool
	IncludeLagFeatures  bool
	Lags               []int  // Lag periods for lag features
	Normalization      string // "zscore", "minmax", "none"

	// Training configuration
	Epochs           int
	BatchSize        int
	LearningRate     float64
	EarlyStoppingPatience int

	// Evaluation
	Metrics          []string // Metrics to compute
}

// NeuralConfig holds neural network specific configuration
type NeuralConfig struct {
	LayerSizes       []int
	Activation       string
	OutputActivation string
	LossFunction     string
	Optimizer        string
	OptimizerParams  map[string]float64
}

// PipelineResult holds pipeline training and evaluation results
type PipelineResult struct {
	Config          PipelineConfig
	TrainingTime    time.Duration
	EvaluationTime  time.Duration
	Metrics         map[string]ForecastMetrics
	ModelPath       string
	DataStats       CSVStats
	FeatureCount    int
	WindowCount     int
}

// ForecastPipeline orchestrates the entire forecasting process
type ForecastPipeline struct {
	config          PipelineConfig
	data            *CSVTimeSeriesData
	normalizedData  []float64
	lastWindow      []float64
	normalization   *NormalizationStats
	features        [][]float64
	inputs          [][]float64
	outputs         [][]float64
	trainInputs     [][]float64
	trainOutputs    [][]float64
	testInputs      [][]float64
	testOutputs     [][]float64
	model           interface{} // Could be neural.Network or baseline config
	results         PipelineResult
	isTrained       bool
}

// NewPipeline creates a new forecasting pipeline with default configuration
func NewPipeline() *ForecastPipeline {
	return &ForecastPipeline{
		config: PipelineConfig{
			WindowSize:       24,
			ForecastHorizon:  1,
			StepSize:         1,
			TestSize:         20,
			ValidationMethod: "walk_forward",
			ModelType:        "neural_network",
			NeuralConfig: NeuralConfig{
				LayerSizes:       []int{24, 16, 8},
				Activation:       "tanh",
				OutputActivation: "linear",
				LossFunction:     "mse",
				Optimizer:        "adam",
				OptimizerParams:  map[string]float64{"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
			},
			IncludeDateFeatures: true,
			IncludeLagFeatures:  true,
			Lags:               []int{1, 2, 3, 7, 14},
			Normalization:      "zscore",
			Epochs:            100,
			BatchSize:         32,
			LearningRate:      0.01,
			EarlyStoppingPatience: 10,
			Metrics:           []string{"rmse", "mae", "mape"},
		},
	}
}

// WithConfig sets the pipeline configuration
func (p *ForecastPipeline) WithConfig(config PipelineConfig) *ForecastPipeline {
	p.config = config
	return p
}

// LoadData loads time series data from CSV
func (p *ForecastPipeline) LoadData(csvConfig CSVConfig) error {
	data, err := LoadCSV(csvConfig)
	if err != nil {
		return fmt.Errorf("failed to load CSV data: %w", err)
	}
	p.data = data

	// Update results with data statistics
	p.results.DataStats = data.Stats

	return nil
}

// LoadBuiltinDataset loads a built-in dataset
func (p *ForecastPipeline) LoadBuiltinDataset(datasetName string) error {
	switch datasetName {
	case "airpassengers":
		p.data = AirPassengersDataset()
	case "sample":
		p.data = CreateSampleData()
	default:
		return fmt.Errorf("unknown dataset: %s", datasetName)
	}

	p.results.DataStats = p.data.Stats
	return nil
}

// Preprocess prepares the data for training
func (p *ForecastPipeline) Preprocess() error {
	if p.data == nil {
		return fmt.Errorf("no data loaded")
	}

	// Extract time series values
	var series []float64
	if len(p.data.Values) > 0 {
		series = p.data.Values
	} else if len(p.data.Series) > 0 {
		// Use first column of multivariate series
		series = make([]float64, len(p.data.Series))
		for i, row := range p.data.Series {
			if len(row) > 0 {
				series[i] = row[0]
			}
		}
	} else {
		return fmt.Errorf("no time series data found")
	}

	// Normalize the series
	switch p.config.Normalization {
	case "zscore":
		normalized, stats := NormalizeZScore(series)
		p.normalizedData = normalized
		p.normalization = &stats
	case "minmax":
		normalized, stats := NormalizeMinMax(series)
		p.normalizedData = normalized
		p.normalization = &stats
	case "none":
		p.normalizedData = make([]float64, len(series))
		copy(p.normalizedData, series)
		p.normalization = &NormalizationStats{}
	default:
		return fmt.Errorf("unknown normalization method: %s", p.config.Normalization)
	}

	// Create feature matrix
	p.features = p.createFeatures()

	// Create sliding windows
	windowConfig := SlidingWindowConfig{
		InputSize:  p.config.WindowSize,
		OutputSize: p.config.ForecastHorizon,
		Step:       p.config.StepSize,
	}

	p.inputs, p.outputs = CreateSlidingWindows(p.normalizedData, windowConfig)
	p.results.WindowCount = len(p.inputs)
	// Save the last window for future predictions
	if len(p.inputs) > 0 {
		p.lastWindow = p.inputs[len(p.inputs)-1]
	}

	// Split into train/test based on validation method
	if err := p.createTrainTestSplit(); err != nil {
		return err
	}

	// Update feature count
	if len(p.features) > 0 && len(p.inputs) > 0 {
		p.results.FeatureCount = len(p.features[0])
	}

	return nil
}

// createFeatures creates feature matrix from the data
func (p *ForecastPipeline) createFeatures() [][]float64 {
	features := make([][]float64, len(p.normalizedData))

	for i := range features {
		features[i] = []float64{}

		// Add lag features
		if p.config.IncludeLagFeatures {
			for _, lag := range p.config.Lags {
				if i >= lag {
					features[i] = append(features[i], p.normalizedData[i-lag])
				} else {
					features[i] = append(features[i], 0.0) // Padding for early values
				}
			}
		}

		// Add date features if timestamps are available
		if p.config.IncludeDateFeatures && len(p.data.Timestamps) > i {
			dateFeatures := ExtractDateFeatures([]time.Time{p.data.Timestamps[i]})
			if len(dateFeatures) > 0 {
				features[i] = append(features[i], dateFeatures[0]...)
			}
		}

		// Add exogenous features if available
		if len(p.data.Features) > i && len(p.data.Features[i]) > 0 {
			features[i] = append(features[i], p.data.Features[i]...)
		}
	}

	return features
}

// createTrainTestSplit splits data based on validation method
func (p *ForecastPipeline) createTrainTestSplit() error {
	if len(p.inputs) == 0 || len(p.outputs) == 0 {
		return fmt.Errorf("no windows created")
	}

	switch p.config.ValidationMethod {
	case "holdout":
		// Simple holdout split
		splitRatio := float64(p.config.TestSize) / float64(len(p.normalizedData))
		p.trainInputs, p.trainOutputs, p.testInputs, p.testOutputs =
			TrainTestSplitWindows(p.inputs, p.outputs, splitRatio)

	case "walk_forward":
		// Walk-forward validation: use last TestSize windows for testing
		if p.config.TestSize >= len(p.inputs) {
			return fmt.Errorf("test size too large for walk-forward validation")
		}

		trainSize := len(p.inputs) - p.config.TestSize
		p.trainInputs = p.inputs[:trainSize]
		p.trainOutputs = p.outputs[:trainSize]
		p.testInputs = p.inputs[trainSize:]
		p.testOutputs = p.outputs[trainSize:]

	case "expanding_window":
		// For expanding window, we'll handle during evaluation
		// Use all data for initial training
		p.trainInputs = p.inputs
		p.trainOutputs = p.outputs
		p.testInputs = p.inputs // Will be overridden during evaluation
		p.testOutputs = p.outputs

	default:
		return fmt.Errorf("unknown validation method: %s", p.config.ValidationMethod)
	}

	return nil
}

// Train trains the forecasting model
func (p *ForecastPipeline) Train() error {
	startTime := time.Now()

	switch p.config.ModelType {
	case "neural_network":
		if err := p.trainNeuralNetwork(); err != nil {
			return err
		}
	case "baseline":
		p.trainBaseline()
	case "ensemble":
		if err := p.trainEnsemble(); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unknown model type: %s", p.config.ModelType)
	}

	p.results.TrainingTime = time.Since(startTime)
	p.isTrained = true

	return nil
}

// trainNeuralNetwork trains a neural network model
func (p *ForecastPipeline) trainNeuralNetwork() error {
	// Determine input size (window size + features if included)
	inputSize := p.config.WindowSize
	if len(p.features) > 0 && len(p.inputs) > 0 {
		// If we have features, we need to incorporate them
		// For simplicity, we'll use window size for now
		// In production, you'd want to properly incorporate features
		inputSize = p.config.WindowSize
	}

	// Build layer sizes
	layerSizes := []int{inputSize}
	layerSizes = append(layerSizes, p.config.NeuralConfig.LayerSizes...)
	layerSizes = append(layerSizes, p.config.ForecastHorizon)

	// Create network configuration
	networkConfig := neural.NetworkConfig{
		LayerSizes:       layerSizes,
		LearningRate:     p.config.LearningRate,
		Activation:       p.getActivation(p.config.NeuralConfig.Activation),
		OutputActivation: p.getActivation(p.config.NeuralConfig.OutputActivation),
		LossFunction:     p.getLossFunction(p.config.NeuralConfig.LossFunction),
	}

	// Create and train network
	network := neural.NewNetwork(networkConfig)

	// Train with early stopping if configured
	bestLoss := math.MaxFloat64
	patienceCounter := 0

	for epoch := 0; epoch < p.config.Epochs; epoch++ {
		loss := network.BatchTrain(p.trainInputs, p.trainOutputs)

		// Early stopping check
		if p.config.EarlyStoppingPatience > 0 {
			if loss < bestLoss {
				bestLoss = loss
				patienceCounter = 0
				// Save best model
				p.model = network
			} else {
				patienceCounter++
				if patienceCounter >= p.config.EarlyStoppingPatience {
					break
				}
			}
		}

		if epoch%100 == 0 {
			fmt.Printf("Epoch %d/%d: Loss = %.6f\n", epoch, p.config.Epochs, loss)
		}
	}

	if p.model == nil {
		p.model = network
	}

	return nil
}

// trainBaseline sets up a baseline forecasting method
func (p *ForecastPipeline) trainBaseline() {
	config := BaselineConfig{
		Method:      p.config.BaselineMethod,
		Horizon:     p.config.ForecastHorizon,
		Window:      p.config.WindowSize,
		Seasonality: 12, // Default monthly seasonality
		Alpha:       0.3,
		Beta:        0.1,
		Gamma:       0.1,
	}
	p.model = config
}

// trainEnsemble trains an ensemble of models
func (p *ForecastPipeline) trainEnsemble() error {
	// For now, train a neural network as the ensemble
	// In production, you'd train multiple models and combine them
	return p.trainNeuralNetwork()
}

// Evaluate evaluates the trained model
func (p *ForecastPipeline) Evaluate() (map[string]ForecastMetrics, error) {
	if !p.isTrained {
		return nil, fmt.Errorf("model not trained")
	}

	startTime := time.Now()
	metrics := make(map[string]ForecastMetrics)

	switch p.config.ModelType {
	case "neural_network":
		nnMetrics, err := p.evaluateNeuralNetwork()
		if err != nil {
			return nil, err
		}
		metrics["neural_network"] = nnMetrics

	case "baseline":
		baselineMetrics, err := p.evaluateBaseline()
		if err != nil {
			return nil, err
		}
		metrics[p.config.BaselineMethod] = baselineMetrics

	case "ensemble":
		ensembleMetrics, err := p.evaluateEnsemble()
		if err != nil {
			return nil, err
		}
		metrics["ensemble"] = ensembleMetrics
	}

	p.results.EvaluationTime = time.Since(startTime)
	p.results.Metrics = metrics

	return metrics, nil
}

// evaluateNeuralNetwork evaluates neural network performance
func (p *ForecastPipeline) evaluateNeuralNetwork() (ForecastMetrics, error) {
	network, ok := p.model.(*neural.Network)
	if !ok {
		return ForecastMetrics{}, fmt.Errorf("model is not a neural network")
	}

	// Make predictions
	predictions := make([][]float64, len(p.testInputs))
	for i, input := range p.testInputs {
		predictions[i] = network.Predict(input)
	}

	// Denormalize predictions and actuals
	denormPredictions := make([][]float64, len(predictions))
	denormActuals := make([][]float64, len(p.testOutputs))

	for i := range predictions {
		denormPredictions[i] = p.denormalizeSlice(predictions[i])
		denormActuals[i] = p.denormalizeSlice(p.testOutputs[i])
	}

	// Calculate metrics
	return p.calculateMetrics(denormActuals, denormPredictions)
}

// evaluateBaseline evaluates baseline method performance
func (p *ForecastPipeline) evaluateBaseline() (ForecastMetrics, error) {
	config, ok := p.model.(BaselineConfig)
	if !ok {
		return ForecastMetrics{}, fmt.Errorf("model is not a baseline config")
	}

	// Use the full series for baseline forecasting
	fullSeries := p.normalizedData

	// For walk-forward validation with baselines
	predictions := make([][]float64, len(p.testInputs))
	actuals := make([][]float64, len(p.testOutputs))

	for i := 0; i < len(p.testInputs); i++ {
		// Determine training series up to this point
		trainEnd := len(p.trainInputs) + i*p.config.StepSize
		if trainEnd > len(fullSeries) {
			trainEnd = len(fullSeries)
		}

		trainSeries := fullSeries[:trainEnd]

		// Generate forecast
		forecast := BaselineForecast(trainSeries, config)

		// Store for metric calculation
		predictions[i] = p.denormalizeSlice(forecast)
		actuals[i] = p.denormalizeSlice(p.testOutputs[i])
	}

	return p.calculateMetrics(actuals, predictions)
}

// evaluateEnsemble evaluates ensemble performance
func (p *ForecastPipeline) evaluateEnsemble() (ForecastMetrics, error) {
	// For now, just evaluate as neural network
	return p.evaluateNeuralNetwork()
}

// calculateMetrics calculates forecast metrics
func (p *ForecastPipeline) calculateMetrics(actuals, predictions [][]float64) (ForecastMetrics, error) {
	if len(actuals) != len(predictions) {
		return ForecastMetrics{}, fmt.Errorf("mismatched actuals and predictions")
	}

	var metrics ForecastMetrics
	var count int

	for i := range actuals {
		for h := 0; h < len(actuals[i]) && h < len(predictions[i]); h++ {
			actual := actuals[i][h]
			predicted := predictions[i][h]

			// RMSE components
			error := actual - predicted
			metrics.RMSE += error * error

			// MAE components
			metrics.MAE += math.Abs(error)

			// MAPE components (only if actual != 0)
			if actual != 0 {
				metrics.MAPE += math.Abs(error/actual) * 100
			}

			// SMAPE components
			denom := math.Abs(actual) + math.Abs(predicted)
			if denom > 0 {
				metrics.SMAPE += (math.Abs(error) / denom) * 200
			}

			count++
		}
	}

	if count > 0 {
		// Finalize metrics
		metrics.RMSE = math.Sqrt(metrics.RMSE / float64(count))
		metrics.MAE = metrics.MAE / float64(count)
		metrics.MAPE = metrics.MAPE / float64(count)
		metrics.SMAPE = metrics.SMAPE / float64(count)

		// Calculate R² (simplified)
		// In production, you'd want a proper R² calculation
	}

	return metrics, nil
}

// Predict generates forecasts for new data
func (p *ForecastPipeline) Predict(steps int) ([][]float64, error) {
	if !p.isTrained {
		return nil, fmt.Errorf("model not trained")
	}

	switch p.config.ModelType {
	case "neural_network":
		return p.predictNeuralNetwork(steps)
	case "baseline":
		return p.predictBaseline(steps)
	case "ensemble":
		return p.predictEnsemble(steps)
	default:
		return nil, fmt.Errorf("unknown model type")
	}
}

// predictNeuralNetwork makes predictions with neural network
func (p *ForecastPipeline) predictNeuralNetwork(steps int) ([][]float64, error) {
	network, ok := p.model.(*neural.Network)
	if !ok {
		return nil, fmt.Errorf("model is not a neural network")
	}

	// Use the most recent window as starting point
	if len(p.inputs) == 0 && len(p.lastWindow) == 0 {
		return nil, fmt.Errorf("no input windows available")
	}

	var lastWindow []float64
	if len(p.inputs) > 0 {
		lastWindow = p.inputs[len(p.inputs)-1]
	} else {
		lastWindow = p.lastWindow
	}
	predictions := make([][]float64, steps)

	// For multi-step forecasting, use recursive prediction
	currentWindow := make([]float64, len(lastWindow))
	copy(currentWindow, lastWindow)

	for step := 0; step < steps; step++ {
		prediction := network.Predict(currentWindow)
		predictions[step] = p.denormalizeSlice(prediction)

		// Update window for next prediction (shift and add predicted value)
		if len(prediction) > 0 {
			for i := 0; i < len(currentWindow)-1; i++ {
				currentWindow[i] = currentWindow[i+1]
			}
			currentWindow[len(currentWindow)-1] = prediction[0]
		}
	}

	return predictions, nil
}

// predictBaseline makes predictions with baseline method
func (p *ForecastPipeline) predictBaseline(steps int) ([][]float64, error) {
	config, ok := p.model.(BaselineConfig)
	if !ok {
		return nil, fmt.Errorf("model is not a baseline config")
	}

	config.Horizon = steps
	forecast := BaselineForecast(p.normalizedData, config)

	// Convert to proper format and denormalize
	predictions := make([][]float64, 1)
	predictions[0] = p.denormalizeSlice(forecast)

	return predictions, nil
}

// predictEnsemble makes predictions with ensemble
func (p *ForecastPipeline) predictEnsemble(steps int) ([][]float64, error) {
	// For now, use neural network prediction
	return p.predictNeuralNetwork(steps)
}

// Save saves the pipeline to a file
func (p *ForecastPipeline) Save(filename string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(filename)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")

	// Save pipeline state
	state := struct {
		Config        PipelineConfig
		Results       PipelineResult
		Normalization *NormalizationStats
		LastWindow    []float64
		IsTrained     bool
	}{
		Config:        p.config,
		Results:       p.results,
		Normalization: p.normalization,
		LastWindow:    p.lastWindow,
		IsTrained:     p.isTrained,
	}

	if err := encoder.Encode(state); err != nil {
		return fmt.Errorf("failed to encode pipeline state: %w", err)
	}

	// Save neural network model separately if it exists
	if p.config.ModelType == "neural_network" && p.isTrained {
		if network, ok := p.model.(*neural.Network); ok {
			modelPath := filename + ".model"
			if err := network.Save(modelPath); err != nil {
				return fmt.Errorf("failed to save neural network: %w", err)
			}
		}
	}

	return nil
}

// Load loads a pipeline from a file
func LoadPipeline(filename string) (*ForecastPipeline, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	var state struct {
		Config        PipelineConfig
		Results       PipelineResult
		Normalization *NormalizationStats
		LastWindow    []float64
		IsTrained     bool
	}

	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&state); err != nil {
		return nil, fmt.Errorf("failed to decode pipeline state: %w", err)
	}

	pipeline := &ForecastPipeline{
		config:        state.Config,
		results:       state.Results,
		normalization: state.Normalization,
		lastWindow:    state.LastWindow,
		isTrained:     state.IsTrained,
	}

	// Load neural network model if it exists
	if state.Config.ModelType == "neural_network" && state.IsTrained {
		modelPath := filename + ".model"
		if _, err := os.Stat(modelPath); err == nil {
			network, err := neural.Load(modelPath)
			if err != nil {
				return nil, fmt.Errorf("failed to load neural network: %w", err)
			}
			pipeline.model = network
		}
	}

	return pipeline, nil
}

// GetResults returns the pipeline results
func (p *ForecastPipeline) GetResults() PipelineResult {
	return p.results
}

// GetConfig returns the pipeline configuration
func (p *ForecastPipeline) GetConfig() PipelineConfig {
	return p.config
}

// GetData returns the loaded data
func (p *ForecastPipeline) GetData() *CSVTimeSeriesData {
	return p.data
}

// helper functions for activation and loss function mapping
func (p *ForecastPipeline) getActivation(name string) neural.ActivationFunc {
	switch name {
	case "sigmoid":
		return neural.Sigmoid
	case "tanh":
		return neural.Tanh
	case "relu":
		return neural.ReLU
	case "linear":
		return neural.Linear
	default:
		return neural.Tanh // Default
	}
}

func (p *ForecastPipeline) getLossFunction(name string) neural.LossFunc {
	switch name {
	case "mse":
		return neural.MeanSquaredError
	case "binary_crossentropy":
		return neural.BinaryCrossEntropy
	default:
		return neural.MeanSquaredError // Default
	}
}

// denormalizeSlice denormalizes a slice of values based on the pipeline's normalization method
func (p *ForecastPipeline) denormalizeSlice(values []float64) []float64 {
	if p.normalization == nil || p.config.Normalization == "none" {
		return values
	}

	switch p.config.Normalization {
	case "zscore":
		return DenormalizeZScore(values, *p.normalization)
	case "minmax":
		return DenormalizeMinMax(values, *p.normalization)
	default:
		return values
	}
}

// denormalizeMatrix denormalizes a matrix of values (2D slice) based on the pipeline's normalization method
func (p *ForecastPipeline) denormalizeMatrix(matrix [][]float64) [][]float64 {
	if p.normalization == nil || p.config.Normalization == "none" {
		return matrix
	}

	result := make([][]float64, len(matrix))
	for i, row := range matrix {
		result[i] = p.denormalizeSlice(row)
	}
	return result
}

// CompareMethods compares multiple forecasting methods
func CompareMethods(data []float64, methods []PipelineConfig) map[string]PipelineResult {
	results := make(map[string]PipelineResult)

	for _, method := range methods {
		pipeline := NewPipeline().WithConfig(method)

		// Load data (simplified - in production you'd use proper data loading)
		pipeline.data = &CSVTimeSeriesData{
			Values: data,
		}

		// Preprocess and train
		if err := pipeline.Preprocess(); err != nil {
			fmt.Printf("Error preprocessing for method %s: %v\n", method.ModelType, err)
			continue
		}

		if err := pipeline.Train(); err != nil {
			fmt.Printf("Error training for method %s: %v\n", method.ModelType, err)
			continue
		}

		// Evaluate
		if _, err := pipeline.Evaluate(); err != nil {
			fmt.Printf("Error evaluating for method %s: %v\n", method.ModelType, err)
			continue
		}

		results[method.ModelType] = pipeline.GetResults()
	}

	return results
}

// WalkForwardEvaluation performs comprehensive walk-forward validation
func WalkForwardEvaluation(pipeline *ForecastPipeline, nFolds int) ([]PipelineResult, error) {
	var foldResults []PipelineResult

	if pipeline.data == nil {
		return nil, fmt.Errorf("no data loaded")
	}

	series := pipeline.data.Values
	totalLen := len(series)
	foldSize := totalLen / nFolds

	for fold := 0; fold < nFolds; fold++ {
		// Create new pipeline for this fold
		foldPipeline := NewPipeline().WithConfig(pipeline.config)
		foldPipeline.data = pipeline.data

		// Calculate test indices for this fold
		testStart := fold * foldSize
		testEnd := testStart + foldSize
		if testEnd > totalLen {
			testEnd = totalLen
		}

		// For walk-forward, we want to test on the last portion
		// This is simplified - in production you'd implement proper walk-forward
		foldPipeline.config.TestSize = foldSize

		// Train and evaluate
		if err := foldPipeline.Preprocess(); err != nil {
			return nil, fmt.Errorf("fold %d preprocessing error: %w", fold, err)
		}

		if err := foldPipeline.Train(); err != nil {
			return nil, fmt.Errorf("fold %d training error: %w", fold, err)
		}

		if _, err := foldPipeline.Evaluate(); err != nil {
			return nil, fmt.Errorf("fold %d evaluation error: %w", fold, err)
		}

		foldResults = append(foldResults, foldPipeline.GetResults())
	}

	return foldResults, nil
}
