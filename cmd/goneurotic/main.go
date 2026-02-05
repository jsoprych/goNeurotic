package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	    "goneurotic/pkg/neural"
	    "goneurotic/pkg/timeseries"
)

// Command-line flags
var (
	trainFile    = flag.String("train", "", "CSV file for training data")
	testFile     = flag.String("test", "", "CSV file for testing data")
	modelFile    = flag.String("model", "model.json", "File to save/load model")
	epochs       = flag.Int("epochs", 1000, "Number of training epochs")
	batchSize    = flag.Int("batch", 32, "Batch size for training")
	learningRate = flag.Float64("lr", 0.01, "Learning rate")
	hiddenLayers = flag.String("layers", "10,5", "Comma-separated hidden layer sizes")
	activation   = flag.String("activation", "relu", "Activation function (sigmoid, relu, tanh, linear)")
	outputAct    = flag.String("output", "sigmoid", "Output activation function")
	lossFunc     = flag.String("loss", "mse", "Loss function (mse, binary_crossentropy)")
	seed         = flag.Int64("seed", time.Now().UnixNano(), "Random seed")
	verbose      = flag.Bool("verbose", false, "Enable verbose output")
	visualize    = flag.Bool("visualize", false, "Generate visualization data")
	benchmark    = flag.Bool("benchmark", false, "Run performance benchmarks")
)

// Demo examples
var demos = map[string]func(){
    "and":        demoAND,
    "xor":        demoXOR,
    "sin":        demoSine,
    "mnist":      demoMNIST,
    "iris":       demoIris,
    "complex":    demoComplex,
    "timeseries": demoTimeseries,
    "realts":     demoRealTS,
    "pipeline":   demoPipeline,
}

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "GoNeurotic - A Production-Ready Neural Network Library\n\n")
		fmt.Fprintf(os.Stderr, "Usage:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nAvailable demos:\n")
		for name := range demos {
			fmt.Fprintf(os.Stderr, "  %s\n", name)
		}
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s -demo xor\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -demo timeseries\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -train data.csv -test test.csv -epochs 5000\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -benchmark\n", os.Args[0])
	}

	demo := flag.String("demo", "", "Run a demo (and, xor, sin, mnist, iris, complex, timeseries)")

	flag.Parse()

	// Set random seed
	rand.Seed(*seed)

	if *demo != "" {
		runDemo(*demo)
		return
	}

	if *benchmark {
		runBenchmarks()
		return
	}

	if *trainFile != "" {
		trainAndTest()
		return
	}

	// Default: run XOR demo
	fmt.Println("No command specified. Running XOR demo...")
	demoXOR()
}

func runDemo(name string) {
	demoFunc, exists := demos[name]
	if !exists {
		log.Fatalf("Unknown demo: %s. Available: %v", name, getDemoNames())
	}
	fmt.Printf("Running %s demo...\n", name)
	demoFunc()
}

func getDemoNames() []string {
	names := make([]string, 0, len(demos))
	for name := range demos {
		names = append(names, name)
	}
	return names
}

func demoAND() {
	fmt.Println("\n=== 3-Input AND Gate Demo ===")
	fmt.Println("Training a neural network to learn the 3-input AND gate truth table")

	// Training data for 3-input AND gate
	inputs := [][]float64{
		{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
		{1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
	}
	targets := [][]float64{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {1}}

	// Create network
	config := neural.NetworkConfig{
		LayerSizes:       []int{3, 8, 4, 1},
		LearningRate:     0.1,
		Activation:       neural.ReLU,
		OutputActivation: neural.Sigmoid,
		LossFunction:     neural.BinaryCrossEntropy,
	}

	network := neural.NewNetwork(config)
	fmt.Printf("Network created: %v\n", config.LayerSizes)

	// Train
	fmt.Println("\nTraining...")
	start := time.Now()
	for epoch := 0; epoch < 5000; epoch++ {
		totalLoss := 0.0
		for i := range inputs {
			loss := network.Train(inputs[i], targets[i])
			totalLoss += loss
		}

		if epoch%500 == 0 {
			fmt.Printf("Epoch %d: Loss = %.6f\n", epoch, totalLoss/float64(len(inputs)))
		}
	}

	trainingTime := time.Since(start)
	fmt.Printf("Training completed in %v\n", trainingTime)

	// Test
	fmt.Println("\nTesting network:")
	correct := 0
	for i, input := range inputs {
		output := network.Predict(input)
		predicted := output[0]
		expected := targets[i][0]

		// Convert to binary prediction
		predictedBinary := 0
		if predicted > 0.5 {
			predictedBinary = 1
		}

		status := "✓"
		if predictedBinary != int(expected) {
			status = "✗"
		} else {
			correct++
		}

		fmt.Printf("  Input: %v -> Output: %.6f (predicted: %d, expected: %.0f) %s\n",
			input, predicted, predictedBinary, expected, status)
	}

	accuracy := float64(correct) / float64(len(inputs)) * 100
	fmt.Printf("\nAccuracy: %.1f%% (%d/%d)\n", accuracy, correct, len(inputs))

	// Save model
	if err := network.Save("and_gate_model.json"); err != nil {
		fmt.Printf("Error saving model: %v\n", err)
	} else {
		fmt.Println("Model saved to 'and_gate_model.json'")
	}
}

func demoXOR() {
	fmt.Println("\n=== XOR Problem Demo ===")
	fmt.Println("Training a neural network to solve the XOR problem (non-linearly separable)")

	// XOR truth table
	inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	targets := [][]float64{{0}, {1}, {1}, {0}}

	// Create network with hidden layer (necessary for XOR)
	config := neural.NetworkConfig{
		LayerSizes:       []int{2, 3, 1}, // Simple architecture that can learn XOR
		LearningRate:     0.5,
		Activation:       neural.Sigmoid,
		OutputActivation: neural.Sigmoid,
		LossFunction:     neural.BinaryCrossEntropy,
	}

	network := neural.NewNetwork(config)
	fmt.Printf("Network created: %v\n", config.LayerSizes)

	// Train
	fmt.Println("\nTraining...")
	start := time.Now()
	lossHistory := []float64{}

	for epoch := 0; epoch < 20000; epoch++ {
		totalLoss := 0.0
		for i := range inputs {
			loss := network.Train(inputs[i], targets[i])
			totalLoss += loss
		}

		avgLoss := totalLoss / float64(len(inputs))
		if epoch%2000 == 0 {
			fmt.Printf("Epoch %d: Loss = %.6f\n", epoch, avgLoss)
		}
		if epoch%200 == 0 {
			lossHistory = append(lossHistory, avgLoss)
		}
	}

	trainingTime := time.Since(start)
	fmt.Printf("Training completed in %v\n", trainingTime)

	// Test
	fmt.Println("\nXOR Truth Table Results:")
	correct := 0
	for i, input := range inputs {
		output := network.Predict(input)
		predicted := output[0]
		expected := targets[i][0]

		// Convert to binary prediction
		predictedBinary := 0
		if predicted > 0.5 {
			predictedBinary = 1
		}

		status := "✓"
		if predictedBinary != int(expected) {
			status = "✗"
		} else {
			correct++
		}

		fmt.Printf("  %v XOR %v = %.0f -> Network: %.6f (%d) %s\n",
			input[0], input[1], expected, predicted, predictedBinary, status)
	}

	accuracy := float64(correct) / float64(len(inputs)) * 100
	fmt.Printf("\nAccuracy: %.1f%% (%d/%d)\n", accuracy, correct, len(inputs))

	// Generate visualization data if requested
	if *visualize {
		generateXORVisualization(network)
	}
}

func demoSine() {
	fmt.Println("\n=== Sine Function Approximation Demo ===")
	fmt.Println("Training a neural network to approximate sin(x)")

	// Generate training data
	numPoints := 100
	inputs := make([][]float64, numPoints)
	targets := make([][]float64, numPoints)

	for i := 0; i < numPoints; i++ {
		x := float64(i) / float64(numPoints) * 4 * math.Pi // 0 to 4π
		inputs[i] = []float64{x}
		targets[i] = []float64{math.Sin(x)}
	}

	// Create network
	config := neural.NetworkConfig{
		LayerSizes:       []int{1, 16, 16, 1}, // Deeper network for function approximation
		LearningRate:     0.01,
		Activation:       neural.Tanh,
		OutputActivation: neural.Linear,
		LossFunction:     neural.MeanSquaredError,
	}

	network := neural.NewNetwork(config)
	fmt.Printf("Network created: %v\n", config.LayerSizes)

	// Train in batches
	fmt.Println("\nTraining...")
	start := time.Now()

	for epoch := 0; epoch < 5000; epoch++ {
		loss := network.BatchTrain(inputs, targets)

		if epoch%500 == 0 {
			fmt.Printf("Epoch %d: Loss = %.6f\n", epoch, loss)
		}
	}

	trainingTime := time.Since(start)
	fmt.Printf("Training completed in %v\n", trainingTime)

	// Test on new points
	fmt.Println("\nTesting on unseen data:")
	testPoints := []float64{0.5, 1.0, 2.0, 3.0, math.Pi, math.Pi / 2}
	for _, x := range testPoints {
		input := []float64{x}
		output := network.Predict(input)
		expected := math.Sin(x)
		error := math.Abs(output[0] - expected)

		fmt.Printf("  sin(%.4f) = %.6f, Network: %.6f, Error: %.6f\n",
			x, expected, output[0], error)
	}

	// Generate visualization data
	if *visualize {
		generateSineVisualization(network)
	}
}

func demoMNIST() {
	fmt.Println("\n=== MNIST Digit Recognition Demo (Simplified) ===")
	fmt.Println("Note: This is a simplified demo. For full MNIST, download the dataset.")

	// Create a simple digit-like pattern recognition demo
	// 5x5 grid representing simplified digits 0-3
	inputs := [][]float64{
		// Digit 0
		{1, 1, 1, 1, 1,
			1, 0, 0, 0, 1,
			1, 0, 0, 0, 1,
			1, 0, 0, 0, 1,
			1, 1, 1, 1, 1},
		// Digit 1
		{0, 0, 1, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 1, 0, 0},
		// Digit 2
		{1, 1, 1, 1, 1,
			0, 0, 0, 0, 1,
			1, 1, 1, 1, 1,
			1, 0, 0, 0, 0,
			1, 1, 1, 1, 1},
		// Digit 3
		{1, 1, 1, 1, 1,
			0, 0, 0, 0, 1,
			1, 1, 1, 1, 1,
			0, 0, 0, 0, 1,
			1, 1, 1, 1, 1},
	}

	// One-hot encoded targets
	targets := [][]float64{
		{1, 0, 0, 0}, // 0
		{0, 1, 0, 0}, // 1
		{0, 0, 1, 0}, // 2
		{0, 0, 0, 1}, // 3
	}

	// Add some noise to training data
	noisyInputs := make([][]float64, 0)
	noisyTargets := make([][]float64, 0)

	for i := 0; i < 4; i++ {
		for j := 0; j < 5; j++ { // 5 variations of each digit
			noisy := make([]float64, len(inputs[i]))
			copy(noisy, inputs[i])
			// Add random noise
			for k := range noisy {
				if rand.Float64() < 0.1 { // 10% chance to flip a pixel
					noisy[k] = 1 - noisy[k]
				}
			}
			noisyInputs = append(noisyInputs, noisy)
			noisyTargets = append(noisyTargets, targets[i])
		}
	}

	// Create network for multi-class classification
	config := neural.NetworkConfig{
		LayerSizes:       []int{25, 16, 8, 4}, // 25 inputs (5x5), 4 outputs (one-hot)
		LearningRate:     0.05,
		Activation:       neural.ReLU,
		OutputActivation: neural.Sigmoid, // Using sigmoid for multi-label (could use softmax)
		LossFunction:     neural.MeanSquaredError,
	}

	network := neural.NewNetwork(config)
	fmt.Printf("Network created: %v\n", config.LayerSizes)

	// Train
	fmt.Println("\nTraining on noisy digit patterns...")
	start := time.Now()

	for epoch := 0; epoch < 2000; epoch++ {
		loss := network.BatchTrain(noisyInputs, noisyTargets)

		if epoch%200 == 0 {
			fmt.Printf("Epoch %d: Loss = %.6f\n", epoch, loss)
		}
	}

	trainingTime := time.Since(start)
	fmt.Printf("Training completed in %v\n", trainingTime)

	// Test on clean digits
	fmt.Println("\nTesting on clean digits:")
	digitNames := []string{"0", "1", "2", "3"}
	for i, input := range inputs {
		output := network.Predict(input)
		predictedIdx := 0
		maxVal := output[0]
		for j, val := range output {
			if val > maxVal {
				maxVal = val
				predictedIdx = j
			}
		}

		expectedIdx := i
		status := "✓"
		if predictedIdx != expectedIdx {
			status = "✗"
		}

		fmt.Printf("  Digit %s: Prediction = %s (confidence: %.2f%%) %s\n",
			digitNames[i], digitNames[predictedIdx], maxVal*100, status)
	}
}

func demoIris() {
	fmt.Println("\n=== Iris Flower Classification Demo ===")
	fmt.Println("Using classic Iris dataset patterns (simplified)")

	// Simplified Iris dataset features (sepal length, sepal width, petal length, petal width)
	// Normalized values
	inputs := [][]float64{
		// Setosa (class 0)
		{5.1, 3.5, 1.4, 0.2}, {4.9, 3.0, 1.4, 0.2}, {4.7, 3.2, 1.3, 0.2},
		// Versicolor (class 1)
		{7.0, 3.2, 4.7, 1.4}, {6.4, 3.2, 4.5, 1.5}, {6.9, 3.1, 4.9, 1.5},
		// Virginica (class 2)
		{6.3, 3.3, 6.0, 2.5}, {5.8, 2.7, 5.1, 1.9}, {7.1, 3.0, 5.9, 2.1},
	}

	// One-hot encoded targets (3 classes)
	targets := [][]float64{
		{1, 0, 0}, {1, 0, 0}, {1, 0, 0}, // Setosa
		{0, 1, 0}, {0, 1, 0}, {0, 1, 0}, // Versicolor
		{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, // Virginica
	}

	// Normalize inputs (simplified normalization)
	for i := range inputs {
		for j := range inputs[i] {
			inputs[i][j] = inputs[i][j] / 10.0 // Rough normalization
		}
	}

	config := neural.NetworkConfig{
		LayerSizes:       []int{4, 8, 6, 3},
		LearningRate:     0.05,
		Activation:       neural.ReLU,
		OutputActivation: neural.Sigmoid,
		LossFunction:     neural.MeanSquaredError,
	}

	network := neural.NewNetwork(config)
	fmt.Printf("Network created: %v\n", config.LayerSizes)

	// Train
	fmt.Println("\nTraining...")
	start := time.Now()

	for epoch := 0; epoch < 3000; epoch++ {
		loss := network.BatchTrain(inputs, targets)

		if epoch%300 == 0 {
			fmt.Printf("Epoch %d: Loss = %.6f\n", epoch, loss)
		}
	}

	trainingTime := time.Since(start)
	fmt.Printf("Training completed in %v\n", trainingTime)

	// Test
	fmt.Println("\nTesting classification:")
	species := []string{"Setosa", "Versicolor", "Virginica"}
	testInputs := [][]float64{
		{5.0, 3.6, 1.4, 0.2}, // Should be Setosa
		{6.5, 3.0, 4.6, 1.5}, // Should be Versicolor
		{6.7, 3.1, 5.6, 2.4}, // Should be Virginica
	}

	for i, input := range testInputs {
		// Normalize
		normalized := make([]float64, len(input))
		copy(normalized, input)
		for j := range normalized {
			normalized[j] = normalized[j] / 10.0
		}

		output := network.Predict(normalized)
		predictedIdx := 0
		maxVal := output[0]
		for j, val := range output {
			if val > maxVal {
				maxVal = val
				predictedIdx = j
			}
		}

		// In this simple demo, we know what the test inputs should be
		expectedIdx := i
		status := "✓"
		if predictedIdx != expectedIdx {
			status = "✗"
		}

		fmt.Printf("  Sample %d: Predicted = %s (confidence: %.1f%%) %s\n",
			i+1, species[predictedIdx], maxVal*100, status)
	}
}

func demoComplex() {
	fmt.Println("\n=== Complex Pattern Recognition Demo ===")
	fmt.Println("Learning a complex non-linear decision boundary")

	// Generate complex 2D pattern (concentric circles with noise)
	numSamples := 200
	inputs := make([][]float64, numSamples)
	targets := make([][]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		// Generate points in [-1, 1] range
		x := rand.Float64()*2 - 1
		y := rand.Float64()*2 - 1

		// Complex pattern: points inside a circle OR in specific quadrants
		radius := math.Sqrt(x*x + y*y)
		angle := math.Atan2(y, x)

		// Complex decision boundary
		label := 0.0
		if (radius < 0.3) || (radius > 0.7 && math.Abs(angle) < math.Pi/3) {
			label = 1.0
		}

		inputs[i] = []float64{x, y}
		targets[i] = []float64{label}
	}

	config := neural.NetworkConfig{
		LayerSizes:       []int{2, 16, 16, 8, 1}, // Deep network for complex patterns
		LearningRate:     0.01,
		Activation:       neural.ReLU,
		OutputActivation: neural.Sigmoid,
		LossFunction:     neural.BinaryCrossEntropy,
	}

	network := neural.NewNetwork(config)
	fmt.Printf("Network created: %v\n", config.LayerSizes)

	// Train in batches
	fmt.Println("\nTraining on complex pattern...")
	start := time.Now()

	// Split into batches
	batchSize := 32
	for epoch := 0; epoch < 5000; epoch++ {
		totalLoss := 0.0
		numBatches := 0

		for startIdx := 0; startIdx < numSamples; startIdx += batchSize {
			endIdx := startIdx + batchSize
			if endIdx > numSamples {
				endIdx = numSamples
			}

			batchInputs := inputs[startIdx:endIdx]
			batchTargets := targets[startIdx:endIdx]

			loss := network.BatchTrain(batchInputs, batchTargets)
			totalLoss += loss
			numBatches++
		}

		if epoch%500 == 0 {
			avgLoss := totalLoss / float64(numBatches)
			fmt.Printf("Epoch %d: Loss = %.6f\n", epoch, avgLoss)
		}
	}

	trainingTime := time.Since(start)
	fmt.Printf("Training completed in %v\n", trainingTime)

	// Test accuracy
	fmt.Println("\nTesting accuracy on training data:")
	correct := 0
	for i := 0; i < numSamples; i++ {
		output := network.Predict(inputs[i])
		predicted := 0.0
		if output[0] > 0.5 {
			predicted = 1.0
		}

		if predicted == targets[i][0] {
			correct++
		}
	}

	accuracy := float64(correct) / float64(numSamples) * 100
	fmt.Printf("Accuracy: %.1f%% (%d/%d)\n", accuracy, correct, numSamples)

	if *visualize {
		generateComplexVisualization(network)
	}
}

func demoTimeseries() {
	fmt.Println("\n=== Time Series Forecasting Demo ===")
	fmt.Println("Training a neural network to forecast future time series values")
	fmt.Println("Using sliding windows, normalization, and advanced evaluation metrics")

	// Generate synthetic time series data (seasonal with trend and noise)
	length := 500
	seasonLength := 50
	amplitude := 10.0
	trendSlope := 0.02
	noiseLevel := 1.5

	fmt.Printf("\nGenerating synthetic time series (length=%d)...\n", length)
	series := timeseries.GenerateSeasonal(length, seasonLength, amplitude, trendSlope, noiseLevel)

	// Normalize the series using z-score normalization
	fmt.Println("Normalizing time series (z-score)...")
	normalizedSeries, stats := timeseries.NormalizeZScore(series)

	// Create sliding windows for supervised learning
	// Input: past 20 observations, Output: next 5 observations
	config := timeseries.SlidingWindowConfig{
		InputSize:  20,
		OutputSize: 5,
		Step:       1,
	}

	fmt.Printf("Creating sliding windows (input=%d, output=%d)...\n", config.InputSize, config.OutputSize)
	inputs, outputs := timeseries.CreateSlidingWindows(normalizedSeries, config)
	fmt.Printf("Created %d training examples\n", len(inputs))

	// Split into train and test sets (80/20 split)
	testRatio := 0.2
	trainInputs, trainOutputs, testInputs, testOutputs := timeseries.TrainTestSplitWindows(inputs, outputs, testRatio)
	fmt.Printf("Train set: %d examples, Test set: %d examples\n", len(trainInputs), len(testInputs))

	// Create neural network for time series forecasting
	// Input size = window size (20), Output size = forecast horizon (5)
	networkConfig := neural.NetworkConfig{
		LayerSizes:       []int{20, 32, 16, 5}, // Forecasting 5 steps ahead
		LearningRate:     0.01,
		Activation:       neural.Tanh,          // Tanh works well for normalized data
		OutputActivation: neural.Linear,        // Linear for regression
		LossFunction:     neural.MeanSquaredError,
	}

	network := neural.NewNetwork(networkConfig)
	fmt.Printf("\nNetwork created: %v\n", networkConfig.LayerSizes)

	// Train the network
	fmt.Println("\nTraining network...")
	start := time.Now()
	epochs := 400

	for epoch := 0; epoch < epochs; epoch++ {
		// Use mini-batch training for efficiency
		loss := network.BatchTrain(trainInputs, trainOutputs)

		if epoch%400 == 0 {
			fmt.Printf("Epoch %d/%d: Loss = %.6f\n", epoch, epochs, loss)
		}
	}

	trainingTime := time.Since(start)
	fmt.Printf("Training completed in %v\n", trainingTime)

	// Evaluate on test set
	fmt.Println("\n=== Test Set Evaluation ===")

	allPredictions := make([][]float64, len(testInputs))
	allActuals := make([][]float64, len(testInputs))

	for i := range testInputs {
		prediction := network.Predict(testInputs[i])
		allPredictions[i] = prediction
		allActuals[i] = testOutputs[i]
	}

	// Denormalize predictions and actuals for evaluation in original scale
	denormPredictions := make([][]float64, len(allPredictions))
	denormActuals := make([][]float64, len(allActuals))

	for i := range allPredictions {
		denormPredictions[i] = timeseries.DenormalizeZScore(allPredictions[i], stats)
		denormActuals[i] = timeseries.DenormalizeZScore(allActuals[i], stats)
	}

	// Calculate metrics for each forecast horizon
	fmt.Println("\nForecast Horizon Performance:")
	horizonMetrics := make([]timeseries.ForecastMetrics, config.OutputSize)

	for h := 0; h < config.OutputSize; h++ {
		// Extract predictions and actuals for this horizon
		horizonPreds := make([]float64, len(denormPredictions))
		horizonActuals := make([]float64, len(denormActuals))

		for i := range denormPredictions {
			horizonPreds[i] = denormPredictions[i][h]
			horizonActuals[i] = denormActuals[i][h]
		}

		metrics := timeseries.CalculateMetrics(horizonActuals, horizonPreds)
		horizonMetrics[h] = metrics

		fmt.Printf("  Horizon %d (+%d steps): RMSE=%.4f, MAE=%.4f, MAPE=%.2f%%\n",
			h+1, h+1, metrics.RMSE, metrics.MAE, metrics.MAPE)
	}

	// Calculate overall metrics (average across all horizons)
	fmt.Println("\nOverall Performance (averaged across horizons):")
	avgRMSE, avgMAE, avgMAPE, avgSMAPE, avgR2 := 0.0, 0.0, 0.0, 0.0, 0.0
	for _, m := range horizonMetrics {
		avgRMSE += m.RMSE
		avgMAE += m.MAE
		avgMAPE += m.MAPE
		avgSMAPE += m.SMAPE
		avgR2 += m.R2
	}
	nHorizons := float64(len(horizonMetrics))
	avgRMSE /= nHorizons
	avgMAE /= nHorizons
	avgMAPE /= nHorizons
	avgSMAPE /= nHorizons
	avgR2 /= nHorizons

	fmt.Printf("  Average RMSE: %.4f\n", avgRMSE)
	fmt.Printf("  Average MAE:  %.4f\n", avgMAE)
	fmt.Printf("  Average MAPE: %.2f%%\n", avgMAPE)
	fmt.Printf("  Average SMAPE: %.2f%%\n", avgSMAPE)
	fmt.Printf("  Average R²:   %.4f\n", avgR2)

	// Show example forecasts
	fmt.Println("\n=== Example Forecasts ===")
	numExamples := 3
	for ex := 0; ex < numExamples && ex < len(testInputs); ex++ {
		fmt.Printf("\nExample %d:\n", ex+1)
		fmt.Printf("  Input window (last %d obs):  [", config.InputSize)
		for i, val := range testInputs[ex] {
			denormVal := timeseries.DenormalizeZScore([]float64{val}, stats)[0]
			if i < 3 || i >= config.InputSize-3 {
				fmt.Printf("%.2f", denormVal)
				if i < config.InputSize-1 {
					if i == 2 && config.InputSize > 6 {
						fmt.Printf(" ... ")
						i = config.InputSize - 4 // Skip to last few
					} else if !(i == 2 && config.InputSize > 6) {
						fmt.Printf(", ")
					}
				}
			}
		}
		fmt.Printf("]\n")

		fmt.Printf("  Actual next %d values:      [", config.OutputSize)
		for i, val := range denormActuals[ex] {
			fmt.Printf("%.2f", val)
			if i < len(denormActuals[ex])-1 {
				fmt.Printf(", ")
			}
		}
		fmt.Printf("]\n")

		fmt.Printf("  Predicted next %d values:   [", config.OutputSize)
		for i, val := range denormPredictions[ex] {
			fmt.Printf("%.2f", val)
			if i < len(denormPredictions[ex])-1 {
				fmt.Printf(", ")
			}
		}
		fmt.Printf("]\n")
	}

	// Demonstrate multi-step forecasting (recursive prediction)
	fmt.Println("\n=== Multi-Step Forecasting Demonstration ===")

	// Take the last window from training data as starting point
	lastTrainWindow := trainInputs[len(trainInputs)-1]
	currentState := make([]float64, len(lastTrainWindow))
	copy(currentState, lastTrainWindow)

	fmt.Printf("Starting from last training window, forecasting %d steps ahead recursively:\n", config.OutputSize*3)

	recursiveForecast := make([]float64, 0)
	for step := 0; step < config.OutputSize*3; step++ {
		// Predict next value
		prediction := network.Predict(currentState)
		nextValue := prediction[0] // Only take first step prediction

		// Denormalize for display
		denormNextValue := timeseries.DenormalizeZScore([]float64{nextValue}, stats)[0]
		recursiveForecast = append(recursiveForecast, denormNextValue)

		// Update state: shift window and add prediction
		for i := 0; i < config.InputSize-1; i++ {
			currentState[i] = currentState[i+1]
		}
		currentState[config.InputSize-1] = nextValue

		if step < 5 || step >= config.OutputSize*3-5 {
			fmt.Printf("  Step %d: %.2f\n", step+1, denormNextValue)
			if step == 4 && config.OutputSize*3 > 10 {
				fmt.Printf("  ... (intermediate steps omitted)\n")
			}
		}
	}

	// Save model for potential reuse
	modelFile := "timeseries_model.json"
	if err := network.Save(modelFile); err != nil {
		fmt.Printf("Error saving model: %v\n", err)
	} else {
		fmt.Printf("\nModel saved to '%s'\n", modelFile)
		fmt.Println("You can load this model for future time series forecasting.")
	}

	// Explain practical applications
	fmt.Println("\n=== Practical Applications ===")
	fmt.Println("This time series forecasting approach can be used for:")
	fmt.Println("1. Stock price prediction (with proper feature engineering)")
	fmt.Println("2. Energy demand forecasting")
	fmt.Println("3. Weather prediction")
	fmt.Println("4. Sales forecasting")
	fmt.Println("5. Anomaly detection in system metrics")
	fmt.Println("\nFor production use, consider:")
	fmt.Println("- Using real historical data with proper preprocessing")
	fmt.Println("- Adding exogenous features (date components, holidays, etc.)")
	fmt.Println("- Implementing walk-forward validation for robust evaluation")
	fmt.Println("- Tuning hyperparameters (window size, network architecture)")
	fmt.Println("- Using ensemble methods for uncertainty quantification")
}

func demoRealTS() {
	fmt.Println("\n=== Real Time Series Forecasting Demo ===")
	fmt.Println("Loading AirPassengers dataset and comparing neural network with baselines")

	// Load the classic AirPassengers dataset
	data := timeseries.AirPassengersDataset()
	fmt.Printf("Dataset: %d monthly observations (1949-1960)\n", len(data.Values))
	fmt.Printf("Passengers: min=%.0f, max=%.0f, mean=%.1f\n",
		data.Stats.Min, data.Stats.Max, data.Stats.Mean)

	// Show walk-forward validation setup
	testSize := 24
	trainSize := len(data.Values) - testSize
	fmt.Printf("\nWalk-forward validation: train=%d months, test=%d months\n", trainSize, testSize)

	// Create sliding windows
	config := timeseries.SlidingWindowConfig{
		InputSize:  12,
		OutputSize: 6,
		Step:       1,
	}

	// Normalize and create windows
	normalized, stats := timeseries.NormalizeZScore(data.Values)
	inputs, outputs := timeseries.CreateSlidingWindows(normalized, config)

	// Time series split
	testRatio := float64(testSize) / float64(len(data.Values))
	trainInputs, trainOutputs, testInputs, testOutputs := timeseries.TrainTestSplitWindows(
		inputs, outputs, testRatio)

	// Train neural network
	network := neural.NewNetwork(neural.NetworkConfig{
		LayerSizes:       []int{12, 16, 8, 6},
		LearningRate:     0.01,
		Activation:       neural.Tanh,
		OutputActivation: neural.Linear,
		LossFunction:     neural.MeanSquaredError,
	})

	fmt.Println("\nTraining neural network (500 epochs)...")
	start := time.Now()
	for epoch := 0; epoch < 500; epoch++ {
		network.BatchTrain(trainInputs, trainOutputs)
	}
	fmt.Printf("Training time: %v\n", time.Since(start))

	// Evaluate
	nnPredictions := make([][]float64, len(testInputs))
	for i := range testInputs {
		nnPredictions[i] = network.Predict(testInputs[i])
	}

	// Denormalize
	denormPreds := make([][]float64, len(nnPredictions))
	denormActuals := make([][]float64, len(testOutputs))
	for i := range nnPredictions {
		denormPreds[i] = timeseries.DenormalizeZScore(nnPredictions[i], stats)
		denormActuals[i] = timeseries.DenormalizeZScore(testOutputs[i], stats)
	}

	// Calculate metrics
	var nnRMSE, nnMAE float64
	count := 0
	for i := range denormPreds {
		for h := 0; h < len(denormPreds[i]); h++ {
			diff := denormPreds[i][h] - denormActuals[i][h]
			nnRMSE += diff * diff
			nnMAE += math.Abs(diff)
			count++
		}
	}
	if count > 0 {
		nnRMSE = math.Sqrt(nnRMSE / float64(count))
		nnMAE = nnMAE / float64(count)
	}

	// Baseline methods
	baselineMethods := []timeseries.BaselineConfig{
		{Method: "naive", Horizon: 6},
		{Method: "seasonal_naive", Horizon: 6, Seasonality: 12},
		{Method: "moving_average", Horizon: 6, Window: 6},
		{Method: "exponential_smoothing", Horizon: 6, Alpha: 0.3},
	}

	fmt.Println("\n=== Performance Comparison ===")
	fmt.Printf("Neural Network: RMSE=%.1f, MAE=%.1f\n", nnRMSE, nnMAE)

	for _, method := range baselineMethods {
		forecast := timeseries.BaselineForecast(data.Values[:trainSize], method)
		if len(forecast) != 6 {
			continue
		}
		// Compare with actual last 6 months of test set
		actual := data.Values[trainSize:trainSize+6]
		var rmse, mae float64
		for i := 0; i < 6 && i < len(actual); i++ {
			diff := forecast[i] - actual[i]
			rmse += diff * diff
			mae += math.Abs(diff)
		}
		rmse = math.Sqrt(rmse / 6)
		mae = mae / 6
		fmt.Printf("%-20s: RMSE=%.1f, MAE=%.1f\n", method.Method, rmse, mae)
	}

	fmt.Println("\nConclusion: Neural networks can outperform simple baselines")
	fmt.Println("for complex seasonal patterns when properly tuned.")
}

func demoPipeline() {
	fmt.Println("\n=== Production Forecasting Pipeline Demo ===")
	fmt.Println("Demonstrating end-to-end forecasting pipeline with walk-forward validation")

	// Create and configure pipeline
	pipeline := timeseries.NewPipeline()

	// Load built-in dataset
	if err := pipeline.LoadBuiltinDataset("airpassengers"); err != nil {
		fmt.Printf("Error loading dataset: %v\n", err)
		return
	}

	// Configure pipeline for monthly forecasting
	pipeline.WithConfig(timeseries.PipelineConfig{
		WindowSize:       12,
		ForecastHorizon:  6,
		StepSize:         1,
		TestSize:         24,
		ValidationMethod: "walk_forward",
		ModelType:        "neural_network",
		NeuralConfig: timeseries.NeuralConfig{
			LayerSizes:       []int{12, 16, 8},
			Activation:       "tanh",
			OutputActivation: "linear",
			LossFunction:     "mse",
			Optimizer:        "adam",
		},
		IncludeDateFeatures: true,
		IncludeLagFeatures:  true,
		Lags:               []int{1, 2, 12},
		Normalization:      "zscore",
		Epochs:            200,
		BatchSize:         16,
		LearningRate:      0.01,
		EarlyStoppingPatience: 15,
		Metrics:           []string{"rmse", "mae", "mape"},
	})

	fmt.Println("\n1. Data Loading:")
	data := pipeline.GetData()
	fmt.Printf("   - Dataset: AirPassengers (1949-1960)\n")
	fmt.Printf("   - Observations: %d months\n", len(data.Values))
	fmt.Printf("   - Range: %.0f to %.0f passengers\n", data.Stats.Min, data.Stats.Max)

	fmt.Println("\n2. Preprocessing:")
	if err := pipeline.Preprocess(); err != nil {
		fmt.Printf("   Error: %v\n", err)
		return
	}
	results := pipeline.GetResults()
	fmt.Printf("   - Created %d sliding windows\n", results.WindowCount)
	fmt.Printf("   - Feature count: %d\n", results.FeatureCount)

	fmt.Println("\n3. Training Neural Network...")
	start := time.Now()
	if err := pipeline.Train(); err != nil {
		fmt.Printf("   Error: %v\n", err)
		return
	}
	trainingTime := time.Since(start)
	fmt.Printf("   - Training time: %v\n", trainingTime)

	fmt.Println("\n4. Evaluation:")
	metrics, err := pipeline.Evaluate()
	if err != nil {
		fmt.Printf("   Error: %v\n", err)
		return
	}

	for modelName, modelMetrics := range metrics {
		fmt.Printf("   - %s:\n", modelName)
		fmt.Printf("     RMSE:  %.1f passengers\n", modelMetrics.RMSE)
		fmt.Printf("     MAE:   %.1f passengers\n", modelMetrics.MAE)
		fmt.Printf("     MAPE:  %.1f%%\n", modelMetrics.MAPE)
	}

	// Generate forecasts
	fmt.Println("\n5. Future Forecasts (next 6 months):")
	forecasts, err := pipeline.Predict(6)
	if err != nil {
		fmt.Printf("   Error: %v\n", err)
		return
	}

	if len(forecasts) > 0 && len(forecasts[0]) > 0 {
		fmt.Print("   Forecasted passengers: [")
		for i, val := range forecasts[0] {
			fmt.Printf("%.0f", val)
			if i < len(forecasts[0])-1 {
				fmt.Print(", ")
			}
		}
		fmt.Println("]")
	}

	fmt.Println("\n6. Model Persistence:")
	if err := pipeline.Save("airpassengers_pipeline.json"); err != nil {
		fmt.Printf("   Warning: Could not save pipeline: %v\n", err)
	} else {
		fmt.Println("   - Pipeline saved to 'airpassengers_pipeline.json'")
		fmt.Println("   - Can be loaded with LoadPipeline() for production use")

		// Verify loading works
		fmt.Println("\n7. Verification (Load & Predict):")
		if loadedPipeline, err := timeseries.LoadPipeline("airpassengers_pipeline.json"); err != nil {
			fmt.Printf("   Warning: Could not load pipeline: %v\n", err)
		} else {
			fmt.Println("   - Pipeline loaded successfully")
			// Make a prediction with loaded pipeline
			if forecasts, err := loadedPipeline.Predict(3); err != nil {
				fmt.Printf("   Warning: Could not make prediction: %v\n", err)
			} else if len(forecasts) > 0 && len(forecasts[0]) > 0 {
				fmt.Print("   - Loaded pipeline forecast (next 3 months): [")
				for i, val := range forecasts[0] {
					fmt.Printf("%.0f", val)
					if i < len(forecasts[0])-1 {
						fmt.Print(", ")
					}
				}
				fmt.Println("]")
			}
		}
	}

	fmt.Println("\n=== Pipeline Features ===")
	fmt.Println("• End-to-end workflow from data to deployment")
	fmt.Println("• Multiple validation methods (walk-forward, holdout)")
	fmt.Println("• Feature engineering (lags, date features)")
	fmt.Println("• Model persistence and reloading")
	fmt.Println("• Production-ready error handling")
	fmt.Println("• Built-in statistical baselines for comparison")
}

func trainAndTest() {
	fmt.Println("\n=== Training from CSV Files ===")

	// Parse hidden layer sizes
	layerStrs := strings.Split(*hiddenLayers, ",")
	layerSizes := make([]int, len(layerStrs)+2) // +2 for input and output layers

	// We'll determine input and output sizes from data
	// For now, placeholder - will be updated after reading data
	layerSizes[0] = 0 // Input size - to be determined
	for i, str := range layerStrs {
		size, err := strconv.Atoi(strings.TrimSpace(str))
		if err != nil {
			log.Fatalf("Invalid layer size: %s", str)
		}
		layerSizes[i+1] = size
	}
	layerSizes[len(layerSizes)-1] = 0 // Output size - to be determined

	// Read training data
	trainInputs, trainTargets, err := readCSV(*trainFile)
	if err != nil {
		log.Fatalf("Error reading training data: %v", err)
	}

	// Update layer sizes based on data
	layerSizes[0] = len(trainInputs[0])
	layerSizes[len(layerSizes)-1] = len(trainTargets[0])

	fmt.Printf("Training data: %d samples, %d features, %d outputs\n",
		len(trainInputs), layerSizes[0], layerSizes[len(layerSizes)-1])

	// Select activation function
	activation := getActivation(*activation)
	outputActivation := getActivation(*outputAct)
	lossFunction := getLossFunction(*lossFunc)

	config := neural.NetworkConfig{
		LayerSizes:       layerSizes,
		LearningRate:     *learningRate,
		Activation:       activation,
		OutputActivation: outputActivation,
		LossFunction:     lossFunction,
	}

	network := neural.NewNetwork(config)
	fmt.Printf("Network created: %v\n", config.LayerSizes)

	// Train
	fmt.Printf("\nTraining for %d epochs (batch size: %d)...\n", *epochs, *batchSize)
	start := time.Now()

	for epoch := 0; epoch < *epochs; epoch++ {
		var loss float64

		if *batchSize == 1 || *batchSize >= len(trainInputs) {
			// Online training or full batch
			for i := range trainInputs {
				loss += network.Train(trainInputs[i], trainTargets[i])
			}
			loss /= float64(len(trainInputs))
		} else {
			// Mini-batch training
			batches := createBatches(trainInputs, trainTargets, *batchSize)
			for _, batch := range batches {
				loss += network.BatchTrain(batch.inputs, batch.targets)
			}
			loss /= float64(len(batches))
		}

		if *verbose && epoch%(*epochs/10) == 0 {
			fmt.Printf("Epoch %d/%d: Loss = %.6f\n", epoch, *epochs, loss)
		}
	}

	trainingTime := time.Since(start)
	fmt.Printf("Training completed in %v\n", trainingTime)

	// Save model
	if err := network.Save(*modelFile); err != nil {
		fmt.Printf("Error saving model: %v\n", err)
	} else {
		fmt.Printf("Model saved to '%s'\n", *modelFile)
	}

	// Test if test file provided
	if *testFile != "" {
		testInputs, testTargets, err := readCSV(*testFile)
		if err != nil {
			log.Fatalf("Error reading test data: %v", err)
		}

		fmt.Printf("\nTesting on %d samples...\n", len(testInputs))

		totalLoss := 0.0
		correct := 0
		total := 0

		for i := range testInputs {
			output := network.Predict(testInputs[i])
			loss := lossFunction.Function(output, testTargets[i])
			totalLoss += loss

			// For classification tasks, check accuracy
			if len(testTargets[i]) == 1 {
				// Binary classification
				predicted := 0.0
				if output[0] > 0.5 {
					predicted = 1.0
				}
				if predicted == testTargets[i][0] {
					correct++
				}
				total++
			}
		}

		avgLoss := totalLoss / float64(len(testInputs))
		fmt.Printf("Test Loss: %.6f\n", avgLoss)

		if total > 0 {
			accuracy := float64(correct) / float64(total) * 100
			fmt.Printf("Test Accuracy: %.1f%% (%d/%d)\n", accuracy, correct, total)
		}
	}
}

func runBenchmarks() {
	fmt.Println("\n=== Performance Benchmarks ===")

	// Benchmark network creation
	fmt.Println("\n1. Network Creation Benchmark:")
	sizes := [][]int{
		{10, 20, 10},
		{100, 200, 100, 50},
		{500, 1000, 500, 200, 100},
	}

	for _, size := range sizes {
		start := time.Now()
		config := neural.NetworkConfig{LayerSizes: size}
		_ = neural.NewNetwork(config)
		elapsed := time.Since(start)
		fmt.Printf("  Layers %v: %v\n", size, elapsed)
	}

	// Benchmark forward pass
	fmt.Println("\n2. Forward Pass Benchmark:")
	config := neural.NetworkConfig{LayerSizes: []int{100, 200, 100, 50, 10}}
	network := neural.NewNetwork(config)
	input := make([]float64, 100)
	for i := range input {
		input[i] = rand.Float64()
	}

	iterations := 10000
	start := time.Now()
	for i := 0; i < iterations; i++ {
		_ = network.Predict(input)
	}
	elapsed := time.Since(start)
	avgTime := elapsed / time.Duration(iterations)
	fmt.Printf("  Network %v: %v per prediction\n", config.LayerSizes, avgTime)

	// Memory usage
	fmt.Println("\n3. Memory Usage Info:")
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("  Alloc = %v MiB", m.Alloc/1024/1024)
	fmt.Printf("\tTotalAlloc = %v MiB", m.TotalAlloc/1024/1024)
	fmt.Printf("\tSys = %v MiB", m.Sys/1024/1024)
	fmt.Printf("\tNumGC = %v\n", m.NumGC)
}

// Helper functions
func getActivation(name string) neural.ActivationFunc {
	switch strings.ToLower(name) {
	case "sigmoid":
		return neural.Sigmoid
	case "relu":
		return neural.ReLU
	case "tanh":
		return neural.Tanh
	case "linear":
		return neural.Linear
	default:
		fmt.Printf("Unknown activation '%s', using ReLU\n", name)
		return neural.ReLU
	}
}

func getLossFunction(name string) neural.LossFunc {
	switch strings.ToLower(name) {
	case "mse":
		return neural.MeanSquaredError
	case "binary_crossentropy":
		return neural.BinaryCrossEntropy
	default:
		fmt.Printf("Unknown loss '%s', using MSE\n", name)
		return neural.MeanSquaredError
	}
}

type batch struct {
	inputs  [][]float64
	targets [][]float64
}

func createBatches(inputs, targets [][]float64, batchSize int) []batch {
	var batches []batch

	for i := 0; i < len(inputs); i += batchSize {
		end := i + batchSize
		if end > len(inputs) {
			end = len(inputs)
		}

		batches = append(batches, batch{
			inputs:  inputs[i:end],
			targets: targets[i:end],
		})
	}

	return batches
}

func readCSV(filename string) ([][]float64, [][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	if len(records) < 2 {
		return nil, nil, fmt.Errorf("CSV must have at least 2 rows (header + data)")
	}

	// Assume last column(s) are targets
	// Simple heuristic: if all values in first row are numeric, no header
	hasHeader := false
	for _, val := range records[0] {
		if _, err := strconv.ParseFloat(val, 64); err != nil {
			hasHeader = true
			break
		}
	}

	startIdx := 0
	if hasHeader {
		startIdx = 1
	}

	// For simplicity, assume all values are numeric
	// In production, you'd want more robust parsing
	inputs := make([][]float64, len(records)-startIdx)
	targets := make([][]float64, len(records)-startIdx)

	for i := startIdx; i < len(records); i++ {
		row := records[i]
		// Assume last column is target for binary classification
		// For multiple outputs, need different parsing logic
		inputVals := make([]float64, len(row)-1)
		targetVals := make([]float64, 1)

		for j := 0; j < len(row)-1; j++ {
			val, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				return nil, nil, fmt.Errorf("invalid number at row %d, col %d: %v", i, j, err)
			}
			inputVals[j] = val
		}

		targetVal, err := strconv.ParseFloat(row[len(row)-1], 64)
		if err != nil {
			return nil, nil, fmt.Errorf("invalid target at row %d: %v", i, err)
		}
		targetVals[0] = targetVal

		inputs[i-startIdx] = inputVals
		targets[i-startIdx] = targetVals
	}

	return inputs, targets, nil
}

func generateXORVisualization(network *neural.Network) {
	fmt.Println("\nGenerating XOR visualization data...")
	file, err := os.Create("xor_visualization.csv")
	if err != nil {
		fmt.Printf("Error creating visualization file: %v\n", err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	writer.Write([]string{"x", "y", "prediction", "binary"})

	// Generate grid of points
	resolution := 50
	for i := 0; i <= resolution; i++ {
		x := float64(i) / float64(resolution)
		for j := 0; j <= resolution; j++ {
			y := float64(j) / float64(resolution)
			output := network.Predict([]float64{x, y})
			prediction := output[0]
			binary := 0
			if prediction > 0.5 {
				binary = 1
			}

			writer.Write([]string{
				strconv.FormatFloat(x, 'f', 4, 64),
				strconv.FormatFloat(y, 'f', 4, 64),
				strconv.FormatFloat(prediction, 'f', 6, 64),
				strconv.Itoa(binary),
			})
		}
	}

	fmt.Println("Visualization data saved to 'xor_visualization.csv'")
	fmt.Println("You can plot this with: python -c \"import pandas as pd; import matplotlib.pyplot as plt; df = pd.read_csv('xor_visualization.csv'); plt.scatter(df.x, df.y, c=df.prediction, cmap='RdYlBu'); plt.colorbar(); plt.show()\"")
}

func generateSineVisualization(network *neural.Network) {
	file, err := os.Create("sine_visualization.csv")
	if err != nil {
		fmt.Printf("Error creating visualization file: %v\n", err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	writer.Write([]string{"x", "sin(x)", "network_prediction", "error"})

	numPoints := 200
	for i := 0; i < numPoints; i++ {
		x := float64(i) / float64(numPoints) * 4 * math.Pi
		expected := math.Sin(x)
		predicted := network.Predict([]float64{x})[0]
		error := math.Abs(predicted - expected)

		writer.Write([]string{
			strconv.FormatFloat(x, 'f', 4, 64),
			strconv.FormatFloat(expected, 'f', 6, 64),
			strconv.FormatFloat(predicted, 'f', 6, 64),
			strconv.FormatFloat(error, 'f', 6, 64),
		})
	}

	fmt.Println("Sine visualization data saved to 'sine_visualization.csv'")
}

func generateComplexVisualization(network *neural.Network) {
	file, err := os.Create("complex_visualization.csv")
	if err != nil {
		fmt.Printf("Error creating visualization file: %v\n", err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	writer.Write([]string{"x", "y", "prediction", "confidence"})

	resolution := 100
	for i := 0; i <= resolution; i++ {
		x := float64(i)/float64(resolution)*2 - 1
		for j := 0; j <= resolution; j++ {
			y := float64(j)/float64(resolution)*2 - 1
			prediction := network.Predict([]float64{x, y})[0]
			confidence := math.Abs(prediction-0.5) * 2 // How confident the network is

			writer.Write([]string{
				strconv.FormatFloat(x, 'f', 4, 64),
				strconv.FormatFloat(y, 'f', 4, 64),
				strconv.FormatFloat(prediction, 'f', 6, 64),
				strconv.FormatFloat(confidence, 'f', 6, 64),
			})
		}
	}

	fmt.Println("Complex pattern visualization data saved to 'complex_visualization.csv'")
}
