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
	"and":     demoAND,
	"xor":     demoXOR,
	"sin":     demoSine,
	"mnist":   demoMNIST,
	"iris":    demoIris,
	"complex": demoComplex,
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
		fmt.Fprintf(os.Stderr, "  %s -train data.csv -test test.csv -epochs 5000\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -benchmark\n", os.Args[0])
	}

	demo := flag.String("demo", "", "Run a demo (and, xor, sin, mnist, iris, complex)")

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
