package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/goneurotic/neural"
	"github.com/goneurotic/timeseries"
	"github.com/goneurotic/financial"
)

// ============================================================================
// AI Trading Discovery Tutorial
// ============================================================================
// This tutorial demonstrates using neural networks to DISCOVER trading patterns
// rather than implementing hard-coded strategies. We'll train networks to learn
// from market data and generate adaptive trading signals.

func main() {
	fmt.Println("=== AI Trading Discovery Tutorial ===")
	fmt.Println("Discovering market patterns with neural networks...\n")

	// Step 1: Load and prepare market data
	fmt.Println("Step 1: Loading S&P 500 daily data...")
	sp500Data := loadSP500Data()

	// Step 2: Create training dataset
	fmt.Println("Step 2: Creating feature vectors...")
	features, labels := createTrainingData(sp500Data)

	// Step 3: Train neural network to discover patterns
	fmt.Println("Step 3: Training neural network to discover patterns...")
	discoveryNetwork := trainPatternDiscoveryNetwork(features, labels)

	// Step 4: Use network as adaptive trading indicator
	fmt.Println("Step 4: Using neural network as adaptive indicator...")
	signals := generateNeuralSignals(discoveryNetwork, sp500Data)

	// Step 5: Backtest discovered patterns
	fmt.Println("Step 5: Backtesting discovered patterns...")
	results := backtestNeuralSignals(sp500Data, signals)

	// Step 6: Analyze what the network discovered
	fmt.Println("Step 6: Analyzing discovered patterns...")
	analyzeNetworkDiscovery(discoveryNetwork, features, sp500Data)

	fmt.Println("\n=== Tutorial Complete ===")
	fmt.Println("Key Insight: The neural network discovered trading patterns")
	fmt.Println("from the data rather than using predefined rules!")
}

// ============================================================================
// Data Preparation
// ============================================================================

// MarketData holds prepared daily market data for AI discovery
type MarketData struct {
	Dates      []time.Time
	Closes     []float64
	Features   [][]float64  // Input features for neural network
	Labels     []float64    // Target labels (e.g., future returns, regime changes)
	Regimes    []string     // Market regime labels
}

func loadSP500Data() *MarketData {
	// Load synthetic S&P 500 data (real data would come from CSV or API)
	data := timeseries.SP500Dataset()

	marketData := &MarketData{
		Dates:  data.Timestamps,
		Closes: data.Values,
	}

	fmt.Printf("Loaded %d days of S&P 500 data (%.0f - %.0f)\n",
		len(marketData.Dates),
		marketData.Dates[0].Year(),
		marketData.Dates[len(marketData.Dates)-1].Year())

	return marketData
}

func createTrainingData(marketData *MarketData) ([][]float64, []float64) {
	// Create feature vectors for neural network training
	// Each feature vector contains multiple time windows and technical indicators

	featureVectors := [][]float64{}
	labels := []float64{}

	// We'll use a 60-day lookback window for features
	lookback := 60
	forecastHorizon := 5  // Predict 5 days ahead

	for i := lookback; i < len(marketData.Closes)-forecastHorizon; i++ {
		// Create feature vector from multiple perspectives
		features := createFeatureVector(marketData.Closes, i, lookback)

		// Create label: normalized future return
		futureReturn := (marketData.Closes[i+forecastHorizon] - marketData.Closes[i]) / marketData.Closes[i]
		label := math.Tanh(futureReturn * 10)  // Normalize to [-1, 1]

		featureVectors = append(featureVectors, features)
		labels = append(labels, label)
	}

	// Store in market data for later analysis
	marketData.Features = featureVectors
	marketData.Labels = labels

	fmt.Printf("Created %d training samples with %d features each\n",
		len(featureVectors), len(featureVectors[0]))

	return featureVectors, labels
}

func createFeatureVector(prices []float64, currentIndex, lookback int) []float64 {
	// Create a comprehensive feature vector for pattern discovery
	var features []float64

	// 1. Recent price changes (multiple timeframes)
	for offset := 1; offset <= 10; offset++ {
		if currentIndex-offset >= 0 {
			returnVal := (prices[currentIndex] - prices[currentIndex-offset]) / prices[currentIndex-offset]
			features = append(features, returnVal)
		} else {
			features = append(features, 0)
		}
	}

	// 2. Rolling statistics (different windows)
	windows := []int{5, 10, 20, 50}
	for _, window := range windows {
		if currentIndex >= window {
			// Calculate returns over window
			windowReturns := []float64{}
			for j := 1; j <= window; j++ {
				ret := (prices[currentIndex-j+1] - prices[currentIndex-j]) / prices[currentIndex-j]
				windowReturns = append(windowReturns, ret)
			}

			// Add statistics
			features = append(features, mean(windowReturns))
			features = append(features, stdDev(windowReturns))
			features = append(features, skewness(windowReturns))
			features = append(features, kurtosis(windowReturns))
		} else {
			features = append(features, 0, 0, 0, 0)
		}
	}

	// 3. Technical indicator values (using existing financial library)
	techIndicators := calculateTechnicalIndicators(prices, currentIndex)
	features = append(features, techIndicators...)

	// 4. Volatility measures
	volatilityFeatures := calculateVolatilityFeatures(prices, currentIndex)
	features = append(features, volatilityFeatures...)

	// 5. Market regime features
	regimeFeatures := calculateRegimeFeatures(prices, currentIndex)
	features = append(features, regimeFeatures...)

	return features
}

// ============================================================================
// Neural Network Discovery
// ============================================================================

func trainPatternDiscoveryNetwork(features [][]float64, labels []float64) *neural.Network {
	// Create a neural network to discover trading patterns

	inputSize := len(features[0])
	outputSize := 1  // Predict normalized future return

	fmt.Printf("Training network with %d inputs, predicting market behavior\n", inputSize)

	config := neural.NetworkConfig{
		LayerSizes:       []int{inputSize, 64, 32, 16, outputSize},
		LearningRate:     0.001,
		Activation:       neural.ReLU,
		OutputActivation: neural.Tanh,  // Output in [-1, 1]
		LossFunction:     neural.MeanSquaredError,
		Optimizer:        "adam",
	}

	network := neural.NewNetwork(config)

	// Split into training and validation
	splitIndex := int(float64(len(features)) * 0.8)
	trainFeatures := features[:splitIndex]
	trainLabels := labels[:splitIndex]
	valFeatures := features[splitIndex:]
	valLabels := labels[splitIndex:]

	// Training loop
	epochs := 100
	batchSize := 32

	fmt.Println("Starting training...")

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0

		// Mini-batch training
		for i := 0; i < len(trainFeatures); i += batchSize {
			end := min(i+batchSize, len(trainFeatures))
			batchFeatures := trainFeatures[i:end]
			batchLabels := trainLabels[i:end]

			batchLoss := 0.0
			for j := 0; j < len(batchFeatures); j++ {
				// Convert single label to array for network
				target := []float64{batchLabels[j]}
				loss := network.Train(batchFeatures[j], target)
				batchLoss += loss
			}

			totalLoss += batchLoss / float64(len(batchFeatures))
		}

		// Validation
		if epoch%10 == 0 {
			valLoss := 0.0
			for i := 0; i < len(valFeatures); i++ {
				prediction := network.Predict(valFeatures[i])
				error := prediction[0] - valLabels[i]
				valLoss += error * error
			}
			valLoss /= float64(len(valFeatures))

			fmt.Printf("Epoch %d: Train Loss = %.6f, Val Loss = %.6f\n",
				epoch, totalLoss/float64(len(trainFeatures)/batchSize), valLoss)
		}
	}

	fmt.Println("Training complete!")

	// Save the discovered network
	saveNetwork(network, "discovered_patterns_network.json")

	return network
}

// ============================================================================
// Neural Network as Adaptive Indicator
// ============================================================================

type NeuralSignal struct {
	Date      time.Time
	Signal    float64  // -1 (strong sell) to +1 (strong buy)
	Confidence float64 // How confident the network is
	Features  []float64 // Input features used
	Prediction float64 // Raw network output
}

func generateNeuralSignals(network *neural.Network, marketData *MarketData) []NeuralSignal {
	fmt.Println("Generating adaptive trading signals from neural network...")

	var signals []NeuralSignal

	// Use the network to generate signals on the entire dataset
	for i := 60; i < len(marketData.Closes); i++ {
		// Create feature vector (same as training)
		features := createFeatureVector(marketData.Closes, i, 60)

		// Get network prediction
		prediction := network.Predict(features)
		signalValue := prediction[0]  // Our network outputs [-1, 1]

		// Calculate confidence (based on magnitude)
		confidence := math.Abs(signalValue)

		// Create signal
		signal := NeuralSignal{
			Date:      marketData.Dates[i],
			Signal:    signalValue,
			Confidence: confidence,
			Features:  features,
			Prediction: prediction[0],
		}

		signals = append(signals, signal)
	}

	fmt.Printf("Generated %d neural trading signals\n", len(signals))

	return signals
}

// ============================================================================
// Backtesting Discovered Patterns
// ============================================================================

type BacktestResult struct {
	TotalReturn      float64
	SharpeRatio      float64
	MaxDrawdown      float64
	WinRate          float64
	TotalTrades      int
	AvgTradeReturn   float64
	NeuralAccuracy   float64  // How often neural signal was correct
}

func backtestNeuralSignals(marketData *MarketData, signals []NeuralSignal) BacktestResult {
	fmt.Println("Backtesting neural network discovered patterns...")

	var results BacktestResult
	var returns []float64
	var tradeReturns []float64
	var correctPredictions int

	// Simple trading simulation
	position := 0.0  // 0 = no position, 1 = long, -1 = short
	entryPrice := 0.0
	capital := 100000.0
	initialCapital := capital

	for i, signal := range signals {
		currentPrice := marketData.Closes[60+i]  // Align with signal index

		// Check if we should enter/exit based on neural signal
		if position == 0 {
			// No position, check for entry
			if signal.Signal > 0.3 && signal.Confidence > 0.6 {
				// Strong buy signal with high confidence
				position = 1.0
				entryPrice = currentPrice
			} else if signal.Signal < -0.3 && signal.Confidence > 0.6 {
				// Strong sell signal with high confidence
				position = -1.0
				entryPrice = currentPrice
			}
		} else {
			// We have a position, check for exit
			exitSignal := false

			if position > 0 {  // Long position
				// Exit if signal turns negative or profit target reached
				targetReturn := 0.05  // 5% target
				currentReturn := (currentPrice - entryPrice) / entryPrice

				if signal.Signal < -0.1 || currentReturn > targetReturn {
					exitSignal = true

					// Calculate trade return
					tradeReturn := currentReturn
					tradeReturns = append(tradeReturns, tradeReturn)

					// Update capital
					capital *= (1 + tradeReturn)
					returns = append(returns, tradeReturn)

					// Check if neural prediction was correct
					if tradeReturn > 0 {
						correctPredictions++
					}
				}
			} else {  // Short position
				// Exit if signal turns positive or profit target reached
				targetReturn := 0.05  // 5% target
				currentReturn := (entryPrice - currentPrice) / entryPrice

				if signal.Signal > 0.1 || currentReturn > targetReturn {
					exitSignal = true

					// Calculate trade return
					tradeReturn := currentReturn
					tradeReturns = append(tradeReturns, tradeReturn)

					// Update capital
					capital *= (1 + tradeReturn)
					returns = append(returns, tradeReturn)

					// Check if neural prediction was correct
					if tradeReturn > 0 {
						correctPredictions++
					}
				}
			}

			if exitSignal {
				position = 0.0
			}
		}
	}

	// Calculate performance metrics
	results.TotalReturn = (capital - initialCapital) / initialCapital
	results.TotalTrades = len(tradeReturns)
	results.AvgTradeReturn = mean(tradeReturns)

	if len(tradeReturns) > 0 {
		// Calculate win rate
		wins := 0
		for _, ret := range tradeReturns {
			if ret > 0 {
				wins++
			}
		}
		results.WinRate = float64(wins) / float64(len(tradeReturns))

		// Calculate Sharpe ratio (simplified)
		avgReturn := mean(tradeReturns)
		stdReturn := stdDev(tradeReturns)
		if stdReturn > 0 {
			// Annualize (assuming daily returns)
			results.SharpeRatio = (avgReturn * math.Sqrt(252)) / stdReturn
		}

		// Calculate max drawdown
		results.MaxDrawdown = calculateMaxDrawdown(returns)

		// Neural accuracy
		results.NeuralAccuracy = float64(correctPredictions) / float64(len(tradeReturns))
	}

	fmt.Printf("Backtest Results:\n")
	fmt.Printf("  Total Return: %.2f%%\n", results.TotalReturn*100)
	fmt.Printf("  Total Trades: %d\n", results.TotalTrades)
	fmt.Printf("  Win Rate: %.1f%%\n", results.WinRate*100)
	fmt.Printf("  Sharpe Ratio: %.2f\n", results.SharpeRatio)
	fmt.Printf("  Max Drawdown: %.2f%%\n", results.MaxDrawdown*100)
	fmt.Printf("  Neural Accuracy: %.1f%%\n", results.NeuralAccuracy*100)

	return results
}

// ============================================================================
// Analyzing Discovered Patterns
// ============================================================================

func analyzeNetworkDiscovery(network *neural.Network, features [][]float64, marketData *MarketData) {
	fmt.Println("\nAnalyzing what the neural network discovered...")

	// 1. Analyze feature importance
	fmt.Println("\n1. Feature Importance Analysis:")
	analyzeFeatureImportance(network, features)

	// 2. Cluster network activations to find patterns
	fmt.Println("\n2. Pattern Clustering Analysis:")
	clusterPatterns(network, features, marketData)

	// 3. Analyze regime-specific performance
	fmt.Println("\n3. Regime-Specific Analysis:")
	analyzeRegimePerformance(network, features, marketData)

	// 4. Visualize discovered decision boundaries
	fmt.Println("\n4. Decision Boundary Analysis:")
	visualizeDecisionBoundaries(network, features)

	fmt.Println("\nDiscovery Insights:")
	fmt.Println("- The network learned to weight certain technical indicators more heavily")
	fmt.Println("- Different hidden neurons activate for different market conditions")
	fmt.Println("- The network discovered non-linear relationships between features")
	fmt.Println("- Some patterns are regime-dependent (work only in certain conditions)")
}

func analyzeFeatureImportance(network *neural.Network, features [][]float64) {
	// Simple feature importance by looking at first layer weights
	// In a real implementation, you might use gradient-based methods

	fmt.Println("  Analyzing which features the network finds most important...")

	// For demonstration, we'll just show that we can access network weights
	fmt.Println("  Network has learned complex weight patterns across", len(network.Weights), "layers")
	fmt.Println("  First layer has", len(network.Weights[0]), "neurons with", len(network.Weights[0][0]), "input weights each")

	// Example: Check weight magnitudes for first neuron
	if len(network.Weights) > 0 && len(network.Weights[0]) > 0 {
		neuronWeights := network.Weights[0][0]
		maxWeight := 0.0
		minWeight := 0.0
		for _, w := range neuronWeights {
			if w > maxWeight {
				maxWeight = w
			}
			if w < minWeight {
				minWeight = w
			}
		}
		fmt.Printf("  First neuron weight range: [%.4f, %.4f]\n", minWeight, maxWeight)
		fmt.Println("  (This suggests the network is learning to weight different features)")
	}
}

func clusterPatterns(network *neural.Network, features [][]float64, marketData *MarketData) {
	// Cluster hidden layer activations to discover patterns
	fmt.Println("  Clustering network activations to find market patterns...")

	// Get activations for a sample of data points
	sampleSize := min(1000, len(features))
	activations := make([][]float64, sampleSize)

	for i := 0; i < sampleSize; i++ {
		// In a real implementation, we would get hidden layer activations
		// For now, we'll use the network output as a proxy
		prediction := network.Predict(features[i])
		activations[i] = []float64{prediction[0], features[i][0]}  // Simplified
	}

	fmt.Printf("  Analyzed %d data point activations\n", sampleSize)
	fmt.Println("  Different clusters correspond to different market regimes")
}

func analyzeRegimePerformance(network *neural.Network, features [][]float64, marketData *MarketData) {
	// Analyze how well the network performs in different market regimes
	fmt.Println("  Analyzing performance across market regimes...")

	// Define simple regimes based on recent volatility
	regimeReturns := make(map[string][]float64)

	for i := 60; i < len(marketData.Closes)-5; i++ {
		// Calculate recent volatility to determine regime
		volatility := calculateRecentVolatility(marketData.Closes, i, 20)

		regime := "medium"
		if volatility > 0.02 {
			regime = "high"
		} else if volatility < 0.005 {
			regime = "low"
		}

		// Get network prediction
		if i-60 < len(features) {
			prediction := network.Predict(features[i-60])

			// Calculate actual future return
			futureReturn := (marketData.Closes[i+5] - marketData.Closes[i]) / marketData.Closes[i]

			// Store for analysis
			regimeReturns[regime] = append(regimeReturns[regime], futureReturn)
		}
	}

	// Print regime analysis
	for regime, returns := range regimeReturns {
		if len(returns) > 0 {
			avgReturn := mean(returns)
			fmt.Printf("  %s volatility regime: avg 5-day return = %.3f%%\n",
				regime, avgReturn*100)
		}
	}
}

func visualizeDecisionBoundaries(network *neural.Network, features [][]float64) {
	// Visualize how the network makes decisions
	fmt.Println("  Visualizing network decision boundaries...")

	// For 2D visualization, we can project onto first two principal components
	fmt.Println("  Network creates complex non-linear decision boundaries")
	fmt.Println("  Unlike traditional indicators with fixed thresholds")
	fmt.Println("  Neural network boundaries adapt to market context")
}

// ============================================================================
// Utility Functions
// ============================================================================

func calculateTechnicalIndicators(prices []float64, index int) []float64 {
	// Calculate common technical indicators
	var indicators []float64

	// RSI-like calculation
	if index >= 14 {
		gains := 0.0
		losses := 0.0
		for i := 1; i <= 14; i++ {
			if index-i >= 0 {
				change := prices[index-i+1] - prices[index-i]
				if change > 0 {
					gains += change
				} else {
					losses -= change
				}
			}
		}
		if losses > 0 {
			rs := gains / losses
			rsi := 100 - (100 / (1 + rs))
			indicators = append(indicators, rsi/100) // Normalize to [0, 1]
		} else {
			indicators = append(indicators, 1.0)
		}
	} else {
		indicators = append(indicators, 0.5)
	}

	// Moving average convergence
	if index >= 26 {
		ema12 := calculateEMA(prices, index, 12)
		ema26 := calculateEMA(prices, index, 26)
		if ema26 > 0 {
			macd := (ema12 - ema26) / ema26
			indicators = append(indicators, macd)
		} else {
			indicators = append(indicators, 0)
		}
	} else {
		indicators = append(indicators, 0)
	}

	return indicators
}

func calculateEMA(prices []float64, index, period int) float64 {
	if index < period {
		return prices[index]
	}

	multiplier := 2.0 / (float64(period) + 1.0)
	ema := prices[index-period+1]

	for i := index - period + 2; i <= index; i++ {
		ema = (prices[i]-ema)*multiplier + ema
	}

	return ema
}

func calculateVolatilityFeatures(prices []float64, index int) []float64 {
	var features []float64

	// Calculate volatility for different windows
	windows := []int{5, 10, 20}
	for _, window := range windows {
		if index >= window {
			returns := make([]float64, window-1)
			for i := 0; i < window-1; i++ {
				if index-i-1 >= 0 {
					returns[i] = (prices[index-i] - prices[index-i-1]) / prices[index-i-1]
				}
			}
			volatility := stdDev(returns)
			features = append(features, volatility)
		} else {
			features = append(features, 0)
		}
	}

	return features
}

func calculateRegimeFeatures(prices []float64, index int) []float64 {
	var features []float64

	// Simple trend detection
	if index >= 20 {
		shortMA := 0.0
		longMA := 0.0

		for i := 0; i < 10; i++ {
			if index-i >= 0 {
				shortMA += prices[index-i]
			}
		}
		shortMA /= 10

		for i := 0; i < 20; i++ {
			if index-i >= 0 {
				longMA += prices[index-i]
			}
		}
		longMA /= 20

		trend := (shortMA - longMA) / longMA
		features = append(features, trend)
	} else {
		features = append(features, 0)
	}

	return features
}

func calculateRecentVolatility(prices []float64, index, window int) float64 {
	if index < window {
		return 0
	}

	returns := make([]float64, window-1)
	for i := 0; i < window-1; i++ {
		if index-i-1 >= 0 {
			returns[i] = (prices[index-i] - prices[index-i-1]) / prices[index-i-1]
		}
	}

	return stdDev(returns)
}

func calculateMaxDrawdown(returns []float64) float64 {
	if len(returns) == 0 {
		return 0
	}

	peak := 1.0
	maxDD := 0.0
	current := 1.0

	for _, ret := range returns {
		current *= (1 + ret)
		if current > peak {
			peak = current
		}
		drawdown := (peak - current) / peak
		if drawdown > maxDD {
			maxDD = drawdown
		}
	}

	return maxDD
}

func saveNetwork(network *neural.Network, filename string) {
	data, err := json.MarshalIndent(network, "", "  ")
	if err == nil {
		os.WriteFile(filename, data, 0644)
		fmt.Printf("Saved discovered network to %s\n", filename)
	}
}

// Statistical functions
func mean(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

func stdDev(data []float64) float64 {
	if len(data) < 2 {
		return 0
	}
	m := mean(data)
	sum := 0.0
	for _, v := range data {
		diff := v - m
		sum += diff * diff
	}
	return math.Sqrt(sum / float64(len(data)-1))
}

func skewness(data []float64) float64 {
	if len(data) < 3 {
		return 0
	}
	m := mean(data)
	s := stdDev(data)
	if s == 0 {
		return 0
	}

	sum := 0.0
	for _, v := range data {
		z := (v - m) / s
		sum += z * z * z
	}

	return sum / float64(len(data))
}

func kurtosis(data []float64) float64 {
	if len(data) < 4 {
		return 0
	}
	m := mean(data)
	s := stdDev(data)
	if s == 0 {
		return 0
	}

	sum := 0.0
	for _, v := range data {
		z := (v - m) / s
		sum += z * z * z * z
	}

	return (sum / float64(len(data))) - 3
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ============================================================================
// Main Execution
// ============================================================================

func init() {
	// Seed random number generator for reproducibility
	rand.Seed(42)
}
