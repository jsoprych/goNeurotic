package timeseries

import (
	"fmt"
	"math"
	"sort"
)

// ============================================================================
// Data Structures
// ============================================================================

// TimeSeries represents a univariate time series
type TimeSeries struct {
	Data []float64
}

// MultivariateTimeSeries represents a multivariate time series
type MultivariateTimeSeries struct {
	Data [][]float64 // [time][feature]
}

// SlidingWindowConfig configures sliding window creation
type SlidingWindowConfig struct {
	InputSize  int // Number of past observations to use as input
	OutputSize int // Number of future observations to predict
	Step       int // Step size between windows (default: 1)
}

// NormalizationStats holds statistics for normalization
type NormalizationStats struct {
	Mean   float64
	StdDev float64
	Min    float64
	Max    float64
}

// ForecastMetrics holds evaluation metrics for time series forecasts
type ForecastMetrics struct {
	RMSE  float64 // Root Mean Square Error
	MAE   float64 // Mean Absolute Error
	MAPE  float64 // Mean Absolute Percentage Error
	SMAPE float64 // Symmetric Mean Absolute Percentage Error
	R2    float64 // R-squared (coefficient of determination)
}

// ============================================================================
// Sliding Window Creation
// ============================================================================

// CreateSlidingWindows creates input-output pairs for time series forecasting
func CreateSlidingWindows(series []float64, config SlidingWindowConfig) ([][]float64, [][]float64) {
	if len(series) < config.InputSize+config.OutputSize {
		panic(fmt.Sprintf("series length %d < input+output size %d",
			len(series), config.InputSize+config.OutputSize))
	}

	if config.Step <= 0 {
		config.Step = 1
	}

	var inputs, outputs [][]float64

	for i := 0; i <= len(series)-config.InputSize-config.OutputSize; i += config.Step {
		// Input window: past observations
		inputWindow := make([]float64, config.InputSize)
		copy(inputWindow, series[i:i+config.InputSize])

		// Output window: future observations to predict
		outputWindow := make([]float64, config.OutputSize)
		copy(outputWindow, series[i+config.InputSize:i+config.InputSize+config.OutputSize])

		inputs = append(inputs, inputWindow)
		outputs = append(outputs, outputWindow)
	}

	return inputs, outputs
}

// CreateMultivariateSlidingWindows creates windows for multivariate time series
func CreateMultivariateSlidingWindows(series [][]float64, config SlidingWindowConfig) ([][][]float64, [][][]float64) {
	if len(series) < config.InputSize+config.OutputSize {
		panic(fmt.Sprintf("series length %d < input+output size %d",
			len(series), config.InputSize+config.OutputSize))
	}

	if config.Step <= 0 {
		config.Step = 1
	}

	numFeatures := len(series[0])
	var inputs, outputs [][][]float64

	for i := 0; i <= len(series)-config.InputSize-config.OutputSize; i += config.Step {
		// Input window: past observations for all features
		inputWindow := make([][]float64, config.InputSize)
		for j := 0; j < config.InputSize; j++ {
			inputWindow[j] = make([]float64, numFeatures)
			copy(inputWindow[j], series[i+j])
		}

		// Output window: future observations (can predict all or subset of features)
		outputWindow := make([][]float64, config.OutputSize)
		for j := 0; j < config.OutputSize; j++ {
			outputWindow[j] = make([]float64, numFeatures)
			copy(outputWindow[j], series[i+config.InputSize+j])
		}

		inputs = append(inputs, inputWindow)
		outputs = append(outputs, outputWindow)
	}

	return inputs, outputs
}

// ============================================================================
// Normalization
// ============================================================================

// NormalizeZScore normalizes series using z-score normalization
func NormalizeZScore(series []float64) ([]float64, NormalizationStats) {
	if len(series) == 0 {
		return series, NormalizationStats{}
	}

	// Calculate statistics
	stats := CalculateStats(series)

	// Normalize
	normalized := make([]float64, len(series))
	for i, val := range series {
		if stats.StdDev == 0 {
			normalized[i] = 0
		} else {
			normalized[i] = (val - stats.Mean) / stats.StdDev
		}
	}

	return normalized, stats
}

// DenormalizeZScore reverses z-score normalization
func DenormalizeZScore(normalized []float64, stats NormalizationStats) []float64 {
	denormalized := make([]float64, len(normalized))
	for i, val := range normalized {
		denormalized[i] = val*stats.StdDev + stats.Mean
	}
	return denormalized
}

// NormalizeMinMax normalizes series to [0, 1] range
func NormalizeMinMax(series []float64) ([]float64, NormalizationStats) {
	if len(series) == 0 {
		return series, NormalizationStats{}
	}

	stats := CalculateStats(series)

	// Handle case where min == max
	if stats.Max == stats.Min {
		normalized := make([]float64, len(series))
		for i := range normalized {
			normalized[i] = 0.5
		}
		return normalized, stats
	}

	// Normalize to [0, 1]
	normalized := make([]float64, len(series))
	rangeVal := stats.Max - stats.Min
	for i, val := range series {
		normalized[i] = (val - stats.Min) / rangeVal
	}

	return normalized, stats
}

// DenormalizeMinMax reverses min-max normalization
func DenormalizeMinMax(normalized []float64, stats NormalizationStats) []float64 {
	if stats.Max == stats.Min {
		denormalized := make([]float64, len(normalized))
		for i := range denormalized {
			denormalized[i] = stats.Min
		}
		return denormalized
	}

	rangeVal := stats.Max - stats.Min
	denormalized := make([]float64, len(normalized))
	for i, val := range normalized {
		denormalized[i] = val*rangeVal + stats.Min
	}
	return denormalized
}

// ============================================================================
// Statistical Utilities
// ============================================================================

// CalculateStats calculates basic statistics for a time series
func CalculateStats(series []float64) NormalizationStats {
	if len(series) == 0 {
		return NormalizationStats{}
	}

	// Initialize with first value
	minVal := series[0]
	maxVal := series[0]
	sum := 0.0

	// Calculate min, max, and sum
	for _, val := range series {
		if val < minVal {
			minVal = val
		}
		if val > maxVal {
			maxVal = val
		}
		sum += val
	}

	mean := sum / float64(len(series))

	// Calculate standard deviation
	variance := 0.0
	for _, val := range series {
		diff := val - mean
		variance += diff * diff
	}
	variance /= float64(len(series))
	stdDev := math.Sqrt(variance)

	return NormalizationStats{
		Mean:   mean,
		StdDev: stdDev,
		Min:    minVal,
		Max:    maxVal,
	}
}

// RollingMean calculates rolling mean with specified window size
func RollingMean(series []float64, window int) []float64 {
	if len(series) < window || window <= 0 {
		return nil
	}

	result := make([]float64, len(series)-window+1)
	for i := 0; i <= len(series)-window; i++ {
		sum := 0.0
		for j := 0; j < window; j++ {
			sum += series[i+j]
		}
		result[i] = sum / float64(window)
	}

	return result
}

// RollingStdDev calculates rolling standard deviation
func RollingStdDev(series []float64, window int) []float64 {
	if len(series) < window || window <= 0 {
		return nil
	}

	result := make([]float64, len(series)-window+1)
	for i := 0; i <= len(series)-window; i++ {
		// Calculate mean
		sum := 0.0
		for j := 0; j < window; j++ {
			sum += series[i+j]
		}
		mean := sum / float64(window)

		// Calculate variance
		variance := 0.0
		for j := 0; j < window; j++ {
			diff := series[i+j] - mean
			variance += diff * diff
		}
		variance /= float64(window)

		result[i] = math.Sqrt(variance)
	}

	return result
}

// ============================================================================
// Feature Engineering
// ============================================================================

// CreateLagFeatures creates lag features for time series
func CreateLagFeatures(series []float64, lags []int) [][]float64 {
	if len(series) == 0 {
		return nil
	}

	// Find maximum lag
	maxLag := 0
	for _, lag := range lags {
		if lag > maxLag {
			maxLag = lag
		}
	}

	// Create feature matrix
	features := make([][]float64, len(series)-maxLag)
	for i := maxLag; i < len(series); i++ {
		row := make([]float64, len(lags))
		for j, lag := range lags {
			row[j] = series[i-lag]
		}
		features[i-maxLag] = row
	}

	return features
}

// CreateDateFeatures creates date-based features from timestamps
func CreateDateFeatures(timestamps []int64, includeComponents []string) [][]float64 {
	if len(timestamps) == 0 {
		return nil
	}

	features := make([][]float64, len(timestamps))
	for i, ts := range timestamps {
		t := ts
		featureVec := []float64{}

		// Convert Unix timestamp to time components
		// Note: In production, you'd use time.Unix(t, 0)
		// This is a simplified version

		for _, component := range includeComponents {
			switch component {
			case "hour":
				featureVec = append(featureVec, float64((t/3600)%24))
			case "day_of_week":
				featureVec = append(featureVec, float64((t/86400)%7))
			case "day_of_month":
				featureVec = append(featureVec, float64((t/86400)%30))
			case "month":
				featureVec = append(featureVec, float64((t/(86400*30))%12))
			case "year":
				featureVec = append(featureVec, float64(1970+t/(86400*365)))
			case "sin_hour":
				hour := (t / 3600) % 24
				featureVec = append(featureVec, math.Sin(2*math.Pi*float64(hour)/24))
			case "cos_hour":
				hour := (t / 3600) % 24
				featureVec = append(featureVec, math.Cos(2*math.Pi*float64(hour)/24))
			}
		}

		features[i] = featureVec
	}

	return features
}

// ============================================================================
// Train-Test Split
// ============================================================================

// TrainTestSplitTimeSeries splits time series into train and test sets
func TrainTestSplitTimeSeries(series []float64, testRatio float64) ([]float64, []float64) {
	if testRatio <= 0 || testRatio >= 1 {
		panic("testRatio must be between 0 and 1")
	}

	testSize := int(float64(len(series)) * testRatio)
	trainSize := len(series) - testSize

	trainSeries := make([]float64, trainSize)
	testSeries := make([]float64, testSize)

	copy(trainSeries, series[:trainSize])
	copy(testSeries, series[trainSize:])

	return trainSeries, testSeries
}

// TrainTestSplitWindows splits sliding windows into train and test sets
func TrainTestSplitWindows(inputs, outputs [][]float64, testRatio float64) ([][]float64, [][]float64, [][]float64, [][]float64) {
	if testRatio <= 0 || testRatio >= 1 {
		panic("testRatio must be between 0 and 1")
	}

	if len(inputs) != len(outputs) {
		panic("inputs and outputs must have same length")
	}

	testSize := int(float64(len(inputs)) * testRatio)
	trainSize := len(inputs) - testSize

	trainInputs := make([][]float64, trainSize)
	trainOutputs := make([][]float64, trainSize)
	testInputs := make([][]float64, testSize)
	testOutputs := make([][]float64, testSize)

	for i := 0; i < trainSize; i++ {
		trainInputs[i] = make([]float64, len(inputs[i]))
		trainOutputs[i] = make([]float64, len(outputs[i]))
		copy(trainInputs[i], inputs[i])
		copy(trainOutputs[i], outputs[i])
	}

	for i := 0; i < testSize; i++ {
		testInputs[trainSize+i] = make([]float64, len(inputs[trainSize+i]))
		testOutputs[trainSize+i] = make([]float64, len(outputs[trainSize+i]))
		copy(testInputs[trainSize+i], inputs[trainSize+i])
		copy(testOutputs[trainSize+i], outputs[trainSize+i])
	}

	return trainInputs, trainOutputs, testInputs, testOutputs
}

// WalkForwardValidation performs walk-forward validation for time series
func WalkForwardValidation(series []float64, trainSize, testSize int) [][2][]float64 {
	if trainSize <= 0 || testSize <= 0 || trainSize+testSize > len(series) {
		panic("invalid trainSize or testSize")
	}

	var splits [][2][]float64
	for i := 0; i <= len(series)-trainSize-testSize; i += testSize {
		trainEnd := i + trainSize
		testEnd := trainEnd + testSize

		trainSet := make([]float64, trainSize)
		testSet := make([]float64, testSize)

		copy(trainSet, series[i:trainEnd])
		copy(testSet, series[trainEnd:testEnd])

		splits = append(splits, [2][]float64{trainSet, testSet})
	}

	return splits
}

// ============================================================================
// Evaluation Metrics
// ============================================================================

// CalculateMetrics calculates forecast evaluation metrics
func CalculateMetrics(actual, predicted []float64) ForecastMetrics {
	if len(actual) != len(predicted) {
		panic("actual and predicted must have same length")
	}

	if len(actual) == 0 {
		return ForecastMetrics{}
	}

	// Calculate errors
	sumSquaredError := 0.0
	sumAbsoluteError := 0.0
	sumPercentageError := 0.0
	sumSymmetricPercentageError := 0.0
	sumActualSquared := 0.0

	actualMean := 0.0
	for _, val := range actual {
		actualMean += val
	}
	actualMean /= float64(len(actual))

	for i := range actual {
		actualVal := actual[i]
		predictedVal := predicted[i]

		// Skip infinite or NaN values
		if math.IsInf(actualVal, 0) || math.IsInf(predictedVal, 0) ||
		   math.IsNaN(actualVal) || math.IsNaN(predictedVal) {
			continue
		}

		error := predictedVal - actualVal
		absError := math.Abs(error)
		sumSquaredError += error * error
		sumAbsoluteError += absError

		// Calculate sum of squares for R²
		diffFromMean := actualVal - actualMean
		sumActualSquared += diffFromMean * diffFromMean

		// Calculate percentage errors (skip if actual is 0)
		if actualVal != 0 {
			percentageError := absError / math.Abs(actualVal)
			sumPercentageError += percentageError

			// SMAPE: symmetric mean absolute percentage error
			denominator := (math.Abs(actualVal) + math.Abs(predictedVal)) / 2
			if denominator != 0 {
				smapeError := absError / denominator
				sumSymmetricPercentageError += smapeError
			}
		}
	}

	n := float64(len(actual))
	rmse := math.Sqrt(sumSquaredError / n)
	mae := sumAbsoluteError / n
	mape := (sumPercentageError / n) * 100 // as percentage
	smape := (sumSymmetricPercentageError / n) * 100 // as percentage

	// R² = 1 - SS_res / SS_tot
	r2 := 0.0
	if sumActualSquared != 0 {
		r2 = 1 - sumSquaredError/sumActualSquared
	}

	return ForecastMetrics{
		RMSE:  rmse,
		MAE:   mae,
		MAPE:  mape,
		SMAPE: smape,
		R2:    r2,
	}
}

// ============================================================================
// Sequence Generation (for testing/demos)
// ============================================================================

// GenerateSinusoidal generates a sinusoidal time series
func GenerateSinusoidal(length int, frequency, amplitude, noiseLevel float64, includeTrend bool) []float64 {
	series := make([]float64, length)
	trend := 0.0
	trendIncrement := 0.01

	for i := 0; i < length; i++ {
		// Base sinusoidal signal
		value := amplitude * math.Sin(2*math.Pi*frequency*float64(i)/float64(length))

		// Add trend if requested
		if includeTrend {
			value += trend
			trend += trendIncrement
		}

		// Add Gaussian noise
		if noiseLevel > 0 {
			// Simple pseudo-random noise (in production, use proper random)
			noise := (float64((i*1103515245+12345)%65536)/65536.0*2 - 1) * noiseLevel
			value += noise
		}

		series[i] = value
	}

	return series
}

// GenerateRandomWalk generates a random walk time series
func GenerateRandomWalk(length int, startValue, stepSize float64) []float64 {
	series := make([]float64, length)
	series[0] = startValue

	for i := 1; i < length; i++ {
		// Simple random step
		step := (float64((i*1103515245+12345)%65536)/65536.0*2 - 1) * stepSize
		series[i] = series[i-1] + step
	}

	return series
}

// GenerateSeasonal generates a seasonal time series with trend and noise
func GenerateSeasonal(length int, seasonLength int, amplitude, trendSlope, noiseLevel float64) []float64 {
	series := make([]float64, length)

	for i := 0; i < length; i++ {
		// Seasonal component
		seasonal := amplitude * math.Sin(2*math.Pi*float64(i%seasonLength)/float64(seasonLength))

		// Trend component
		trend := trendSlope * float64(i)

		// Noise component
		noise := (float64((i*1103515245+12345)%65536)/65536.0*2 - 1) * noiseLevel

		series[i] = seasonal + trend + noise
	}

	return series
}

// ============================================================================
// Visualization Helpers
// ============================================================================

// FormatMetrics formats forecast metrics for display
func FormatMetrics(metrics ForecastMetrics) string {
	return fmt.Sprintf("RMSE: %.4f, MAE: %.4f, MAPE: %.2f%%, SMAPE: %.2f%%, R²: %.4f",
		metrics.RMSE, metrics.MAE, metrics.MAPE, metrics.SMAPE, metrics.R2)
}

// PrintMetrics prints forecast metrics to stdout
func PrintMetrics(metrics ForecastMetrics) {
	fmt.Println("=== Forecast Evaluation Metrics ===")
	fmt.Printf("Root Mean Square Error (RMSE): %.4f\n", metrics.RMSE)
	fmt.Printf("Mean Absolute Error (MAE): %.4f\n", metrics.MAE)
	fmt.Printf("Mean Absolute Percentage Error (MAPE): %.2f%%\n", metrics.MAPE)
	fmt.Printf("Symmetric MAPE (SMAPE): %.2f%%\n", metrics.SMAPE)
	fmt.Printf("R-squared (R²): %.4f\n", metrics.R2)
}

// ============================================================================
// Utility Functions
// ============================================================================

// FlattenMultivariate flattens multivariate series for univariate models
func FlattenMultivariate(series [][]float64, featureIndex int) []float64 {
	if len(series) == 0 {
		return nil
	}

	if featureIndex < 0 || featureIndex >= len(series[0]) {
		panic(fmt.Sprintf("featureIndex %d out of bounds [0, %d]", featureIndex, len(series[0])-1))
	}

	flattened := make([]float64, len(series))
	for i, row := range series {
		flattened[i] = row[featureIndex]
	}
	return flattened
}

// CombineFeatures combines multiple univariate series into multivariate
func CombineFeatures(series ...[]float64) [][]float64 {
	if len(series) == 0 {
		return nil
	}

	// Check all series have same length
	length := len(series[0])
	for _, s := range series {
		if len(s) != length {
			panic("all series must have same length")
		}
	}

	combined := make([][]float64, length)
	for i := 0; i < length; i++ {
		row := make([]float64, len(series))
		for j := range series {
			row[j] = series[j][i]
		}
		combined[i] = row
	}

	return combined
}

// DetectOutliers detects outliers using IQR method
func DetectOutliers(series []float64, multiplier float64) []int {
	if len(series) == 0 {
		return nil
	}

	// Create copy and sort
	sorted := make([]float64, len(series))
	copy(sorted, series)
	sort.Float64s(sorted)

	// Calculate quartiles
	q1Index := len(sorted) / 4
	q3Index := 3 * len(sorted) / 4
	q1 := sorted[q1Index]
	q3 := sorted[q3Index]
	iqr := q3 - q1

	// Define outlier bounds
	lowerBound := q1 - multiplier*iqr
	upperBound := q3 + multiplier*iqr

	// Find outliers
	var outliers []int
	for i, val := range series {
		if val < lowerBound || val > upperBound {
			outliers = append(outliers, i)
		}
	}

	return outliers
}

// ImputeMissing linear interpolation for missing values (represented as NaN)
func ImputeMissing(series []float64) []float64 {
	result := make([]float64, len(series))
	copy(result, series)

	for i := 0; i < len(result); i++ {
		if math.IsNaN(result[i]) {
			// Find previous non-NaN value
			prevIdx := -1
			for j := i - 1; j >= 0; j-- {
				if !math.IsNaN(result[j]) {
					prevIdx = j
					break
				}
			}

			// Find next non-NaN value
			nextIdx := -1
			for j := i + 1; j < len(result); j++ {
				if !math.IsNaN(result[j]) {
					nextIdx = j
					break
				}
			}

			// Interpolate or use edge values
			if prevIdx >= 0 && nextIdx >= 0 {
				// Linear interpolation
				t := float64(i-prevIdx) / float64(nextIdx-prevIdx)
				result[i] = result[prevIdx]*(1-t) + result[nextIdx]*t
			} else if prevIdx >= 0 {
				// Use previous value
				result[i] = result[prevIdx]
			} else if nextIdx >= 0 {
				// Use next value
				result[i] = result[nextIdx]
			} else {
				// All values are NaN, set to 0
				result[i] = 0
			}
		}
	}

	return result
}
