package timeseries

import (
	"math"
)

// BaselineConfig holds configuration for baseline forecasting methods
type BaselineConfig struct {
	Method      string  // "naive", "seasonal_naive", "moving_average", "exponential_smoothing", "holt_winters", "linear_trend"
	Horizon     int     // Number of steps to forecast
	Window      int     // Window size for moving average
	Seasonality int     // Seasonality period for seasonal methods
	Alpha       float64 // Smoothing parameter for exponential smoothing
	Beta        float64 // Trend smoothing parameter for Holt-Winters
	Gamma       float64 // Seasonal smoothing parameter for Holt-Winters
}

// BaselineForecast performs a baseline forecast using the specified method
func BaselineForecast(series []float64, config BaselineConfig) []float64 {
	if len(series) == 0 {
		return nil
	}

	switch config.Method {
	case "naive":
		return NaiveForecast(series, config.Horizon)
	case "seasonal_naive":
		if config.Seasonality <= 0 {
			config.Seasonality = 1
		}
		return SeasonalNaiveForecast(series, config.Seasonality, config.Horizon)
	case "moving_average":
		if config.Window <= 0 {
			config.Window = 3
		}
		return MovingAverageForecast(series, config.Window, config.Horizon)
	case "exponential_smoothing":
		if config.Alpha <= 0 || config.Alpha > 1 {
			config.Alpha = 0.3
		}
		return ExponentialSmoothing(series, config.Alpha, config.Horizon)
	case "holt_winters":
		if config.Seasonality <= 0 {
			config.Seasonality = 12 // Default monthly seasonality
		}
		if config.Alpha <= 0 || config.Alpha > 1 {
			config.Alpha = 0.3
		}
		if config.Beta <= 0 || config.Beta > 1 {
			config.Beta = 0.1
		}
		if config.Gamma <= 0 || config.Gamma > 1 {
			config.Gamma = 0.1
		}
		return HoltWinters(series, config.Seasonality, config.Alpha, config.Beta, config.Gamma, config.Horizon)
	case "linear_trend":
		return LinearTrendForecast(series, config.Horizon)
	default:
		// Default to naive forecast
		return NaiveForecast(series, config.Horizon)
	}
}

// NaiveForecast returns the naive forecast (last observation carried forward)
func NaiveForecast(series []float64, horizon int) []float64 {
	if len(series) == 0 || horizon <= 0 {
		return nil
	}

	forecast := make([]float64, horizon)
	lastValue := series[len(series)-1]
	for i := 0; i < horizon; i++ {
		forecast[i] = lastValue
	}
	return forecast
}

// SeasonalNaiveForecast returns the seasonal naive forecast
// Uses the observation from the same position in the previous season
func SeasonalNaiveForecast(series []float64, seasonality int, horizon int) []float64 {
	if len(series) == 0 || horizon <= 0 || seasonality <= 0 {
		return nil
	}

	// Ensure we have enough data for at least one full season
	if len(series) < seasonality {
		return NaiveForecast(series, horizon)
	}

	forecast := make([]float64, horizon)
	for i := 0; i < horizon; i++ {
		// Find the corresponding position in the last available season
		seasonPos := (len(series) - seasonality + i) % seasonality
		if seasonPos < 0 {
			seasonPos = 0
		}
		// Walk back through seasons until we find data
		for j := len(series) - seasonality + seasonPos; j >= 0; j -= seasonality {
			if j < len(series) {
				forecast[i] = series[j]
				break
			}
		}
	}
	return forecast
}

// MovingAverageForecast returns forecast based on moving average
func MovingAverageForecast(series []float64, window int, horizon int) []float64 {
	if len(series) == 0 || horizon <= 0 || window <= 0 {
		return nil
	}

	// Use available window size if series is shorter
	if window > len(series) {
		window = len(series)
	}

	// Calculate the moving average of the last window observations
	var sum float64
	for i := len(series) - window; i < len(series); i++ {
		sum += series[i]
	}
	ma := sum / float64(window)

	// Forecast is the moving average for all horizons
	forecast := make([]float64, horizon)
	for i := 0; i < horizon; i++ {
		forecast[i] = ma
	}
	return forecast
}

// ExponentialSmoothing returns forecast using simple exponential smoothing
func ExponentialSmoothing(series []float64, alpha float64, horizon int) []float64 {
	if len(series) == 0 || horizon <= 0 {
		return nil
	}

	// Initialize with first observation
	level := series[0]

	// Update level for each observation
	for i := 1; i < len(series); i++ {
		level = alpha*series[i] + (1-alpha)*level
	}

	// Forecast is the final level for all horizons
	forecast := make([]float64, horizon)
	for i := 0; i < horizon; i++ {
		forecast[i] = level
	}
	return forecast
}

// HoltWinters implements the Holt-Winters method with additive seasonality
func HoltWinters(series []float64, seasonality int, alpha, beta, gamma float64, horizon int) []float64 {
	if len(series) == 0 || horizon <= 0 || seasonality <= 0 {
		return nil
	}

	// Need at least 2 full seasons for initialization
	if len(series) < 2*seasonality {
		// Fall back to exponential smoothing
		return ExponentialSmoothing(series, alpha, horizon)
	}

	n := len(series)

	// Initialize level, trend, and seasonal components
	level := make([]float64, n)
	trend := make([]float64, n)
	seasonal := make([]float64, n)

	// Initial seasonal components (average of each seasonal position)
	for i := 0; i < seasonality; i++ {
		sum := 0.0
		count := 0
		for j := i; j < n; j += seasonality {
			sum += series[j]
			count++
		}
		if count > 0 {
			seasonal[i] = sum / float64(count)
		}
	}

	// Normalize seasonal components
	seasonalSum := 0.0
	for i := 0; i < seasonality; i++ {
		seasonalSum += seasonal[i]
	}
	adjustment := seasonalSum / float64(seasonality)
	for i := 0; i < seasonality; i++ {
		seasonal[i] -= adjustment
	}

	// Initialize level and trend
	level[seasonality-1] = series[seasonality-1] - seasonal[seasonality-1]
	trend[seasonality-1] = 0.0

	// Update equations for each time period
	for t := seasonality; t < n; t++ {
		// Update level, trend, and seasonal
		prevLevel := level[t-1]
		prevTrend := trend[t-1]
		seasonIdx := t % seasonality

		newLevel := alpha*(series[t]-seasonal[seasonIdx]) + (1-alpha)*(prevLevel+prevTrend)
		newTrend := beta*(newLevel-prevLevel) + (1-beta)*prevTrend
		newSeasonal := gamma*(series[t]-newLevel) + (1-gamma)*seasonal[seasonIdx]

		level[t] = newLevel
		trend[t] = newTrend
		seasonal[seasonIdx] = newSeasonal
	}

	// Generate forecasts
	forecast := make([]float64, horizon)
	lastLevel := level[n-1]
	lastTrend := trend[n-1]
	lastSeasonIdx := (n - 1) % seasonality

	for i := 0; i < horizon; i++ {
		h := i + 1
		seasonIdx := (lastSeasonIdx + h) % seasonality
		if seasonIdx < 0 {
			seasonIdx += seasonality
		}
		forecast[i] = lastLevel + float64(h)*lastTrend + seasonal[seasonIdx]
	}

	return forecast
}

// LinearTrendForecast forecasts using linear regression on the time index
func LinearTrendForecast(series []float64, horizon int) []float64 {
	if len(series) < 2 || horizon <= 0 {
		return NaiveForecast(series, horizon)
	}

	// Simple linear regression: y = a + b*t
	// where t is time index (0, 1, 2, ...)
	n := float64(len(series))

	// Calculate means
	var sumT, sumY, sumTT, sumTY float64
	for i := 0; i < len(series); i++ {
		t := float64(i)
		y := series[i]
		sumT += t
		sumY += y
		sumTT += t * t
		sumTY += t * y
	}

	// Calculate slope (b) and intercept (a)
	denom := n*sumTT - sumT*sumT
	if math.Abs(denom) < 1e-10 {
		// Avoid division by zero, fall back to naive
		return NaiveForecast(series, horizon)
	}

	b := (n*sumTY - sumT*sumY) / denom
	a := (sumY - b*sumT) / n

	// Generate forecasts
	forecast := make([]float64, horizon)
	for i := 0; i < horizon; i++ {
		t := float64(len(series) + i)
		forecast[i] = a + b*t
	}

	return forecast
}

// EvaluateBaselines performs walk-forward validation for multiple baseline methods
func EvaluateBaselines(series []float64, testSize int, methods []BaselineConfig) map[string][]ForecastMetrics {
	if len(series) == 0 || testSize <= 0 || len(methods) == 0 {
		return nil
	}

	results := make(map[string][]ForecastMetrics)

	// Perform walk-forward validation
	trainSize := len(series) - testSize
	if trainSize <= 0 {
		return results
	}

	for _, method := range methods {
		methodName := method.Method
		if methodName == "" {
			methodName = "unknown"
		}

		horizonMetrics := make([]ForecastMetrics, method.Horizon)
		horizonCounts := make([]int, method.Horizon)

		// For each step in the test set
		for step := 0; step < testSize-method.Horizon+1; step++ {
			trainSeries := series[step : trainSize+step]
			testSeries := series[trainSize+step : trainSize+step+method.Horizon]

			// Generate forecast
			forecast := BaselineForecast(trainSeries, method)

			if len(forecast) != method.Horizon {
				continue
			}

			// Calculate metrics for each horizon
			for h := 0; h < method.Horizon; h++ {
				if h < len(testSeries) {
					actual := testSeries[h]
					predicted := forecast[h]

					// Accumulate for later averaging
					horizonMetrics[h].RMSE += (actual - predicted) * (actual - predicted)
					horizonMetrics[h].MAE += math.Abs(actual - predicted)
					if actual != 0 {
						horizonMetrics[h].MAPE += math.Abs((actual-predicted)/actual) * 100
					}
					horizonMetrics[h].SMAPE += (math.Abs(actual-predicted) / ((math.Abs(actual) + math.Abs(predicted)) / 2)) * 100
					horizonMetrics[h].R2 += 0 // R² calculation requires more context

					horizonCounts[h]++
				}
			}
		}

		// Calculate average metrics for each horizon
		for h := 0; h < method.Horizon; h++ {
			if horizonCounts[h] > 0 {
				count := float64(horizonCounts[h])
				horizonMetrics[h].RMSE = math.Sqrt(horizonMetrics[h].RMSE / count)
				horizonMetrics[h].MAE = horizonMetrics[h].MAE / count
				horizonMetrics[h].MAPE = horizonMetrics[h].MAPE / count
				horizonMetrics[h].SMAPE = horizonMetrics[h].SMAPE / count
			}
		}

		results[methodName] = horizonMetrics
	}

	return results
}

// CompareWithNeuralNetwork compares baseline methods with neural network predictions
func CompareWithNeuralNetwork(actuals, nnPredictions [][]float64, series []float64, methods []BaselineConfig) map[string]ForecastMetrics {
	if len(actuals) == 0 || len(nnPredictions) == 0 || len(actuals) != len(nnPredictions) {
		return nil
	}

	results := make(map[string]ForecastMetrics)
	horizon := len(actuals[0])

	// Calculate neural network metrics
	var nnMetrics ForecastMetrics
	nnCount := 0

	for i := 0; i < len(actuals); i++ {
		for h := 0; h < horizon && h < len(actuals[i]) && h < len(nnPredictions[i]); h++ {
			actual := actuals[i][h]
			predicted := nnPredictions[i][h]

			nnMetrics.RMSE += (actual - predicted) * (actual - predicted)
			nnMetrics.MAE += math.Abs(actual - predicted)
			if actual != 0 {
				nnMetrics.MAPE += math.Abs((actual-predicted)/actual) * 100
			}
			nnMetrics.SMAPE += (math.Abs(actual-predicted) / ((math.Abs(actual) + math.Abs(predicted)) / 2)) * 100
			nnCount++
		}
	}

	if nnCount > 0 {
		count := float64(nnCount)
		nnMetrics.RMSE = math.Sqrt(nnMetrics.RMSE / count)
		nnMetrics.MAE = nnMetrics.MAE / count
		nnMetrics.MAPE = nnMetrics.MAPE / count
		nnMetrics.SMAPE = nnMetrics.SMAPE / count
		results["neural_network"] = nnMetrics
	}

	// Calculate baseline metrics for the same test windows
	for _, method := range methods {
		var baselineMetrics ForecastMetrics
		baselineCount := 0

		for i := 0; i < len(actuals); i++ {
			// Get the training series that would have been available
			// This is a simplification - in real walk-forward, we'd need the training window
			if i*method.Horizon >= len(series) {
				continue
			}

			trainEnd := i * method.Horizon
			if trainEnd > len(series) {
				trainEnd = len(series)
			}

			trainSeries := series[:trainEnd]
			forecast := BaselineForecast(trainSeries, method)

			for h := 0; h < horizon && h < len(actuals[i]) && h < len(forecast); h++ {
				actual := actuals[i][h]
				predicted := forecast[h]

				baselineMetrics.RMSE += (actual - predicted) * (actual - predicted)
				baselineMetrics.MAE += math.Abs(actual - predicted)
				if actual != 0 {
					baselineMetrics.MAPE += math.Abs((actual-predicted)/actual) * 100
				}
				baselineMetrics.SMAPE += (math.Abs(actual-predicted) / ((math.Abs(actual) + math.Abs(predicted)) / 2)) * 100
				baselineCount++
			}
		}

		if baselineCount > 0 {
			count := float64(baselineCount)
			baselineMetrics.RMSE = math.Sqrt(baselineMetrics.RMSE / count)
			baselineMetrics.MAE = baselineMetrics.MAE / count
			baselineMetrics.MAPE = baselineMetrics.MAPE / count
			baselineMetrics.SMAPE = baselineMetrics.SMAPE / count
			results[method.Method] = baselineMetrics
		}
	}

	return results
}

// PersistenceModel returns a persistence forecast (naive for horizon=1, recursive for horizon>1)
func PersistenceModel(series []float64, horizon int) []float64 {
	if len(series) == 0 || horizon <= 0 {
		return nil
	}

	if horizon == 1 {
		return NaiveForecast(series, horizon)
	}

	// For multi-step, use recursive naive forecasting
	forecast := make([]float64, horizon)
	lastValue := series[len(series)-1]

	// First step is last observation
	forecast[0] = lastValue

	// Subsequent steps use the previous forecast (persistence)
	for i := 1; i < horizon; i++ {
		forecast[i] = forecast[i-1]
	}

	return forecast
}

// DriftModel returns a forecast with drift (naive with trend)
func DriftModel(series []float64, horizon int) []float64 {
	if len(series) < 2 || horizon <= 0 {
		return NaiveForecast(series, horizon)
	}

	// Calculate drift (average change per period)
	firstValue := series[0]
	lastValue := series[len(series)-1]
	nPeriods := float64(len(series) - 1)

	if nPeriods == 0 {
		return NaiveForecast(series, horizon)
	}

	drift := (lastValue - firstValue) / nPeriods

	// Generate forecast with drift
	forecast := make([]float64, horizon)
	for i := 0; i < horizon; i++ {
		forecast[i] = lastValue + drift*float64(i+1)
	}

	return forecast
}

// ThetaModel implements a simplified Theta method (theta = 2 is often used)
func ThetaModel(series []float64, theta float64, horizon int) []float64 {
	if len(series) == 0 || horizon <= 0 {
		return nil
	}

	if theta == 0 {
		theta = 2.0 // Default theta value
	}

	// Calculate linear trend forecast
	linearForecast := LinearTrendForecast(series, horizon)

	// Calculate naive forecast
	naiveForecast := NaiveForecast(series, horizon)

	// Theta forecast is weighted combination
	forecast := make([]float64, horizon)
	for i := 0; i < horizon; i++ {
		// theta * linear + (1 - theta) * naive
		forecast[i] = theta*linearForecast[i] + (1-theta)*naiveForecast[i]
	}

	return forecast
}
