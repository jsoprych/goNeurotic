package financial

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// ============================================================================
// Core Structures
// ============================================================================

// TimeSeriesAnalysis holds comprehensive analysis of a single financial time series
type TimeSeriesAnalysis struct {
	// Basic data
	Timestamp []time.Time
	Prices    []float64
	Returns   []float64
	LogReturns []float64

	// Basic statistics
	Stats BasicStats

	// Risk metrics
	RiskMetrics RiskMetrics

	// Technical indicators
	Indicators map[string]interface{}

	// Derived features
	Features map[string][]float64
}

// PortfolioAnalysis holds analysis of multiple correlated assets
type PortfolioAnalysis struct {
	// Asset data
	AssetNames []string
	AssetReturns [][]float64
	AssetPrices [][]float64

	// Portfolio weights
	Weights []float64

	// Analysis results
	PortfolioReturns []float64
	CorrelationMatrix [][]float64
	CovarianceMatrix [][]float64

	// Portfolio metrics
	PortfolioRisk float64
	PortfolioReturn float64
	SharpeRatio float64
	SortinoRatio float64
}

// BasicStats holds basic statistical properties
type BasicStats struct {
	Count    int
	Mean     float64
	StdDev   float64
	Variance float64
	Min      float64
	Max      float64
	Median   float64
	Skewness float64
	Kurtosis float64
	Q1       float64 // First quartile
	Q3       float64 // Third quartile
	IQR      float64 // Interquartile range
}

// RiskMetrics holds comprehensive risk measurements
type RiskMetrics struct {
	// Volatility measures
	AnnualVolatility  float64
	MonthlyVolatility float64
	DailyVolatility   float64

	// Downside risk
	DownsideDeviation float64
	SemiVariance      float64

	// Drawdown analysis
	MaxDrawdown       float64
	MaxDrawdownPeriod int
	AvgDrawdown       float64
	DrawdownDuration  int

	// Value at Risk
	VaR95            float64 // 95% confidence
	VaR99            float64 // 99% confidence
	CVaR95           float64 // Conditional VaR at 95%
	CVaR99           float64 // Conditional VaR at 99%

	// Performance ratios
	SharpeRatio      float64
	SortinoRatio     float64
	CalmarRatio      float64 // Return / Max Drawdown
	OmegaRatio       float64
}

// MarketRegime identifies different market conditions
type MarketRegime string

const (
	RegimeBull      MarketRegime = "bull"
	RegimeBear      MarketRegime = "bear"
	RegimeSideways  MarketRegime = "sideways"
	RegimeHighVol   MarketRegime = "high_volatility"
	RegimeLowVol    MarketRegime = "low_volatility"
	RegimeCrisis    MarketRegime = "crisis"
	RegimeRecovery  MarketRegime = "recovery"
)

// ============================================================================
// Analysis Functions
// ============================================================================

// AnalyzeTimeSeries performs comprehensive analysis on a price series
func AnalyzeTimeSeries(prices []float64, timestamps []time.Time) *TimeSeriesAnalysis {
	if len(prices) < 2 {
		return nil
	}

	analysis := &TimeSeriesAnalysis{
		Prices:     prices,
		Timestamp: timestamps,
		Indicators: make(map[string]interface{}),
		Features:   make(map[string][]float64),
	}

	// Calculate returns
	analysis.Returns = CalculateReturns(prices)
	analysis.LogReturns = LogReturns(prices)

	// Calculate basic statistics
	analysis.calculateBasicStats()

	// Calculate risk metrics
	analysis.calculateRiskMetrics()

	// Generate features
	analysis.generateFeatures()

	return analysis
}

// calculateBasicStats computes basic statistical properties
func (tsa *TimeSeriesAnalysis) calculateBasicStats() {
	if len(tsa.Returns) == 0 {
		return
	}

	stats := BasicStats{
		Count: len(tsa.Returns),
	}

	// Calculate mean, min, max
	sum := 0.0
	sumSq := 0.0
	stats.Min = tsa.Returns[0]
	stats.Max = tsa.Returns[0]

	for _, ret := range tsa.Returns {
		sum += ret
		sumSq += ret * ret
		if ret < stats.Min {
			stats.Min = ret
		}
		if ret > stats.Max {
			stats.Max = ret
		}
	}

	stats.Mean = sum / float64(stats.Count)

	// Calculate variance and standard deviation
	if stats.Count > 1 {
		stats.Variance = (sumSq - sum*sum/float64(stats.Count)) / float64(stats.Count-1)
		stats.StdDev = math.Sqrt(stats.Variance)
	}

	// Calculate quartiles and median
	sortedReturns := make([]float64, len(tsa.Returns))
	copy(sortedReturns, tsa.Returns)
	sort.Float64s(sortedReturns)

	stats.Median = median(sortedReturns)
	stats.Q1 = percentile(sortedReturns, 25)
	stats.Q3 = percentile(sortedReturns, 75)
	stats.IQR = stats.Q3 - stats.Q1

	// Calculate higher moments (skewness and kurtosis)
	stats.calculateHigherMoments(tsa.Returns)

	tsa.Stats = stats
}

// calculateHigherMoments computes skewness and kurtosis
func (bs *BasicStats) calculateHigherMoments(data []float64) {
	if len(data) < 3 || bs.StdDev == 0 {
		return
	}

	sum3 := 0.0
	sum4 := 0.0

	for _, x := range data {
		z := (x - bs.Mean) / bs.StdDev
		sum3 += z * z * z
		sum4 += z * z * z * z
	}

	n := float64(len(data))
	bs.Skewness = sum3 / n
	bs.Kurtosis = sum4/n - 3.0 // Excess kurtosis
}

// calculateRiskMetrics computes comprehensive risk measures
func (tsa *TimeSeriesAnalysis) calculateRiskMetrics() {
	if len(tsa.Returns) < 2 {
		return
	}

	rm := RiskMetrics{}

	// Calculate volatility measures
	rm.DailyVolatility = tsa.Stats.StdDev
	rm.MonthlyVolatility = rm.DailyVolatility * math.Sqrt(21) // Approx trading days in month
	rm.AnnualVolatility = rm.DailyVolatility * math.Sqrt(252) // Trading days in year

	// Calculate downside deviation and semi-variance
	rm.calculateDownsideRisk(tsa.Returns, tsa.Stats.Mean)

	// Calculate drawdowns
	rm.calculateDrawdowns(tsa.Prices)

	// Calculate Value at Risk
	rm.calculateValueAtRisk(tsa.Returns)

	// Calculate performance ratios
	rm.calculatePerformanceRatios(tsa.Returns)

	tsa.RiskMetrics = rm
}

// calculateDownsideRisk computes downside-focused risk measures
func (rm *RiskMetrics) calculateDownsideRisk(returns []float64, meanReturn float64) {
	downsideSum := 0.0
	semiVarSum := 0.0
	downsideCount := 0

	for _, ret := range returns {
		if ret < meanReturn {
			diff := ret - meanReturn
			downsideSum += diff * diff
			semiVarSum += math.Min(diff, 0) * math.Min(diff, 0)
			downsideCount++
		}
	}

	if downsideCount > 0 {
		rm.DownsideDeviation = math.Sqrt(downsideSum / float64(len(returns)))
		rm.SemiVariance = math.Sqrt(semiVarSum / float64(len(returns)))
	}
}

// calculateDrawdowns analyzes price drawdowns
func (rm *RiskMetrics) calculateDrawdowns(prices []float64) {
	if len(prices) == 0 {
		return
	}

	peak := prices[0]
	maxDrawdown := 0.0
	currentDrawdown := 0.0
	drawdownStart := 0
	maxDrawdownPeriod := 0
	totalDrawdown := 0.0
	drawdownCount := 0

	for i, price := range prices {
		if price > peak {
			peak = price
			if currentDrawdown > 0 {
				// Drawdown ended
				totalDrawdown += currentDrawdown
				drawdownCount++
				currentDrawdown = 0
			}
		} else {
			drawdown := (peak - price) / peak
			if drawdown > currentDrawdown {
				currentDrawdown = drawdown
				if drawdown > maxDrawdown {
					maxDrawdown = drawdown
					maxDrawdownPeriod = i - drawdownStart
				}
			}
		}

		if currentDrawdown == 0 {
			drawdownStart = i
		}
	}

	rm.MaxDrawdown = maxDrawdown
	rm.MaxDrawdownPeriod = maxDrawdownPeriod
	if drawdownCount > 0 {
		rm.AvgDrawdown = totalDrawdown / float64(drawdownCount)
	}
}

// calculateValueAtRisk computes Value at Risk metrics
func (rm *RiskMetrics) calculateValueAtRisk(returns []float64) {
	if len(returns) < 100 {
		return // Need sufficient data for reliable VaR
	}

	// Sort returns for percentile calculation
	sortedReturns := make([]float64, len(returns))
	copy(sortedReturns, returns)
	sort.Float64s(sortedReturns)

	// Historical VaR
	rm.VaR95 = -percentile(sortedReturns, 5)   // 5th percentile of losses
	rm.VaR99 = -percentile(sortedReturns, 1)   // 1st percentile of losses

	// Calculate Conditional VaR (expected shortfall)
	rm.calculateConditionalVaR(sortedReturns)
}

// calculateConditionalVaR computes expected shortfall
func (rm *RiskMetrics) calculateConditionalVaR(sortedReturns []float64) {
	// CVaR95: average of worst 5% of returns
	var sum95 float64
	count95 := 0
	threshold95 := int(float64(len(sortedReturns)) * 0.05)

	for i := 0; i < threshold95; i++ {
		sum95 += sortedReturns[i]
		count95++
	}

	if count95 > 0 {
		rm.CVaR95 = -sum95 / float64(count95)
	}

	// CVaR99: average of worst 1% of returns
	var sum99 float64
	count99 := 0
	threshold99 := int(float64(len(sortedReturns)) * 0.01)

	for i := 0; i < threshold99; i++ {
		sum99 += sortedReturns[i]
		count99++
	}

	if count99 > 0 {
		rm.CVaR99 = -sum99 / float64(count99)
	}
}

// calculatePerformanceRatios computes risk-adjusted performance measures
func (rm *RiskMetrics) calculatePerformanceRatios(returns []float64) {
	if len(returns) == 0 || rm.AnnualVolatility == 0 {
		return
	}

	// Calculate mean return
	meanReturn := 0.0
	for _, ret := range returns {
		meanReturn += ret
	}
	meanReturn /= float64(len(returns))

	// Annualized return (assuming daily returns)
	annualReturn := meanReturn * 252

	// Risk-free rate assumption (can be parameterized)
	riskFreeRate := 0.02 // 2% annual

	// Sharpe Ratio
	if rm.AnnualVolatility > 0 {
		rm.SharpeRatio = (annualReturn - riskFreeRate) / rm.AnnualVolatility
	}

	// Sortino Ratio (uses downside deviation)
	if rm.DownsideDeviation > 0 {
		rm.SortinoRatio = (annualReturn - riskFreeRate) / (rm.DownsideDeviation * math.Sqrt(252))
	}

	// Calmar Ratio (return / max drawdown)
	if rm.MaxDrawdown > 0 {
		rm.CalmarRatio = annualReturn / rm.MaxDrawdown
	}

	// Omega Ratio
	rm.OmegaRatio = calculateOmegaRatio(returns, riskFreeRate/252)
}

// generateFeatures creates derived features for analysis
func (tsa *TimeSeriesAnalysis) generateFeatures() {
	if len(tsa.Prices) < 20 {
		return
	}

	// Rolling statistics
	tsa.Features["rolling_mean_20"] = rollingMean(tsa.Prices, 20)
	tsa.Features["rolling_std_20"] = rollingStd(tsa.Prices, 20)
	tsa.Features["rolling_min_20"] = rollingMin(tsa.Prices, 20)
	tsa.Features["rolling_max_20"] = rollingMax(tsa.Prices, 20)

	// Momentum features
	tsa.Features["momentum_5"] = momentum(tsa.Prices, 5)
	tsa.Features["momentum_10"] = momentum(tsa.Prices, 10)
	tsa.Features["momentum_20"] = momentum(tsa.Prices, 20)

	// Volatility features
	tsa.Features["volatility_ratio"] = volatilityRatio(tsa.Prices, 5, 20)
	tsa.Features["bollinger_position"] = bollingerPosition(tsa.Prices, 20, 2.0)

	// Return-based features
	tsa.Features["positive_return_ratio"] = positiveReturnRatio(tsa.Returns, 20)
	tsa.Features["return_autocorrelation"] = returnAutocorrelation(tsa.Returns, 1)
}

// ============================================================================
// Portfolio Analysis
// ============================================================================

// AnalyzePortfolio performs comprehensive portfolio analysis
func AnalyzePortfolio(prices [][]float64, weights []float64, assetNames []string) *PortfolioAnalysis {
	if len(prices) == 0 || len(prices[0]) < 2 {
		return nil
	}

	// Validate inputs
	if weights == nil {
		weights = equalWeights(len(prices))
	}

	if assetNames == nil {
		assetNames = generateAssetNames(len(prices))
	}

	analysis := &PortfolioAnalysis{
		AssetNames:    assetNames,
		AssetPrices:   prices,
		Weights:       weights,
	}

	// Calculate asset returns
	analysis.calculateAssetReturns()

	// Calculate portfolio returns
	analysis.calculatePortfolioReturns()

	// Calculate correlation and covariance
	analysis.calculateCorrelationCovariance()

	// Calculate portfolio metrics
	analysis.calculatePortfolioMetrics()

	return analysis
}

// calculateAssetReturns computes returns for each asset
func (pa *PortfolioAnalysis) calculateAssetReturns() {
	pa.AssetReturns = make([][]float64, len(pa.AssetPrices))

	for i, prices := range pa.AssetPrices {
		pa.AssetReturns[i] = CalculateReturns(prices)
	}
}

// calculatePortfolioReturns computes weighted portfolio returns
func (pa *PortfolioAnalysis) calculatePortfolioReturns() {
	if len(pa.AssetReturns) == 0 {
		return
	}

	nReturns := len(pa.AssetReturns[0])
	pa.PortfolioReturns = make([]float64, nReturns)

	for t := 0; t < nReturns; t++ {
		weightedReturn := 0.0
		for i := range pa.AssetReturns {
			weightedReturn += pa.Weights[i] * pa.AssetReturns[i][t]
		}
		pa.PortfolioReturns[t] = weightedReturn
	}
}

// calculateCorrelationCovariance computes correlation and covariance matrices
func (pa *PortfolioAnalysis) calculateCorrelationCovariance() {
	nAssets := len(pa.AssetReturns)

	pa.CorrelationMatrix = make([][]float64, nAssets)
	pa.CovarianceMatrix = make([][]float64, nAssets)

	for i := 0; i < nAssets; i++ {
		pa.CorrelationMatrix[i] = make([]float64, nAssets)
		pa.CovarianceMatrix[i] = make([]float64, nAssets)

		for j := 0; j < nAssets; j++ {
			if i == j {
				pa.CorrelationMatrix[i][j] = 1.0
				// Calculate variance
				pa.CovarianceMatrix[i][j] = variance(pa.AssetReturns[i])
			} else if j > i {
				// Calculate correlation and covariance
				corr := correlation(pa.AssetReturns[i], pa.AssetReturns[j])
				covar := covariance(pa.AssetReturns[i], pa.AssetReturns[j])

				pa.CorrelationMatrix[i][j] = corr
				pa.CorrelationMatrix[j][i] = corr

				pa.CovarianceMatrix[i][j] = covar
				pa.CovarianceMatrix[j][i] = covar
			}
		}
	}
}

// calculatePortfolioMetrics computes portfolio-level risk and return metrics
func (pa *PortfolioAnalysis) calculatePortfolioMetrics() {
	if len(pa.PortfolioReturns) == 0 {
		return
	}

	// Calculate portfolio return
	pa.PortfolioReturn = mean(pa.PortfolioReturns) * 252 // Annualized

	// Calculate portfolio risk (standard deviation)
	pa.PortfolioRisk = stdDev(pa.PortfolioReturns) * math.Sqrt(252)

	// Calculate Sharpe ratio (assuming 2% risk-free rate)
	riskFreeRate := 0.02
	if pa.PortfolioRisk > 0 {
		pa.SharpeRatio = (pa.PortfolioReturn - riskFreeRate) / pa.PortfolioRisk
	}

	// Calculate Sortino ratio
	downsideDev := downsideDeviation(pa.PortfolioReturns, 0.0)
	if downsideDev > 0 {
		pa.SortinoRatio = (pa.PortfolioReturn - riskFreeRate) / (downsideDev * math.Sqrt(252))
	}
}

// ============================================================================
// Market Regime Detection
// ============================================================================

// DetectMarketRegimes identifies different market conditions
func DetectMarketRegimes(prices []float64, window int) []MarketRegime {
	if len(prices) < window*2 {
		return nil
	}

	regimes := make([]MarketRegime, len(prices)-window+1)
	returns := CalculateReturns(prices)

	for i := 0; i <= len(prices)-window; i++ {
		windowPrices := prices[i : i+window]
		windowReturns := returns[i:min(i+window-1, len(returns))]

		regimes[i] = classifyRegime(windowPrices, windowReturns)
	}

	return regimes
}

// classifyRegime determines the market regime for a window
func classifyRegime(prices []float64, returns []float64) MarketRegime {
	if len(prices) < 2 || len(returns) < 1 {
		return RegimeSideways
	}

	// Calculate metrics
	meanReturn := mean(returns)
	volatility := stdDev(returns)

	// Calculate trend
	priceChange := (prices[len(prices)-1] - prices[0]) / prices[0]
	trendStrength := math.Abs(priceChange)

	// Classify based on rules
	if volatility > 0.03 { // High volatility threshold
		if meanReturn < -0.01 {
			return RegimeCrisis
		}
		return RegimeHighVol
	}

	if volatility < 0.01 { // Low volatility threshold
		return RegimeLowVol
	}

	if trendStrength > 0.1 { // Strong trend
		if priceChange > 0 {
			return RegimeBull
		} else {
			return RegimeBear
		}
	}

	if trendStrength > 0.05 { // Moderate trend
		if priceChange > 0 {
			return RegimeBull
		} else {
			return RegimeBear
		}
	}

	return RegimeSideways
}

// ============================================================================
// Statistical Tests for Financial Data
// ============================================================================

// TestNormality performs normality tests on returns
func TestNormality(returns []float64) (jarqueBera float64, isNormal bool) {
	if len(returns) < 20 {
		return 0, false
	}

	// Calculate skewness and kurtosis
	meanRet := mean(returns)
	stdRet := stdDev(returns)

	skewness := 0.0
	kurtosis := 0.0

	for _, ret := range returns {
		z := (ret - meanRet) / stdRet
		skewness += z * z * z
		kurtosis += z * z * z * z
	}

	n := float64(len(returns))
	skewness /= n
	kurtosis = kurtosis/n - 3.0 // Excess kurtosis

	// Jarque-Bera test statistic
	jb := n/6.0 * (skewness*skewness + kurtosis*kurtosis/4.0)

	// Critical value for 95% confidence (chi-square with 2 dof)
	criticalValue := 5.991

	return jb, jb < criticalValue
}

// TestStationarity performs basic stationarity checks
func TestStationarity(prices []float64, returns []float64) (priceStationary bool, returnStationary bool) {
	if len(prices) < 50 || len(returns) < 49 {
		return false, false
	}

	// Simple heuristic: compare variance of first and second halves
	mid := len(prices) / 2
	firstHalf := prices[:mid]
	secondHalf := prices[mid:]

	// Price stationarity (likely non-stationary)
	priceVarRatio := variance(firstHalf) / variance(secondHalf)
	priceStationary = priceVarRatio > 0.5 && priceVarRatio < 2.0

	// Return stationarity (likely stationary)
	retMid := len(returns) / 2
	retFirstHalf := returns[:retMid]
	retSecondHalf := returns[retMid:]

	retVarRatio := variance(retFirstHalf) / variance(retSecondHalf)
	returnStationary = retVarRatio > 0.5 && retVarRatio < 2.0

	return priceStationary, returnStationary
}

// TestAutocorrelation checks for serial correlation in returns
func TestAutocorrelation(returns []float64, lags []int) map[int]float64 {
	results := make(map[int]float64)

	for _, lag := range lags {
		if lag >= len(returns) {
			continue
		}

		corr := 0.0
		count := 0

		for i := lag; i < len(returns); i++ {
			corr += returns[i] * returns[i-lag]
			count++
		}

		if count > 0 {
			results[lag] = corr / float64(count)
		}
	}

	return results
}

// ============================================================================
// Feature Engineering Functions
// ============================================================================

func rollingMean(data []float64, window int) []float64 {
	if len(data) < window {
		return nil
	}

	result := make([]float64, len(data)-window+1)
	for i := 0; i <= len(data)-window; i++ {
		sum := 0.0
		for j := 0; j < window; j++ {
			sum += data[i+j]
		}
		result[i] = sum / float64(window)
	}
	return result
}

func rollingStd(data []float64, window int) []float64 {
	if len(data) < window {
		return nil
	}

	result := make([]float64, len(data)-window+1)
	for i := 0; i <= len(data)-window; i++ {
		// Calculate mean
		sum := 0.0
		for j := 0; j < window; j++ {
			sum += data[i+j]
		}
		mean := sum / float64(window)

		// Calculate variance
		varSum := 0.0
		for j := 0; j < window; j++ {
			diff := data[i+j] - mean
			varSum += diff * diff
		}

		result[i] = math.Sqrt(varSum / float64(window))
	}
	return result
}

func rollingMin(data []float64, window int) []float64 {
	if len(data) < window {
		return nil
	}

	result := make([]float64, len(data)-window+1)
	for i := 0; i <= len(data)-window; i++ {
		minVal := data[i]
		for j := 1; j < window; j++ {
			if data[i+j] < minVal {
				minVal = data[i+j]
			}
		}
		result[i] = minVal
	}
	return result
}

func rollingMax(data []float64, window int) []float64 {
	if len(data) < window {
		return nil
	}

	result := make([]float64, len(data)-window+1)
	for i := 0; i <= len(data)-window; i++ {
		maxVal := data[i]
		for j := 1; j < window; j++ {
			if data[i+j] > maxVal {
				maxVal = data[i+j]
			}
		}
		result[i] = maxVal
	}
	return result
}

func momentum(data []float64, period int) []float64 {
	if len(data) < period {
		return nil
	}

	result := make([]float64, len(data)-period)
	for i := 0; i < len(data)-period; i++ {
		result[i] = (data[i+period] - data[i]) / data[i]
	}
	return result
}

func volatilityRatio(data []float64, shortWindow, longWindow int) []float64 {
	if len(data) < longWindow {
		return nil
	}

	shortVol := rollingStd(data, shortWindow)
	longVol := rollingStd(data, longWindow)

	// Align lengths
	startIdx := len(longVol) - len(shortVol)
	alignedLongVol := longVol[startIdx:]

	result := make([]float64, len(shortVol))
	for i := range result {
		if alignedLongVol[i] > 0 {
			result[i] = shortVol[i] / alignedLongVol[i]
		} else {
			result[i] = 1.0
		}
	}
	return result
}

func bollingerPosition(data []float64, window int, stdDevs float64) []float64 {
	if len(data) < window {
		return nil
	}

	upper, middle, lower, _ := BollingerBands(data, window, stdDevs)

	result := make([]float64, len(middle))
	for i := range result {
		priceIdx := i + window - 1
		if priceIdx < len(data) {
			if upper[i] != lower[i] {
				// Normalize position between -1 (lower band) and 1 (upper band)
				result[i] = 2*(data[priceIdx]-lower[i])/(upper[i]-lower[i]) - 1
			} else {
				result[i] = 0
			}
		}
	}
	return result
}

func positiveReturnRatio(returns []float64, window int) []float64 {
	if len(returns) < window {
		return nil
	}

	result := make([]float64, len(returns)-window+1)
	for i := 0; i <= len(returns)-window; i++ {
		positiveCount := 0
		for j := 0; j < window; j++ {
			if returns[i+j] > 0 {
				positiveCount++
			}
		}
		result[i] = float64(positiveCount) / float64(window)
	}
	return result
}

func returnAutocorrelation(returns []float64, lag int) []float64 {
	if len(returns) < lag*2 {
		return nil
	}

	window := 20
	result := make([]float64, len(returns)-window-lag+1)

	for i := 0; i <= len(returns)-window-lag; i++ {
		// Calculate correlation between returns[i:i+window] and returns[i+lag:i+window+lag]
		x := returns[i : i+window]
		y := returns[i+lag : i+window+lag]
		result[i] = correlation(x, y)
	}

	return result
}

// ============================================================================
// Utility Functions
// ============================================================================

func median(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}

	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)

	n := len(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2
	}
	return sorted[n/2]
}

func percentile(data []float64, p float64) float64 {
	if len(data) == 0 {
		return 0
	}

	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)

	index := float64(len(sorted)-1) * p / 100.0
	lower := int(math.Floor(index))
	upper := int(math.Ceil(index))

	if lower == upper {
		return sorted[lower]
	}

	weight := index - float64(lower)
	return sorted[lower]*(1-weight) + sorted[upper]*weight
}

func mean(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}

	sum := 0.0
	for _, x := range data {
		sum += x
	}
	return sum / float64(len(data))
}

func variance(data []float64) float64 {
	if len(data) < 2 {
		return 0
	}

	m := mean(data)
	sum := 0.0
	for _, x := range data {
		diff := x - m
		sum += diff * diff
	}
	return sum / float64(len(data)-1)
}

func stdDev(data []float64) float64 {
	return math.Sqrt(variance(data))
}

func covariance(x, y []float64) float64 {
	if len(x) != len(y) || len(x) < 2 {
		return 0
	}

	meanX := mean(x)
	meanY := mean(y)

	sum := 0.0
	for i := range x {
		sum += (x[i] - meanX) * (y[i] - meanY)
	}

	return sum / float64(len(x)-1)
}

func correlation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) < 2 {
		return 0
	}

	covar := covariance(x, y)
	stdX := stdDev(x)
	stdY := stdDev(y)

	if stdX == 0 || stdY == 0 {
		return 0
	}

	return covar / (stdX * stdY)
}

func downsideDeviation(returns []float64, mar float64) float64 { // MAR = Minimum Acceptable Return
	if len(returns) == 0 {
		return 0
	}

	sum := 0.0
	count := 0
	for _, ret := range returns {
		if ret < mar {
			diff := ret - mar
			sum += diff * diff
			count++
		}
	}

	if count == 0 {
		return 0
	}
	return math.Sqrt(sum / float64(count))
}

func calculateOmegaRatio(returns []float64, threshold float64) float64 {
	if len(returns) == 0 {
		return 0
	}

	gains := 0.0
	losses := 0.0

	for _, ret := range returns {
		excess := ret - threshold
		if excess > 0 {
			gains += excess
		} else {
			losses += math.Abs(excess)
		}
	}

	if losses == 0 {
		return math.Inf(1)
	}

	return gains / losses
}

func equalWeights(n int) []float64 {
	weights := make([]float64, n)
	equalWeight := 1.0 / float64(n)
	for i := range weights {
		weights[i] = equalWeight
	}
	return weights
}

func generateAssetNames(n int) []string {
	names := make([]string, n)
	for i := range names {
		names[i] = fmt.Sprintf("Asset_%d", i+1)
	}
	return names
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
