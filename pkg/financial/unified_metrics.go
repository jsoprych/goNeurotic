package financial

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// ============================================================================
// Core Unified Metrics System for Financial Time-Series Analysis
// ============================================================================
// This file provides a unified, modular API for financial metrics and indicators.
// All functions are designed for reusability across different time-series studies.
// Documentation includes mathematical formulas, typical usage, and interpretation.

// ============================================================================
// Core Data Structures
// ============================================================================

// TimeSeriesData represents a generic financial time series
type TimeSeriesData struct {
	Timestamps []time.Time // Time points (must be sorted)
	Values     []float64   // Price/Value data
	Returns    []float64   // Computed returns (optional)
}

// OHLCVSeries represents Open-High-Low-Close-Volume data
type OHLCVSeries struct {
	Timestamps []time.Time
	Open       []float64
	High       []float64
	Low        []float64
	Close      []float64
	Volume     []float64
}

// MetricConfig provides configuration for metric calculations
type MetricConfig struct {
	WindowSize   int     // Lookback window for calculations
	Period       int     // Indicator period
	StdDev       float64 // Standard deviations for bands
	SmoothPeriod int     // Smoothing period
	RiskFreeRate float64 // Risk-free rate for performance ratios
	AnnualizationFactor float64 // Days/year for annualization (default: 252)
}

// DefaultMetricConfig returns a sensible default configuration
func DefaultMetricConfig() MetricConfig {
	return MetricConfig{
		WindowSize:   20,
		Period:       14,
		StdDev:       2.0,
		SmoothPeriod: 3,
		RiskFreeRate: 0.02,
		AnnualizationFactor: 252.0,
	}
}

// ============================================================================
// Technical Indicators (Price-Based Analysis)
// ============================================================================

// Moving Averages ------------------------------------------------------------

// SMA calculates Simple Moving Average: MA = (Σ P_i) / n
// Where P_i are prices, n is period
// Typical usage: Trend identification, support/resistance
func SMA(prices []float64, period int) ([]float64, error) {
	if len(prices) < period || period <= 0 {
		return nil, fmt.Errorf("SMA: insufficient data or invalid period (got %d, need ≥%d)", len(prices), period)
	}

	sma := make([]float64, len(prices)-period+1)
	for i := 0; i <= len(prices)-period; i++ {
		sum := 0.0
		for j := 0; j < period; j++ {
			sum += prices[i+j]
		}
		sma[i] = sum / float64(period)
	}
	return sma, nil
}

// EMA calculates Exponential Moving Average: EMA_t = α * P_t + (1-α) * EMA_{t-1}
// Where α = 2/(n+1), n is period
// Typical usage: Faster trend detection than SMA
func EMA(prices []float64, period int) ([]float64, error) {
	if len(prices) < period || period <= 0 {
		return nil, fmt.Errorf("EMA: insufficient data or invalid period")
	}

	ema := make([]float64, len(prices))
	alpha := 2.0 / (float64(period) + 1.0)

	// First EMA is SMA of first period
	sum := 0.0
	for i := 0; i < period; i++ {
		sum += prices[i]
	}
	ema[period-1] = sum / float64(period)

	// Calculate EMA for remaining points
	for i := period; i < len(prices); i++ {
		ema[i] = (prices[i]-ema[i-1])*alpha + ema[i-1]
	}

	return ema[period-1:], nil
}

// Momentum Indicators --------------------------------------------------------

// RSI calculates Relative Strength Index: RSI = 100 - 100/(1 + RS)
// Where RS = AvgGain / AvgLoss over period
// Interpretation: 0-30 oversold, 30-70 neutral, 70-100 overbought
func RSI(prices []float64, period int) ([]float64, error) {
	if len(prices) < period+1 || period <= 0 {
		return nil, fmt.Errorf("RSI: need at least %d data points, got %d", period+1, len(prices))
	}

	gains := make([]float64, len(prices))
	losses := make([]float64, len(prices))

	// Calculate gains and losses
	for i := 1; i < len(prices); i++ {
		change := prices[i] - prices[i-1]
		if change > 0 {
			gains[i] = change
			losses[i] = 0
		} else {
			gains[i] = 0
			losses[i] = -change
		}
	}

	rsi := make([]float64, len(prices)-period)

	for i := period; i < len(prices); i++ {
		avgGain := 0.0
		avgLoss := 0.0

		for j := i - period; j < i; j++ {
			avgGain += gains[j]
			avgLoss += losses[j]
		}

		avgGain /= float64(period)
		avgLoss /= float64(period)

		if avgLoss == 0 {
			rsi[i-period] = 100
		} else {
			rs := avgGain / avgLoss
			rsi[i-period] = 100 - (100 / (1 + rs))
		}
	}

	return rsi, nil
}

// MACD calculates Moving Average Convergence Divergence
// MACD Line = EMA(12) - EMA(26), Signal = EMA(MACD, 9), Histogram = MACD - Signal
// Interpretation: Bullish when MACD crosses above signal, bearish when below
func MACD(prices []float64, fast, slow, signal int) ([]float64, []float64, []float64, error) {
	if len(prices) < slow+signal || fast >= slow {
		return nil, nil, nil, fmt.Errorf("MACD: invalid parameters or insufficient data")
	}

	fastEMA, err := EMA(prices, fast)
	if err != nil {
		return nil, nil, nil, err
	}

	slowEMA, err := EMA(prices, slow)
	if err != nil {
		return nil, nil, nil, err
	}

	// Align lengths
	startIdx := len(prices) - len(fastEMA)
	if len(fastEMA) != len(slowEMA) {
		if len(fastEMA) > len(slowEMA) {
			startIdx = len(prices) - len(slowEMA)
		}
	}

	// Calculate MACD line
	macd := make([]float64, len(fastEMA)-startIdx)
	for i := startIdx; i < len(fastEMA); i++ {
		macd[i-startIdx] = fastEMA[i] - slowEMA[i]
	}

	// Calculate signal line (EMA of MACD)
	signalLine, err := EMA(macd, signal)
	if err != nil {
		return nil, nil, nil, err
	}

	// Calculate histogram
	histogram := make([]float64, len(signalLine))
	for i := 0; i < len(signalLine); i++ {
		histogram[i] = macd[i+signal-1] - signalLine[i]
	}

	return macd[signal-1:], signalLine, histogram, nil
}

// Volatility Indicators ------------------------------------------------------

// BollingerBands calculates Bollinger Bands: Upper = SMA + kσ, Lower = SMA - kσ
// Where σ is standard deviation, k is number of standard deviations
// Interpretation: Price near upper band = overbought, near lower band = oversold
func BollingerBands(prices []float64, period int, stdDev float64) ([]float64, []float64, []float64, error) {
	if len(prices) < period || period <= 0 || stdDev <= 0 {
		return nil, nil, nil, fmt.Errorf("BollingerBands: invalid parameters")
	}

	middle, err := SMA(prices, period)
	if err != nil {
		return nil, nil, nil, err
	}

	upper := make([]float64, len(middle))
	lower := make([]float64, len(middle))

	for i := 0; i < len(middle); i++ {
		// Calculate standard deviation of this window
		sum := 0.0
		for j := 0; j < period; j++ {
			diff := prices[i+j] - middle[i]
			sum += diff * diff
		}
		std := math.Sqrt(sum / float64(period))

		upper[i] = middle[i] + stdDev*std
		lower[i] = middle[i] - stdDev*std
	}

	return upper, middle, lower, nil
}

// ATR calculates Average True Range: TR = max(H-L, |H-C_prev|, |L-C_prev|)
// ATR = smoothed TR over period (Wilder's smoothing)
// Interpretation: Higher ATR = higher volatility, useful for position sizing
func ATR(high, low, close []float64, period int) ([]float64, error) {
	if len(high) != len(low) || len(high) != len(close) {
		return nil, fmt.Errorf("ATR: input arrays must have same length")
	}
	if len(high) < period || period <= 0 {
		return nil, fmt.Errorf("ATR: insufficient data")
	}

	// Calculate True Range
	tr := make([]float64, len(high))
	for i := 0; i < len(high); i++ {
		if i == 0 {
			tr[i] = high[i] - low[i]
		} else {
			method1 := high[i] - low[i]
			method2 := math.Abs(high[i] - close[i-1])
			method3 := math.Abs(low[i] - close[i-1])
			tr[i] = math.Max(method1, math.Max(method2, method3))
		}
	}

	// First ATR is average of first period TR values
	sum := 0.0
	for i := 0; i < period; i++ {
		sum += tr[i]
	}
	atr := make([]float64, len(tr)-period+1)
	atr[0] = sum / float64(period)

	// Subsequent ATR values using Wilder's smoothing
	for i := 1; i < len(atr); i++ {
		atr[i] = (atr[i-1]*float64(period-1) + tr[i+period-1]) / float64(period)
	}

	return atr, nil
}

// ============================================================================
// Risk Metrics (Performance & Risk Analysis)
// ============================================================================

// ReturnsMetrics calculates comprehensive return statistics
type ReturnsMetrics struct {
	Returns          []float64 // Raw returns
	LogReturns       []float64 // Logarithmic returns
	MeanReturn       float64   // Arithmetic mean
	GeometricMean    float64   // Geometric mean
	AnnualizedReturn float64   // Annualized return
	StdDev           float64   // Standard deviation
	AnnualizedVol    float64   // Annualized volatility
	Skewness         float64   // Third moment (asymmetry)
	Kurtosis         float64   // Fourth moment (tail heaviness)
	MinReturn        float64   // Minimum return
	MaxReturn        float64   // Maximum return
}

// CalculateReturnsMetrics computes comprehensive return statistics
func CalculateReturnsMetrics(prices []float64, config MetricConfig) (ReturnsMetrics, error) {
	if len(prices) < 2 {
		return ReturnsMetrics{}, fmt.Errorf("need at least 2 price points")
	}

	var metrics ReturnsMetrics

	// Calculate simple returns
	metrics.Returns = make([]float64, len(prices)-1)
	for i := 1; i < len(prices); i++ {
		if prices[i-1] != 0 {
			metrics.Returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1]
		}
	}

	// Calculate log returns
	metrics.LogReturns = make([]float64, len(prices)-1)
	for i := 1; i < len(prices); i++ {
		if prices[i] > 0 && prices[i-1] > 0 {
			metrics.LogReturns[i-1] = math.Log(prices[i] / prices[i-1])
		}
	}

	// Calculate statistics
	metrics.MeanReturn = mean(metrics.Returns)
	metrics.GeometricMean = geometricMean(metrics.Returns)
	metrics.AnnualizedReturn = metrics.GeometricMean * config.AnnualizationFactor
	metrics.StdDev = stdDev(metrics.Returns)
	metrics.AnnualizedVol = metrics.StdDev * math.Sqrt(config.AnnualizationFactor)
	metrics.Skewness = skewness(metrics.Returns)
	metrics.Kurtosis = kurtosis(metrics.Returns)
	metrics.MinReturn = minSlice(metrics.Returns)
	metrics.MaxReturn = maxSlice(metrics.Returns)

	return metrics, nil
}

// DrawdownMetrics analyzes price drawdowns
type DrawdownMetrics struct {
	Drawdowns      []float64 // Individual drawdowns
	MaxDrawdown    float64   // Maximum drawdown
	AvgDrawdown    float64   // Average drawdown
	DrawdownPeriod int       // Duration of max drawdown (days)
	RecoveryPeriod int       // Recovery period (days)
	CalmarRatio    float64   // Return / MaxDrawdown
}

// CalculateDrawdownMetrics computes drawdown analysis
func CalculateDrawdownMetrics(prices []float64) (DrawdownMetrics, error) {
	if len(prices) < 2 {
		return DrawdownMetrics{}, fmt.Errorf("need at least 2 price points")
	}

	var metrics DrawdownMetrics
	metrics.Drawdowns = make([]float64, len(prices))

	peak := prices[0]
	maxDD := 0.0
	maxDDStart := 0
	maxDDEnd := 0
	currentDDStart := 0
	totalDrawdown := 0.0
	drawdownCount := 0

	for i := 0; i < len(prices); i++ {
		if prices[i] > peak {
			peak = prices[i]
			if i > 0 && metrics.Drawdowns[i-1] > 0 {
				// Drawdown ended
				totalDrawdown += metrics.Drawdowns[i-1]
				drawdownCount++
			}
			currentDDStart = i
		} else {
			drawdown := (peak - prices[i]) / peak
			metrics.Drawdowns[i] = drawdown

			if drawdown > maxDD {
				maxDD = drawdown
				maxDDStart = currentDDStart
				maxDDEnd = i
			}
		}
	}

	metrics.MaxDrawdown = maxDD
	metrics.DrawdownPeriod = maxDDEnd - maxDDStart

	// Calculate recovery period (simplified)
	if maxDDEnd < len(prices)-1 {
		for i := maxDDEnd; i < len(prices); i++ {
			if prices[i] >= prices[maxDDStart] {
				metrics.RecoveryPeriod = i - maxDDEnd
				break
			}
		}
	}

	if drawdownCount > 0 {
		metrics.AvgDrawdown = totalDrawdown / float64(drawdownCount)
	}

	// Calculate Calmar ratio
	totalReturn := (prices[len(prices)-1] - prices[0]) / prices[0]
	annualizedReturn := math.Pow(1+totalReturn, 365.0/float64(len(prices))) - 1
	if metrics.MaxDrawdown > 0 {
		metrics.CalmarRatio = annualizedReturn / metrics.MaxDrawdown
	}

	return metrics, nil
}

// PerformanceRatios calculates risk-adjusted performance metrics
type PerformanceRatios struct {
	SharpeRatio  float64 // Excess return per unit of total risk
	SortinoRatio float64 // Excess return per unit of downside risk
	OmegaRatio   float64 // Ratio of gains to losses relative to threshold
	TreynorRatio float64 // Excess return per unit of systematic risk (requires beta)
	InformationRatio float64 // Active return per unit of tracking error
}

// CalculatePerformanceRatios computes risk-adjusted performance metrics
func CalculatePerformanceRatios(returns []float64, config MetricConfig) PerformanceRatios {
	var ratios PerformanceRatios

	if len(returns) == 0 {
		return ratios
	}

	meanRet := mean(returns)
	stdRet := stdDev(returns)

	// Sharpe Ratio (annualized)
	if stdRet > 0 {
		excessReturn := meanRet*config.AnnualizationFactor - config.RiskFreeRate
		ratios.SharpeRatio = excessReturn / (stdRet * math.Sqrt(config.AnnualizationFactor))
	}

	// Sortino Ratio (uses downside deviation)
	downsideDev := downsideDeviation(returns, 0)
	if downsideDev > 0 {
		excessReturn := meanRet*config.AnnualizationFactor - config.RiskFreeRate
		ratios.SortinoRatio = excessReturn / (downsideDev * math.Sqrt(config.AnnualizationFactor))
	}

	// Omega Ratio
	ratios.OmegaRatio = calculateOmegaRatio(returns, config.RiskFreeRate/config.AnnualizationFactor)

	return ratios
}

// ============================================================================
// Statistical Tests for Financial Data
// ============================================================================

// NormalityTestResult contains results of normality tests
type NormalityTestResult struct {
	JarqueBera   float64 // JB test statistic
	JBpValue     float64 // p-value for JB test
	IsNormal     bool    // Whether data appears normal (α=0.05)
	Skewness     float64
	Kurtosis     float64
}

// TestNormality performs Jarque-Bera test for normality
// H₀: Data is normally distributed
// Interpretation: p-value < 0.05 suggests non-normality
func TestNormality(returns []float64) NormalityTestResult {
	var result NormalityTestResult

	if len(returns) < 20 {
		return result
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

	result.Skewness = skewness
	result.Kurtosis = kurtosis

	// Jarque-Bera test statistic
	jb := n/6.0 * (skewness*skewness + kurtosis*kurtosis/4.0)
	result.JarqueBera = jb

	// Critical value for 95% confidence (chi-square with 2 dof)
	criticalValue := 5.991
	result.IsNormal = jb < criticalValue

	return result
}

// StationarityTestResult contains stationarity test results
type StationarityTestResult struct {
	PriceStationary  bool    // Whether price series appears stationary
	ReturnStationary bool    // Whether return series appears stationary
	VarianceRatio    float64 // Ratio of variances (first vs second half)
	MeanDifference   float64 // Difference in means (first vs second half)
}

// TestStationarity performs basic stationarity checks
func TestStationarity(prices []float64, returns []float64) StationarityTestResult {
	var result StationarityTestResult

	if len(prices) < 50 || len(returns) < 49 {
		return result
	}

	// Split data into halves
	mid := len(prices) / 2
	firstHalfPrices := prices[:mid]
	secondHalfPrices := prices[mid:]

	retMid := len(returns) / 2
	firstHalfReturns := returns[:retMid]
	secondHalfReturns := returns[retMid:]

	// Compare variances
	var1 := variance(firstHalfPrices)
	var2 := variance(secondHalfPrices)
	if var2 > 0 {
		result.VarianceRatio = var1 / var2
		result.PriceStationary = result.VarianceRatio > 0.5 && result.VarianceRatio < 2.0
	}

	// Compare means
	mean1 := mean(firstHalfReturns)
	mean2 := mean(secondHalfReturns)
	result.MeanDifference = math.Abs(mean1 - mean2)

	// Returns are typically stationary
	retVar1 := variance(firstHalfReturns)
	retVar2 := variance(secondHalfReturns)
	if retVar2 > 0 {
		retVarRatio := retVar1 / retVar2
		result.ReturnStationary = retVarRatio > 0.5 && retVarRatio < 2.0
	}

	return result
}

// AutocorrelationTest measures serial correlation in returns
// Significant autocorrelation suggests predictable patterns
func AutocorrelationTest(returns []float64, maxLag int) map[int]float64 {
	results := make(map[int]float64)

	if len(returns) < maxLag*2 {
		return results
	}

	meanRet := mean(returns)
	variance := 0.0
	for _, ret := range returns {
		diff := ret - meanRet
		variance += diff * diff
	}
	variance /= float64(len(returns))

	for lag := 1; lag <= maxLag; lag++ {
		if lag >= len(returns) {
			break
		}

		autocov := 0.0
		for i := lag; i < len(returns); i++ {
			autocov += (returns[i] - meanRet) * (returns[i-lag] - meanRet)
		}
		autocov /= float64(len(returns) - lag)

		if variance > 0 {
			results[lag] = autocov / variance
		}
	}

	return results
}

// ============================================================================
// Market Regime Detection
// ============================================================================

// MarketRegime represents different market conditions
type MarketRegime string

const (
	RegimeUnknown      MarketRegime = "unknown"
	RegimeStrongBull   MarketRegime = "strong_bull"      // Clear uptrend, high momentum
	RegimeWeakBull     MarketRegime = "weak_bull"        // Moderate uptrend
	RegimeStrongBear   MarketRegime = "strong_bear"      // Clear downtrend
	RegimeWeakBear     MarketRegime = "weak_bear"        // Moderate downtrend
	RegimeSideways     MarketRegime = "sideways"         // Range-bound, low trend
	RegimeHighVol      MarketRegime = "high_volatility"  // Elevated volatility
	RegimeLowVol       MarketRegime = "low_volatility"   // Suppressed volatility
	RegimeMeanReverting MarketRegime = "mean_reverting"  // Oscillating around mean
	RegimeTrending     MarketRegime = "trending"         // Strong directional movement
)

// RegimeMetrics contains regime classification metrics
type RegimeMetrics struct {
	CurrentRegime MarketRegime
	RegimeStrength float64     // 0-1 confidence in classification
	TrendStrength  float64     // 0-1 strength of trend
	VolatilityLevel float64    // Normalized volatility (0-1)
	Momentum       float64     // Recent price momentum
	MeanReversionScore float64 // Evidence of mean reversion
}

// DetectMarketRegime classifies current market conditions
func DetectMarketRegime(prices []float64, window int) (RegimeMetrics, error) {
	var metrics RegimeMetrics

	if len(prices) < window*2 {
		return metrics, fmt.Errorf("need at least %d prices for regime detection", window*2)
	}

	// Use last window for regime detection
	start := len(prices) - window
	if start < 0 {
		start = 0
	}
	windowPrices := prices[start:]

	// Calculate returns
	returns := make([]float64, len(windowPrices)-1)
	for i := 1; i < len(windowPrices); i++ {
		if windowPrices[i-1] > 0 {
			returns[i-1] = (windowPrices[i] - windowPrices[i-1]) / windowPrices[i-1]
		}
	}

	// Calculate metrics
	meanRet := mean(returns)
	stdRet := stdDev(returns)
	annualizedVol := stdRet * math.Sqrt(252)

	// Trend strength
	priceChange := (windowPrices[len(windowPrices)-1] - windowPrices[0]) / windowPrices[0]
	metrics.TrendStrength = math.Abs(priceChange) / (annualizedVol * math.Sqrt(float64(window)/252))
	metrics.TrendStrength = math.Min(metrics.TrendStrength, 1.0)

	// Volatility level (normalized)
	metrics.VolatilityLevel = math.Min(annualizedVol/0.3, 1.0) // Cap at 30% annual vol

	// Momentum
	shortWindow := min(10, len(returns))
	if shortWindow > 0 {
		metrics.Momentum = mean(returns[len(returns)-shortWindow:]) * math.Sqrt(252)
	}

	// Mean reversion score (negative autocorrelation at lag 1)
	if len(returns) > 2 {
		autocorr := AutocorrelationTest(returns, 1)
		if len(autocorr) > 0 {
			metrics.MeanReversionScore = math.Max(0, -autocorr[1])
		}
	}

	// Classify regime
	metrics.CurrentRegime = classifyRegimeFromMetrics(metrics)
	metrics.RegimeStrength = calculateRegimeConfidence(metrics)

	return metrics, nil
}

// classifyRegimeFromMetrics determines regime based on calculated metrics
func classifyRegimeFromMetrics(metrics RegimeMetrics) MarketRegime {
	// High volatility regime
	if metrics.VolatilityLevel > 0.7 {
		return RegimeHighVol
	}

	// Low volatility regime
	if metrics.VolatilityLevel < 0.3 {
		return RegimeLowVol
	}

	// Strong trends
	if metrics.TrendStrength > 0.7 {
		if metrics.Momentum > 0 {
			return RegimeStrongBull
		} else {
			return RegimeStrongBear
		}
	}

	// Weak trends
	if metrics.TrendStrength > 0.3 {
		if metrics.Momentum > 0 {
			return RegimeWeakBull
		} else {
			return RegimeWeakBear
		}
	}

	// Mean reverting
	if metrics.MeanReversionScore > 0.5 {
		return RegimeMeanReverting
	}

	// Default to sideways
	return RegimeSideways
}

// calculateRegimeConfidence computes confidence in regime classification
func calculateRegimeConfidence(metrics RegimeMetrics) float64 {
	// Confidence based on how clear the signals are
	confidence := 0.0

	// Trend-based confidence
	if metrics.TrendStrength > 0.5 {
		confidence += metrics.TrendStrength * 0.4
	}

	// Volatility-based confidence
	if metrics.VolatilityLevel > 0.7 || metrics.VolatilityLevel < 0.3 {
		confidence += 0.3
	}

	// Mean reversion confidence
	if metrics.MeanReversionScore > 0.5 {
		confidence += metrics.MeanReversionScore * 0.3
	}

	return math.Min(confidence, 1.0)
}

// ============================================================================
// Utility Functions (Internal)
// ============================================================================

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

func geometricMean(returns []float64) float64 {
	if len(returns) == 0 {
		return 0
	}
	sum := 0.0
	for _, r := range returns {
		sum += math.Log(1 + r)
	}
	return math.Exp(sum/float64(len(returns))) - 1
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

func variance(data []float64) float64 {
	if len(data) < 2 {
		return 0
	}
	m := mean(data)
	sum := 0.0
	for _, v := range data {
		diff := v - m
		sum += diff * diff
	}
	return sum / float64(len(data)-1)
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

func minSlice(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	minVal := data[0]
	for _, v := range data {
		if v < minVal {
			minVal = v
		}
	}
	return minVal
}

func maxSlice(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	maxVal := data[0]
	for _, v := range data {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

func downsideDeviation(returns []float64, mar float64) float64 {
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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ============================================================================
// Example Usage Documentation
// ============================================================================
/*
Example 1: Comprehensive Time-Series Analysis

func AnalyzeTimeSeries(prices []float64) {
    config := financial.DefaultMetricConfig()

    // 1. Calculate returns metrics
    returnsMetrics, _ := financial.CalculateReturnsMetrics(prices, config)
    fmt.Printf("Annualized Return: %.2f%%\n", returnsMetrics.AnnualizedReturn*100)
    fmt.Printf("Annualized Volatility: %.2f%%\n", returnsMetrics.AnnualizedVol*100)

    // 2. Calculate risk metrics
    drawdownMetrics, _ := financial.CalculateDrawdownMetrics(prices)
    fmt.Printf("Max Drawdown: %.2f%%\n", drawdownMetrics.MaxDrawdown*100)

    // 3. Calculate performance ratios
    ratios := financial.CalculatePerformanceRatios(returnsMetrics.Returns, config)
    fmt.Printf("Sharpe Ratio: %.2f\n", ratios.SharpeRatio)

    // 4. Technical indicators
    rsi, _ := financial.RSI(prices, 14)
    fmt.Printf("Latest RSI: %.1f\n", rsi[len(rsi)-1])

    // 5. Market regime detection
    regimeMetrics, _ := financial.DetectMarketRegime(prices, 60)
    fmt.Printf("Current Market Regime: %s (confidence: %.1f%%)\n",
               regimeMetrics.CurrentRegime, regimeMetrics.RegimeStrength*100)

    // 6. Statistical tests
    normality := financial.TestNormality(returnsMetrics.Returns)
    fmt.Printf("Data appears normal: %v (JB: %.2f)\n", normality.IsNormal, normality.JarqueBera)
}

Example 2: Portfolio Construction

func OptimizePortfolio(assets [][]float64) {
    config := financial.DefaultMetricConfig()

    // Calculate correlation matrix
    n := len(assets)
    corrMatrix := make([][]float64, n)
    for i := 0; i < n; i++ {
        corrMatrix[i] = make([]float64, n)
        for j := 0; j < n; j++ {
            if i == j {
                corrMatrix[i][j] = 1.0
            } else {
                // Calculate correlation between assets
                // Implementation depends on correlation function
            }
        }
    }

    // Use metrics for portfolio optimization
    var metrics []financial.ReturnsMetrics
    for _, prices := range assets {
        m, _ := financial.CalculateReturnsMetrics(prices, config)
        metrics = append(metrics, m)
    }

    // Portfolio optimization logic here...
}
*/
