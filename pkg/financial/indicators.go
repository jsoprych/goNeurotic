package financial

import (
	"fmt"
	"math"
)

// ============================================================================
// Data Structures
// ============================================================================

// OHLCV represents a single period's Open, High, Low, Close, Volume data
type OHLCV struct {
	Timestamp int64   // Unix timestamp
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
}

// IndicatorResult represents the output of a technical indicator
type IndicatorResult struct {
	Values  []float64   // Indicator values
	Signals []int       // Trading signals (-1, 0, 1)
	Metadata map[string]interface{} // Additional metadata (upper/lower bands, etc.)
}

// ============================================================================
// Moving Averages
// ============================================================================

// SMA calculates Simple Moving Average
func SMA(data []float64, period int) ([]float64, error) {
	if len(data) < period || period <= 0 {
		return nil, fmt.Errorf("insufficient data or invalid period")
	}

	sma := make([]float64, len(data)-period+1)
	for i := 0; i <= len(data)-period; i++ {
		sum := 0.0
		for j := 0; j < period; j++ {
			sum += data[i+j]
		}
		sma[i] = sum / float64(period)
	}
	return sma, nil
}

// EMA calculates Exponential Moving Average
func EMA(data []float64, period int) ([]float64, error) {
	if len(data) < period || period <= 0 {
		return nil, fmt.Errorf("insufficient data or invalid period")
	}

	ema := make([]float64, len(data))
	multiplier := 2.0 / (float64(period) + 1.0)

	// First EMA is SMA of first period
	sum := 0.0
	for i := 0; i < period; i++ {
		sum += data[i]
	}
	ema[period-1] = sum / float64(period)

	// Calculate EMA for remaining points
	for i := period; i < len(data); i++ {
		ema[i] = (data[i]-ema[i-1])*multiplier + ema[i-1]
	}

	return ema[period-1:], nil
}

// WMA calculates Weighted Moving Average
func WMA(data []float64, period int) ([]float64, error) {
	if len(data) < period || period <= 0 {
		return nil, fmt.Errorf("insufficient data or invalid period")
	}

	wma := make([]float64, len(data)-period+1)
	weightSum := float64(period * (period + 1) / 2)

	for i := 0; i <= len(data)-period; i++ {
		sum := 0.0
		for j := 0; j < period; j++ {
			sum += data[i+j] * float64(j+1)
		}
		wma[i] = sum / weightSum
	}
	return wma, nil
}

// ============================================================================
// Volatility Indicators
// ============================================================================

// BollingerBands calculates Bollinger Bands with SMA and standard deviation
func BollingerBands(data []float64, period int, stdDev float64) ([]float64, []float64, []float64, error) {
	if len(data) < period || period <= 0 || stdDev <= 0 {
		return nil, nil, nil, fmt.Errorf("invalid parameters")
	}

	middleBand, err := SMA(data, period)
	if err != nil {
		return nil, nil, nil, err
	}

	upperBand := make([]float64, len(middleBand))
	lowerBand := make([]float64, len(middleBand))

	for i := 0; i < len(middleBand); i++ {
		// Calculate standard deviation of this window
		sum := 0.0
		for j := 0; j < period; j++ {
			diff := data[i+j] - middleBand[i]
			sum += diff * diff
		}
		std := math.Sqrt(sum / float64(period))

		upperBand[i] = middleBand[i] + stdDev*std
		lowerBand[i] = middleBand[i] - stdDev*std
	}

	return upperBand, middleBand, lowerBand, nil
}

// ATR calculates Average True Range
func ATR(high, low, close []float64, period int) ([]float64, error) {
	if len(high) != len(low) || len(high) != len(close) {
		return nil, fmt.Errorf("input arrays must have same length")
	}
	if len(high) < period || period <= 0 {
		return nil, fmt.Errorf("insufficient data or invalid period")
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

	// First ATR is simple average of first period TR values
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
// Momentum Indicators
// ============================================================================

// RSI calculates Relative Strength Index
func RSI(data []float64, period int) ([]float64, error) {
	if len(data) < period+1 || period <= 0 {
		return nil, fmt.Errorf("insufficient data or invalid period")
	}

	gains := make([]float64, len(data))
	losses := make([]float64, len(data))

	// Calculate gains and losses
	for i := 1; i < len(data); i++ {
		change := data[i] - data[i-1]
		if change > 0 {
			gains[i] = change
			losses[i] = 0
		} else {
			gains[i] = 0
			losses[i] = -change
		}
	}

	rsi := make([]float64, len(data)-period)

	for i := period; i < len(data); i++ {
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
func MACD(data []float64, fastPeriod, slowPeriod, signalPeriod int) ([]float64, []float64, []float64, error) {
	if len(data) < slowPeriod+signalPeriod || fastPeriod >= slowPeriod {
		return nil, nil, nil, fmt.Errorf("invalid parameters")
	}

	// Calculate EMAs
	fastEMA, err := EMA(data, fastPeriod)
	if err != nil {
		return nil, nil, nil, err
	}

	slowEMA, err := EMA(data, slowPeriod)
	if err != nil {
		return nil, nil, nil, err
	}

	// Align lengths (EMAs may have different starting points)
	startIdx := len(data) - len(fastEMA)
	if len(fastEMA) != len(slowEMA) {
		// Find common length
		if len(fastEMA) > len(slowEMA) {
			startIdx = len(data) - len(slowEMA)
		}
	}

	// Calculate MACD line
	macd := make([]float64, len(fastEMA)-startIdx)
	for i := startIdx; i < len(fastEMA); i++ {
		macd[i-startIdx] = fastEMA[i] - slowEMA[i]
	}

	// Calculate signal line (EMA of MACD)
	signalLine, err := EMA(macd, signalPeriod)
	if err != nil {
		return nil, nil, nil, err
	}

	// Calculate histogram
	histogram := make([]float64, len(signalLine))
	for i := 0; i < len(signalLine); i++ {
		histogram[i] = macd[i+signalPeriod-1] - signalLine[i]
	}

	return macd[signalPeriod-1:], signalLine, histogram, nil
}

// Stochastic calculates Stochastic Oscillator
func Stochastic(high, low, close []float64, kPeriod, dPeriod, smoothK int) ([]float64, []float64, error) {
	if len(high) != len(low) || len(high) != len(close) {
		return nil, nil, fmt.Errorf("input arrays must have same length")
	}
	if len(high) < kPeriod+dPeriod {
		return nil, nil, fmt.Errorf("insufficient data")
	}

	// Calculate %K
	percentK := make([]float64, len(close)-kPeriod+1)
	for i := 0; i <= len(close)-kPeriod; i++ {
		// Find highest high and lowest low in period
		highest := high[i]
		lowest := low[i]
		for j := 1; j < kPeriod; j++ {
			if high[i+j] > highest {
				highest = high[i+j]
			}
			if low[i+j] < lowest {
				lowest = low[i+j]
			}
		}

		if highest == lowest {
			percentK[i] = 50 // Neutral if no range
		} else {
			percentK[i] = 100 * (close[i+kPeriod-1] - lowest) / (highest - lowest)
		}
	}

	// Smooth %K if requested
	if smoothK > 1 && len(percentK) >= smoothK {
		smoothedK := make([]float64, len(percentK)-smoothK+1)
		for i := 0; i <= len(percentK)-smoothK; i++ {
			sum := 0.0
			for j := 0; j < smoothK; j++ {
				sum += percentK[i+j]
			}
			smoothedK[i] = sum / float64(smoothK)
		}
		percentK = smoothedK
	}

	// Calculate %D (SMA of %K)
	percentD, err := SMA(percentK, dPeriod)
	if err != nil {
		return nil, nil, err
	}

	// Align lengths
	startIdx := len(percentK) - len(percentD)
	return percentK[startIdx:], percentD, nil
}

// ============================================================================
// Volume Indicators
// ============================================================================

// OBV calculates On-Balance Volume
func OBV(close, volume []float64) ([]float64, error) {
	if len(close) != len(volume) {
		return nil, fmt.Errorf("close and volume arrays must have same length")
	}

	obv := make([]float64, len(close))
	obv[0] = volume[0]

	for i := 1; i < len(close); i++ {
		if close[i] > close[i-1] {
			obv[i] = obv[i-1] + volume[i]
		} else if close[i] < close[i-1] {
			obv[i] = obv[i-1] - volume[i]
		} else {
			obv[i] = obv[i-1]
		}
	}

	return obv, nil
}

// MFI calculates Money Flow Index
func MFI(high, low, close, volume []float64, period int) ([]float64, error) {
	if len(high) != len(low) || len(high) != len(close) || len(high) != len(volume) {
		return nil, fmt.Errorf("all input arrays must have same length")
	}
	if len(high) < period || period <= 0 {
		return nil, fmt.Errorf("insufficient data or invalid period")
	}

	// Calculate Typical Price and Money Flow
	typicalPrice := make([]float64, len(close))
	moneyFlow := make([]float64, len(close))

	for i := 0; i < len(close); i++ {
		typicalPrice[i] = (high[i] + low[i] + close[i]) / 3.0
		moneyFlow[i] = typicalPrice[i] * volume[i]
	}

	mfi := make([]float64, len(close)-period+1)

	for i := 0; i <= len(close)-period; i++ {
		positiveFlow := 0.0
		negativeFlow := 0.0

		for j := i; j < i+period; j++ {
			if j == 0 {
				continue
			}
			if typicalPrice[j] > typicalPrice[j-1] {
				positiveFlow += moneyFlow[j]
			} else if typicalPrice[j] < typicalPrice[j-1] {
				negativeFlow += moneyFlow[j]
			}
		}

		if negativeFlow == 0 {
			mfi[i] = 100
		} else {
			moneyRatio := positiveFlow / negativeFlow
			mfi[i] = 100 - (100 / (1 + moneyRatio))
		}
	}

	return mfi, nil
}

// ============================================================================
// Trend Indicators
// ============================================================================

// ADX calculates Average Directional Index
func ADX(high, low, close []float64, period int) ([]float64, []float64, []float64, error) {
	if len(high) != len(low) || len(high) != len(close) {
		return nil, nil, nil, fmt.Errorf("input arrays must have same length")
	}
	if len(high) < period*2 {
		return nil, nil, nil, fmt.Errorf("insufficient data")
	}

	n := len(high)
	plusDM := make([]float64, n)
	minusDM := make([]float64, n)
	tr := make([]float64, n)

	// Calculate +DM, -DM, and TR
	for i := 1; i < n; i++ {
		upMove := high[i] - high[i-1]
		downMove := low[i-1] - low[i]

		if upMove > downMove && upMove > 0 {
			plusDM[i] = upMove
		}
		if downMove > upMove && downMove > 0 {
			minusDM[i] = downMove
		}

		// True Range
		tr[i] = math.Max(high[i]-low[i], math.Max(math.Abs(high[i]-close[i-1]), math.Abs(low[i]-close[i-1])))
	}

	// Calculate smoothed values
	plusDI := make([]float64, n-period+1)
	minusDI := make([]float64, n-period+1)
	dx := make([]float64, n-period+1)

	for i := period; i < n; i++ {
		sumPlusDM := 0.0
		sumMinusDM := 0.0
		sumTR := 0.0

		for j := i - period + 1; j <= i; j++ {
			sumPlusDM += plusDM[j]
			sumMinusDM += minusDM[j]
			sumTR += tr[j]
		}

		if sumTR > 0 {
			plusDI[i-period] = 100 * sumPlusDM / sumTR
			minusDI[i-period] = 100 * sumMinusDM / sumTR
			diff := math.Abs(plusDI[i-period] - minusDI[i-period])
			sum := plusDI[i-period] + minusDI[i-period]
			if sum > 0 {
				dx[i-period] = 100 * diff / sum
			}
		}
	}

	// Calculate ADX (SMA of DX)
	adx, err := SMA(dx, period)
	if err != nil {
		return nil, nil, nil, err
	}

	return plusDI[period-1:], minusDI[period-1:], adx, nil
}

// ============================================================================
// Support and Resistance
// ============================================================================

// SupportResistance identifies support and resistance levels
func SupportResistance(high, low []float64, lookback int, threshold float64) ([]float64, []float64) {
	if len(high) != len(low) || lookback <= 0 {
		return nil, nil
	}

	var supportLevels []float64
	var resistanceLevels []float64

	for i := lookback; i < len(high)-lookback; i++ {
		// Check for resistance (local maximum)
		isResistance := true
		currentHigh := high[i]

		for j := i - lookback; j <= i+lookback; j++ {
			if j == i {
				continue
			}
			if j >= 0 && j < len(high) && high[j] > currentHigh {
				isResistance = false
				break
			}
		}

		if isResistance {
			resistanceLevels = append(resistanceLevels, currentHigh)
		}

		// Check for support (local minimum)
		isSupport := true
		currentLow := low[i]

		for j := i - lookback; j <= i+lookback; j++ {
			if j == i {
				continue
			}
			if j >= 0 && j < len(low) && low[j] < currentLow {
				isSupport = false
				break
			}
		}

		if isSupport {
			supportLevels = append(supportLevels, currentLow)
		}
	}

	return supportLevels, resistanceLevels
}

// ============================================================================
// Pattern Recognition
// ============================================================================

// DetectDoubleTopBottom detects double top/bottom patterns
func DetectDoubleTopBottom(high, low []float64, window int, tolerance float64) ([]int, []int) {
	if len(high) != len(low) || window <= 0 {
		return nil, nil
	}

	doubleTops := []int{}
	doubleBottoms := []int{}

	for i := window; i < len(high)-window; i++ {
		// Check for double top
		leftHigh := high[i-window]
		rightHigh := high[i+window]
		middleHigh := high[i]

		if math.Abs(leftHigh-rightHigh) < tolerance &&
		   middleHigh > leftHigh && middleHigh > rightHigh &&
		   math.Abs(middleHigh-leftHigh) > tolerance {
			doubleTops = append(doubleTops, i)
		}

		// Check for double bottom
		leftLow := low[i-window]
		rightLow := low[i+window]
		middleLow := low[i]

		if math.Abs(leftLow-rightLow) < tolerance &&
		   middleLow < leftLow && middleLow < rightLow &&
		   math.Abs(middleLow-leftLow) > tolerance {
			doubleBottoms = append(doubleBottoms, i)
		}
	}

	return doubleTops, doubleBottoms
}

// ============================================================================
// Utility Functions
// ============================================================================

// CalculateReturns calculates percentage returns
func CalculateReturns(prices []float64) []float64 {
	if len(prices) <= 1 {
		return nil
	}

	returns := make([]float64, len(prices)-1)
	for i := 1; i < len(prices); i++ {
		if prices[i-1] != 0 {
			returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1]
		}
	}
	return returns
}

// LogReturns calculates logarithmic returns
func LogReturns(prices []float64) []float64 {
	if len(prices) <= 1 {
		return nil
	}

	logReturns := make([]float64, len(prices)-1)
	for i := 1; i < len(prices); i++ {
		if prices[i] > 0 && prices[i-1] > 0 {
			logReturns[i-1] = math.Log(prices[i] / prices[i-1])
		}
	}
	return logReturns
}

// RollingVolatility calculates rolling volatility of returns
func RollingVolatility(returns []float64, window int) ([]float64, error) {
	if len(returns) < window || window <= 0 {
		return nil, fmt.Errorf("insufficient data or invalid window")
	}

	volatility := make([]float64, len(returns)-window+1)

	for i := 0; i <= len(returns)-window; i++ {
		sum := 0.0
		sumSq := 0.0

		for j := 0; j < window; j++ {
			sum += returns[i+j]
			sumSq += returns[i+j] * returns[i+j]
		}

		mean := sum / float64(window)
		variance := (sumSq/float64(window) - mean*mean)
		if variance > 0 {
			volatility[i] = math.Sqrt(variance) * math.Sqrt(252) // Annualize (trading days)
		}
	}

	return volatility, nil
}
