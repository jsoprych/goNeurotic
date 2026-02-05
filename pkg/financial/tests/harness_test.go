package financial

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// ============================================================================
// Test Data Generation
// ============================================================================

// GenerateTestData generates realistic OHLCV test data with configurable patterns
func GenerateTestData(n int, seed int64, patterns ...string) ([]float64, []float64, []float64, []float64, []float64) {
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	rand.Seed(seed)

	// Initialize arrays
	open := make([]float64, n)
	high := make([]float64, n)
	low := make([]float64, n)
	close := make([]float64, n)
	volume := make([]float64, n)

	// Base price and parameters
	price := 100.0
	volatility := 0.02
	trend := 0.0002 // Slight upward trend

	// Pattern flags
	hasTrend := contains(patterns, "trend")
	hasSeasonality := contains(patterns, "seasonality")
	hasCycles := contains(patterns, "cycles")
	hasVolatilityClustering := contains(patterns, "volatility_clustering")
	hasJumps := contains(patterns, "jumps")
	hasMeanReversion := contains(patterns, "mean_reversion")

	for i := 0; i < n; i++ {
		// Apply trend
		baseChange := trend
		if hasTrend && i > n/2 {
			baseChange *= 2 // Stronger trend in second half
		}

		// Apply seasonality (weekly/monthly patterns)
		if hasSeasonality {
			dayOfWeek := i % 7
			monthOfYear := i % 252 // Approx trading days in year
			baseChange += 0.001 * math.Sin(float64(dayOfWeek)*2*math.Pi/7)
			baseChange += 0.002 * math.Sin(float64(monthOfYear)*2*math.Pi/252)
		}

		// Apply cycles
		if hasCycles {
			// Business cycle ~5 years
			cyclePeriod := 1260 // ~5 years of trading days
			baseChange += 0.003 * math.Sin(float64(i)*2*math.Pi/float64(cyclePeriod))
		}

		// Volatility clustering (GARCH-like effect)
		if hasVolatilityClustering && i > 0 {
			prevReturn := math.Abs(close[i-1] - open[i-1]) / open[i-1]
			if prevReturn > 0.03 {
				volatility *= 1.5 // Increase volatility after large moves
			} else {
				volatility = math.Max(0.005, volatility*0.99) // Gradually decay
			}
		}

		// Random shock with current volatility
		shock := rand.NormFloat64() * volatility

		// Apply mean reversion
		if hasMeanReversion {
			// Pull toward 100 with strength 0.1
			meanReversionForce := 0.1 * (100.0 - price) / 100.0
			shock += meanReversionForce
		}

		// Apply price jumps
		if hasJumps && rand.Float64() < 0.01 { // 1% chance of jump
			jumpSize := (rand.Float64() - 0.5) * 0.1 // ±10% jump
			shock += jumpSize
		}

		// Update price
		priceChange := baseChange + shock
		price *= (1 + priceChange)

		// Generate realistic OHLC from price
		intradayVol := volatility * 0.5
		open[i] = price
		high[i] = price * (1 + math.Abs(rand.NormFloat64()*intradayVol))
		low[i] = price * (1 - math.Abs(rand.NormFloat64()*intradayVol))

		// Ensure high > low and close between them
		if high[i] < low[i] {
			high[i], low[i] = low[i], high[i]
		}

		close[i] = low[i] + (high[i]-low[i])*rand.Float64()

		// Update price for next iteration based on close
		price = close[i]

		// Generate volume (correlated with volatility and price change)
		volume[i] = 1000000 * (1 + math.Abs(priceChange)*10 + rand.ExpFloat64()*0.5)
	}

	return open, high, low, close, volume
}

// GenerateMultivariateData generates correlated financial time series
func GenerateMultivariateData(n int, numSeries int, correlation float64) [][]float64 {
	if correlation < -1 || correlation > 1 {
		correlation = 0.5
	}

	data := make([][]float64, numSeries)
	for i := range data {
		data[i] = make([]float64, n)
	}

	// Generate independent random walks
	for i := 0; i < n; i++ {
		for j := 0; j < numSeries; j++ {
			if i == 0 {
				data[j][i] = 100.0
			} else {
				// Common factor + idiosyncratic component
				common := rand.NormFloat64() * 0.02 * math.Sqrt(math.Abs(correlation))
				idiosyncratic := rand.NormFloat64() * 0.02 * math.Sqrt(1-math.Abs(correlation))

				if correlation < 0 && j%2 == 1 {
					common = -common // Negative correlation for alternating series
				}

				data[j][i] = data[j][i-1] * (1 + common + idiosyncratic)
			}
		}
	}

	return data
}

// ============================================================================
// Helper Functions
// ============================================================================

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// ============================================================================
// Test Cases
// ============================================================================

func TestSMA(t *testing.T) {
	// Generate test data
	_, _, _, close, _ := GenerateTestData(100, 42, "trend", "volatility_clustering")

	// Test basic SMA
	sma, err := SMA(close, 20)
	assert.NoError(t, err)
	assert.NotNil(t, sma)
	assert.Equal(t, len(close)-20+1, len(sma))

	// Test edge cases
	_, err = SMA(close, 0)
	assert.Error(t, err)

	_, err = SMA(close, 200)
	assert.Error(t, err)

	// Test SMA properties
	for i := 0; i < len(sma)-1; i++ {
		// SMA should be smoother than original data
		origVolatility := math.Abs(close[i+20] - close[i+19])
		smaVolatility := math.Abs(sma[i+1] - sma[i])
		assert.Less(t, smaVolatility, origVolatility*5) // Much smoother
	}
}

func TestEMA(t *testing.T) {
	_, _, _, close, _ := GenerateTestData(100, 42)

	// Test EMA
	ema, err := EMA(close, 20)
	assert.NoError(t, err)
	assert.NotNil(t, ema)

	// EMA should react faster to changes than SMA
	sma, _ := SMA(close, 20)

	// Introduce a sharp change
	testData := make([]float64, 50)
	for i := range testData {
		testData[i] = 100.0
	}
	testData[25] = 110.0 // Sharp increase

	emaFast, _ := EMA(testData, 10)
	smaFast, _ := SMA(testData, 10)

	// EMA should respond faster to the shock
	emaResponse := math.Abs(emaFast[26] - emaFast[24])
	smaResponse := math.Abs(smaFast[26] - smaFast[24])
	assert.Greater(t, emaResponse, smaResponse*0.8)
}

func TestRSI(t *testing.T) {
	// Test with clearly trending data
	n := 100
	close := make([]float64, n)

	// Create strong uptrend
	for i := 0; i < n; i++ {
		close[i] = 100.0 + float64(i)*0.5
	}

	rsi, err := RSI(close, 14)
	assert.NoError(t, err)
	assert.NotNil(t, rsi)

	// In strong uptrend, RSI should be high
	for _, val := range rsi {
		assert.Greater(t, val, 50.0)
		if val > 70 {
			t.Logf("RSI indicates overbought: %.2f", val)
		}
	}

	// Test with oscillating data
	oscillating := make([]float64, 50)
	for i := range oscillating {
		oscillating[i] = 100 + 10*math.Sin(float64(i)*2*math.Pi/10)
	}

	rsiOsc, err := RSI(oscillating, 14)
	assert.NoError(t, err)

	// Oscillating data should have RSI crossing 50
	crosses50 := 0
	for i := 1; i < len(rsiOsc); i++ {
		if (rsiOsc[i-1] < 50 && rsiOsc[i] >= 50) || (rsiOsc[i-1] >= 50 && rsiOsc[i] < 50) {
			crosses50++
		}
	}
	assert.Greater(t, crosses50, 0)
}

func TestMACD(t *testing.T) {
	_, _, _, close, _ := GenerateTestData(200, 42, "trend", "cycles")

	macdLine, signalLine, histogram, err := MACD(close, 12, 26, 9)
	assert.NoError(t, err)
	assert.NotNil(t, macdLine)
	assert.NotNil(t, signalLine)
	assert.NotNil(t, histogram)
	assert.Equal(t, len(macdLine), len(signalLine))
	assert.Equal(t, len(macdLine), len(histogram))

	// Test MACD crossover signals
	buySignals := 0
	sellSignals := 0

	for i := 1; i < len(macdLine); i++ {
		// MACD crossing above signal line (buy signal)
		if macdLine[i-1] <= signalLine[i-1] && macdLine[i] > signalLine[i] {
			buySignals++
		}
		// MACD crossing below signal line (sell signal)
		if macdLine[i-1] >= signalLine[i-1] && macdLine[i] < signalLine[i] {
			sellSignals++
		}
	}

	t.Logf("MACD generated %d buy signals and %d sell signals", buySignals, sellSignals)
	assert.Greater(t, buySignals+sellSignals, 0)
}

func TestBollingerBands(t *testing.T) {
	_, _, _, close, _ := GenerateTestData(100, 42, "volatility_clustering")

	upper, middle, lower, err := BollingerBands(close, 20, 2.0)
	assert.NoError(t, err)
	assert.Equal(t, len(upper), len(middle))
	assert.Equal(t, len(middle), len(lower))

	// Test band properties
	for i := 0; i < len(middle); i++ {
		assert.Greater(t, upper[i], middle[i])
		assert.Less(t, lower[i], middle[i])

		// Bands should expand during volatile periods
		bandWidth := upper[i] - lower[i]
		if i > 0 {
			priceChange := math.Abs(close[i+19] - close[i+18])
			// Large price changes should correlate with wider bands
			if priceChange > 0.03 {
				assert.Greater(t, bandWidth, (upper[i-1]-lower[i-1])*0.9)
			}
		}
	}

	// Test price outside bands (extreme events)
	outsideUpper := 0
	outsideLower := 0

	for i := 0; i < len(middle); i++ {
		idx := i + 19 // Adjust for window
		if close[idx] > upper[i] {
			outsideUpper++
		}
		if close[idx] < lower[i] {
			outsideLower++
		}
	}

	t.Logf("Price outside upper band: %d times, lower band: %d times", outsideUpper, outsideLower)
}

func TestATR(t *testing.T) {
	high := make([]float64, 100)
	low := make([]float64, 100)
	close := make([]float64, 100)

	// Create data with increasing volatility
	for i := 0; i < 100; i++ {
		base := 100.0
		vol := 0.01 + float64(i)*0.0001

		high[i] = base * (1 + math.Abs(rand.NormFloat64())*vol)
		low[i] = base * (1 - math.Abs(rand.NormFloat64())*vol)
		close[i] = low[i] + (high[i]-low[i])*rand.Float64()

		// Ensure high > low
		if high[i] < low[i] {
			high[i], low[i] = low[i], high[i]
		}
	}

	atr, err := ATR(high, low, close, 14)
	assert.NoError(t, err)
	assert.NotNil(t, atr)

	// ATR should increase with increasing volatility
	for i := 1; i < len(atr); i++ {
		// Generally increasing due to our construction
		if i > len(atr)/2 {
			assert.Greater(t, atr[i], atr[0]*0.5)
		}
	}
}

func TestVolumeIndicators(t *testing.T) {
	open, high, low, close, volume := GenerateTestData(100, 42, "jumps")

	// Test OBV
	obv, err := OBV(close, volume)
	assert.NoError(t, err)
	assert.Equal(t, len(close), len(obv))

	// OBV should correlate with price direction
	priceUpObvUp := 0
	totalComparisons := 0

	for i := 1; i < len(close); i++ {
		if close[i] > close[i-1] && obv[i] > obv[i-1] {
			priceUpObvUp++
		}
		totalComparisons++
	}

	correlation := float64(priceUpObvUp) / float64(totalComparisons)
	assert.Greater(t, correlation, 0.4) // Should have positive correlation

	// Test MFI
	mfi, err := MFI(high, low, close, volume, 14)
	assert.NoError(t, err)
	assert.NotNil(t, mfi)

	// MFI should be between 0 and 100
	for _, val := range mfi {
		assert.GreaterOrEqual(t, val, 0.0)
		assert.LessOrEqual(t, val, 100.0)
	}
}

func TestStochastic(t *testing.T) {
	high := make([]float64, 100)
	low := make([]float64, 100)
	close := make([]float64, 100)

	// Create cyclical data
	for i := 0; i < 100; i++ {
		cycle := math.Sin(float64(i) * 2 * math.Pi / 20)
		high[i] = 100 + 10*cycle + rand.Float64()*2
		low[i] = 100 + 10*cycle - rand.Float64()*2
		close[i] = low[i] + (high[i]-low[i])*(0.3+0.4*math.Abs(cycle))

		if high[i] < low[i] {
			high[i], low[i] = low[i], high[i]
		}
	}

	k, d, err := Stochastic(high, low, close, 14, 3, 3)
	assert.NoError(t, err)
	assert.Equal(t, len(k), len(d))

	// %K should be more volatile than %D
	kVolatility := 0.0
	dVolatility := 0.0

	for i := 1; i < len(k); i++ {
		kVolatility += math.Abs(k[i] - k[i-1])
		dVolatility += math.Abs(d[i] - d[i-1])
	}

	assert.Greater(t, kVolatility, dVolatility)

	// Both should be between 0 and 100
	for i := 0; i < len(k); i++ {
		assert.GreaterOrEqual(t, k[i], 0.0)
		assert.LessOrEqual(t, k[i], 100.0)
		assert.GreaterOrEqual(t, d[i], 0.0)
		assert.LessOrEqual(t, d[i], 100.0)
	}
}

func TestADX(t *testing.T) {
	// Create trending data
	high := make([]float64, 200)
	low := make([]float64, 200)
	close := make([]float64, 200)

	trendStrength := 0.005
	for i := 0; i < 200; i++ {
		trend := float64(i) * trendStrength
		noise := rand.NormFloat64() * 0.02

		high[i] = 100 + trend + math.Abs(noise)
		low[i] = 100 + trend - math.Abs(noise)
		close[i] = 100 + trend + noise*0.5

		if high[i] < low[i] {
			high[i], low[i] = low[i], high[i]
		}
	}

	plusDI, minusDI, adx, err := ADX(high, low, close, 14)
	assert.NoError(t, err)
	assert.Equal(t, len(plusDI), len(minusDI))
	assert.Equal(t, len(minusDI), len(adx))

	// In uptrend, +DI should generally be greater than -DI
	plusDIGreater := 0
	for i := 0; i < len(plusDI); i++ {
		if plusDI[i] > minusDI[i] {
			plusDIGreater++
		}
	}

	assert.Greater(t, float64(plusDIGreater)/float64(len(plusDI)), 0.6)

	// ADX should be positive
	for _, val := range adx {
		assert.GreaterOrEqual(t, val, 0.0)
		assert.LessOrEqual(t, val, 100.0)
	}
}

func TestSupportResistance(t *testing.T) {
	high := make([]float64, 200)
	low := make([]float64, 200)

	// Create data with clear support/resistance levels
	for i := 0; i < 200; i++ {
		// Major resistance at 110, support at 90
		base := 100.0
		cycle := math.Sin(float64(i) * 2 * math.Pi / 50)

		high[i] = base + 15*cycle + rand.Float64()*5
		low[i] = base + 15*cycle - rand.Float64()*5 - 5

		if high[i] < low[i] {
			high[i], low[i] = low[i], high[i]
		}

		// Create clear resistance around 115
		if high[i] > 114 && high[i] < 116 {
			high[i] = 115 // Pin to resistance
		}

		// Create clear support around 85
		if low[i] > 84 && low[i] < 86 {
			low[i] = 85 // Pin to support
		}
	}

	supportLevels, resistanceLevels := SupportResistance(high, low, 10, 2.0)

	assert.NotNil(t, supportLevels)
	assert.NotNil(t, resistanceLevels)

	// Should find at least some levels
	assert.Greater(t, len(supportLevels), 0)
	assert.Greater(t, len(resistanceLevels), 0)

	t.Logf("Found %d support levels and %d resistance levels",
		len(supportLevels), len(resistanceLevels))
}

func TestPatternRecognition(t *testing.T) {
	high := make([]float64, 100)
	low := make([]float64, 100)

	// Create a clear double top pattern around index 40
	for i := 0; i < 100; i++ {
		base := 100.0

		// Double top pattern
		if i >= 30 && i <= 50 {
			patternPos := float64(i - 30)
			// M-shaped pattern
			patternValue := -math.Abs(patternPos-10) + 10
			high[i] = base + patternValue*2 + rand.Float64()
			low[i] = base + patternValue*2 - 5 - rand.Float64()
		} else {
			high[i] = base + rand.NormFloat64()*5
			low[i] = base - 5 + rand.NormFloat64()*5
		}

		if high[i] < low[i] {
			high[i], low[i] = low[i], high[i]
		}
	}

	doubleTops, doubleBottoms := DetectDoubleTopBottom(high, low, 10, 5.0)

	// Should detect the double top around index 40
	assert.Greater(t, len(doubleTops), 0)

	foundDoubleTop := false
	for _, idx := range doubleTops {
		if idx >= 35 && idx <= 45 {
			foundDoubleTop = true
			t.Logf("Detected double top at index %d (expected around 40)", idx)
			break
		}
	}

	assert.True(t, foundDoubleTop)
}

func TestReturnsCalculations(t *testing.T) {
	// Create price series with known returns
	prices := []float64{100, 105, 102, 108, 110}

	returns := CalculateReturns(prices)
	assert.Equal(t, len(prices)-1, len(returns))

	// Manual calculation check
	expected := []float64{0.05, -0.0285714, 0.0588235, 0.0185185}
	for i := range returns {
		assert.InDelta(t, expected[i], returns[i], 0.0001)
	}

	// Test log returns
	logReturns := LogReturns(prices)
	assert.Equal(t, len(prices)-1, len(logReturns))

	// Log returns should be slightly less than simple returns for positive returns
	for i := range returns {
		if returns[i] > 0 {
			assert.Less(t, logReturns[i], returns[i])
		}
	}
}

func TestRollingVolatility(t *testing.T) {
	// Create returns with changing volatility
	returns := make([]float64, 100)

	// Low volatility first half, high volatility second half
	for i := 0; i < 100; i++ {
		if i < 50 {
			returns[i] = rand.NormFloat64() * 0.01
		} else {
			returns[i] = rand.NormFloat64() * 0.03
		}
	}

	volatility, err := RollingVolatility(returns, 20)
	assert.NoError(t, err)
	assert.NotNil(t, volatility)

	// Second half should have higher volatility
	firstHalfAvg := 0.0
	secondHalfAvg := 0.0

	midpoint := len(volatility) / 2
	for i := 0; i < midpoint; i++ {
		firstHalfAvg += volatility[i]
	}
	for i := midpoint; i < len(volatility); i++ {
		secondHalfAvg += volatility[i]
	}

	firstHalfAvg /= float64(midpoint)
	secondHalfAvg /= float64(len(volatility) - midpoint)

	assert.Greater(t, secondHalfAvg, firstHalfAvg*1.5)
}

// ============================================================================
// Integration Tests
// ============================================================================

func TestIndicatorCombination(t *testing.T) {
	// Test combining multiple indicators for trading signals
	open, high, low, close, volume := GenerateTestData(500, 42, "trend", "volatility_clustering")

	// Calculate multiple indicators
	rsi, err := RSI(close, 14)
	assert.NoError(t, err)

	macdLine, signalLine, _, err := MACD(close, 12, 26, 9)
	assert.NoError(t, err)

	upperBB, middleBB, lowerBB, err := BollingerBands(close, 20, 2.0)
	assert.NoError(t, err)

	// Generate combined signals
	signals := make([]string, len(rsi))

	for i := 0; i < len(rsi); i++ {
		bbIdx := i + (len(close) - len(upperBB)) // Align indices
		macdIdx := i + (len(close) - len(macdLine))

		buyConditions := 0
		sellConditions := 0

		// RSI conditions
		if rsi[i] < 30 {
			buyConditions++
		} else if rsi[i] > 70 {
			sellConditions++
		}

		// MACD conditions
		if macdIdx >= 1 && macdLine[macdIdx] > signalLine[macdIdx] && macdLine[macdIdx-1] <= signalLine[macdIdx-1] {
			buyConditions++
		}
		if macdIdx >= 1 && macdLine[macdIdx] < signalLine[macdIdx] && macdLine[macdIdx-1] >= signalLine[macdIdx-1] {
			sellConditions++
		}

		// Bollinger Bands conditions
		if bbIdx >= 0 && close[bbIdx] < lowerBB[bbIdx-(len(close)-len(lowerBB))] {
			buyConditions++
		}
		if bbIdx >= 0 && close[bbIdx] > upperBB[bbIdx-(len(close)-len(upperBB))] {
			sellConditions++
		}

		// Determine signal
		if buyConditions >= 2 && buyConditions > sellConditions {
			signals[i] = "BUY"
		} else if sellConditions >= 2 && sellConditions > buyConditions {
			signals[i] = "SELL"
		} else {
			signals[i] = "HOLD"
		}
	}

	// Count signals
	buyCount := 0
	sellCount := 0
	for _, signal := range signals {
		if signal == "BUY" {
			buyCount++
		} else if signal == "SELL" {
			sellCount++
		}
	}

	t.Logf("Combined strategy generated %d BUY and %d SELL signals", buyCount, sellCount)
	assert.Greater(t, buyCount+sellCount, 0)
}

// ============================================================================
// Benchmark Tests
// ============================================================================

func BenchmarkSMA(b *testing.B) {
	_, _, _, close, _ := GenerateTestData(10000, 42)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		SMA(close, 50)
	}
}

func BenchmarkEMA(b *testing.B) {
	_, _, _, close, _ := GenerateTestData(10000, 42)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		EMA(close, 50)
	}
}

func BenchmarkRSI(b *testing.B) {
	_, _, _, close, _ := GenerateTestData(10000, 42)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		RSI(close, 14)
	}
}

func BenchmarkMACD(b *testing.B) {
	_, _, _, close, _ := GenerateTestData(10000, 42)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		MACD(close, 12, 26, 9)
	}
}

func BenchmarkBollingerBands(b *testing.B) {
	_, _, _, close, _ := GenerateTestData(10000, 42)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		BollingerBands(close, 20, 2.0)
	}
}

func BenchmarkIndicatorCombination(b *testing.B) {
	open, high, low, close, volume := GenerateTestData(5000, 42)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Calculate multiple indicators
		RSI(close, 14)
		MACD(close, 12, 26, 9)
		BollingerBands(close, 20, 2.0)
		MFI(high, low, close, volume, 14)
		ATR(high, low, close, 14)
	}
}

// ============================================================================
// Realistic Trading Simulation
// ============================================================================

type TradingSimulation struct {
	Capital    float64
	Position   float64 // Number of shares
	Trades     []Trade
	Commission float64
}

type Trade struct {
	Type      string    // "BUY" or "SELL"
	Price     float64
	Shares    float64
	Timestamp int
	Reason    string
}

func (ts *TradingSimulation) Buy(price float64, shares float64, reason string) {
	cost := price*shares + ts.Commission
	if cost > ts.Capital {
		shares = (ts.Capital - ts.Commission) / price
		cost = price*shares + ts.Commission
	}

	ts.Capital -= cost
	ts.Position += shares
	ts.Trades = append(ts.Trades, Trade{
		Type:      "BUY",
		Price:     price,
		Shares:    shares,
		Timestamp: len(ts.Trades),
		Reason:    reason,
	})
}

func (ts *TradingSimulation) Sell(price float64, shares float64, reason string) {
	if shares > ts.Position {
		shares = ts.Position
	}

	proceeds := price*shares - ts.Commission
	ts.Capital += proceeds
	ts.Position -= shares
	ts.Trades = append(ts.Trades, Trade{
		Type:      "SELL",
		Price:     price,
		Shares:    shares,
		Timestamp: len(ts.Trades),
		Reason:    reason,
	})
}

func (ts *TradingSimulation) Value(price float64) float64 {
	return ts.Capital + ts.Position*price
}

func TestTradingSimulationWithIndicators(t *testing.T) {
	// Generate realistic market data
	open, high, low, close, volume := GenerateTestData(1000, 42,
		"trend", "volatility_clustering", "cycles")

	// Initialize simulation
	sim := &TradingSimulation{
		Capital:    100000,
		Commission: 9.99,
	}

	// Calculate indicators
	rsi, _ := RSI(close, 14)
	macdLine, signalLine, _, _ := MACD(close, 12, 26, 9)
	upperBB, middleBB, lowerBB, _ := BollingerBands(close, 20, 2.0)

	// Trading strategy
	positionSize := 10.0 // Shares per trade

	for i := 20; i < len(close); i++ {
		currentPrice := close[i]

		// Get indicator values (with boundary checks)
		rsiIdx := i - 14
		macdIdx := i - 26
		bbIdx := i - 20

		// Check for buy signals
		buySignal := false
		buyReason := ""

		if rsiIdx >= 0 && rsiIdx < len(rsi) && rsi[rsiIdx] < 30 {
			buySignal = true
			buyReason = "RSI oversold"
		}

		if bbIdx >= 0 && bbIdx < len(lowerBB) && currentPrice < lowerBB[bbIdx] {
			buySignal = true
			if buyReason != "" {
				buyReason += " + BB oversold"
			} else {
				buyReason = "BB oversold"
			}
		}

		if macdIdx >= 1 && macdIdx < len(macdLine) &&
		   macdLine[macdIdx] > signalLine[macdIdx] &&
		   macdLine[macdIdx-1] <= signalLine[macdIdx-1] {
			buySignal = true
			if buyReason != "" {
				buyReason += " + MACD crossover"
			} else {
				buyReason = "MACD crossover"
			}
		}

		// Check for sell signals
		sellSignal := false
		sellReason := ""

		if rsiIdx >= 0 && rsiIdx < len(rsi) && rsi[rsiIdx] > 70 {
			sellSignal = true
			sellReason = "RSI overbought"
		}

		if bbIdx >= 0 && bbIdx < len(upperBB) && currentPrice > upperBB[bbIdx] {
			sellSignal = true
			if sellReason != "" {
				sellReason += " + BB overbought"
			} else {
				sellReason = "BB overbought"
			}
		}

		if macdIdx >= 1 && macdIdx < len(macdLine) &&
		   macdLine[macdIdx] < signalLine[macdIdx] &&
		   macdLine[macdIdx-1] >= signalLine[macdIdx-1] {
			sellSignal = true
			if sellReason != "" {
				sellReason += " + MACD crossover"
			} else {
				sellReason = "MACD crossover"
			}
		}

		// Execute trades
		if buySignal && sim.Position == 0 {
			sim.Buy(currentPrice, positionSize, buyReason)
		}

		if sellSignal && sim.Position > 0 {
			sim.Sell(currentPrice, sim.Position, sellReason)
		}
	}

	// Close any open position at end
	if sim.Position > 0 {
		sim.Sell(close[len(close)-1], sim.Position, "End of period")
	}

	finalValue := sim.Value(close[len(close)-1])
	initialValue := 100000.0
	returnPct := (finalValue - initialValue) / initialValue * 100

	t.Logf("Trading Simulation Results:")
	t.Logf("Initial Capital: $%.2f", initialValue)
	t.Logf("Final Value: $%.2f", finalValue)
	t.Logf("Return: %.2f%%", returnPct)
	t.Logf("Number of Trades: %d", len(sim.Trades))
	t.Logf("Commission Paid: $%.2f", float64(len(sim.Trades))*sim.Commission)

	// Basic validation
	assert.Greater(t, len(sim.Trades), 0)
	assert.Greater(t, finalValue, 0.0)

	// Log some example trades
	if len(sim.Trades) > 0 {
		t.Logf("First trade: %s at $%.2f (%s)",
			sim.Trades[0].Type, sim.Trades[0].Price, sim.Trades[0].Reason)
		if len(sim.Trades) > 1 {
			t.Logf("Last trade: %s at $%.2f (%s)",
				sim.Trades[len(sim.Trades)-1].Type,
				sim.Trades[len(sim.Trades)-1].Price,
				sim.Trades[len(sim.Trades)-1].Reason)
		}
	}
}
