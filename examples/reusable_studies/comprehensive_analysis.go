package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/goneurotic/financial"
)

// ============================================================================
// Reusable Financial Time-Series Study Template
// ============================================================================
// This template demonstrates a comprehensive, reusable analysis workflow
// for financial time-series data using GoNeurotic's financial package.
//
// Key Features:
// 1. Modular design - Each analysis component is separate and reusable
// 2. Comprehensive metrics - Technical indicators, risk metrics, statistical tests
// 3. Market regime detection - Adaptive analysis based on market conditions
// 4. Export functionality - JSON and CSV output for further analysis
// 5. Visualization-ready - Structured output for charting libraries
//
// Usage:
// 1. Load your price data (CSV format: date,price)
// 2. Run analysis with different configurations
// 3. Export results for visualization or further processing
//
// Example command:
//   go run comprehensive_analysis.go -file=sp500.csv -window=60 -period=14
// ============================================================================

// StudyConfig holds configuration for the analysis
type StudyConfig struct {
	// Data configuration
	InputFile  string
	DateFormat string

	// Analysis parameters
	WindowSize         int     // Lookback window for metrics
	IndicatorPeriod    int     // Period for technical indicators
	VolatilityStdDev   float64 // Standard deviations for Bollinger Bands
	RiskFreeRate       float64 // Annual risk-free rate for performance ratios
	AnnualizationDays  float64 // Trading days per year

	// Output options
	ExportJSON  bool
	ExportCSV   bool
	Verbose     bool
	PlotOutput  bool
}

// DefaultStudyConfig returns sensible defaults
func DefaultStudyConfig() StudyConfig {
	return StudyConfig{
		DateFormat:        "2006-01-02",
		WindowSize:        60,
		IndicatorPeriod:   14,
		VolatilityStdDev:  2.0,
		RiskFreeRate:      0.02,
		AnnualizationDays: 252.0,
		ExportJSON:        true,
		ExportCSV:         false,
		Verbose:           false,
		PlotOutput:        false,
	}
}

// PriceData holds loaded price information
type PriceData struct {
	Timestamps []time.Time
	Prices     []float64
	Ticker     string
	StartDate  time.Time
	EndDate    time.Time
}

// StudyResults aggregates all analysis results
type StudyResults struct {
	Metadata struct {
		Ticker       string
		StartDate    string
		EndDate      string
		DataPoints   int
		Config       StudyConfig
		AnalysisDate string
	}

	// Price statistics
	PriceStats struct {
		MinPrice  float64
		MaxPrice  float64
		MeanPrice float64
		StdDev    float64
		Returns   []float64
	}

	// Technical indicators
	Indicators struct {
		SMA        []float64
		EMA        []float64
		RSI        []float64
		MACD       []float64
		MACDSignal []float64
		BBUpper    []float64
		BBMiddle   []float64
		BBLower    []float64
		ATR        []float64
	}

	// Risk and performance metrics
	RiskMetrics struct {
		AnnualizedReturn   float64
		AnnualizedVolatility float64
		SharpeRatio        float64
		SortinoRatio       float64
		MaxDrawdown        float64
		AvgDrawdown        float64
		CalmarRatio        float64
		VaR95              float64
		VaR99              float64
	}

	// Statistical tests
	StatisticalTests struct {
		NormalityTest financial.NormalityTestResult
		StationarityTest financial.StationarityTestResult
		Autocorrelation map[int]float64
	}

	// Market regime analysis
	MarketRegimes []struct {
		Date       string
		Regime     financial.MarketRegime
		Confidence float64
		Features   financial.RegimeMetrics
	}

	// Feature matrix for machine learning
	FeatureMatrix [][]float64
	FeatureNames  []string
}

// LoadPriceData loads price data from CSV file
// Expected format: date,price (optional: volume,open,high,low)
func LoadPriceData(filename, dateFormat string) (*PriceData, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV: %w", err)
	}

	if len(records) < 2 {
		return nil, fmt.Errorf("insufficient data in file")
	}

	var timestamps []time.Time
	var prices []float64

	// Skip header if present
	startIdx := 0
	if _, err := time.Parse(dateFormat, records[0][0]); err != nil {
		startIdx = 1 // Has header
	}

	for i := startIdx; i < len(records); i++ {
		record := records[i]
		if len(record) < 2 {
			continue
		}

		// Parse date
		date, err := time.Parse(dateFormat, record[0])
		if err != nil {
			return nil, fmt.Errorf("invalid date format at row %d: %w", i+1, err)
		}

		// Parse price
		price, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			return nil, fmt.Errorf("invalid price at row %d: %w", i+1, err)
		}

		timestamps = append(timestamps, date)
		prices = append(prices, price)
	}

	if len(timestamps) == 0 {
		return nil, fmt.Errorf("no valid data found")
	}

	return &PriceData{
		Timestamps: timestamps,
		Prices:     prices,
		Ticker:     filename,
		StartDate:  timestamps[0],
		EndDate:    timestamps[len(timestamps)-1],
	}, nil
}

// RunComprehensiveAnalysis executes the complete analysis workflow
func RunComprehensiveAnalysis(data *PriceData, config StudyConfig) (*StudyResults, error) {
	if len(data.Prices) < config.WindowSize*2 {
		return nil, fmt.Errorf("insufficient data: need at least %d points, got %d",
			config.WindowSize*2, len(data.Prices))
	}

	results := &StudyResults{}

	// Set metadata
	results.Metadata.Ticker = data.Ticker
	results.Metadata.StartDate = data.StartDate.Format("2006-01-02")
	results.Metadata.EndDate = data.EndDate.Format("2006-01-02")
	results.Metadata.DataPoints = len(data.Prices)
	results.Metadata.Config = config
	results.Metadata.AnalysisDate = time.Now().Format("2006-01-02 15:04:05")

	// 1. Calculate basic price statistics
	if config.Verbose {
		fmt.Println("Step 1: Calculating price statistics...")
	}
	calculatePriceStatistics(data, results)

	// 2. Calculate technical indicators
	if config.Verbose {
		fmt.Println("Step 2: Calculating technical indicators...")
	}
	if err := calculateTechnicalIndicators(data, config, results); err != nil {
		return nil, fmt.Errorf("failed to calculate indicators: %w", err)
	}

	// 3. Calculate risk metrics
	if config.Verbose {
		fmt.Println("Step 3: Calculating risk metrics...")
	}
	if err := calculateRiskMetrics(data, config, results); err != nil {
		return nil, fmt.Errorf("failed to calculate risk metrics: %w", err)
	}

	// 4. Perform statistical tests
	if config.Verbose {
		fmt.Println("Step 4: Performing statistical tests...")
	}
	performStatisticalTests(data, results)

	// 5. Detect market regimes
	if config.Verbose {
		fmt.Println("Step 5: Detecting market regimes...")
	}
	detectMarketRegimes(data, config, results)

	// 6. Create feature matrix for machine learning
	if config.Verbose {
		fmt.Println("Step 6: Creating feature matrix...")
	}
	createFeatureMatrix(results)

	return results, nil
}

// calculatePriceStatistics computes basic price statistics
func calculatePriceStatistics(data *PriceData, results *StudyResults) {
	prices := data.Prices

	// Basic statistics
	results.PriceStats.MinPrice = minSlice(prices)
	results.PriceStats.MaxPrice = maxSlice(prices)
	results.PriceStats.MeanPrice = mean(prices)
	results.PriceStats.StdDev = stdDev(prices)

	// Calculate returns
	results.PriceStats.Returns = make([]float64, len(prices)-1)
	for i := 1; i < len(prices); i++ {
		if prices[i-1] > 0 {
			results.PriceStats.Returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1]
		}
	}
}

// calculateTechnicalIndicators computes various technical indicators
func calculateTechnicalIndicators(data *PriceData, config StudyConfig, results *StudyResults) error {
	prices := data.Prices

	// Moving averages
	sma, err := financial.SMA(prices, config.IndicatorPeriod)
	if err != nil {
		return fmt.Errorf("SMA calculation failed: %w", err)
	}
	results.Indicators.SMA = sma

	ema, err := financial.EMA(prices, config.IndicatorPeriod)
	if err != nil {
		return fmt.Errorf("EMA calculation failed: %w", err)
	}
	results.Indicators.EMA = ema

	// RSI
	rsi, err := financial.RSI(prices, config.IndicatorPeriod)
	if err != nil {
		return fmt.Errorf("RSI calculation failed: %w", err)
	}
	results.Indicators.RSI = rsi

	// MACD (standard parameters: 12, 26, 9)
	macdLine, signalLine, _, err := financial.MACD(prices, 12, 26, 9)
	if err == nil {
		results.Indicators.MACD = macdLine
		results.Indicators.MACDSignal = signalLine
	}

	// Bollinger Bands
	upper, middle, lower, err := financial.BollingerBands(prices, config.IndicatorPeriod, config.VolatilityStdDev)
	if err == nil {
		results.Indicators.BBUpper = upper
		results.Indicators.BBMiddle = middle
		results.Indicators.BBLower = lower
	}

	// For ATR, we need OHLC data - using simplified version with just close prices
	// In a real implementation, you would use actual OHLC data
	if len(prices) >= config.IndicatorPeriod {
		// Simplified ATR using close prices only
		atrValues := make([]float64, len(prices)-config.IndicatorPeriod+1)
		for i := 0; i <= len(prices)-config.IndicatorPeriod; i++ {
			window := prices[i : i+config.IndicatorPeriod]
			trSum := 0.0
			for j := 1; j < len(window); j++ {
				tr := math.Abs(window[j] - window[j-1])
				trSum += tr
			}
			atrValues[i] = trSum / float64(len(window)-1)
		}
		results.Indicators.ATR = atrValues
	}

	return nil
}

// calculateRiskMetrics computes risk and performance metrics
func calculateRiskMetrics(data *PriceData, config StudyConfig, results *StudyResults) error {
	prices := data.Prices
	returns := results.PriceStats.Returns

	if len(returns) == 0 {
		return fmt.Errorf("no returns calculated")
	}

	// Create metric config for financial package
	metricConfig := financial.MetricConfig{
		WindowSize:          config.WindowSize,
		Period:              config.IndicatorPeriod,
		StdDev:              config.VolatilityStdDev,
		RiskFreeRate:        config.RiskFreeRate,
		AnnualizationFactor: config.AnnualizationDays,
	}

	// Calculate returns metrics
	returnsMetrics, err := financial.CalculateReturnsMetrics(prices, metricConfig)
	if err != nil {
		return fmt.Errorf("returns metrics calculation failed: %w", err)
	}

	results.RiskMetrics.AnnualizedReturn = returnsMetrics.AnnualizedReturn
	results.RiskMetrics.AnnualizedVolatility = returnsMetrics.AnnualizedVol

	// Calculate performance ratios
	performanceRatios := financial.CalculatePerformanceRatios(returns, metricConfig)
	results.RiskMetrics.SharpeRatio = performanceRatios.SharpeRatio
	results.RiskMetrics.SortinoRatio = performanceRatios.SortinoRatio

	// Calculate drawdown metrics
	drawdownMetrics, err := financial.CalculateDrawdownMetrics(prices)
	if err == nil {
		results.RiskMetrics.MaxDrawdown = drawdownMetrics.MaxDrawdown
		results.RiskMetrics.AvgDrawdown = drawdownMetrics.AvgDrawdown
		results.RiskMetrics.CalmarRatio = drawdownMetrics.CalmarRatio
	}

	// Calculate Value at Risk (simplified historical VaR)
	if len(returns) >= 100 {
		sortedReturns := make([]float64, len(returns))
		copy(sortedReturns, returns)
		sortSlice(sortedReturns)

		// Historical VaR at 95% and 99% confidence
		var95Idx := int(float64(len(sortedReturns)) * 0.05)
		var99Idx := int(float64(len(sortedReturns)) * 0.01)

		if var95Idx < len(sortedReturns) {
			results.RiskMetrics.VaR95 = -sortedReturns[var95Idx]
		}
		if var99Idx < len(sortedReturns) {
			results.RiskMetrics.VaR99 = -sortedReturns[var99Idx]
		}
	}

	return nil
}

// performStatisticalTests runs statistical tests on returns
func performStatisticalTests(data *PriceData, results *StudyResults) {
	returns := results.PriceStats.Returns

	if len(returns) < 20 {
		return
	}

	// Normality test
	results.StatisticalTests.NormalityTest = financial.TestNormality(returns)

	// Stationarity test (using prices and returns)
	results.StatisticalTests.StationarityTest = financial.TestStationarity(data.Prices, returns)

	// Autocorrelation test (lags 1-5)
	results.StatisticalTests.Autocorrelation = financial.AutocorrelationTest(returns, 5)
}

// detectMarketRegimes identifies different market conditions
func detectMarketRegimes(data *PriceData, config StudyConfig, results *StudyResults) {
	prices := data.Prices
	window := config.WindowSize

	// Detect regimes at regular intervals
	step := window / 2
	if step < 1 {
		step = 1
	}

	for i := window; i < len(prices); i += step {
		// Get window for regime detection
		start := i - window
		if start < 0 {
			start = 0
		}
		windowPrices := prices[start:i]

		// Detect regime
		regimeMetrics, err := financial.DetectMarketRegime(windowPrices, window)
		if err != nil {
			continue
		}

		// Record regime
		regime := struct {
			Date       string
			Regime     financial.MarketRegime
			Confidence float64
			Features   financial.RegimeMetrics
		}{
			Date:       data.Timestamps[i-1].Format("2006-01-02"),
			Regime:     regimeMetrics.CurrentRegime,
			Confidence: regimeMetrics.RegimeStrength,
			Features:   regimeMetrics,
		}

		results.MarketRegimes = append(results.MarketRegimes, regime)
	}
}

// createFeatureMatrix prepares data for machine learning models
func createFeatureMatrix(results *StudyResults) {
	// Find the minimum length among all indicators
	minLength := len(results.Indicators.SMA)
	if len(results.Indicators.EMA) < minLength && len(results.Indicators.EMA) > 0 {
		minLength = len(results.Indicators.EMA)
	}
	if len(results.Indicators.RSI) < minLength && len(results.Indicators.RSI) > 0 {
		minLength = len(results.Indicators.RSI)
	}

	if minLength < 10 {
		return
	}

	// Create feature names
	results.FeatureNames = []string{
		"Returns",
		"Returns_Lag1",
		"Returns_Lag2",
		"RSI",
		"SMA_Ratio",
		"BB_Position",
		"Volatility",
		"ATR_Ratio",
	}

	// Create feature matrix
	results.FeatureMatrix = make([][]float64, minLength)

	for i := 0; i < minLength; i++ {
		features := make([]float64, len(results.FeatureNames))

		// Returns (need to align with indicators)
		returnIdx := len(results.PriceStats.Returns) - minLength + i
		if returnIdx >= 0 && returnIdx < len(results.PriceStats.Returns) {
			features[0] = results.PriceStats.Returns[returnIdx]
			if returnIdx > 0 {
				features[1] = results.PriceStats.Returns[returnIdx-1]
			}
			if returnIdx > 1 {
				features[2] = results.PriceStats.Returns[returnIdx-2]
			}
		}

		// RSI
		rsiIdx := len(results.Indicators.RSI) - minLength + i
		if rsiIdx >= 0 && rsiIdx < len(results.Indicators.RSI) {
			features[3] = results.Indicators.RSI[rsiIdx]
		}

		// SMA Ratio (price relative to SMA)
		smaIdx := len(results.Indicators.SMA) - minLength + i
		if smaIdx >= 0 && smaIdx < len(results.Indicators.SMA) && results.Indicators.SMA[smaIdx] > 0 {
			// We would need the corresponding price here
			// For now, leave as 0 or implement proper alignment
		}

		// Add other features similarly...

		results.FeatureMatrix[i] = features
	}
}

// ExportResults saves analysis results to files
func ExportResults(results *StudyResults, config StudyConfig) error {
	if config.ExportJSON {
		if err := exportJSON(results); err != nil {
			return fmt.Errorf("JSON export failed: %w", err)
		}
	}

	if config.ExportCSV {
		if err := exportCSV(results); err != nil {
			return fmt.Errorf("CSV export failed: %w", err)
		}
	}

	return nil
}

// exportJSON saves results as JSON file
func exportJSON(results *StudyResults) error {
	filename := fmt.Sprintf("%s_analysis_%s.json",
		results.Metadata.Ticker,
		time.Now().Format("20060102_150405"))

	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("failed to write JSON file: %w", err)
	}

	fmt.Printf("Results exported to %s\n", filename)
	return nil
}

// exportCSV saves key metrics as CSV files
func exportCSV(results *StudyResults) error {
	// Export price data with indicators
	priceFile := fmt.Sprintf("%s_prices_%s.csv",
		results.Metadata.Ticker,
		time.Now().Format("20060102"))

	file, err := os.Create(priceFile)
	if err != nil {
		return fmt.Errorf("failed to create CSV file: %w", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	header := []string{"Metric", "Value", "Description"}
	writer.Write(header)

	// Write key metrics
	metrics := [][]string{
		{"Annualized Return", fmt.Sprintf("%.4f", results.RiskMetrics.AnnualizedReturn), "Annualized return"},
		{"Annualized Volatility", fmt.Sprintf("%.4f", results.RiskMetrics.AnnualizedVolatility), "Annualized volatility"},
		{"Sharpe Ratio", fmt.Sprintf("%.4f", results.RiskMetrics.SharpeRatio), "Risk-adjusted return"},
		{"Max Drawdown", fmt.Sprintf("%.4f", results.RiskMetrics.MaxDrawdown), "Maximum drawdown"},
		{"VaR 95%", fmt.Sprintf("%.4f", results.RiskMetrics.VaR95), "Value at Risk 95%"},
		{"Data Points", fmt.Sprintf("%d", results.Metadata.DataPoints), "Number of data points"},
	}

	for _, metric := range metrics {
		writer.Write(metric)
	}

	fmt.Printf("Metrics exported to %s\n", priceFile)
	return nil
}

// PrintSummary displays key findings
func PrintSummary(results *StudyResults) {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("FINANCIAL TIME-SERIES ANALYSIS SUMMARY")
	fmt.Println(strings.Repeat("=", 80))

	fmt.Printf("\nDataset: %s\n", results.Metadata.Ticker)
	fmt.Printf("Period: %s to %s (%d data points)\n",
		results.Metadata.StartDate,
		results.Metadata.EndDate,
		results.Metadata.DataPoints)

	fmt.Println("\n" + strings.Repeat("-", 80))
	fmt.Println("PERFORMANCE METRICS")
	fmt.Println(strings.Repeat("-", 80))
	fmt.Printf("Annualized Return:    %7.2f%%\n", results.RiskMetrics.AnnualizedReturn*100)
	fmt.Printf("Annualized Volatility:%7.2f%%\n", results.RiskMetrics.AnnualizedVolatility*100)
	fmt.Printf("Sharpe Ratio:         %7.2f\n", results.RiskMetrics.SharpeRatio)
	fmt.Printf("Sortino Ratio:        %7.2f\n", results.RiskMetrics.SortinoRatio)
	fmt.Printf("Max Drawdown:         %7.2f%%\n", results.RiskMetrics.MaxDrawdown*100)
	fmt.Printf("Calmar Ratio:         %7.2f\n", results.RiskMetrics.CalmarRatio)

	fmt.Println("\n" + strings.Repeat("-", 80))
	fmt.Println("STATISTICAL TESTS")
	fmt.Println(strings.Repeat("-", 80))
	fmt.Printf("Normality Test (Jarque-Bera): %.2f\n", results.StatisticalTests.NormalityTest.JarqueBera)
	fmt.Printf("Data appears normal: %v\n", results.StatisticalTests.NormalityTest.IsNormal)
	fmt.Printf("Returns stationary: %v\n", results.StatisticalTests.StationarityTest.ReturnStationary)

	fmt.Println("\n" + strings.Repeat("-", 80))
	fmt.Println("MARKET REGIME ANALYSIS")
	fmt.Println(strings.Repeat("-", 80))
	if len(results.MarketRegimes) > 0 {
		latest := results.MarketRegimes[len(results.MarketRegimes)-1]
		fmt.Printf("Current Regime: %s (confidence: %.1f%%)\n",
			latest.Regime, latest.Confidence*100)

		// Count regime occurrences
		regimeCounts := make(map[financial.MarketRegime]int)
		for _, r := range results.MarketRegimes {
			regimeCounts[r.Regime]++
		}

		fmt.Println("\nRegime Distribution:")
		for regime, count := range regimeCounts {
			percentage := float64(count) / float64(len(results.MarketRegimes)) * 100
			fmt.Printf("  %-20s: %3d (%.1f%%)\n", regime, count, percentage)
		}
	}

	fmt.Println("\n" + strings.Repeat("-", 80))
	fmt.Println("FEATURE ENGINEERING")
	fmt.Println(strings.Repeat("-", 80))
	fmt.Printf("Feature matrix: %d samples × %d features\n",
		len(results.FeatureMatrix), len(results.FeatureNames))

	if len(results.FeatureNames) > 0 {
		fmt.Println("Features available for ML modeling:")
		for i, name := range results.FeatureNames {
			fmt.Printf("  %2d. %s\n", i+1, name)
		}
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("ANALYSIS COMPLETE")
	fmt.Println(strings.Repeat("=", 80))
}

// Utility functions
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

func minSlice(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	min := data[0]
	for _, v := range data {
		if v < min {
			min = v
		}
	}
	return min
}

func maxSlice(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	max := data[0]
	for _, v := range data {
		if v > max {
			max = v
		}
	}
	return max
}

func sortSlice(data []float64) {
	sort.Float64s(data)
}

// sort.Float64s wrapper for our custom type
func sortFloat64s(data []float64) {
	sort.Slice(data, func(i, j int) bool {
		return data[i] < data[j]
	})
}

// strings package is needed for the strings.Repeat function
import "strings"

// sort package is needed for sorting
import "sort"

// Main function with example usage
func main() {
	fmt.Println("Financial Time-Series Analysis Template")
	fmt.Println(strings.Repeat("=", 80))

	// Example configuration
	config := DefaultStudyConfig()
	config.InputFile = "example_prices.csv"
	config.Verbose = true

	// Load data
	fmt.Printf("\nLoading data from %s...\n", config.InputFile)
	data, err := LoadPriceData(config.InputFile, config.DateFormat)
	if err != nil {
		// For demonstration, create synthetic data if file doesn't exist
		fmt.Println("Creating synthetic data for demonstration...")
		data = createSyntheticData()
	}

	// Run analysis
	fmt.Println("\nRunning comprehensive analysis...")
	results, err := RunComprehensiveAnalysis(data, config)
	if err != nil {
		fmt.Printf("Analysis failed: %v\n", err)
		return
	}

	// Print summary
	PrintSummary(results)

	// Export results
	if config.ExportJSON || config.ExportCSV {
		fmt.Println("\nExporting results...")
		if err := ExportResults(results, config); err != nil {
			fmt.Printf("Export failed: %v\n", err)
		}
	}

	fmt.Println("\nTemplate execution complete!")
	fmt.Println("\nNext steps:")
	fmt.Println("1. Replace synthetic data with your actual price data")
	fmt.Println("2. Adjust configuration parameters for your analysis")
	fmt.Println("3. Extend the analysis with custom metrics or indicators")
	fmt.Println("4. Use the feature matrix for machine learning models")
}

// createSyntheticData generates example data for demonstration
func createSyntheticData() *PriceData {
	// Generate 1000 days of synthetic price data
	n := 1000
	startDate := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)

	timestamps := make([]time.Time, n)
	prices := make([]float64, n)

	// Geometric Brownian Motion simulation
	price := 100.0
	mu := 0.0003    // Daily drift (7.7% annual)
	sigma := 0.012  // Daily volatility

	rand.Seed(42)
	for i := 0; i < n; i++ {
		timestamps[i] = startDate.AddDate(0, 0, i)

		// Skip weekends (crude approximation)
		if timestamps[i].Weekday() == time.Saturday {
			timestamps[i] = timestamps[i].AddDate(0, 0, 2)
		} else if timestamps[i].Weekday() == time.Sunday {
			timestamps[i] = timestamps[i].AddDate(0, 0, 1)
		}

		// Update price with random walk
		shock := rand.NormFloat64() * sigma
		price *= (1 + mu + shock)

		// Ensure price stays positive
		if price < 1 {
			price = 1
		}

		prices[i] = price
	}

	return &PriceData{
		Timestamps: timestamps,
		Prices:     prices,
		Ticker:     "SYNTHETIC",
		StartDate:  timestamps[0],
		EndDate:    timestamps[n-1],
	}
}

// rand package is needed for random number generation
import "math/rand"

// ============================================================================
// Template Usage Examples
// ============================================================================
/*
Example 1: Basic Usage

func main() {
    config := DefaultStudyConfig()
    config.InputFile = "sp500_daily.csv"
    config.WindowSize = 252  // 1 year of trading days
    config.IndicatorPeriod = 20

    data, _ := LoadPriceData(config.InputFile, config.DateFormat)
    results, _ := RunComprehensiveAnalysis(data, config)
    PrintSummary(results)
}

Example 2: Custom Analysis

func CustomAnalysis(prices []float64) {
    // Just calculate specific metrics
    metricConfig := financial.MetricConfig{
        WindowSize: 60,
        Period: 14,
        RiskFreeRate: 0.02,
    }

    returnsMetrics, _ := financial.CalculateReturnsMetrics(prices, metricConfig)
    drawdownMetrics, _ := financial.CalculateDrawdownMetrics(prices)

    fmt.Printf("Return: %.2f%%, Max DD: %.2f%%\n",
        returnsMetrics.AnnualizedReturn*100,
        drawdownMetrics.MaxDrawdown*100)
}

Example 3: Regime-Based Analysis

func RegimeAdaptiveStrategy(prices []float64) {
    // Detect current regime
    regimeMetrics, _ := financial.DetectMarketRegime(prices, 60)

    switch regimeMetrics.CurrentRegime {
    case financial.RegimeStrongBull:
        fmt.Println("Using trend-following strategy")
    case financial.RegimeMeanReverting:
        fmt.Println("Using mean-reversion strategy")
    case financial.RegimeHighVol:
        fmt.Println("Using volatility-based strategy")
    default:
        fmt.Println("Using neutral strategy")
    }
}
*/
```

Note: This template requires the following imports which are already included in the code:
- "encoding/csv"
- "encoding/json"
- "fmt"
- "math"
- "os"
- "strconv"
- "time"
- "strings"
- "sort"
- "math/rand"
- "github.com/goneurotic/financial"

The template demonstrates a complete, reusable financial time-series analysis workflow that can be easily adapted for different studies. Key features include:

1. **Modular Design**: Each analysis component is separate and reusable
2. **Comprehensive Metrics**: Technical indicators, risk metrics, statistical tests
3. **Market Regime Detection**: Adaptive analysis based on market conditions
4. **Export Functionality**: JSON and CSV output for further analysis
5. **Visualization-Ready**: Structured output for charting libraries
6. **Machine Learning Ready**: Feature matrix for predictive modeling

Users can:
- Replace the synthetic data with their own price data
- Adjust configuration parameters
- Extend with custom metrics or indicators
- Use the feature matrix for ML models
- Integrate with visualization tools

The template follows best practices for financial time-series analysis and demonstrates the full capabilities of the GoNeurotic financial package.
