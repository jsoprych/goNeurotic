# GoNeurotic Financial Time-Series Analysis Guide

## 🎯 Overview

GoNeurotic provides a comprehensive, production-ready financial time-series analysis system built in Go. This system combines traditional technical analysis with modern statistical methods and AI-driven pattern discovery, offering a unified framework for both educational exploration and professional financial analysis.

### Key Design Principles

1. **Modularity**: Each component is independent and reusable
2. **Performance**: Optimized for large datasets with efficient algorithms
3. **Completeness**: Covers technical indicators, risk metrics, statistical tests, and market regime detection
4. **Educational Value**: Well-documented with mathematical formulas and interpretation guides
5. **Production Readiness**: Error handling, alignment utilities, and export capabilities

## 📊 Core Components

### 1. Technical Indicators (`pkg/financial/indicators.go`)
Traditional price-based analysis tools with clear mathematical foundations:

- **Moving Averages**: SMA, EMA, WMA for trend identification
- **Momentum Indicators**: RSI, MACD, Stochastic Oscillator
- **Volatility Indicators**: Bollinger Bands, ATR (Average True Range)
- **Trend Indicators**: ADX (Average Directional Index)
- **Volume Indicators**: OBV (On-Balance Volume), MFI (Money Flow Index)
- **Pattern Recognition**: Support/Resistance, Double Top/Bottom detection

### 2. Unified Metrics System (`pkg/financial/unified_metrics.go`)
Modern statistical and risk analysis framework:

- **Returns Metrics**: Arithmetic/geometric means, annualization, skewness, kurtosis
- **Risk Metrics**: Sharpe/Sortino ratios, max drawdown, Calmar ratio, Value at Risk (VaR)
- **Statistical Tests**: Normality (Jarque-Bera), stationarity, autocorrelation
- **Market Regime Detection**: Bull/bear/sideways classification with confidence scores
- **Performance Ratios**: Comprehensive risk-adjusted performance measurement

### 3. Alignment Utilities (`pkg/financial/alignment.go`)
Essential for handling time-series data alignment:

- **Indicator Alignment**: Match indicator outputs with original price series
- **Time Series Alignment**: Align multiple series by timestamp
- **Data Imputation**: Forward/backward fill, linear interpolation
- **Feature Engineering**: Lag creation, differencing, rolling statistics
- **Feature Matrix Creation**: Prepare data for machine learning models

### 4. Adaptive Trading System (`pkg/financial/adaptive/`)
AI-driven strategy selection framework:

- **Market Regime Classification**: Detect current market conditions
- **Strategy Interfaces**: Define trading strategies with performance tracking
- **Meta-Learning**: Select optimal strategies based on market conditions
- **Portfolio Management**: Position sizing, risk management, trade execution

## 🚀 Getting Started

### Installation

```go
import "github.com/goneurotic/financial"
```

### Basic Usage Example

```go
package main

import (
    "fmt"
    "github.com/goneurotic/financial"
)

func main() {
    // Load price data (simplified example)
    prices := []float64{100, 101, 102, 103, 104, 105, 106, 107, 108, 109}
    
    // 1. Calculate technical indicators
    sma, _ := financial.SMA(prices, 3)
    rsi, _ := financial.RSI(prices, 14)
    
    // 2. Calculate risk metrics
    config := financial.DefaultMetricConfig()
    returnsMetrics, _ := financial.CalculateReturnsMetrics(prices, config)
    
    // 3. Detect market regime
    regimeMetrics, _ := financial.DetectMarketRegime(prices, 20)
    
    fmt.Printf("Latest SMA(3): %.2f\n", sma[len(sma)-1])
    fmt.Printf("Annualized Return: %.2f%%\n", returnsMetrics.AnnualizedReturn*100)
    fmt.Printf("Current Market Regime: %s\n", regimeMetrics.CurrentRegime)
}
```

## 📈 Comprehensive Analysis Workflow

### Step 1: Data Preparation

```go
func LoadAndPrepareData(filename string) (*financial.TimeSeriesData, error) {
    // Load from CSV, database, or API
    // Handle missing values, outliers, and alignment
    // Calculate returns and log returns
}
```

### Step 2: Technical Analysis

```go
func TechnicalAnalysis(prices []float64, config financial.MetricConfig) {
    // Moving averages for trend
    sma20, _ := financial.SMA(prices, 20)
    ema12, _ := financial.EMA(prices, 12)
    
    // Momentum indicators
    rsi14, _ := financial.RSI(prices, 14)
    macd, signal, histogram, _ := financial.MACD(prices, 12, 26, 9)
    
    // Volatility indicators
    upperBB, middleBB, lowerBB, _ := financial.BollingerBands(prices, 20, 2.0)
    atr14, _ := financial.ATR(high, low, close, 14) // Requires OHLC data
    
    // Align all indicators
    alignments, _ := financial.AlignMultipleIndicators(prices, map[string][]float64{
        "SMA20": sma20,
        "RSI14": rsi14,
        // ... other indicators
    }, map[string]int{
        "SMA20": 19,
        "RSI14": 13,
        // ... corresponding lookbacks
    })
}
```

### Step 3: Risk and Performance Analysis

```go
func RiskAnalysis(returns []float64, config financial.MetricConfig) {
    // Returns statistics
    returnsMetrics, _ := financial.CalculateReturnsMetrics(prices, config)
    
    // Drawdown analysis
    drawdownMetrics, _ := financial.CalculateDrawdownMetrics(prices)
    
    // Performance ratios
    performanceRatios := financial.CalculatePerformanceRatios(returns, config)
    
    // Value at Risk
    fmt.Printf("95%% VaR: %.2f%%\n", returnsMetrics.VaR95*100)
    fmt.Printf("Max Drawdown: %.2f%%\n", drawdownMetrics.MaxDrawdown*100)
    fmt.Printf("Sharpe Ratio: %.2f\n", performanceRatios.SharpeRatio)
}
```

### Step 4: Statistical Validation

```go
func StatisticalValidation(returns []float64) {
    // Normality test
    normality := financial.TestNormality(returns)
    fmt.Printf("Data appears normal: %v (JB: %.2f)\n", 
               normality.IsNormal, normality.JarqueBera)
    
    // Stationarity test
    stationarity := financial.TestStationarity(prices, returns)
    fmt.Printf("Returns stationary: %v\n", stationarity.ReturnStationary)
    
    // Autocorrelation analysis
    autocorr := financial.AutocorrelationTest(returns, 5)
    for lag, corr := range autocorr {
        fmt.Printf("Lag %d autocorrelation: %.3f\n", lag, corr)
    }
}
```

### Step 5: Market Regime Analysis

```go
func RegimeAnalysis(prices []float64, window int) {
    // Detect current regime
    regimeMetrics, _ := financial.DetectMarketRegime(prices, window)
    
    fmt.Printf("Current Regime: %s\n", regimeMetrics.CurrentRegime)
    fmt.Printf("Regime Confidence: %.1f%%\n", regimeMetrics.RegimeStrength*100)
    fmt.Printf("Trend Strength: %.2f\n", regimeMetrics.TrendStrength)
    fmt.Printf("Volatility Level: %.2f\n", regimeMetrics.VolatilityLevel)
    
    // Adaptive strategy based on regime
    switch regimeMetrics.CurrentRegime {
    case financial.RegimeStrongBull:
        // Use trend-following strategies
    case financial.RegimeMeanReverting:
        // Use mean-reversion strategies
    case financial.RegimeHighVol:
        // Reduce position sizes, use volatility strategies
    default:
        // Neutral or defensive positioning
    }
}
```

## 🤖 AI-Enhanced Analysis

### Neural Network as Adaptive Indicator

```go
func NeuralNetworkAnalysis(prices []float64, network *neural.Network) {
    // Create feature vectors from financial metrics
    features := createFinancialFeatures(prices)
    
    // Use trained network to generate signals
    signal := network.Predict(features)
    
    // Interpret network output
    if signal[0] > 0.7 {
        fmt.Println("Strong buy signal from neural network")
    } else if signal[0] < 0.3 {
        fmt.Println("Strong sell signal from neural network")
    } else {
        fmt.Println("Neutral signal from neural network")
    }
}
```

### Meta-Learning for Strategy Selection

```go
type AdaptiveTradingSystem struct {
    strategies map[string]TradingStrategy
    regimeDetector RegimeDetector
    performanceTracker PerformanceTracker
}

func (ats *AdaptiveTradingSystem) SelectStrategy(prices []float64) TradingStrategy {
    // 1. Detect current market regime
    regime := ats.regimeDetector.Detect(prices)
    
    // 2. Filter strategies suitable for this regime
    suitableStrategies := filterStrategies(ats.strategies, regime)
    
    // 3. Select best performing strategy in similar historical conditions
    bestStrategy := ats.performanceTracker.SelectBest(suitableStrategies, regime)
    
    return bestStrategy
}
```

## 📊 Example Studies

### Study 1: Trend-Following System Evaluation

```go
func EvaluateTrendFollowing(prices []float64) {
    // Calculate moving average crossovers
    sma50, _ := financial.SMA(prices, 50)
    sma200, _ := financial.SMA(prices, 200)
    
    // Generate signals
    signals := make([]string, len(sma50))
    for i := 0; i < len(sma50); i++ {
        if sma50[i] > sma200[i] {
            signals[i] = "BUY"
        } else {
            signals[i] = "SELL"
        }
    }
    
    // Calculate performance
    returns := calculateStrategyReturns(prices, signals)
    metrics, _ := financial.CalculateReturnsMetrics(returns, config)
    
    fmt.Printf("Trend-following Sharpe Ratio: %.2f\n", metrics.SharpeRatio)
}
```

### Study 2: Mean-Reversion Strategy Backtest

```go
func BacktestMeanReversion(prices []float64) {
    // Calculate Bollinger Bands
    upper, middle, lower, _ := financial.BollingerBands(prices, 20, 2.0)
    
    // Mean-reversion signals
    signals := make([]string, len(prices))
    for i := 20; i < len(prices); i++ {
        if prices[i] < lower[i-20] {
            signals[i] = "BUY"  // Oversold
        } else if prices[i] > upper[i-20] {
            signals[i] = "SELL" // Overbought
        }
    }
    
    // Analyze regime-specific performance
    regimes := financial.DetectMarketRegimes(prices, 60)
    analyzeRegimePerformance(signals, regimes)
}
```

### Study 3: Volatility-Regime Adaptive Portfolio

```go
func AdaptivePortfolio(prices []float64) {
    // Detect volatility regime
    regimeMetrics, _ := financial.DetectMarketRegime(prices, 60)
    
    // Adjust portfolio based on volatility
    var allocation map[string]float64
    switch {
    case regimeMetrics.VolatilityLevel > 0.7:
        // High volatility: defensive allocation
        allocation = map[string]float64{
            "Cash": 0.4,
            "Bonds": 0.4,
            "Stocks": 0.2,
        }
    case regimeMetrics.VolatilityLevel < 0.3:
        // Low volatility: aggressive allocation
        allocation = map[string]float64{
            "Cash": 0.1,
            "Bonds": 0.2,
            "Stocks": 0.7,
        }
    default:
        // Normal volatility: balanced allocation
        allocation = map[string]float64{
            "Cash": 0.2,
            "Bonds": 0.3,
            "Stocks": 0.5,
        }
    }
    
    fmt.Printf("Adaptive allocation: %v\n", allocation)
}
```

## 🔧 Advanced Usage

### Custom Indicator Development

```go
type CustomIndicator struct {
    period int
    buffer []float64
}

func (ci *CustomIndicator) Calculate(prices []float64) ([]float64, error) {
    if len(prices) < ci.period {
        return nil, fmt.Errorf("insufficient data")
    }
    
    values := make([]float64, len(prices)-ci.period+1)
    for i := 0; i <= len(prices)-ci.period; i++ {
        // Custom calculation logic
        window := prices[i : i+ci.period]
        values[i] = ci.customFormula(window)
    }
    
    return values, nil
}

func (ci *CustomIndicator) customFormula(window []float64) float64 {
    // Implement your custom indicator logic
    // Example: Custom volatility measure
    mean := financial.mean(window)
    sum := 0.0
    for _, price := range window {
        sum += math.Abs(price - mean)
    }
    return sum / float64(len(window))
}
```

### Real-time Analysis Pipeline

```go
type RealTimeAnalyzer struct {
    priceBuffer []float64
    maxBufferSize int
    indicators map[string]Indicator
    regimeDetector RegimeDetector
}

func (rta *RealTimeAnalyzer) ProcessNewPrice(price float64, timestamp time.Time) {
    // Add to buffer
    rta.priceBuffer = append(rta.priceBuffer, price)
    if len(rta.priceBuffer) > rta.maxBufferSize {
        rta.priceBuffer = rta.priceBuffer[1:]
    }
    
    // Update indicators when enough data
    if len(rta.priceBuffer) >= 20 {
        // Calculate all indicators
        for name, indicator := range rta.indicators {
            values, _ := indicator.Calculate(rta.priceBuffer)
            // Process indicator values...
        }
        
        // Update market regime
        regime, _ := rta.regimeDetector.Detect(rta.priceBuffer)
        
        // Generate signals or alerts
        rta.generateSignals(regime)
    }
}
```

## 📋 Best Practices

### 1. Data Quality First
- Always check for NaN/inf values
- Handle missing data appropriately (imputation vs. exclusion)
- Validate timestamp consistency and sorting
- Be aware of look-ahead bias in backtesting

### 2. Proper Indicator Alignment
```go
// Always align indicators with price data
alignment, err := financial.AlignIndicator(prices, indicatorValues, lookback, 0)
if err != nil {
    // Handle error appropriately
}

// Use aligned values for analysis
alignedValues := alignment.AlignedValues
```

### 3. Comprehensive Validation
- Use walk-forward validation for time-series models
- Test across multiple market regimes
- Validate statistical assumptions (normality, stationarity)
- Conduct sensitivity analysis on parameters

### 4. Risk Management Integration
- Always calculate position sizes based on volatility (e.g., ATR)
- Implement drawdown limits
- Use Value at Risk for portfolio risk assessment
- Monitor correlation between strategies

### 5. Performance Attribution
- Separate skill from luck using statistical tests
- Attribute returns to specific strategies or regimes
- Track turnover and transaction costs
- Compare against appropriate benchmarks

## 🚨 Common Pitfalls to Avoid

1. **Overfitting**: Avoid optimizing too many parameters on limited data
2. **Look-Ahead Bias**: Ensure no future information leaks into calculations
3. **Survivorship Bias**: Use datasets that include delisted securities
4. **Transaction Costs**: Account for slippage, commissions, and market impact
5. **Regime Changes**: Strategies that work in one regime may fail in another
6. **Data Snooping**: Avoid testing too many strategies on the same data

## 📚 Educational Resources

### Built-in Tutorials
1. `examples/reusable_studies/comprehensive_analysis.go` - Complete analysis template
2. `examples/ai_trading_tutorial/neural_discovery_tutorial.go` - AI-powered analysis
3. `pkg/financial/tests/harness_test.go` - Test cases and examples

### Key Concepts to Master
1. **Time-Series Properties**: Stationarity, autocorrelation, seasonality
2. **Risk Metrics**: Sharpe ratio, maximum drawdown, Value at Risk
3. **Statistical Tests**: Normality, stationarity, hypothesis testing
4. **Market Microstructure**: Bid-ask spreads, liquidity, market impact
5. **Portfolio Theory**: Modern Portfolio Theory, efficient frontier, risk parity

## 🔮 Future Extensions

### Planned Features
1. **Alternative Data Integration**: News sentiment, options data, economic indicators
2. **High-Frequency Analysis**: Tick data processing, order book analysis
3. **Advanced Machine Learning**: LSTM networks, attention mechanisms, reinforcement learning
4. **Portfolio Optimization**: Mean-variance optimization, risk parity, Black-Litterman
5. **Live Trading Integration**: Broker API connectors, execution algorithms

### Research Directions
1. **Explainable AI**: Interpretable machine learning models for finance
2. **Causal Inference**: Understanding cause-effect relationships in markets
3. **Network Analysis**: Interconnectedness of financial markets
4. **Behavioral Finance**: Incorporating investor psychology into models
5. **Sustainable Finance**: ESG integration and impact investing analytics

## 🎯 Conclusion

GoNeurotic's financial time-series analysis system provides a comprehensive toolkit for both traditional quantitative analysis and modern AI-driven approaches. By combining technical indicators, statistical tests, risk metrics, and market regime detection with a modular, reusable architecture, it supports a wide range of financial analysis tasks from basic educational exercises to sophisticated trading system development.

The system's design emphasizes:
- **Practical utility**: Ready-to-use functions with clear documentation
- **Educational value**: Mathematical formulas and interpretation guides
- **Production readiness**: Error handling, alignment utilities, export capabilities
- **Extensibility**: Easy to add custom indicators and analysis methods

Whether you're learning quantitative finance, developing trading strategies, or conducting financial research, this system provides the foundational tools needed for rigorous, reproducible analysis.

---

*For questions, issues, or contributions, please refer to the main GoNeurotic documentation or open an issue on the GitHub repository.*