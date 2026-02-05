package adaptive

import (
	"fmt"
	"math"
	"time"
)

// ============================================================================
// Daily OHLCV Data Structures
// ============================================================================

// DailyOHLCV represents a single day's Open, High, Low, Close, Volume data
type DailyOHLCV struct {
	Date   time.Time
	Open   float64
	High   float64
	Low    float64
	Close  float64
	Volume float64
	// Additional derived metrics
	Returns    float64 // Daily returns (Close/PreviousClose - 1)
	Range      float64 // Daily range (High - Low) / PreviousClose
	Body       float64 // Candle body (Close - Open) / PreviousClose
	UpperShadow float64 // Upper shadow (High - max(Open, Close)) / PreviousClose
	LowerShadow float64 // Lower shadow (min(Open, Close) - Low) / PreviousClose
}

// DailyOHLCVSeries represents a time series of daily OHLCV data
type DailyOHLCVSeries struct {
	Data        []DailyOHLCV
	Ticker      string
	Description string
	// Precomputed technical indicators for performance
	SMAs       map[int][]float64 // Simple Moving Averages by period
	EMAs       map[int][]float64 // Exponential Moving Averages by period
	RSI        map[int][]float64 // RSI by period
	ATR        map[int][]float64 // Average True Range by period
	// Derived features
	Volatility []float64 // Rolling volatility
	Trend      []float64 // Trend strength indicators
	Momentum   []float64 // Momentum indicators
}

// NewDailyOHLCVSeries creates a new daily OHLCV series
func NewDailyOHLCVSeries(ticker string, description string) *DailyOHLCVSeries {
	return &DailyOHLCVSeries{
		Ticker:      ticker,
		Description: description,
		Data:        []DailyOHLCV{},
		SMAs:        make(map[int][]float64),
		EMAs:        make(map[int][]float64),
		RSI:         make(map[int][]float64),
		ATR:         make(map[int][]float64),
	}
}

// AddDailyData adds a new day's OHLCV data and calculates derived metrics
func (s *DailyOHLCVSeries) AddDailyData(date time.Time, open, high, low, close, volume float64) {
	ohlcv := DailyOHLCV{
		Date:   date,
		Open:   open,
		High:   high,
		Low:    low,
		Close:  close,
		Volume: volume,
	}

	// Calculate derived metrics if we have previous data
	if len(s.Data) > 0 {
		prev := s.Data[len(s.Data)-1].Close
		if prev > 0 {
			ohlcv.Returns = (close - prev) / prev
			ohlcv.Range = (high - low) / prev
			ohlcv.Body = (close - open) / prev

			// Calculate shadows
			upper := math.Max(open, close)
			lower := math.Min(open, close)
			ohlcv.UpperShadow = (high - upper) / prev
			ohlcv.LowerShadow = (lower - low) / prev
		}
	}

	s.Data = append(s.Data, ohlcv)
}

// CalculateTechnicalIndicators precomputes common technical indicators
func (s *DailyOHLCVSeries) CalculateTechnicalIndicators() {
	if len(s.Data) < 20 {
		return
	}

	// Extract price series for calculations
	closes := make([]float64, len(s.Data))
	highs := make([]float64, len(s.Data))
	lows := make([]float64, len(s.Data))
	volumes := make([]float64, len(s.Data))

	for i, d := range s.Data {
		closes[i] = d.Close
		highs[i] = d.High
		lows[i] = d.Low
		volumes[i] = d.Volume
	}

	// Calculate SMAs for common periods
	smaPeriods := []int{5, 10, 20, 50, 200}
	for _, period := range smaPeriods {
		s.SMAs[period] = calculateSMA(closes, period)
	}

	// Calculate EMAs for common periods
	emaPeriods := []int{9, 12, 26, 50}
	for _, period := range emaPeriods {
		s.EMAs[period] = calculateEMA(closes, period)
	}

	// Calculate RSI
	rsiPeriods := []int{14, 21}
	for _, period := range rsiPeriods {
		s.RSI[period] = calculateRSI(closes, period)
	}

	// Calculate ATR
	atrPeriods := []int{14, 21}
	for _, period := range atrPeriods {
		s.ATR[period] = calculateATR(highs, lows, closes, period)
	}

	// Calculate volatility (20-day rolling std dev of returns)
	s.Volatility = calculateRollingVolatility(closes, 20)

	// Calculate trend strength (ADX-like)
	s.Trend = calculateTrendStrength(highs, lows, closes, 14)

	// Calculate momentum
	s.Momentum = calculateMomentum(closes, 10)
}

// GetFeatures extracts features for machine learning/adaptive systems
func (s *DailyOHLCVSeries) GetFeatures(index int) *DailyFeatures {
	if index < 0 || index >= len(s.Data) {
		return nil
	}

	features := &DailyFeatures{
		PriceFeatures: PriceFeatures{
			Returns:      s.Data[index].Returns,
			Range:        s.Data[index].Range,
			Body:         s.Data[index].Body,
			UpperShadow:  s.Data[index].UpperShadow,
			LowerShadow:  s.Data[index].LowerShadow,
			Volume:       s.Data[index].Volume,
			VolumeChange: 0,
		},
		TechnicalFeatures: TechnicalFeatures{},
		MarketRegime:      RegimeUnknown,
		Date:              s.Data[index].Date,
	}

	// Calculate volume change if we have previous data
	if index > 0 {
		prevVolume := s.Data[index-1].Volume
		if prevVolume > 0 {
			features.VolumeChange = (s.Data[index].Volume - prevVolume) / prevVolume
		}
	}

	// Add technical indicators if calculated
	if len(s.SMAs) > 0 {
		if sma20, ok := s.SMAs[20]; ok && index >= 20 {
			features.TechnicalFeatures.SMA20 = sma20[index-20]
			features.TechnicalFeatures.PriceVsSMA20 = (s.Data[index].Close - sma20[index-20]) / sma20[index-20]
		}
		if sma50, ok := s.SMAs[50]; ok && index >= 50 {
			features.TechnicalFeatures.SMA50 = sma50[index-50]
		}
		if sma200, ok := s.SMAs[200]; ok && index >= 200 {
			features.TechnicalFeatures.SMA200 = sma200[index-200]
		}
	}

	if len(s.EMAs) > 0 {
		if ema12, ok := s.EMAs[12]; ok && index >= 12 {
			features.TechnicalFeatures.EMA12 = ema12[index-12]
		}
		if ema26, ok := s.EMAs[26]; ok && index >= 26 {
			features.TechnicalFeatures.EMA26 = ema26[index-26]
		}
	}

	if len(s.RSI) > 0 {
		if rsi14, ok := s.RSI[14]; ok && index >= 14 {
			features.TechnicalFeatures.RSI14 = rsi14[index-14]
		}
	}

	if len(s.ATR) > 0 {
		if atr14, ok := s.ATR[14]; ok && index >= 14 {
			features.TechnicalFeatures.ATR14 = atr14[index-14]
			features.TechnicalFeatures.RangeVsATR = s.Data[index].Range / atr14[index-14]
		}
	}

	// Add volatility and trend if calculated
	if len(s.Volatility) > 0 && index >= 20 {
		features.TechnicalFeatures.Volatility20 = s.Volatility[index-20]
	}

	if len(s.Trend) > 0 && index >= 14 {
		features.TechnicalFeatures.TrendStrength = s.Trend[index-14]
	}

	if len(s.Momentum) > 0 && index >= 10 {
		features.TechnicalFeatures.Momentum10 = s.Momentum[index-10]
	}

	// Detect market regime
	features.MarketRegime = s.detectMarketRegime(index)

	return features
}

// ============================================================================
// Feature Structures
// ============================================================================

// DailyFeatures represents the complete feature set for a trading day
type DailyFeatures struct {
	PriceFeatures      PriceFeatures
	TechnicalFeatures  TechnicalFeatures
	MarketRegime       MarketRegime
	Date               time.Time
}

// PriceFeatures contains raw price and volume features
type PriceFeatures struct {
	Returns      float64 // Daily return
	Range        float64 // Daily range (normalized)
	Body         float64 // Candle body (normalized)
	UpperShadow  float64 // Upper shadow (normalized)
	LowerShadow  float64 // Lower shadow (normalized)
	Volume       float64 // Daily volume
	VolumeChange float64 // Volume change from previous day
}

// TechnicalFeatures contains calculated technical indicators
type TechnicalFeatures struct {
	SMA20        float64 // 20-day Simple Moving Average
	SMA50        float64 // 50-day SMA
	SMA200       float64 // 200-day SMA
	EMA12        float64 // 12-day Exponential Moving Average
	EMA26        float64 // 26-day EMA
	RSI14        float64 // 14-day Relative Strength Index
	ATR14        float64 // 14-day Average True Range
	PriceVsSMA20 float64 // Price relative to 20-day SMA
	RangeVsATR   float64 // Daily range relative to ATR
	Volatility20 float64 // 20-day rolling volatility
	TrendStrength float64 // ADX-like trend strength
	Momentum10   float64 // 10-day momentum
}

// ============================================================================
// Market Regime Detection
// ============================================================================

// MarketRegime represents different market conditions
type MarketRegime string

const (
	RegimeUnknown      MarketRegime = "unknown"
	RegimeStrongBull   MarketRegime = "strong_bull"
	RegimeWeakBull     MarketRegime = "weak_bull"
	RegimeStrongBear   MarketRegime = "strong_bear"
	RegimeWeakBear     MarketRegime = "weak_bear"
	RegimeSideways     MarketRegime = "sideways"
	RegimeHighVol      MarketRegime = "high_volatility"
	RegimeLowVol       MarketRegime = "low_volatility"
	RegimeTrending     MarketRegime = "trending"
	RegimeMeanReverting MarketRegime = "mean_reverting"
)

// detectMarketRegime determines the market regime based on recent data
func (s *DailyOHLCVSeries) detectMarketRegime(index int) MarketRegime {
	if index < 50 {
		return RegimeUnknown
	}

	// Look at last 20 days for regime detection
	lookback := 20
	start := max(0, index-lookback+1)

	// Calculate statistics for the lookback period
	var returns []float64
	var closes []float64
	var highs []float64
	var lows []float64

	for i := start; i <= index; i++ {
		returns = append(returns, s.Data[i].Returns)
		closes = append(closes, s.Data[i].Close)
		highs = append(highs, s.Data[i].High)
		lows = append(lows, s.Data[i].Low)
	}

	// Calculate metrics
	meanReturn := mean(returns)
	stdDevReturn := stdDev(returns)

	// Calculate trend
	priceChange := (closes[len(closes)-1] - closes[0]) / closes[0]
	absPriceChange := math.Abs(priceChange)

	// Calculate volatility level
	volatilityLevel := stdDevReturn * math.Sqrt(252) // Annualized

	// Determine regime based on rules
	if volatilityLevel > 0.25 {
		return RegimeHighVol
	}

	if volatilityLevel < 0.1 {
		return RegimeLowVol
	}

	// Check for strong trends
	if absPriceChange > 0.08 { // > 8% move in 20 days
		if priceChange > 0 {
			return RegimeStrongBull
		} else {
			return RegimeStrongBear
		}
	}

	// Check for weaker trends
	if absPriceChange > 0.03 { // 3-8% move
		if priceChange > 0 {
			return RegimeWeakBull
		} else {
			return RegimeWeakBear
		}
	}

	// Check for mean reversion vs trending
	if isMeanReverting(returns) {
		return RegimeMeanReverting
	}

	// Default to sideways
	return RegimeSideways
}

// ============================================================================
// Trading Strategy Interface
// ============================================================================

// TradingStrategy defines the interface for daily trading strategies
type TradingStrategy interface {
	Name() string
	Description() string
	Category() StrategyCategory
	// Generate signal based on daily OHLCV data and features
	GenerateSignal(features *DailyFeatures, series *DailyOHLCVSeries, positionSize float64) TradeSignal
	// Calculate position size based on risk
	CalculatePositionSize(accountSize, riskPercent, stopLoss float64) float64
	// Check if strategy is suitable for current market regime
	IsSuitableForRegime(regime MarketRegime) bool
	// Performance metrics
	GetPerformance() StrategyPerformance
	// Update performance after trade
	UpdatePerformance(trade *Trade)
}

// StrategyCategory categorizes trading strategies
type StrategyCategory string

const (
	CategoryTrendFollowing StrategyCategory = "trend_following"
	CategoryMeanReversion  StrategyCategory = "mean_reversion"
	CategoryBreakout       StrategyCategory = "breakout"
	CategoryMomentum       StrategyCategory = "momentum"
	CategoryVolatility     StrategyCategory = "volatility"
	CategoryArbitrage      StrategyCategory = "arbitrage"
	CategoryMarketMaking   StrategyCategory = "market_making"
)

// TradeSignal represents a trading decision
type TradeSignal struct {
	Action       TradeAction
	Confidence   float64 // 0 to 1
	Price        float64 // Entry price
	StopLoss     float64
	TakeProfit   float64
	Size         float64 // Position size
	Reason       string
	StrategyName string
}

// TradeAction defines possible trading actions
type TradeAction string

const (
	ActionBuy      TradeAction = "buy"
	ActionSell     TradeAction = "sell"
	ActionShort    TradeAction = "short"
	ActionCover    TradeAction = "cover"
	ActionHold     TradeAction = "hold"
	ActionReduce   TradeAction = "reduce"
	ActionIncrease TradeAction = "increase"
)

// StrategyPerformance tracks strategy performance metrics
type StrategyPerformance struct {
	TotalTrades      int
	WinningTrades    int
	LosingTrades     int
	TotalReturn      float64
	MaxDrawdown      float64
	SharpeRatio      float64
	SortinoRatio     float64
	WinRate          float64
	ProfitFactor     float64 // Gross profits / gross losses
	AvgWin           float64
	AvgLoss          float64
	AvgTrade         float64
	BestTrade        float64
	WorstTrade       float64
}

// ============================================================================
// Adaptive Trading System Core
// ============================================================================

// AdaptiveTradingSystem manages multiple strategies and selects the best one
type AdaptiveTradingSystem struct {
	Strategies      map[string]TradingStrategy
	CurrentStrategy TradingStrategy
	Performance     map[string]StrategyPerformance
	MarketRegime    MarketRegime
	AccountSize     float64
	RiskPerTrade    float64 // e.g., 0.02 for 2%
	// Meta-learning components
	RegimeHistory   []MarketRegime
	StrategyHistory []StrategySelection
	PerformanceLog  []PerformanceRecord
	// Configuration
	MinDataPoints  int
	LookbackPeriod int
}

// StrategySelection records which strategy was selected and why
type StrategySelection struct {
	Date           time.Time
	Strategy       string
	Regime         MarketRegime
	Confidence     float64
	Features       *DailyFeatures
	Alternatives   []string
	Reason         string
}

// PerformanceRecord tracks daily performance
type PerformanceRecord struct {
	Date            time.Time
	Strategy        string
	Regime          MarketRegime
	DailyReturn     float64
	CumulativeReturn float64
	Drawdown        float64
	PositionSize    float64
	Signal          TradeSignal
}

// NewAdaptiveTradingSystem creates a new adaptive trading system
func NewAdaptiveTradingSystem(accountSize, riskPerTrade float64) *AdaptiveTradingSystem {
	return &AdaptiveTradingSystem{
		Strategies:      make(map[string]TradingStrategy),
		Performance:     make(map[string]StrategyPerformance),
		AccountSize:     accountSize,
		RiskPerTrade:    riskPerTrade,
		RegimeHistory:   []MarketRegime{},
		StrategyHistory: []StrategySelection{},
		PerformanceLog:  []PerformanceRecord{},
		MinDataPoints:   50,
		LookbackPeriod:  20,
	}
}

// RegisterStrategy adds a strategy to the system
func (ats *AdaptiveTradingSystem) RegisterStrategy(strategy TradingStrategy) {
	ats.Strategies[strategy.Name()] = strategy
	ats.Performance[strategy.Name()] = StrategyPerformance{}
}

// SelectOptimalStrategy chooses the best strategy for current market conditions
func (ats *AdaptiveTradingSystem) SelectOptimalStrategy(features *DailyFeatures, series *DailyOHLCVSeries) TradingStrategy {
	// Update market regime
	ats.MarketRegime = features.MarketRegime
	ats.RegimeHistory = append(ats.RegimeHistory, features.MarketRegime)

	// Filter strategies suitable for current regime
	var suitableStrategies []TradingStrategy
	for _, strategy := range ats.Strategies {
		if strategy.IsSuitableForRegime(features.MarketRegime) {
			suitableStrategies = append(suitableStrategies, strategy)
		}
	}

	if len(suitableStrategies) == 0 {
		// No suitable strategies, use default or hold
		return nil
	}

	// If we have performance data, use meta-learning
	if len(ats.PerformanceLog) > ats.MinDataPoints {
		// Use performance-based selection
		return ats.selectByPerformance(features, suitableStrategies)
	}

	// Otherwise use rule-based selection
	return ats.selectByRules(features, suitableStrategies)
}

// selectByPerformance selects strategy based on historical performance in similar conditions
func (ats *AdaptiveTradingSystem) selectByPerformance(features *DailyFeatures, candidates []TradingStrategy) TradingStrategy {
	// Simple implementation: choose strategy with best recent Sharpe ratio
	bestStrategy := candidates[0]
	bestSharpe := -math.MaxFloat64

	for _, strategy := range candidates {
		perf := ats.Performance[strategy.Name()]
		if perf.SharpeRatio > bestSharpe {
			bestSharpe = perf.SharpeRatio
			bestStrategy = strategy
		}
	}

	return bestStrategy
}

// selectByRules uses rule-based selection when performance data is limited
func (ats *AdaptiveTradingSystem) selectByRules(features *DailyFeatures, candidates []TradingStrategy) TradingStrategy {
	// Rule-based selection based on market regime
	switch features.MarketRegime {
	case RegimeStrongBull, RegimeWeakBull:
		// Prefer trend following in bull markets
		for _, strategy := range candidates {
			if strategy.Category() == CategoryTrendFollowing {
				return strategy
			}
		}
	case RegimeStrongBear, RegimeWeakBear:
		// Prefer mean reversion or short strategies in bear markets
		for _, strategy := range candidates {
			if strategy.Category() == CategoryMeanReversion {
				return strategy
			}
		}
	case RegimeSideways, RegimeMeanReverting:
		// Prefer mean reversion in sideways markets
		for _, strategy := range candidates {
			if strategy.Category() == CategoryMeanReversion {
				return strategy
			}
		}
	case RegimeHighVol:
		// Prefer volatility strategies in high volatility
		for _, strategy := range candidates {
			if strategy.Category() == CategoryVolatility {
				return strategy
			}
		}
	}

	// Default to first candidate
	return candidates[0]
}

// GenerateTradeSignal generates a trade signal using the optimal strategy
func (ats *AdaptiveTradingSystem) GenerateTradeSignal(features *DailyFeatures, series *DailyOHLCVSeries) TradeSignal {
	// Select optimal strategy
	strategy := ats.SelectOptimalStrategy(features, series)
	if strategy == nil {
		return TradeSignal{
			Action:     ActionHold,
			Confidence: 0,
			Reason:     "No suitable strategy found for current market regime",
		}
	}

	// Calculate position size based on risk
	positionSize := strategy.CalculatePositionSize(ats.AccountSize, ats.RiskPerTrade, 0)

	// Generate signal
	signal := strategy.GenerateSignal(features, series, positionSize)

	// Record strategy selection
	selection := StrategySelection{
		Date:         features.Date,
		Strategy:     strategy.Name(),
		Regime:       features.MarketRegime,
		Confidence:   signal.Confidence,
		Features:     features,
		Reason:       fmt.Sprintf("Selected based on %s regime", features.MarketRegime),
	}
	ats.StrategyHistory = append(ats.StrategyHistory, selection)

	// Update current strategy
	ats.CurrentStrategy = strategy

	return signal
}

// UpdatePerformance updates system performance after a trade
func (ats *AdaptiveTradingSystem) UpdatePerformance(trade *Trade) {
	if trade.Strategy == "" {
		return
	}

	// Update strategy performance
	strategy, exists := ats.Strategies[trade.Strategy]
	if !exists {
		return
	}

	strategy.UpdatePerformance(trade)

	// Update system performance log
	record := PerformanceRecord{
		Date:            trade.ExitDate,
		Strategy:        trade.Strategy,
		Regime:          ats.MarketRegime,
		DailyReturn:     trade.ReturnPercent,
		CumulativeReturn: 0, // Would need to calculate from account
		Drawdown:        trade.Drawdown,
		PositionSize:    trade.Size,
		Signal:          trade.Signal,
	}
	ats.PerformanceLog = append(ats.PerformanceLog, record)
}

// ============================================================================
// Trade Execution and Management
// ============================================================================

// Trade represents a completed trade
type Trade struct {
	EntryDate    time.Time
	ExitDate     time.Time
	EntryPrice   float64
	ExitPrice    float64
	Size         float64
	Direction    TradeAction
	Strategy     string
	ReturnAmount float64
	ReturnPercent float64
	Drawdown     float64
	Duration     int // Days
	Signal       TradeSignal
	Metadata     map[string]interface{}
}

// Portfolio manages multiple positions and trades
type Portfolio struct {
	Cash          float64
	Positions     map[string]*Position
	Trades        []*Trade
	TotalValue    float64
	PeakValue     float64
	CurrentDrawdown float64
	MaxDrawdown   float64
	DailyReturns  []float64
}

// Position represents an open position
type Position struct {
	Ticker       string
	EntryDate    time.Time
	EntryPrice   float64
	CurrentPrice float64
	Size         float64
	Direction    TradeAction
	StopLoss     float64
	TakeProfit   float64
	Strategy     string
	UnrealizedPL float64
	UnrealizedPLPercent float64
}

// ExecuteTrade executes a trade signal
func (p *Portfolio) ExecuteTrade(signal TradeSignal, currentPrice float64, date time.Time) *Trade {
	if signal.Action == ActionHold {
		return nil
	}

	// Create trade
	trade := &Trade{
		EntryDate:  date,
		EntryPrice: currentPrice,
		Size:       signal.Size,
		Direction:  signal.Action,
		Strategy:   signal.StrategyName,
		Signal:     signal,
	}

	// Update portfolio based on trade
	switch signal.Action {
	case ActionBuy:
		cost := currentPrice * signal.Size
		if cost <= p.Cash {
			p.Cash -= cost
			// Create or update position
			// Implementation depends on position management logic
		}
	case ActionSell:
		// Close long position
		// Implementation depends on position management logic
	case ActionShort:
		// Open short position
		// Implementation depends on position management logic
	case ActionCover:
		// Close short position
		// Implementation depends on position management logic
	}

	return trade
}

// ============================================================================
// Utility Functions
// ============================================================================

func calculateSMA(data []float64, period int) []float64 {
	if len(data) < period {
		return nil
	}

	sma := make([]float64, len(data)-period+1)
	for i := 0; i <= len(data)-period; i++ {
		sum := 0.0
		for j := 0; j < period; j++ {
			sum += data[i+j]
		}
		sma[i] = sum / float64(period)
	}
	return sma
}

func calculateEMA(data []float64, period int) []float64 {
	if len(data) < period {
		return nil
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

	return ema[period-1:]
}

func calculateRSI(data []float64, period int) []float64 {
	if len(data) < period+1 {
		return nil
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

	return rsi
}

func calculateATR(high, low, close []float64, period int) []float64 {
	if len(high) != len(low) || len(high) != len(close) {
		return nil
	}
	if len(high) < period {
		return nil
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

	return atr
}

func calculateRollingVolatility(prices []float64, window int) []float64 {
	if len(prices) < window+1 {
		return nil
	}

	volatility := make([]float64, len(prices)-window)

	for i := 0; i <= len(prices)-window-1; i++ {
		returns := make([]float64, window)
		for j := 0; j < window; j++ {
			returns[j] = (prices[i+j+1] - prices[i+j]) / prices[i+j]
		}
		volatility[i] = stdDev(returns) * math.Sqrt(252)
	}

	return volatility
}

func calculateTrendStrength(high, low, close []float64, period int) []float64 {
	// Simplified trend strength calculation
	if len(high) < period*2 {
		return nil
	}

	trend := make([]float64, len(close)-period+1)

	for i := 0; i <= len(close)-period; i++ {
		// Calculate average upward movement vs downward movement
		upMove := 0.0
		downMove := 0.0

		for j := i; j < i+period; j++ {
			if j == 0 {
				continue
			}
			if high[j] > high[j-1] {
				upMove += high[j] - high[j-1]
			}
			if low[j] < low[j-1] {
				downMove += low[j-1] - low[j]
			}
		}

		totalMove := upMove + downMove
		if totalMove > 0 {
			// Trend strength from 0 to 1
			trend[i] = math.Abs(upMove-downMove) / totalMove
		} else {
			trend[i] = 0
		}
	}

	return trend
}

func calculateMomentum(prices []float64, period int) []float64 {
	if len(prices) < period {
		return nil
	}

	momentum := make([]float64, len(prices)-period)
	for i := 0; i < len(prices)-period; i++ {
		momentum[i] = (prices[i+period] - prices[i]) / prices[i]
	}
	return momentum
}

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

func isMeanReverting(returns []float64) bool {
	if len(returns) < 10 {
		return false
	}

	// Simple mean reversion test: check autocorrelation at lag 1
	meanRet := mean(returns)

	numerator := 0.0
	denominator := 0.0

	for i := 1; i < len(returns); i++ {
		numerator += (returns[i] - meanRet) * (returns[i-1] - meanRet)
		denominator += (returns[i] - meanRet) * (returns[i] - meanRet)
	}

	if denominator == 0 {
		return false
	}

	autocorr := numerator / denominator
	return autocorr < 0 // Negative autocorrelation suggests mean reversion
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
