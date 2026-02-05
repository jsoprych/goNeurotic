package financial

import (
	"fmt"
	"time"
)

// ============================================================================
// Alignment Functions for Financial Time-Series Indicators
// ============================================================================
// These functions handle alignment of indicator outputs with original price data,
// accounting for different lookback periods and missing values.

// IndicatorAlignment handles alignment of indicator values with price data
type IndicatorAlignment struct {
	OriginalLength int      // Length of original price series
	IndicatorStart int      // Index where indicator values start
	IndicatorEnd   int      // Index where indicator values end
	AlignedValues []float64 // Indicator values aligned with original series
	HasValue      []bool    // Boolean mask indicating valid indicator values
}

// AlignIndicator aligns indicator output with original price series
// indicatorValues: Output from technical indicator (may be shorter than prices)
// lookback: Number of price points needed to compute first indicator value
// fillValue: Value to use for positions without indicator data (default: NaN placeholder)
func AlignIndicator(prices []float64, indicatorValues []float64, lookback int, fillValue float64) (*IndicatorAlignment, error) {
	if len(prices) == 0 {
		return nil, fmt.Errorf("price series is empty")
	}

	if len(indicatorValues) == 0 {
		return nil, fmt.Errorf("indicator values are empty")
	}

	if lookback < 0 {
		return nil, fmt.Errorf("lookback must be non-negative")
	}

	// Special value for NaN placeholder
	nanPlaceholder := fillValue
	if nanPlaceholder == 0 {
		// Use a value unlikely to appear in financial data
		nanPlaceholder = -999999.0
	}

	aligned := make([]float64, len(prices))
	hasValue := make([]bool, len(prices))

	// Indicator values start after lookback period
	startIdx := lookback

	// Fill beginning with placeholder
	for i := 0; i < startIdx && i < len(prices); i++ {
		aligned[i] = nanPlaceholder
		hasValue[i] = false
	}

	// Align indicator values
	indicatorIdx := 0
	for i := startIdx; i < len(prices); i++ {
		if indicatorIdx < len(indicatorValues) {
			aligned[i] = indicatorValues[indicatorIdx]
			hasValue[i] = true
			indicatorIdx++
		} else {
			aligned[i] = nanPlaceholder
			hasValue[i] = false
		}
	}

	return &IndicatorAlignment{
		OriginalLength: len(prices),
		IndicatorStart: startIdx,
		IndicatorEnd:   startIdx + len(indicatorValues) - 1,
		AlignedValues:  aligned,
		HasValue:       hasValue,
	}, nil
}

// AlignMultipleIndicators aligns multiple indicators with price series
// indicators: Map of indicator names to their values
// lookbacks: Map of indicator names to their required lookback periods
func AlignMultipleIndicators(prices []float64, indicators map[string][]float64, lookbacks map[string]int) (map[string]*IndicatorAlignment, error) {
	if len(prices) == 0 {
		return nil, fmt.Errorf("price series is empty")
	}

	results := make(map[string]*IndicatorAlignment)

	for name, values := range indicators {
		lookback, exists := lookbacks[name]
		if !exists {
			// Default lookback of 0
			lookback = 0
		}

		alignment, err := AlignIndicator(prices, values, lookback, 0)
		if err != nil {
			return nil, fmt.Errorf("failed to align indicator %s: %w", name, err)
		}

		results[name] = alignment
	}

	return results, nil
}

// TimeSeriesAlignment aligns time series data with timestamps
type TimeSeriesAlignment struct {
	Timestamps     []time.Time
	AlignedSeries  map[string][]float64 // Map of series name to aligned values
	ValidMask      []bool               // Mask indicating valid data points
}

// AlignTimeSeries aligns multiple time series by timestamp
// baseTimestamps: Reference timestamps to align to
// seriesData: Map of series names to timestamp-value pairs
// tolerance: Maximum time difference for matching (default: 1 hour)
func AlignTimeSeries(baseTimestamps []time.Time, seriesData map[string][]TimePoint, tolerance time.Duration) (*TimeSeriesAlignment, error) {
	if len(baseTimestamps) == 0 {
		return nil, fmt.Errorf("base timestamps are empty")
	}

	if tolerance == 0 {
		tolerance = time.Hour
	}

	alignedSeries := make(map[string][]float64)

	// Initialize all series with NaN placeholders
	for name := range seriesData {
		alignedSeries[name] = make([]float64, len(baseTimestamps))
		for i := range alignedSeries[name] {
			alignedSeries[name][i] = -999999.0 // NaN placeholder
		}
	}

	validMask := make([]bool, len(baseTimestamps))

	// For each base timestamp, find matching data points
	for i, baseTs := range baseTimestamps {
		validCount := 0

		for name, points := range seriesData {
			// Find closest matching timestamp
			closestIdx := -1
			minDiff := tolerance + 1

			for j, point := range points {
				diff := absDuration(baseTs.Sub(point.Timestamp))
				if diff <= tolerance && diff < minDiff {
					minDiff = diff
					closestIdx = j
				}
			}

			if closestIdx >= 0 {
				alignedSeries[name][i] = points[closestIdx].Value
				validCount++
			}
		}

		// Mark as valid if we found data for at least one series
		validMask[i] = validCount > 0
	}

	return &TimeSeriesAlignment{
		Timestamps:    baseTimestamps,
		AlignedSeries: alignedSeries,
		ValidMask:     validMask,
	}, nil
}

// TimePoint represents a value at a specific timestamp
type TimePoint struct {
	Timestamp time.Time
	Value     float64
}

// ForwardFill fills NaN values with last valid observation
func ForwardFill(aligned *IndicatorAlignment) []float64 {
	if aligned == nil {
		return nil
	}

	filled := make([]float64, len(aligned.AlignedValues))
	lastValid := -999999.0 // NaN placeholder

	for i, val := range aligned.AlignedValues {
		if aligned.HasValue[i] {
			filled[i] = val
			lastValid = val
		} else if lastValid != -999999.0 {
			filled[i] = lastValid
		} else {
			filled[i] = val // Keep original placeholder
		}
	}

	return filled
}

// BackwardFill fills NaN values with next valid observation
func BackwardFill(aligned *IndicatorAlignment) []float64 {
	if aligned == nil {
		return nil
	}

	filled := make([]float64, len(aligned.AlignedValues))
	nextValid := -999999.0

	// First pass: find next valid values
	nextValidQueue := make([]float64, len(aligned.AlignedValues))
	for i := len(aligned.AlignedValues) - 1; i >= 0; i-- {
		if aligned.HasValue[i] {
			nextValid = aligned.AlignedValues[i]
		}
		nextValidQueue[i] = nextValid
	}

	// Second pass: fill with next valid values
	for i := range filled {
		if aligned.HasValue[i] {
			filled[i] = aligned.AlignedValues[i]
		} else if nextValidQueue[i] != -999999.0 {
			filled[i] = nextValidQueue[i]
		} else {
			filled[i] = aligned.AlignedValues[i]
		}
	}

	return filled
}

// LinearInterpolate linearly interpolates NaN values between valid observations
func LinearInterpolate(aligned *IndicatorAlignment) []float64 {
	if aligned == nil {
		return nil
	}

	interpolated := make([]float64, len(aligned.AlignedValues))
	copy(interpolated, aligned.AlignedValues)

	i := 0
	for i < len(interpolated) {
		// Find start of NaN sequence
		if aligned.HasValue[i] {
			i++
			continue
		}

		start := i - 1
		// Find end of NaN sequence
		for i < len(interpolated) && !aligned.HasValue[i] {
			i++
		}
		end := i

		if start >= 0 && end < len(interpolated) {
			// We have valid values before and after the NaN sequence
			startVal := interpolated[start]
			endVal := interpolated[end]

			steps := end - start
			stepSize := (endVal - startVal) / float64(steps)

			for j := start + 1; j < end; j++ {
				interpolated[j] = startVal + stepSize*float64(j-start)
			}
		}

		i++
	}

	return interpolated
}

// CreateFeatureMatrix creates a feature matrix from aligned indicators
// Each row represents a time point, each column represents an indicator
// Only includes time points where all indicators have valid values
func CreateFeatureMatrix(alignments map[string]*IndicatorAlignment) ([][]float64, []int, error) {
	if len(alignments) == 0 {
		return nil, nil, fmt.Errorf("no alignments provided")
	}

	// Find common length
	var commonLength int
	for name, alignment := range alignments {
		if commonLength == 0 {
			commonLength = alignment.OriginalLength
		} else if alignment.OriginalLength != commonLength {
			return nil, nil, fmt.Errorf("inconsistent lengths: %s has %d, expected %d",
				name, alignment.OriginalLength, commonLength)
		}
	}

	// Find indices where all indicators have valid values
	var validIndices []int
	for i := 0; i < commonLength; i++ {
		allValid := true
		for _, alignment := range alignments {
			if !alignment.HasValue[i] {
				allValid = false
				break
			}
		}

		if allValid {
			validIndices = append(validIndices, i)
		}
	}

	if len(validIndices) == 0 {
		return nil, nil, fmt.Errorf("no time points with all indicators valid")
	}

	// Create feature matrix
	featureMatrix := make([][]float64, len(validIndices))
	for rowIdx, timeIdx := range validIndices {
		features := make([]float64, len(alignments))
		colIdx := 0
		for _, alignment := range alignments {
			features[colIdx] = alignment.AlignedValues[timeIdx]
			colIdx++
		}
		featureMatrix[rowIdx] = features
	}

	return featureMatrix, validIndices, nil
}

// IndicatorLag creates lagged versions of indicators for time-series modeling
func IndicatorLag(aligned *IndicatorAlignment, lag int) (*IndicatorAlignment, error) {
	if aligned == nil {
		return nil, fmt.Errorf("alignment is nil")
	}

	if lag <= 0 {
		return nil, fmt.Errorf("lag must be positive")
	}

	if lag >= len(aligned.AlignedValues) {
		return nil, fmt.Errorf("lag exceeds series length")
	}

	laggedValues := make([]float64, len(aligned.AlignedValues))
	laggedHasValue := make([]bool, len(aligned.AlignedValues))

	// Fill beginning with NaN (no data before lag)
	for i := 0; i < lag; i++ {
		laggedValues[i] = -999999.0
		laggedHasValue[i] = false
	}

	// Shift values
	for i := lag; i < len(aligned.AlignedValues); i++ {
		laggedValues[i] = aligned.AlignedValues[i-lag]
		laggedHasValue[i] = aligned.HasValue[i-lag]
	}

	return &IndicatorAlignment{
		OriginalLength: aligned.OriginalLength,
		IndicatorStart: aligned.IndicatorStart + lag,
		IndicatorEnd:   min(aligned.IndicatorEnd + lag, aligned.OriginalLength - 1),
		AlignedValues:  laggedValues,
		HasValue:       laggedHasValue,
	}, nil
}

// IndicatorDelta calculates differences between indicator values
// period: Number of periods to difference (e.g., 1 for first difference)
func IndicatorDelta(aligned *IndicatorAlignment, period int) (*IndicatorAlignment, error) {
	if aligned == nil {
		return nil, fmt.Errorf("alignment is nil")
	}

	if period <= 0 {
		return nil, fmt.Errorf("period must be positive")
	}

	if period >= len(aligned.AlignedValues) {
		return nil, fmt.Errorf("period exceeds series length")
	}

	deltaValues := make([]float64, len(aligned.AlignedValues))
	deltaHasValue := make([]bool, len(aligned.AlignedValues))

	// Fill beginning with NaN
	for i := 0; i < period; i++ {
		deltaValues[i] = -999999.0
		deltaHasValue[i] = false
	}

	// Calculate differences
	for i := period; i < len(aligned.AlignedValues); i++ {
		if aligned.HasValue[i] && aligned.HasValue[i-period] {
			deltaValues[i] = aligned.AlignedValues[i] - aligned.AlignedValues[i-period]
			deltaHasValue[i] = true
		} else {
			deltaValues[i] = -999999.0
			deltaHasValue[i] = false
		}
	}

	return &IndicatorAlignment{
		OriginalLength: aligned.OriginalLength,
		IndicatorStart: aligned.IndicatorStart + period,
		IndicatorEnd:   aligned.IndicatorEnd,
		AlignedValues:  deltaValues,
		HasValue:       deltaHasValue,
	}, nil
}

// RollingAlignment aligns rolling window statistics
func RollingAlignment(prices []float64, window int, statFunc func([]float64) float64) (*IndicatorAlignment, error) {
	if len(prices) < window {
		return nil, fmt.Errorf("insufficient data for window size %d", window)
	}

	if window <= 0 {
		return nil, fmt.Errorf("window must be positive")
	}

	values := make([]float64, len(prices)-window+1)
	for i := 0; i <= len(prices)-window; i++ {
		windowData := prices[i : i+window]
		values[i] = statFunc(windowData)
	}

	return AlignIndicator(prices, values, window-1, 0)
}

// Utility functions
func absDuration(d time.Duration) time.Duration {
	if d < 0 {
		return -d
	}
	return d
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Example usage:
/*
func ExampleAlignment() {
	prices := []float64{100, 101, 102, 103, 104, 105, 106, 107, 108, 109}

	// Calculate SMA with period 3
	sma, _ := SMA(prices, 3)

	// Align SMA with original prices
	alignment, _ := AlignIndicator(prices, sma, 2, 0)

	// Forward fill missing values
	filled := ForwardFill(alignment)

	// Create lagged version for prediction
	lagged, _ := IndicatorLag(alignment, 1)

	// Calculate differences
	delta, _ := IndicatorDelta(alignment, 1)

	// Multiple indicators
	indicators := map[string][]float64{
		"SMA_3": sma,
		"SMA_5": sma5, // Assuming sma5 is calculated
	}

	lookbacks := map[string]int{
		"SMA_3": 2,
		"SMA_5": 4,
	}

	multiAlign, _ := AlignMultipleIndicators(prices, indicators, lookbacks)

	// Create feature matrix
	features, indices, _ := CreateFeatureMatrix(multiAlign)

	fmt.Printf("Created %d features with %d time points\n",
		len(features[0]), len(features))
}
*/
