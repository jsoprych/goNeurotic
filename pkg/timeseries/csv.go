package timeseries

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"
)

// CSVConfig holds configuration for loading CSV files
type CSVConfig struct {
	FilePath          string    // Path to CSV file
	HasHeader         bool      // Whether CSV has a header row
	Separator         rune      // CSV separator (default comma)
	TimeColumn        string    // Name or index of time column (empty if no time column)
	ValueColumn       string    // Name or index of value column (for univariate)
	ValueColumns      []string  // Names or indices of value columns (for multivariate)
	ExogenousColumns  []string  // Names or indices of exogenous feature columns
	DateFormat        string    // Format string for parsing dates (if TimeColumn is date)
	StartDate         time.Time // Optional start date filter
	EndDate           time.Time // Optional end date filter
	SkipRows          int       // Number of rows to skip at beginning
	MaxRows           int       // Maximum number of rows to load (0 for all)
	FillMissing       bool      // Whether to fill missing values
	MissingValue      float64   // Value to use for filling missing data
}

// CSVTimeSeriesData holds loaded time series data from CSV
type CSVTimeSeriesData struct {
	Timestamps  []time.Time // Time values (empty if no time column)
	Values      []float64   // Univariate values (if univariate)
	Series      [][]float64 // Multivariate values [time][feature]
	Features    [][]float64 // Exogenous features [time][feature]
	ColumnNames []string    // Column names from header
	Stats       CSVStats    // Statistical summary
}

// CSVStats holds statistical summary of loaded data
type CSVStats struct {
	Count     int
	Min       float64
	Max       float64
	Mean      float64
	StdDev    float64
	StartTime time.Time
	EndTime   time.Time
}

// DefaultCSVConfig returns a default CSV configuration
func DefaultCSVConfig(filePath string) CSVConfig {
	return CSVConfig{
		FilePath:     filePath,
		HasHeader:    true,
		Separator:    ',',
		DateFormat:   "2006-01-02", // Go's reference date format
		FillMissing:  false,
		MissingValue: 0.0,
		SkipRows:     0,
		MaxRows:      0, // Load all rows
	}
}

// LoadCSV loads time series data from a CSV file
func LoadCSV(config CSVConfig) (*CSVTimeSeriesData, error) {
	file, err := os.Open(config.FilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open CSV file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = config.Separator

	// Skip rows if configured
	for i := 0; i < config.SkipRows; i++ {
		if _, err := reader.Read(); err != nil {
			return nil, fmt.Errorf("failed to skip row %d: %w", i, err)
		}
	}

	// Read all records
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV: %w", err)
	}

	if len(records) == 0 {
		return nil, fmt.Errorf("CSV file is empty")
	}

	// Handle header
	startRow := 0
	columnNames := make([]string, 0)

	if config.HasHeader && len(records) > 0 {
		columnNames = records[0]
		startRow = 1
	} else {
		// Generate column names like col0, col1, ...
		for i := 0; i < len(records[0]); i++ {
			columnNames = append(columnNames, fmt.Sprintf("col%d", i))
		}
	}

	// Find column indices
	timeColIdx := -1
	valueColIdx := -1
	valueColIndices := make([]int, 0)
	exogColIndices := make([]int, 0)

	// Parse time column
	if config.TimeColumn != "" {
		if idx, ok := findColumnIndex(config.TimeColumn, columnNames); ok {
			timeColIdx = idx
		} else {
			// Try parsing as index
			if idx, err := strconv.Atoi(config.TimeColumn); err == nil && idx >= 0 && idx < len(columnNames) {
				timeColIdx = idx
			} else {
				return nil, fmt.Errorf("time column '%s' not found", config.TimeColumn)
			}
		}
	}

	// Parse value column(s)
	if config.ValueColumn != "" {
		if idx, ok := findColumnIndex(config.ValueColumn, columnNames); ok {
			valueColIdx = idx
		} else if idx, err := strconv.Atoi(config.ValueColumn); err == nil && idx >= 0 && idx < len(columnNames) {
			valueColIdx = idx
		} else {
			return nil, fmt.Errorf("value column '%s' not found", config.ValueColumn)
		}
	} else if len(config.ValueColumns) > 0 {
		for _, col := range config.ValueColumns {
			if idx, ok := findColumnIndex(col, columnNames); ok {
				valueColIndices = append(valueColIndices, idx)
			} else if idx, err := strconv.Atoi(col); err == nil && idx >= 0 && idx < len(columnNames) {
				valueColIndices = append(valueColIndices, idx)
			} else {
				return nil, fmt.Errorf("value column '%s' not found", col)
			}
		}
	} else {
		// Default: use first column as value
		valueColIdx = 0
	}

	// Parse exogenous columns
	for _, col := range config.ExogenousColumns {
		if idx, ok := findColumnIndex(col, columnNames); ok {
			exogColIndices = append(exogColIndices, idx)
		} else if idx, err := strconv.Atoi(col); err == nil && idx >= 0 && idx < len(columnNames) {
			exogColIndices = append(exogColIndices, idx)
		} else {
			return nil, fmt.Errorf("exogenous column '%s' not found", col)
		}
	}

	// Apply max rows limit
	dataRows := records[startRow:]
	if config.MaxRows > 0 && config.MaxRows < len(dataRows) {
		dataRows = dataRows[:config.MaxRows]
	}

	// Parse data
	result := &CSVTimeSeriesData{
		ColumnNames: columnNames,
	}

	var sum, sumSq float64
	count := 0
	minSet := false

	for rowIdx, row := range dataRows {
		// Parse timestamp if time column exists
		var timestamp time.Time
		if timeColIdx >= 0 && timeColIdx < len(row) {
			timestamp, err = parseTime(row[timeColIdx], config.DateFormat)
			if err != nil {
				if config.FillMissing {
					timestamp = time.Time{}
				} else {
					return nil, fmt.Errorf("row %d: invalid time '%s': %w",
						rowIdx+startRow+1, row[timeColIdx], err)
				}
			}

			// Apply date filters
			if !config.StartDate.IsZero() && timestamp.Before(config.StartDate) {
				continue
			}
			if !config.EndDate.IsZero() && timestamp.After(config.EndDate) {
				continue
			}
		}

		// Parse value(s)
		var values []float64
		if valueColIdx >= 0 {
			// Univariate
			val, err := parseFloat(row[valueColIdx], config)
			if err != nil {
				if config.FillMissing {
					val = config.MissingValue
				} else {
					return nil, fmt.Errorf("row %d: invalid value '%s': %w",
						rowIdx+startRow+1, row[valueColIdx], err)
				}
			}
			values = []float64{val}
		} else if len(valueColIndices) > 0 {
			// Multivariate
			values = make([]float64, len(valueColIndices))
			for i, colIdx := range valueColIndices {
				val, err := parseFloat(row[colIdx], config)
				if err != nil {
					if config.FillMissing {
						val = config.MissingValue
					} else {
						return nil, fmt.Errorf("row %d col %d: invalid value '%s': %w",
							rowIdx+startRow+1, colIdx, row[colIdx], err)
					}
				}
				values[i] = val
			}
		}

		// Parse exogenous features
		var features []float64
		if len(exogColIndices) > 0 {
			features = make([]float64, len(exogColIndices))
			for i, colIdx := range exogColIndices {
				val, err := parseFloat(row[colIdx], config)
				if err != nil {
					if config.FillMissing {
						val = config.MissingValue
					} else {
						return nil, fmt.Errorf("row %d col %d: invalid feature '%s': %w",
							rowIdx+startRow+1, colIdx, row[colIdx], err)
					}
				}
				features[i] = val
			}
		}

		// Store data
		if !timestamp.IsZero() {
			result.Timestamps = append(result.Timestamps, timestamp)
		}

		if valueColIdx >= 0 {
			// Univariate
			result.Values = append(result.Values, values[0])
			if !minSet {
				result.Stats.Min = values[0]
				result.Stats.Max = values[0]
				minSet = true
			}
			sum += values[0]
			sumSq += values[0] * values[0]
			if values[0] < result.Stats.Min {
				result.Stats.Min = values[0]
			}
			if values[0] > result.Stats.Max {
				result.Stats.Max = values[0]
			}
		} else {
			// Multivariate
			result.Series = append(result.Series, values)
		}

		if len(features) > 0 {
			result.Features = append(result.Features, features)
		}
		count++
	}

	// Calculate statistics
	result.Stats.Count = count
	if count > 0 {
		result.Stats.Mean = sum / float64(count)
		if count > 1 {
			variance := (sumSq - sum*sum/float64(count)) / float64(count-1)
			if variance > 0 {
				result.Stats.StdDev = math.Sqrt(variance)
			}
		}

		if len(result.Timestamps) > 0 {
			result.Stats.StartTime = result.Timestamps[0]
			result.Stats.EndTime = result.Timestamps[len(result.Timestamps)-1]
		}
	}

	return result, nil
}

// findColumnIndex finds a column by name or returns -1, false if not found
func findColumnIndex(col string, columnNames []string) (int, bool) {
	for i, name := range columnNames {
		if strings.EqualFold(name, col) {
			return i, true
		}
	}
	return -1, false
}

// parseTime parses a time string with the given format
func parseTime(timeStr, format string) (time.Time, error) {
	timeStr = strings.TrimSpace(timeStr)
	if timeStr == "" {
		return time.Time{}, fmt.Errorf("empty time string")
	}

	if format == "" {
		// Try common formats
		formats := []string{
			"2006-01-02 15:04:05",
			"2006-01-02T15:04:05",
			"2006-01-02",
			"01/02/2006",
			"01/02/2006 15:04:05",
			time.RFC3339,
			time.RFC3339Nano,
		}

		for _, f := range formats {
			if t, err := time.Parse(f, timeStr); err == nil {
				return t, nil
			}
		}

		// Try Unix timestamp
		if unixTime, err := strconv.ParseInt(timeStr, 10, 64); err == nil {
			return time.Unix(unixTime, 0), nil
		}

		return time.Time{}, fmt.Errorf("could not parse time '%s'", timeStr)
	}

	return time.Parse(format, timeStr)
}

// parseFloat parses a float value with error handling
func parseFloat(valStr string, config CSVConfig) (float64, error) {
	valStr = strings.TrimSpace(valStr)
	if valStr == "" {
		return 0, fmt.Errorf("empty value")
	}

	val, err := strconv.ParseFloat(valStr, 64)
	if err != nil {
		return 0, err
	}

	return val, nil
}

// ExtractDateFeatures extracts date-based features from timestamps
func ExtractDateFeatures(timestamps []time.Time) [][]float64 {
	features := make([][]float64, len(timestamps))

	for i, ts := range timestamps {
		// Basic temporal features
		feat := []float64{
			float64(ts.Year()),
			float64(ts.Month()),
			float64(ts.Day()),
			float64(ts.Hour()),
			float64(ts.Minute()),
			float64(ts.Second()),
			float64(ts.Weekday()), // 0=Sunday, 6=Saturday
			float64(ts.YearDay()), // Day of year (1-365/366)
		}

		// Cyclical features for seasonality
		monthRad := 2 * math.Pi * float64(ts.Month()-1) / 12.0
		dayOfYearRad := 2 * math.Pi * float64(ts.YearDay()-1) / 365.25
		hourRad := 2 * math.Pi * float64(ts.Hour()) / 24.0
		weekdayRad := 2 * math.Pi * float64(ts.Weekday()) / 7.0

		feat = append(feat,
			monthRad,               // Month as angle
			dayOfYearRad,           // Day of year as angle
			hourRad,                // Hour as angle
			weekdayRad,             // Weekday as angle
			math.Sin(monthRad),     // Sin(month)
			math.Cos(monthRad),     // Cos(month)
			math.Sin(dayOfYearRad), // Sin(day of year)
			math.Cos(dayOfYearRad), // Cos(day of year)
			math.Sin(hourRad),      // Sin(hour)
			math.Cos(hourRad),      // Cos(hour)
			math.Sin(weekdayRad),   // Sin(weekday)
			math.Cos(weekdayRad),   // Cos(weekday)
		)

		// Business day indicator (Mon-Fri)
		isBusinessDay := 0.0
		if ts.Weekday() >= time.Monday && ts.Weekday() <= time.Friday {
			isBusinessDay = 1.0
		}
		feat = append(feat, isBusinessDay)

		// Quarter
		quarter := float64((ts.Month()-1)/3 + 1)
		feat = append(feat, quarter)

		// Month end/beginning indicators
		isMonthEnd := 0.0
		nextDay := ts.Add(24 * time.Hour)
		if nextDay.Month() != ts.Month() {
			isMonthEnd = 1.0
		}
		feat = append(feat, isMonthEnd)

		// Weekend indicator
		isWeekend := 0.0
		if ts.Weekday() == time.Saturday || ts.Weekday() == time.Sunday {
			isWeekend = 1.0
		}
		feat = append(feat, isWeekend)

		features[i] = feat
	}

	return features
}

// LoadUnivariateCSV is a convenience function for loading univariate time series
func LoadUnivariateCSV(filePath, timeCol, valueCol string) (*CSVTimeSeriesData, error) {
	config := DefaultCSVConfig(filePath)
	config.TimeColumn = timeCol
	config.ValueColumn = valueCol
	return LoadCSV(config)
}

// LoadMultivariateCSV is a convenience function for loading multivariate time series
func LoadMultivariateCSV(filePath, timeCol string, valueCols []string) (*CSVTimeSeriesData, error) {
	config := DefaultCSVConfig(filePath)
	config.TimeColumn = timeCol
	config.ValueColumns = valueCols
	return LoadCSV(config)
}

// CreateSampleData creates sample time series data for testing/demo purposes
func CreateSampleData() *CSVTimeSeriesData {
	// Create a year of daily data
	start := time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)
	data := &CSVTimeSeriesData{
		Timestamps: make([]time.Time, 365),
		Values:     make([]float64, 365),
	}

	// Synthetic time series with trend + seasonality + noise
	trend := 0.05
	seasonAmplitude := 10.0
	noiseLevel := 2.0

	for i := 0; i < 365; i++ {
		t := start.AddDate(0, 0, i)
		data.Timestamps[i] = t

		// Trend component
		trendComp := trend * float64(i)

		// Seasonal component (weekly and yearly)
		dayOfYear := float64(t.YearDay())
		weekly := math.Sin(2 * math.Pi * float64(t.Weekday()) / 7.0)
		yearly := math.Sin(2 * math.Pi * dayOfYear / 365.25)

		// Noise (deterministic pseudo-random using sine)
		noise := math.Sin(float64(i)*1.3+float64(t.Weekday())*2.1) * noiseLevel

		// Combined series
		data.Values[i] = 100.0 + trendComp + seasonAmplitude*(weekly+yearly) + noise
	}

	// Calculate stats
	var sum, sumSq float64
	data.Stats.Min = data.Values[0]
	data.Stats.Max = data.Values[0]
	data.Stats.Count = 365

	for _, val := range data.Values {
		sum += val
		sumSq += val * val
		if val < data.Stats.Min {
			data.Stats.Min = val
		}
		if val > data.Stats.Max {
			data.Stats.Max = val
		}
	}

	data.Stats.Mean = sum / 365.0
	variance := (sumSq - sum*sum/365.0) / 364.0
	data.Stats.StdDev = math.Sqrt(variance)
	data.Stats.StartTime = data.Timestamps[0]
	data.Stats.EndTime = data.Timestamps[364]

	return data
}

// ToTimeSeries converts CSVTimeSeriesData to the standard TimeSeries struct
func (data *CSVTimeSeriesData) ToTimeSeries() TimeSeries {
	return TimeSeries{Data: data.Values}
}

// ToMultivariateTimeSeries converts CSVTimeSeriesData to the standard MultivariateTimeSeries struct
func (data *CSVTimeSeriesData) ToMultivariateTimeSeries() MultivariateTimeSeries {
	if len(data.Series) > 0 {
		return MultivariateTimeSeries{Data: data.Series}
	}
	// Convert univariate to multivariate
	series := make([][]float64, len(data.Values))
	for i, val := range data.Values {
		series[i] = []float64{val}
	}
	return MultivariateTimeSeries{Data: series}
}

// GetUnixTimestamps returns timestamps as Unix timestamps (seconds since epoch)
func (data *CSVTimeSeriesData) GetUnixTimestamps() []int64 {
	timestamps := make([]int64, len(data.Timestamps))
	for i, ts := range data.Timestamps {
		timestamps[i] = ts.Unix()
	}
	return timestamps
}

// AirPassengersDataset creates the classic AirPassengers time series dataset (monthly totals from 1949-1960)
func AirPassengersDataset() *CSVTimeSeriesData {
	// Monthly airline passenger numbers from 1949-1960 (in thousands)
	// Source: Box & Jenkins (1970) Series G
	passengers := []float64{
		112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
		115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
		145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
		171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
		196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
		204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
		242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
		284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
		315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
		340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
		360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
		417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
	}

	start := time.Date(1949, time.January, 1, 0, 0, 0, 0, time.UTC)
	data := &CSVTimeSeriesData{
		Timestamps: make([]time.Time, len(passengers)),
		Values:     make([]float64, len(passengers)),
	}

	for i := 0; i < len(passengers); i++ {
		// Add i months to start date
		data.Timestamps[i] = start.AddDate(0, i, 0)
		data.Values[i] = passengers[i]
	}

	// Calculate statistics
	var sum, sumSq float64
	data.Stats.Min = data.Values[0]
	data.Stats.Max = data.Values[0]
	data.Stats.Count = len(passengers)

	for _, val := range data.Values {
		sum += val
		sumSq += val * val
		if val < data.Stats.Min {
			data.Stats.Min = val
		}
		if val > data.Stats.Max {
			data.Stats.Max = val
		}
	}

	data.Stats.Mean = sum / float64(len(passengers))
	if len(passengers) > 1 {
		variance := (sumSq - sum*sum/float64(len(passengers))) / float64(len(passengers)-1)
		if variance > 0 {
			data.Stats.StdDev = math.Sqrt(variance)
		}
	}

	data.Stats.StartTime = data.Timestamps[0]
	data.Stats.EndTime = data.Timestamps[len(passengers)-1]

	return data
}
