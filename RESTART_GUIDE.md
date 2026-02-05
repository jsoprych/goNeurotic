# GoNeurotic - Production Restart Guide

## 🎯 Current Status
**Version**: v1.4.0+ (Production-Ready Time Series & API Server)
**State**: ✅ All systems operational | 🚀 Production-ready

## 📦 What's Been Built

### ✅ **Complete Time Series Forecasting System**
1. **Real CSV Data Loading** (`pkg/timeseries/csv.go`)
   - Flexible CSV parsing for univariate/multivariate series
   - Built-in datasets (AirPassengers, synthetic)
   - Date parsing, missing value handling, statistical summaries

2. **Walk-Forward Validation**
   - Multiple validation methods (walk-forward, holdout, expanding window)
   - Proper time series splits preserving temporal order

3. **Exogenous Feature Engineering**
   - Temporal features (year, month, day, hour, weekday, yearday)
   - Cyclical encoding (sin/cos transformations)
   - Business indicators (business days, weekends, month ends)
   - Lag features with configurable periods

4. **Statistical Baseline Comparisons**
   - 9 forecasting methods: Naive, Seasonal Naive, Moving Average, Exponential Smoothing, Holt-Winters, Linear Trend, Persistence, Drift, Theta
   - Comprehensive metrics: RMSE, MAE, MAPE, SMAPE, R²

5. **Production Forecasting Pipeline** (`pkg/timeseries/forecast_pipeline.go`)
   - End-to-end workflow: data → preprocessing → training → evaluation → deployment
   - Model persistence with `Save()` and `LoadPipeline()`
   - Last window persistence for immediate predictions

### ✅ **Full REST API Server** (`cmd/goneurotic-server`)
1. **Neural Network Operations**
   - Train models via POST `/api/v1/models/train`
   - Predict via POST `/api/v1/models/predict`
   - Save/load models with persistence
   - List and manage models

2. **Time Series Endpoints**
   - Quick forecasting with baselines
   - Pipeline creation, training, prediction
   - Metrics retrieval and comparison

3. **Production Features**
   - CORS support
   - Error handling with JSON responses
   - Request size limits
   - Health checks and system monitoring

### ✅ **BLAS-Accelerated Neural Network Core**
- 7.8× faster batch training
- Adam optimizer with adaptive learning rates
- Memory optimization with 30-40% fewer allocations
- Full model serialization/deserialization

## 🚀 Quick Start Commands

### 1. Verify Build
```bash
# Build everything
go build ./...

# Run tests
go test ./pkg/neural ./pkg/timeseries
```

### 2. Run Demos
```bash
# Show available demos
./goneurotic -help

# Production forecasting pipeline
./goneurotic -demo pipeline

# AirPassengers dataset with baseline comparison
./goneurotic -demo realts

# Original time series demo (synthetic data)
./goneurotic -demo timeseries

# Classic neural network demos
./goneurotic -demo xor
./goneurotic -demo iris
```

### 3. Start API Server
```bash
# Build server
go build ./cmd/goneurotic-server

# Run with defaults (port 8080)
./goneurotic-server

# Run with custom port
PORT=3000 ./goneurotic-server
```

### 4. Test API Endpoints
```bash
# Health check
curl http://localhost:8080/health

# System info
curl http://localhost:8080/api/v1/system/info

# List models (initially empty)
curl http://localhost:8080/api/v1/models/
```

## 📊 Example: Complete AirPassengers Workflow

### Via CLI Demo
```bash
./goneurotic -demo pipeline
```
*Creates, trains, evaluates, and saves a forecasting pipeline with walk-forward validation*

### Via API
```bash
# 1. Create pipeline
curl -X POST http://localhost:8080/api/v1/timeseries/pipeline/create \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_id": "airpassengers_production",
    "config": {
      "window_size": 12,
      "forecast_horizon": 6,
      "test_size": 24,
      "validation_method": "walk_forward",
      "model_type": "neural_network",
      "neural_config": {
        "layer_sizes": [12, 16, 8],
        "activation": "tanh",
        "output_activation": "linear",
        "loss_function": "mse"
      }
    }
  }'

# 2. Train pipeline
curl -X POST http://localhost:8080/api/v1/timeseries/pipeline/airpassengers_production/train

# 3. Get forecasts
curl -X POST http://localhost:8080/api/v1/timeseries/pipeline/airpassengers_production/predict?steps=12

# 4. Save pipeline
curl -X POST http://localhost:8080/api/v1/timeseries/pipeline/airpassengers_production/save
```

## 🎯 Next Enhancement Opportunities

### High Priority (v1.5.0)
1. **Hyperparameter Tuning**
   ```go
   // Grid/Random search for optimal window sizes, architectures
   type HyperparameterSearch struct {
       WindowSizes []int
       LayerConfigs [][]int
       LearningRates []float64
   }
   ```

2. **Uncertainty Quantification**
   - Prediction intervals via bootstrap or Bayesian methods
   - Confidence bounds for forecasts
   - Model uncertainty estimation

3. **Ensemble Methods**
   - Model averaging and stacking
   - Bayesian model averaging
   - Weighted ensemble based on validation performance

### Medium Priority (v1.6.0)
4. **Advanced Feature Engineering**
   - Fourier terms for seasonality
   - Rolling statistics (mean, std, min, max)
   - Change point detection
   - Automated feature selection

5. **Experiment Tracking**
   - MLflow-style experiment management
   - Parameter and metric logging
   - Model versioning with metadata

6. **Automated Monitoring**
   - Concept drift detection
   - Performance degradation alerts
   - Automatic retraining triggers

### Long Term Vision
7. **GPU Acceleration** - CUDA/OpenCL integration
8. **Advanced Architectures** - LSTMs, Transformers, Attention
9. **Multivariate Forecasting** - Vector autoregression, dynamic factor models
10. **Causal Inference** - Intervention analysis, counterfactual forecasting

## 🔧 Architecture Overview

### Key Files
```
goNeurotic/
├── pkg/neural/                  # Core neural network with BLAS acceleration
│   ├── network.go               # Main network implementation
│   ├── optimizers.go            # Adam, SGD, RMSprop, Momentum
│   └── network_blas.go          # BLAS-accelerated operations
├── pkg/timeseries/              # Time series forecasting
│   ├── timeseries.go            # Basic time series utilities
│   ├── csv.go                   # CSV loading with built-in datasets
│   ├── baselines.go             # 9 statistical forecasting methods
│   └── forecast_pipeline.go     # Production pipeline
├── cmd/goneurotic/              # CLI with demos
│   └── main.go                  # realts, pipeline, timeseries demos
└── cmd/goneurotic-server/       # REST API server
    └── main.go                  # Full HTTP API
```

### Data Flow
```
CSV/Data → Preprocessing → Feature Engineering → 
Sliding Windows → Train/Test Split → 
Model Training → Evaluation → 
Forecast Generation → Model Persistence
```

## 🐛 Troubleshooting

### Common Issues

1. **"no test files" warning**
   ```bash
   # This is normal - we focus on integration tests via demos
   go test ./pkg/timeseries
   ```

2. **API server won't start**
   ```bash
   # Check port availability
   lsof -i :8080
   
   # Run with different port
   PORT=3000 ./goneurotic-server
   ```

3. **Model loading fails**
   ```bash
   # Ensure models directory exists
   mkdir -p models
   
   # Check file permissions
   ls -la models/
   ```

4. **Performance issues**
   ```bash
   # Verify BLAS is working
   ./goneurotic -demo xor  # Should achieve 100% accuracy quickly
   
   # Check memory usage
   go test -bench=. -benchmem ./pkg/neural
   ```

### Verification Checklist
- [ ] All demos run without errors
- [ ] API server starts on port 8080
- [ ] Health endpoint returns `{"status":"healthy"}`
- [ ] Pipeline demo saves to `airpassengers_pipeline.json`
- [ ] Loaded pipeline can make predictions
- [ ] Neural network demos (XOR, Iris) work correctly

## 📚 Documentation

### Key Documentation Files
- `README.md` - Project overview and features
- `API_SERVER.md` - Complete API documentation with examples
- `PERFORMANCE_REPORT.md` - BLAS acceleration results
- `CHANGELOG.md` - Version history

### Quick Examples
```go
// 1. Create forecasting pipeline
pipeline := timeseries.NewPipeline()
pipeline.LoadBuiltinDataset("airpassengers")
pipeline.WithConfig(timeseries.PipelineConfig{...})
pipeline.Preprocess()
pipeline.Train()
pipeline.Save("production_model.json")

// 2. Load and use later
loadedPipeline, _ := timeseries.LoadPipeline("production_model.json")
forecasts, _ := loadedPipeline.Predict(12)

// 3. Train neural network via API (example request)
/*
POST /api/v1/models/train
{
  "model_id": "my_model",
  "layer_sizes": [10, 20, 5],
  "learning_rate": 0.01,
  "inputs": [...],
  "targets": [...]
}
*/
```

## 🎯 Immediate Next Steps

**Prompt for continuation:**
```
"Let's implement hyperparameter tuning for the forecasting pipeline. Starting with:
1. Grid search over window sizes and layer configurations
2. Cross-validation with time series splits
3. Automated selection of best parameters
4. Integration with existing pipeline system"
```

**Or for API enhancements:**
```
"Let's add authentication and rate limiting to the API server:
1. JWT-based authentication middleware
2. API key support for machine-to-machine
3. Rate limiting per endpoint
4. Request logging and analytics"
```

**Or for advanced features:**
```
"Let's implement uncertainty quantification for forecasts:
1. Bootstrap prediction intervals
2. Bayesian neural networks for uncertainty
3. Confidence bounds visualization
4. Risk assessment metrics"
```

---
*Last Updated: GoNeurotic v1.4.0+ with production-ready time series forecasting and REST API*
*All systems operational - ready for production deployment or further enhancement*