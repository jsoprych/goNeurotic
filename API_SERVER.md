# GoNeurotic API Server

A production-ready REST API server for neural network training, prediction, and time series forecasting. Built on top of the GoNeurotic neural network library with BLAS acceleration, optimizer system, and comprehensive time series forecasting capabilities.

## Features

- **RESTful API**: Complete HTTP/JSON interface for all GoNeurotic functionality
- **Neural Network Operations**: Train, predict, save, load neural networks
- **Time Series Forecasting**: End-to-end forecasting pipelines with walk-forward validation
- **Production Ready**: Model persistence, error handling, CORS support
- **Scalable**: BLAS-accelerated backend with batch training
- **Extensible**: Easy to add new models, features, and endpoints

## Quick Start

### Installation

```bash
# Build from source
go build ./cmd/goneurotic-server

# Or install globally
go install ./cmd/goneurotic-server
```

### Running the Server

```bash
# Run with default settings (port 8080)
./goneurotic-server

# Run with custom port
PORT=3000 ./goneurotic-server

# Run with custom models directory
MODELS_DIR=./my_models ./goneurotic-server
```

### Configuration

Environment variables:
- `PORT`: Server port (default: 8080)
- `MODELS_DIR`: Directory for model storage (default: ./models)
- `MAX_UPLOAD_SIZE`: Maximum upload size in bytes (default: 10MB)

Or modify `ServerConfig` in `cmd/goneurotic-server/main.go`.

## API Endpoints

### Health Check

**GET /health**

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-02-03T21:00:00Z",
  "version": "1.0.0"
}
```

### System Information

**GET /api/v1/system/info**

```bash
curl http://localhost:8080/api/v1/system/info
```

Response:
```json
{
  "server_version": "1.0.0",
  "go_version": "1.24.0",
  "models_dir": "./models",
  "models_loaded": 3,
  "pipelines_loaded": 2,
  "timestamp": "2024-02-03T21:00:00Z"
}
```

## Neural Network Operations

### Train a Model

**POST /api/v1/models/train**

Train a neural network with specified architecture and data.

Request:
```bash
curl -X POST http://localhost:8080/api/v1/models/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "xor_classifier",
    "layer_sizes": [2, 4, 1],
    "learning_rate": 0.5,
    "activation": "sigmoid",
    "output_activation": "sigmoid",
    "loss_function": "mse",
    "epochs": 1000,
    "batch_size": 4,
    "inputs": [[0, 0], [0, 1], [1, 0], [1, 1]],
    "targets": [[0], [1], [1], [0]]
  }'
```

Response:
```json
{
  "model_id": "xor_classifier",
  "training_time": "245.172799ms",
  "final_loss": 0.002345,
  "message": "Model trained successfully"
}
```

### Make Predictions

**POST /api/v1/models/predict**

Use a trained model to make predictions.

Request:
```bash
curl -X POST http://localhost:8080/api/v1/models/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "xor_classifier",
    "input": [0, 1]
  }'
```

Response:
```json
{
  "model_id": "xor_classifier",
  "input": [0, 1],
  "output": [0.987],
  "timestamp": "2024-02-03T21:00:00Z"
}
```

### List Models

**GET /api/v1/models/**

List all loaded models.

```bash
curl http://localhost:8080/api/v1/models/
```

Response:
```json
{
  "models": [
    {
      "id": "xor_classifier",
      "layer_sizes": [2, 4, 1],
      "input_size": 2,
      "output_size": 1
    },
    {
      "id": "iris_classifier",
      "layer_sizes": [4, 8, 3],
      "input_size": 4,
      "output_size": 3
    }
  ],
  "count": 2,
  "timestamp": "2024-02-03T21:00:00Z"
}
```

### Save Model

**POST /api/v1/models/{id}/save**

Save a model to disk for persistence.

```bash
curl -X POST http://localhost:8080/api/v1/models/xor_classifier/save
```

Response:
```json
{
  "model_id": "xor_classifier",
  "filename": "./models/xor_classifier.json",
  "message": "Model saved successfully",
  "timestamp": "2024-02-03T21:00:00Z"
}
```

### Load Model

**POST /api/v1/models/{id}/load**

Load a previously saved model.

```bash
curl -X POST http://localhost:8080/api/v1/models/xor_classifier/load
```

Response:
```json
{
  "model_id": "xor_classifier",
  "filename": "./models/xor_classifier.json",
  "message": "Model loaded successfully",
  "timestamp": "2024-02-03T21:00:00Z"
}
```

### Delete Model

**DELETE /api/v1/models/{id}**

Delete a model from memory and disk.

```bash
curl -X DELETE http://localhost:8080/api/v1/models/xor_classifier
```

Response:
```json
{
  "model_id": "xor_classifier",
  "message": "Model deleted successfully",
  "timestamp": "2024-02-03T21:00:00Z"
}
```

## Time Series Forecasting

### Quick Forecast

**POST /api/v1/timeseries/forecast**

Generate forecasts using statistical baselines.

Request:
```bash
curl -X POST http://localhost:8080/api/v1/timeseries/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "series": [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118],
    "window_size": 6,
    "horizon": 3,
    "method": "seasonal_naive"
  }'
```

Response:
```json
{
  "forecast": [115, 126, 141],
  "metrics": {
    "RMSE": 0.0,
    "MAE": 0.0,
    "MAPE": 0.0,
    "SMAPE": 0.0,
    "R2": 0.0
  },
  "method": "seasonal_naive",
  "timestamp": "2024-02-03T21:00:00Z"
}
```

Available methods: `naive`, `seasonal_naive`, `moving_average`, `exponential_smoothing`

### Create Forecasting Pipeline

**POST /api/v1/timeseries/pipeline/create**

Create a production forecasting pipeline with neural network backend.

Request:
```bash
curl -X POST http://localhost:8080/api/v1/timeseries/pipeline/create \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_id": "airpassengers_pipeline",
    "config": {
      "window_size": 12,
      "forecast_horizon": 6,
      "step_size": 1,
      "test_size": 24,
      "validation_method": "walk_forward",
      "model_type": "neural_network",
      "neural_config": {
        "layer_sizes": [12, 16, 8],
        "activation": "tanh",
        "output_activation": "linear",
        "loss_function": "mse",
        "optimizer": "adam"
      },
      "include_date_features": true,
      "include_lag_features": true,
      "lags": [1, 2, 12],
      "normalization": "zscore",
      "epochs": 200,
      "batch_size": 16,
      "learning_rate": 0.01,
      "early_stopping_patience": 15
    }
  }'
```

Response:
```json
{
  "pipeline_id": "airpassengers_pipeline",
  "message": "Pipeline created successfully",
  "created_at": "2024-02-03T21:00:00Z"
}
```

### Train Pipeline

**POST /api/v1/timeseries/pipeline/{id}/train**

Train a forecasting pipeline (uses AirPassengers dataset for demo).

```bash
curl -X POST http://localhost:8080/api/v1/timeseries/pipeline/airpassengers_pipeline/train
```

Response:
```json
{
  "pipeline_id": "airpassengers_pipeline",
  "metrics": {
    "neural_network": {
      "RMSE": 98.2,
      "MAE": 73.9,
      "MAPE": 15.3,
      "SMAPE": 0.0,
      "R2": 0.0
    }
  },
  "message": "Pipeline trained successfully",
  "timestamp": "2024-02-03T21:00:00Z"
}
```

### Make Pipeline Predictions

**POST /api/v1/timeseries/pipeline/{id}/predict**

Generate forecasts using a trained pipeline.

```bash
curl -X POST http://localhost:8080/api/v1/timeseries/pipeline/airpassengers_pipeline/predict?steps=6
```

Response:
```json
{
  "pipeline_id": "airpassengers_pipeline",
  "steps": 6,
  "forecasts": [[395, 375, 424, 407, 363, 378]],
  "timestamp": "2024-02-03T21:00:00Z"
}
```

### Get Pipeline Metrics

**GET /api/v1/timeseries/pipeline/{id}/metrics**

Retrieve evaluation metrics for a pipeline.

```bash
curl http://localhost:8080/api/v1/timeseries/pipeline/airpassengers_pipeline/metrics
```

Response:
```json
{
  "pipeline_id": "airpassengers_pipeline",
  "metrics": {
    "neural_network": {
      "RMSE": 98.2,
      "MAE": 73.9,
      "MAPE": 15.3,
      "SMAPE": 0.0,
      "R2": 0.0
    }
  },
  "config": {
    "window_size": 12,
    "forecast_horizon": 6,
    "test_size": 24,
    "validation_method": "walk_forward",
    "model_type": "neural_network"
  },
  "data_stats": {
    "count": 144,
    "min": 104,
    "max": 622,
    "mean": 280.3,
    "std_dev": 119.6,
    "start_time": "1949-01-01T00:00:00Z",
    "end_time": "1960-12-01T00:00:00Z"
  },
  "training_time": "140.128492ms",
  "evaluation_time": "15.234567ms",
  "timestamp": "2024-02-03T21:00:00Z"
}
```

### Save/Load Pipeline

**POST /api/v1/timeseries/pipeline/{id}/save**
**POST /api/v1/timeseries/pipeline/{id}/load**

Persist pipelines to disk and reload them.

## Complete Example: XOR Classifier

```bash
# 1. Train XOR classifier
curl -X POST http://localhost:8080/api/v1/models/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "xor_demo",
    "layer_sizes": [2, 4, 1],
    "learning_rate": 0.5,
    "activation": "sigmoid",
    "output_activation": "sigmoid",
    "loss_function": "mse",
    "epochs": 1000,
    "batch_size": 4,
    "inputs": [[0, 0], [0, 1], [1, 0], [1, 1]],
    "targets": [[0], [1], [1], [0]]
  }'

# 2. Test predictions
curl -X POST http://localhost:8080/api/v1/models/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "xor_demo",
    "input": [1, 0]
  }'

# 3. Save model
curl -X POST http://localhost:8080/api/v1/models/xor_demo/save

# 4. List models
curl http://localhost:8080/api/v1/models/
```

## Complete Example: AirPassengers Forecasting

```bash
# 1. Create forecasting pipeline
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
      },
      "include_date_features": true,
      "normalization": "zscore",
      "epochs": 200
    }
  }'

# 2. Train pipeline
curl -X POST http://localhost:8080/api/v1/timeseries/pipeline/airpassengers_production/train

# 3. Get forecasts
curl -X POST http://localhost:8080/api/v1/timeseries/pipeline/airpassengers_production/predict?steps=12

# 4. Save pipeline
curl -X POST http://localhost:8080/api/v1/timeseries/pipeline/airpassengers_production/save
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200 OK`: Success
- `400 Bad Request`: Invalid input
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error response format:
```json
{
  "error": "Model not found",
  "message": "Model 'nonexistent' not found",
  "code": 404,
  "timestamp": "2024-02-03T21:00:00Z"
}
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM golang:1.24-alpine AS builder
WORKDIR /app
COPY . .
RUN go build -o goneurotic-server ./cmd/goneurotic-server

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/goneurotic-server .
COPY --from=builder /app/models ./models
EXPOSE 8080
CMD ["./goneurotic-server"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: goneurotic-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: goneurotic
  template:
    metadata:
      labels:
        app: goneurotic
    spec:
      containers:
      - name: server
        image: goneurotic/server:latest
        ports:
        - containerPort: 8080
        env:
        - name: PORT
          value: "8080"
        - name: MODELS_DIR
          value: "/data/models"
        volumeMounts:
        - name: models-data
          mountPath: /data/models
      volumes:
      - name: models-data
        persistentVolumeClaim:
          claimName: models-pvc
```

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name api.goneurotic.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## Monitoring and Observability

### Health Checks

```bash
# Basic health check
curl -f http://localhost:8080/health

# With timeout
curl -m 5 http://localhost:8080/health

# With retry
curl --retry 3 --retry-delay 1 http://localhost:8080/health
```

### Metrics Integration

The API server can be integrated with Prometheus/Grafana by adding metrics middleware:

```go
import "github.com/prometheus/client_golang/prometheus/promhttp"

// Add to router
s.router.Get("/metrics", promhttp.Handler().ServeHTTP)
```

## Security Considerations

1. **Authentication**: Add API key or JWT authentication for production
2. **Rate Limiting**: Implement rate limiting for public endpoints
3. **Input Validation**: Validate all incoming JSON data
4. **CORS**: Configure CORS origins for browser access
5. **HTTPS**: Always use HTTPS in production

## Development

### Adding New Endpoints

1. Add route in `setupRoutes()` function
2. Create handler method
3. Define request/response structures
4. Implement business logic
5. Add error handling

### Testing

```bash
# Run server in background
./goneurotic-server &

# Test endpoints
curl http://localhost:8080/health
curl http://localhost:8080/api/v1/system/info

# Kill server
pkill goneurotic-server
```

## License

MIT License - see LICENSE file for details.

## Support

- GitHub Issues: https://github.com/yourusername/goneurotic/issues
- Documentation: https://goneurotic.readthedocs.io/
- Email: support@goneurotic.com