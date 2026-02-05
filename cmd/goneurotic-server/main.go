package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"

	"goneurotic/pkg/neural"
	"goneurotic/pkg/timeseries"
)

// ServerConfig holds server configuration
type ServerConfig struct {
	Port          int    `json:"port"`
	ModelsDir     string `json:"models_dir"`
	MaxUploadSize int64  `json:"max_upload_size"`
	EnableCORS    bool   `json:"enable_cors"`
	LogLevel      string `json:"log_level"`
}

// DefaultConfig returns default server configuration
func DefaultConfig() ServerConfig {
	return ServerConfig{
		Port:          8080,
		ModelsDir:     "./models",
		MaxUploadSize: 10 * 1024 * 1024, // 10MB
		EnableCORS:    true,
		LogLevel:      "info",
	}
}

// ModelManager manages loaded neural network models
type ModelManager struct {
	models map[string]*neural.Network
	mu     sync.RWMutex
}

// NewModelManager creates a new model manager
func NewModelManager() *ModelManager {
	return &ModelManager{
		models: make(map[string]*neural.Network),
	}
}

// PipelineManager manages forecasting pipelines
type PipelineManager struct {
	pipelines map[string]*timeseries.ForecastPipeline
	mu        sync.RWMutex
}

// NewPipelineManager creates a new pipeline manager
func NewPipelineManager() *PipelineManager {
	return &PipelineManager{
		pipelines: make(map[string]*timeseries.ForecastPipeline),
	}
}

// Server structure
type Server struct {
	config          ServerConfig
	modelManager    *ModelManager
	pipelineManager *PipelineManager
	router          *chi.Mux
}

// NewServer creates a new API server
func NewServer(config ServerConfig) *Server {
	// Ensure models directory exists
	if err := os.MkdirAll(config.ModelsDir, 0755); err != nil {
		log.Fatalf("Failed to create models directory: %v", err)
	}

	s := &Server{
		config:          config,
		modelManager:    NewModelManager(),
		pipelineManager: NewPipelineManager(),
		router:          chi.NewRouter(),
	}

	s.setupMiddleware()
	s.setupRoutes()

	return s
}

// setupMiddleware configures server middleware
func (s *Server) setupMiddleware() {
	// Basic middleware
	s.router.Use(middleware.RequestID)
	s.router.Use(middleware.RealIP)
	s.router.Use(middleware.Logger)
	s.router.Use(middleware.Recoverer)
	s.router.Use(middleware.Timeout(60 * time.Second))

	// CORS
	if s.config.EnableCORS {
		s.router.Use(cors.Handler(cors.Options{
			AllowedOrigins:   []string{"*"},
			AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
			AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-CSRF-Token"},
			ExposedHeaders:   []string{"Link"},
			AllowCredentials: false,
			MaxAge:           300,
		}))
	}

	// Request size limit
	s.router.Use(middleware.RequestSize(s.config.MaxUploadSize))
}

// setupRoutes configures API routes
func (s *Server) setupRoutes() {
	// Health check
	s.router.Get("/health", s.handleHealth)

	// API v1 routes
	s.router.Route("/api/v1", func(r chi.Router) {
		// Neural Network endpoints
		r.Route("/models", func(r chi.Router) {
			r.Post("/train", s.handleTrain)
			r.Post("/predict", s.handlePredict)
			r.Get("/", s.handleListModels)
			r.Post("/{id}/save", s.handleSaveModel)
			r.Post("/{id}/load", s.handleLoadModel)
			r.Delete("/{id}", s.handleDeleteModel)
		})

		// Time Series endpoints
		r.Route("/timeseries", func(r chi.Router) {
			r.Post("/forecast", s.handleTimeSeriesForecast)
			r.Post("/pipeline/create", s.handleCreatePipeline)
			r.Post("/pipeline/{id}/train", s.handleTrainPipeline)
			r.Post("/pipeline/{id}/predict", s.handlePredictPipeline)
			r.Post("/pipeline/{id}/save", s.handleSavePipeline)
			r.Post("/pipeline/{id}/load", s.handleLoadPipeline)
			r.Get("/pipeline/{id}/metrics", s.handlePipelineMetrics)
		})

		// System endpoints
		r.Get("/system/info", s.handleSystemInfo)
		r.Get("/system/stats", s.handleSystemStats)
	})
}

// API Request/Response structures

// TrainRequest represents a training request
type TrainRequest struct {
	ModelID          string      `json:"model_id"`
	LayerSizes       []int       `json:"layer_sizes"`
	LearningRate     float64     `json:"learning_rate"`
	Activation       string      `json:"activation"`
	OutputActivation string      `json:"output_activation"`
	LossFunction     string      `json:"loss_function"`
	Epochs           int         `json:"epochs"`
	BatchSize        int         `json:"batch_size"`
	Inputs           [][]float64 `json:"inputs"`
	Targets          [][]float64 `json:"targets"`
}

// TrainResponse represents a training response
type TrainResponse struct {
	ModelID      string        `json:"model_id"`
	TrainingTime time.Duration `json:"training_time"`
	FinalLoss    float64       `json:"final_loss"`
	Message      string        `json:"message"`
}

// PredictRequest represents a prediction request
type PredictRequest struct {
	ModelID string    `json:"model_id"`
	Input   []float64 `json:"input"`
}

// PredictResponse represents a prediction response
type PredictResponse struct {
	ModelID   string      `json:"model_id"`
	Input     []float64   `json:"input"`
	Output    []float64   `json:"output"`
	Timestamp time.Time   `json:"timestamp"`
}

// TimeSeriesForecastRequest represents a time series forecast request
type TimeSeriesForecastRequest struct {
	Series    []float64 `json:"series"`
	WindowSize int      `json:"window_size"`
	Horizon   int       `json:"horizon"`
	Method    string    `json:"method"` // "neural", "naive", "seasonal_naive", etc.
}

// TimeSeriesForecastResponse represents a time series forecast response
type TimeSeriesForecastResponse struct {
	Forecast  []float64                    `json:"forecast"`
	Metrics   timeseries.ForecastMetrics   `json:"metrics"`
	Method    string                       `json:"method"`
	Timestamp time.Time                    `json:"timestamp"`
}

// PipelineCreateRequest represents a pipeline creation request
type PipelineCreateRequest struct {
	PipelineID string                     `json:"pipeline_id"`
	Config     timeseries.PipelineConfig  `json:"config"`
}

// PipelineCreateResponse represents a pipeline creation response
type PipelineCreateResponse struct {
	PipelineID string    `json:"pipeline_id"`
	Message    string    `json:"message"`
	CreatedAt  time.Time `json:"created_at"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error     string    `json:"error"`
	Message   string    `json:"message"`
	Code      int       `json:"code"`
	Timestamp time.Time `json:"timestamp"`
}

// Helper function to send JSON responses
func sendJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(data); err != nil {
		log.Printf("Failed to encode JSON response: %v", err)
	}
}

// Helper function to send error responses
func sendError(w http.ResponseWriter, status int, err error, message string) {
	resp := ErrorResponse{
		Error:     err.Error(),
		Message:   message,
		Code:      status,
		Timestamp: time.Now(),
	}
	sendJSON(w, status, resp)
}

// Handler implementations

// handleHealth handles health check requests
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now(),
		"version":   "1.0.0",
	}
	sendJSON(w, http.StatusOK, response)
}

// handleTrain handles model training requests
func (s *Server) handleTrain(w http.ResponseWriter, r *http.Request) {
	var req TrainRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, http.StatusBadRequest, err, "Invalid request body")
		return
	}

	if req.ModelID == "" {
		sendError(w, http.StatusBadRequest, fmt.Errorf("model_id is required"), "Model ID is required")
		return
	}

	if len(req.Inputs) == 0 || len(req.Targets) == 0 {
		sendError(w, http.StatusBadRequest, fmt.Errorf("training data required"), "Inputs and targets are required")
		return
	}

	// Create neural network configuration
	config := neural.NetworkConfig{
		LayerSizes:       req.LayerSizes,
		LearningRate:     req.LearningRate,
		Activation:       getActivation(req.Activation),
		OutputActivation: getActivation(req.OutputActivation),
		LossFunction:     getLossFunction(req.LossFunction),
	}

	// Create and train network
	network := neural.NewNetwork(config)
	startTime := time.Now()

	// Train with mini-batches if batch size specified
	var finalLoss float64
	if req.BatchSize > 0 && req.BatchSize < len(req.Inputs) {
		finalLoss = network.BatchTrain(req.Inputs, req.Targets)
	} else {
		// Individual training
		for epoch := 0; epoch < req.Epochs; epoch++ {
			for i := range req.Inputs {
				finalLoss = network.Train(req.Inputs[i], req.Targets[i])
			}
		}
	}

	trainingTime := time.Since(startTime)

	// Store model
	s.modelManager.mu.Lock()
	s.modelManager.models[req.ModelID] = network
	s.modelManager.mu.Unlock()

	response := TrainResponse{
		ModelID:      req.ModelID,
		TrainingTime: trainingTime,
		FinalLoss:    finalLoss,
		Message:      "Model trained successfully",
	}

	sendJSON(w, http.StatusOK, response)
}

// handlePredict handles prediction requests
func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	var req PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, http.StatusBadRequest, err, "Invalid request body")
		return
	}

	if req.ModelID == "" {
		sendError(w, http.StatusBadRequest, fmt.Errorf("model_id is required"), "Model ID is required")
		return
	}

	// Get model
	s.modelManager.mu.RLock()
	network, exists := s.modelManager.models[req.ModelID]
	s.modelManager.mu.RUnlock()

	if !exists {
		sendError(w, http.StatusNotFound, fmt.Errorf("model not found"), "Model not found")
		return
	}

	// Make prediction
	output := network.Predict(req.Input)

	response := PredictResponse{
		ModelID:   req.ModelID,
		Input:     req.Input,
		Output:    output,
		Timestamp: time.Now(),
	}

	sendJSON(w, http.StatusOK, response)
}

// handleListModels lists all loaded models
func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	s.modelManager.mu.RLock()
	defer s.modelManager.mu.RUnlock()

	models := make([]map[string]interface{}, 0, len(s.modelManager.models))
	for id, network := range s.modelManager.models {
		layerSizes := []int{}
		if len(network.LayerSizes) > 0 {
			layerSizes = network.LayerSizes
		}
		inputSize := 0
		if len(layerSizes) > 0 {
			inputSize = layerSizes[0]
		}
		outputSize := 0
		if len(layerSizes) > 0 {
			outputSize = layerSizes[len(layerSizes)-1]
		}

		modelInfo := map[string]interface{}{
			"id":          id,
			"layer_sizes": layerSizes,
			"input_size":  inputSize,
			"output_size": outputSize,
		}
		models = append(models, modelInfo)
	}

	response := map[string]interface{}{
		"models":    models,
		"count":     len(models),
		"timestamp": time.Now(),
	}

	sendJSON(w, http.StatusOK, response)
}

// handleSaveModel saves a model to disk
func (s *Server) handleSaveModel(w http.ResponseWriter, r *http.Request) {
	modelID := chi.URLParam(r, "id")
	if modelID == "" {
		sendError(w, http.StatusBadRequest, fmt.Errorf("model_id is required"), "Model ID is required")
		return
	}

	// Get model
	s.modelManager.mu.RLock()
	network, exists := s.modelManager.models[modelID]
	s.modelManager.mu.RUnlock()

	if !exists {
		sendError(w, http.StatusNotFound, fmt.Errorf("model not found"), "Model not found")
		return
	}

	// Save model
	filename := filepath.Join(s.config.ModelsDir, modelID+".json")
	if err := network.Save(filename); err != nil {
		sendError(w, http.StatusInternalServerError, err, "Failed to save model")
		return
	}

	response := map[string]interface{}{
		"model_id":  modelID,
		"filename":  filename,
		"message":   "Model saved successfully",
		"timestamp": time.Now(),
	}

	sendJSON(w, http.StatusOK, response)
}

// handleLoadModel loads a model from disk
func (s *Server) handleLoadModel(w http.ResponseWriter, r *http.Request) {
	modelID := chi.URLParam(r, "id")
	if modelID == "" {
		sendError(w, http.StatusBadRequest, fmt.Errorf("model_id is required"), "Model ID is required")
		return
	}

	// Load model
	filename := filepath.Join(s.config.ModelsDir, modelID+".json")
	network, err := neural.Load(filename)
	if err != nil {
		sendError(w, http.StatusInternalServerError, err, "Failed to load model")
		return
	}

	// Store model
	s.modelManager.mu.Lock()
	s.modelManager.models[modelID] = network
	s.modelManager.mu.Unlock()

	response := map[string]interface{}{
		"model_id":  modelID,
		"filename":  filename,
		"message":   "Model loaded successfully",
		"timestamp": time.Now(),
	}

	sendJSON(w, http.StatusOK, response)
}

// handleDeleteModel deletes a model
func (s *Server) handleDeleteModel(w http.ResponseWriter, r *http.Request) {
	modelID := chi.URLParam(r, "id")
	if modelID == "" {
		sendError(w, http.StatusBadRequest, fmt.Errorf("model_id is required"), "Model ID is required")
		return
	}

	// Remove from memory
	s.modelManager.mu.Lock()
	delete(s.modelManager.models, modelID)
	s.modelManager.mu.Unlock()

	// Remove from disk if exists
	filename := filepath.Join(s.config.ModelsDir, modelID+".json")
	if _, err := os.Stat(filename); err == nil {
		if err := os.Remove(filename); err != nil {
			log.Printf("Warning: Failed to delete model file %s: %v", filename, err)
		}
	}

	response := map[string]interface{}{
		"model_id":  modelID,
		"message":   "Model deleted successfully",
		"timestamp": time.Now(),
	}

	sendJSON(w, http.StatusOK, response)
}

// handleTimeSeriesForecast handles time series forecasting requests
func (s *Server) handleTimeSeriesForecast(w http.ResponseWriter, r *http.Request) {
	var req TimeSeriesForecastRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, http.StatusBadRequest, err, "Invalid request body")
		return
	}

	if len(req.Series) == 0 {
		sendError(w, http.StatusBadRequest, fmt.Errorf("series is empty"), "Time series data is required")
		return
	}

	if req.WindowSize <= 0 {
		req.WindowSize = 24 // Default window size
	}

	if req.Horizon <= 0 {
		req.Horizon = 1 // Default horizon
	}

	// Perform forecast based on method
	var forecast []float64
	var metrics timeseries.ForecastMetrics

	switch req.Method {
	case "neural":
		// For neural network forecasting, we'd need to train a model first
		// This is a simplified version - in production you'd want a trained model
		fallthrough
	case "naive":
		forecast = timeseries.BaselineForecast(req.Series, timeseries.BaselineConfig{
			Method:  "naive",
			Horizon: req.Horizon,
		})
	case "seasonal_naive":
		forecast = timeseries.BaselineForecast(req.Series, timeseries.BaselineConfig{
			Method:      "seasonal_naive",
			Horizon:     req.Horizon,
			Seasonality: 12, // Default monthly seasonality
		})
	case "moving_average":
		forecast = timeseries.BaselineForecast(req.Series, timeseries.BaselineConfig{
			Method:  "moving_average",
			Horizon: req.Horizon,
			Window:  3, // Default window
		})
	case "exponential_smoothing":
		forecast = timeseries.BaselineForecast(req.Series, timeseries.BaselineConfig{
			Method:  "exponential_smoothing",
			Horizon: req.Horizon,
			Alpha:   0.3, // Default smoothing
		})
	default:
		// Default to naive
		forecast = timeseries.BaselineForecast(req.Series, timeseries.BaselineConfig{
			Method:  "naive",
			Horizon: req.Horizon,
		})
		req.Method = "naive"
	}

	// Calculate simple metrics (in production you'd want actual vs predicted)
	// For demo purposes, we'll return placeholder metrics
	metrics = timeseries.ForecastMetrics{
		RMSE:  0.0,
		MAE:   0.0,
		MAPE:  0.0,
		SMAPE: 0.0,
		R2:    0.0,
	}

	response := TimeSeriesForecastResponse{
		Forecast:  forecast,
		Metrics:   metrics,
		Method:    req.Method,
		Timestamp: time.Now(),
	}

	sendJSON(w, http.StatusOK, response)
}

// handleCreatePipeline creates a new forecasting pipeline
func (s *Server) handleCreatePipeline(w http.ResponseWriter, r *http.Request) {
	var req PipelineCreateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, http.StatusBadRequest, err, "Invalid request body")
		return
	}

	if req.PipelineID == "" {
		sendError(w, http.StatusBadRequest, fmt.Errorf("pipeline_id is required"), "Pipeline ID is required")
		return
	}

	// Create pipeline
	pipeline := timeseries.NewPipeline().WithConfig(req.Config)

	// Store pipeline
	s.pipelineManager.mu.Lock()
	s.pipelineManager.pipelines[req.PipelineID] = pipeline
	s.pipelineManager.mu.Unlock()

	response := PipelineCreateResponse{
		PipelineID: req.PipelineID,
		Message:    "Pipeline created successfully",
		CreatedAt:  time.Now(),
	}

	sendJSON(w, http.StatusOK, response)
}

// handleTrainPipeline trains a forecasting pipeline
func (s *Server) handleTrainPipeline(w http.ResponseWriter, r *http.Request) {
	pipelineID := chi.URLParam(r, "id")
	if pipelineID == "" {
		sendError(w, http.StatusBadRequest, fmt.Errorf("pipeline_id is required"), "Pipeline ID is required")
		return
	}

	// Get pipeline
	s.pipelineManager.mu.RLock()
	pipeline, exists := s.pipelineManager.pipelines[pipelineID]
	s.pipelineManager.mu.RUnlock()

	if !exists {
		sendError(w, http.StatusNotFound, fmt.Errorf("pipeline not found"), "Pipeline not found")
		return
	}

	// For this demo, we'll load the AirPassengers dataset
	// In production, you'd load data from the request
	if err := pipeline.LoadBuiltinDataset("airpassengers"); err != nil {
		sendError(w, http.StatusInternalServerError, err, "Failed to load dataset")
		return
	}

	// Preprocess and train
	if err := pipeline.Preprocess(); err != nil {
		sendError(w, http.StatusInternalServerError, err, "Failed to preprocess data")
		return
	}

	if err := pipeline.Train(); err != nil {
		sendError(w, http.StatusInternalServerError, err, "Failed to train pipeline")
		return
	}

	// Evaluate to get metrics
	metrics, err := pipeline.Evaluate()
	if err != nil {
		sendError(w, http.StatusInternalServerError, err, "Failed to evaluate pipeline")
		return
	}

	response := map[string]interface{}{
		"pipeline_id": pipelineID,
		"metrics":     metrics,
		"message":     "Pipeline trained successfully",
		"timestamp":   time.Now(),
	}

	sendJSON(w, http.StatusOK, response)
}

// handlePredictPipeline makes predictions with a pipeline
func (s *Server) handlePredictPipeline(w http.ResponseWriter, r *http.Request) {
	pipelineID := chi.URLParam(r, "id")
	if pipelineID == "" {
		sendError(w, http.StatusBadRequest, fmt.Errorf("pipeline_id is required"), "Pipeline ID is required")
		return
	}

	// Get pipeline
	s.pipelineManager.mu.RLock()
	pipeline, exists := s.pipelineManager.pipelines[pipelineID]
	s.pipelineManager.mu.RUnlock()

	if !exists {
		sendError(w, http.StatusNotFound, fmt.Errorf("pipeline not found"), "Pipeline not found")
		return
	}

	// Parse steps from query parameter or request body
	steps := 6 // Default
	if stepsStr := r.URL.Query().Get("steps"); stepsStr != "" {
		if s, err := strconv.Atoi(stepsStr); err == nil && s > 0 {
			steps = s
		}
	}

	// Make prediction
	forecasts, err := pipeline.Predict(steps)
	if err != nil {
		sendError(w, http.StatusInternalServerError, err, "Failed to make prediction")
		return
	}

	response := map[string]interface{}{
		"pipeline_id": pipelineID,
		"steps":       steps,
		"forecasts":   forecasts,
		"timestamp":   time.Now(),
	}

	sendJSON(w, http.StatusOK, response)
}

// handleSavePipeline saves a pipeline to disk
func (s *Server) handleSavePipeline(w http.ResponseWriter, r *http.Request) {
	pipelineID := chi.URLParam(r, "id")
	if pipelineID == "" {
		sendError(w, http.StatusBadRequest, fmt.Errorf("pipeline_id is required"), "Pipeline ID is required")
		return
	}

	// Get pipeline
	s.pipelineManager.mu.RLock()
	pipeline, exists := s.pipelineManager.pipelines[pipelineID]
	s.pipelineManager.mu.RUnlock()

	if !exists {
		sendError(w, http.StatusNotFound, fmt.Errorf("pipeline not found"), "Pipeline not found")
		return
	}

	// Save pipeline
	pipelineDir := filepath.Join(s.config.ModelsDir, "pipelines")
	if err := os.MkdirAll(pipelineDir, 0755); err != nil {
		sendError(w, http.StatusInternalServerError, err, "Failed to create pipeline directory")
		return
	}

	filename := filepath.Join(pipelineDir, pipelineID+".json")
	if err := pipeline.Save(filename); err != nil {
		sendError(w, http.StatusInternalServerError, err, "Failed to save pipeline")
		return
	}

	response := map[string]interface{}{
		"pipeline_id": pipelineID,
		"filename":    filename,
		"message":     "Pipeline saved successfully",
		"timestamp":   time.Now(),
	}

	sendJSON(w, http.StatusOK, response)
}

// handleLoadPipeline loads a pipeline from disk
func (s *Server) handleLoadPipeline(w http.ResponseWriter, r *http.Request) {
	pipelineID := chi.URLParam(r, "id")
	if pipelineID == "" {
		sendError(w, http.StatusBadRequest, fmt.Errorf("pipeline_id is required"), "Pipeline ID is required")
		return
	}

	// Load pipeline
	filename := filepath.Join(s.config.ModelsDir, "pipelines", pipelineID+".json")
	pipeline, err := timeseries.LoadPipeline(filename)
	if err != nil {
		sendError(w, http.StatusInternalServerError, err, "Failed to load pipeline")
		return
	}

	// Store pipeline
	s.pipelineManager.mu.Lock()
	s.pipelineManager.pipelines[pipelineID] = pipeline
	s.pipelineManager.mu.Unlock()

	response := map[string]interface{}{
		"pipeline_id": pipelineID,
		"filename":    filename,
		"message":     "Pipeline loaded successfully",
		"timestamp":   time.Now(),
	}

	sendJSON(w, http.StatusOK, response)
}

// handlePipelineMetrics gets pipeline evaluation metrics
func (s *Server) handlePipelineMetrics(w http.ResponseWriter, r *http.Request) {
	pipelineID := chi.URLParam(r, "id")
	if pipelineID == "" {
		sendError(w, http.StatusBadRequest, fmt.Errorf("pipeline_id is required"), "Pipeline ID is required")
		return
	}

	// Get pipeline
	s.pipelineManager.mu.RLock()
	pipeline, exists := s.pipelineManager.pipelines[pipelineID]
	s.pipelineManager.mu.RUnlock()

	if !exists {
		sendError(w, http.StatusNotFound, fmt.Errorf("pipeline not found"), "Pipeline not found")
		return
	}

	results := pipeline.GetResults()

	response := map[string]interface{}{
		"pipeline_id":     pipelineID,
		"metrics":         results.Metrics,
		"config":          results.Config,
		"data_stats":      results.DataStats,
		"training_time":   results.TrainingTime,
		"evaluation_time": results.EvaluationTime,
		"timestamp":       time.Now(),
	}

	sendJSON(w, http.StatusOK, response)
}

// handleSystemInfo returns system information
func (s *Server) handleSystemInfo(w http.ResponseWriter, r *http.Request) {
	info := map[string]interface{}{
		"server_version":   "1.0.0",
		"go_version":       "1.24.0",
		"models_dir":       s.config.ModelsDir,
		"models_loaded":    len(s.modelManager.models),
		"pipelines_loaded": len(s.pipelineManager.pipelines),
		"timestamp":        time.Now(),
	}

	sendJSON(w, http.StatusOK, info)
}

// handleSystemStats returns system statistics
func (s *Server) handleSystemStats(w http.ResponseWriter, r *http.Request) {
	stats := map[string]interface{}{
		"memory_usage":    "N/A",
		"goroutines":      0,
		"uptime":          "N/A",
		"total_requests":  0,
		"timestamp":       time.Now(),
	}

	sendJSON(w, http.StatusOK, stats)
}

// Helper functions for activation/loss function mapping
func getActivation(name string) neural.ActivationFunc {
	switch name {
	case "sigmoid":
		return neural.Sigmoid
	case "tanh":
		return neural.Tanh
	case "relu":
		return neural.ReLU
	case "linear":
		return neural.Linear
	default:
		return neural.Tanh // Default
	}
}

func getLossFunction(name string) neural.LossFunc {
	switch name {
	case "mse":
		return neural.MeanSquaredError
	case "binary_crossentropy":
		return neural.BinaryCrossEntropy
	default:
		return neural.MeanSquaredError // Default
	}
}

// Run starts the HTTP server
func (s *Server) Run() error {
	addr := fmt.Sprintf(":%d", s.config.Port)
	log.Printf("Starting GoNeurotic API server on %s", addr)
	log.Printf("Models directory: %s", s.config.ModelsDir)
	log.Printf("CORS enabled: %v", s.config.EnableCORS)

	return http.ListenAndServe(addr, s.router)
}

func main() {
	// Load configuration (could be from environment variables or config file)
	config := DefaultConfig()

	// Override from environment variables if present
	if portStr := os.Getenv("PORT"); portStr != "" {
		if port, err := strconv.Atoi(portStr); err == nil {
			config.Port = port
		}
	}

	if modelsDir := os.Getenv("MODELS_DIR"); modelsDir != "" {
		config.ModelsDir = modelsDir
	}

	// Create and run server
	server := NewServer(config)
	if err := server.Run(); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
