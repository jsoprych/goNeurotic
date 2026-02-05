package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
)

// ServerConfig holds server configuration
type ServerConfig struct {
	Port            int    `json:"port"`
	APIServerURL    string `json:"api_server_url"`
	TemplatesDir    string `json:"templates_dir"`
	StaticDir       string `json:"static_dir"`
	EnableCORS      bool   `json:"enable_cors"`
	LogLevel        string `json:"log_level"`
	CacheTemplates  bool   `json:"cache_templates"`
}

// DefaultConfig returns default server configuration
func DefaultConfig() *ServerConfig {
	return &ServerConfig{
		Port:            3000,
		APIServerURL:    "http://localhost:8080",
		TemplatesDir:    "./web/templates",
		StaticDir:       "./web/static",
		EnableCORS:      true,
		LogLevel:        "info",
		CacheTemplates:  true,
	}
}

// Tutorial represents a learning tutorial
type Tutorial struct {
	ID          string   `json:"id"`
	Title       string   `json:"title"`
	Description string   `json:"description"`
	Chapter     int      `json:"chapter"`
	Part        string   `json:"part"`
	Difficulty  string   `json:"difficulty"` // beginner, intermediate, advanced
	Estimated   string   `json:"estimated"`  // estimated time
	Objectives  []string `json:"objectives"`
	Topics      []string `json:"topics"`
	Exercises   []string `json:"exercises"`
	Data        []string `json:"data"`       // required datasets
	HasVisual   bool     `json:"has_visual"` // has visualization component
	HasCode     bool     `json:"has_code"`   // has code examples
	Completed   bool     `json:"completed"`  // user completion status
}

// TemplateData holds data for template rendering
type TemplateData struct {
	Title       string
	Tutorial    *Tutorial
	Tutorials   []Tutorial
	CurrentTime string
	Config      *ServerConfig
	User        *UserProgress
	Data        interface{} // Additional data for specific pages
}

// UserProgress tracks user learning progress
type UserProgress struct {
	ID         string                 `json:"id"`
	Name       string                 `json:"name"`
	Started    time.Time              `json:"started"`
	LastActive time.Time              `json:"last_active"`
	Completed  map[string]time.Time   `json:"completed"` // tutorial_id -> completion time
	Scores     map[string]float64     `json:"scores"`    // tutorial_id -> score (0-100)
	Badges     []string               `json:"badges"`
	TimeSpent  map[string]time.Duration `json:"time_spent"` // tutorial_id -> time spent
}

// Server represents the educational web server
type Server struct {
	config        *ServerConfig
	router        *chi.Mux
	templates     map[string]*template.Template
	templatesMu   sync.RWMutex
	apiProxy      *httputil.ReverseProxy
	apiClient     *http.Client
	tutorials     []Tutorial
	tutorialsMu   sync.RWMutex
	userProgress  map[string]*UserProgress
	progressMu    sync.RWMutex
}

// NewServer creates a new educational server
func NewServer(config *ServerConfig) *Server {
	s := &Server{
		config:       config,
		router:       chi.NewRouter(),
		templates:    make(map[string]*template.Template),
		apiClient:    &http.Client{Timeout: 30 * time.Second},
		tutorials:    make([]Tutorial, 0),
		userProgress: make(map[string]*UserProgress),
	}

	// Setup API proxy
	if apiURL, err := url.Parse(config.APIServerURL); err == nil {
		s.apiProxy = httputil.NewSingleHostReverseProxy(apiURL)
		s.apiProxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
			log.Printf("API proxy error: %v", err)
			http.Error(w, "API server unavailable", http.StatusBadGateway)
		}
	} else {
		log.Printf("Warning: Invalid API server URL: %v", err)
		s.apiProxy = nil
	}

	// Load tutorials
	s.loadTutorials()

	// Setup middleware and routes
	s.setupMiddleware()
	s.setupRoutes()

	// Load templates
	if config.CacheTemplates {
		s.loadTemplates()
	}

	return s
}

// setupMiddleware configures server middleware
func (s *Server) setupMiddleware() {
	// Request logging
	s.router.Use(middleware.Logger)

	// Recover from panics
	s.router.Use(middleware.Recoverer)

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

	// Request ID
	s.router.Use(middleware.RequestID)

	// Real IP
	s.router.Use(middleware.RealIP)

	// Timeout
	s.router.Use(middleware.Timeout(60 * time.Second))
}

// setupRoutes configures server routes
func (s *Server) setupRoutes() {
	// Static files
	s.router.Handle("/static/*", http.StripPrefix("/static/",
		http.FileServer(http.Dir(s.config.StaticDir))))

	// Health check
	s.router.Get("/health", s.handleHealth)

	// API proxy (forward to API server)
	s.router.HandleFunc("/api/*", s.handleAPIProxy)

	// Home page
	s.router.Get("/", s.handleHome)

	// Tutorial listing
	s.router.Get("/tutorials", s.handleTutorialList)

	// Individual tutorial
	s.router.Get("/tutorial/{id}", s.handleTutorial)

	// Tutorial chapters by part
	s.router.Get("/tutorials/part/{part}", s.handleTutorialsByPart)

	// Interactive visualizations
	s.router.Get("/visualize/network", s.handleNetworkVisualization)
	s.router.Get("/visualize/xor", s.handleXORVisualization)
	s.router.Get("/visualize/timeseries", s.handleTimeSeriesVisualization)
	s.router.Get("/visualize/activations", s.handleActivationVisualization)

	// Progress tracking
	s.router.Get("/progress", s.handleProgress)
	s.router.Post("/progress/{tutorial_id}/complete", s.handleProgressComplete)
	s.router.Get("/progress/badges", s.handleBadges)

	// About and help
	s.router.Get("/about", s.handleAbout)
	s.router.Get("/help", s.handleHelp)
	s.router.Get("/syllabus", s.handleSyllabus)

	// 404 handler
	s.router.NotFound(s.handleNotFound)
}

// loadTutorials loads the tutorial catalog
func (s *Server) loadTutorials() {
	s.tutorials = []Tutorial{
		{
			ID:          "intro-neural-networks",
			Title:       "Introduction to Neural Networks",
			Description: "Learn the fundamentals of neural networks, from biological inspiration to artificial neurons.",
			Chapter:     1,
			Part:        "fundamentals",
			Difficulty:  "beginner",
			Estimated:   "2-3 hours",
			Objectives: []string{
				"Understand what neural networks are and why they're powerful",
				"Learn the basic components: neurons, layers, activation functions",
				"Explore the history and applications of neural networks",
			},
			Topics: []string{
				"What is Machine Learning?",
				"Biological vs Artificial Neurons",
				"The Perceptron: First Neural Network",
				"Forward Propagation Explained",
				"Activation Functions: Sigmoid, Tanh, ReLU",
			},
			Exercises: []string{
				"Interactive neuron visualization",
				"Activation function playground",
				"Simple forward pass calculation",
			},
			Data:       []string{"None (conceptual)"},
			HasVisual:  true,
			HasCode:    false,
			Completed:  false,
		},
		{
			ID:          "training-backpropagation",
			Title:       "Training Neural Networks with Backpropagation",
			Description: "Master the mathematics of backpropagation and gradient descent.",
			Chapter:     2,
			Part:        "fundamentals",
			Difficulty:  "beginner",
			Estimated:   "3-4 hours",
			Objectives: []string{
				"Master the mathematics of backpropagation",
				"Understand loss functions and gradient descent",
				"Implement training loops from scratch",
			},
			Topics: []string{
				"The Need for Learning",
				"Loss Functions: MSE, MAE, Cross-Entropy",
				"Gradient Descent: Intuition & Mathematics",
				"Backpropagation Step-by-Step",
				"Learning Rates and Convergence",
			},
			Exercises: []string{
				"Visual gradient descent on simple functions",
				"Backpropagation tracing through network layers",
				"Learning rate optimization game",
			},
			Data:       []string{"Simple linear dataset"},
			HasVisual:  true,
			HasCode:    true,
			Completed:  false,
		},
		{
			ID:          "xor-problem",
			Title:       "The Classic XOR Problem",
			Description: "Solve the XOR problem - the 'Hello World' of neural networks.",
			Chapter:     3,
			Part:        "fundamentals",
			Difficulty:  "beginner",
			Estimated:   "2-3 hours",
			Objectives: []string{
				"Understand why XOR is the 'Hello World' of neural networks",
				"Build and train a network that solves XOR",
				"Visualize decision boundaries and learning process",
			},
			Topics: []string{
				"Linear Separability Challenge",
				"Network Architecture for XOR",
				"Training Strategy",
				"Decision Boundary Evolution",
				"Hyperparameter Tuning",
			},
			Exercises: []string{
				"Interactive XOR training with real-time visualization",
				"Architecture experimenter (2-4-1 vs 2-3-1 vs 2-8-4-1)",
				"Training progress dashboard",
				"Prediction playground",
			},
			Data:       []string{"XOR truth table"},
			HasVisual:  true,
			HasCode:    true,
			Completed:  false,
		},
		{
			ID:          "intro-time-series",
			Title:       "Introduction to Time Series Analysis",
			Description: "Learn time series components and basic statistical forecasting methods.",
			Chapter:     7,
			Part:        "timeseries",
			Difficulty:  "intermediate",
			Estimated:   "3-4 hours",
			Objectives: []string{
				"Understand time series components: trend, seasonality, noise",
				"Learn basic statistical forecasting methods",
				"Prepare time series data for neural networks",
			},
			Topics: []string{
				"Time Series Components Decomposition",
				"Stationarity and Differencing",
				"Autocorrelation and Partial Autocorrelation",
				"Simple Forecasting Methods: Naïve, Drift, Average",
				"Data Splitting for Time Series",
			},
			Exercises: []string{
				"Time series decomposition visualizer",
				"Stationarity test playground",
				"ACF/PACF plot generator",
				"Simple forecast comparison",
			},
			Data:       []string{"AirPassengers dataset", "synthetic trends"},
			HasVisual:  true,
			HasCode:    true,
			Completed:  false,
		},
		{
			ID:          "airpassengers-challenge",
			Title:       "The AirPassengers Challenge",
			Description: "Apply everything learned to the classic AirPassengers forecasting problem.",
			Chapter:     11,
			Part:        "timeseries",
			Difficulty:  "intermediate",
			Estimated:   "5-6 hours",
			Objectives: []string{
				"Apply everything learned to a classic forecasting problem",
				"Compare neural networks vs statistical baselines",
				"Implement walk-forward validation properly",
			},
			Topics: []string{
				"AirPassengers Dataset Analysis",
				"Comprehensive Feature Engineering",
				"Walk-Forward Validation Implementation",
				"Model Comparison Framework",
				"Production Pipeline Design",
			},
			Exercises: []string{
				"Complete AirPassengers forecasting pipeline",
				"Model comparison dashboard (9 baselines + NN)",
				"Walk-forward validation visualization",
				"Horizon-by-horizon accuracy analysis",
			},
			Data:       []string{"Full AirPassengers dataset (144 points)"},
			HasVisual:  true,
			HasCode:    true,
			Completed:  false,
		},
	}
}

// loadTemplates loads and caches HTML templates
func (s *Server) loadTemplates() {
	s.templatesMu.Lock()
	defer s.templatesMu.Unlock()

	// Define base template
	baseTemplate := filepath.Join(s.config.TemplatesDir, "base.html")

	// Load individual templates
	templates := []string{
		"home.html",
		"tutorial_list.html",
		"tutorial.html",
		"visualization.html",
		"progress.html",
		"about.html",
		"help.html",
		"syllabus.html",
		"404.html",
	}

	for _, tmpl := range templates {
		tmplPath := filepath.Join(s.config.TemplatesDir, tmpl)
		if _, err := os.Stat(tmplPath); err == nil {
			parsed, err := template.ParseFiles(baseTemplate, tmplPath)
			if err != nil {
				log.Printf("Error parsing template %s: %v", tmpl, err)
				continue
			}
			s.templates[tmpl] = parsed
		}
	}
}

// renderTemplate renders a template with data
func (s *Server) renderTemplate(w http.ResponseWriter, tmpl string, data *TemplateData) {
	s.templatesMu.RLock()
	t, ok := s.templates[tmpl]
	s.templatesMu.RUnlock()

	if !ok {
		// Template not cached, load it
		tmplPath := filepath.Join(s.config.TemplatesDir, tmpl)
		basePath := filepath.Join(s.config.TemplatesDir, "base.html")

		var err error
		if _, err := os.Stat(basePath); err == nil {
			t, err = template.ParseFiles(basePath, tmplPath)
		} else {
			t, err = template.ParseFiles(tmplPath)
		}

		if err != nil {
			log.Printf("Error loading template %s: %v", tmpl, err)
			http.Error(w, "Template error", http.StatusInternalServerError)
			return
		}

		if s.config.CacheTemplates {
			s.templatesMu.Lock()
			s.templates[tmpl] = t
			s.templatesMu.Unlock()
		}
	}

	// Add current time if not set
	if data.CurrentTime == "" {
		data.CurrentTime = time.Now().Format(time.RFC1123)
	}

	// Execute template
	var buf bytes.Buffer
	if err := t.Execute(&buf, data); err != nil {
		log.Printf("Error executing template %s: %v", tmpl, err)
		http.Error(w, "Template execution error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(buf.Bytes())
}

// handleHealth handles health check requests
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"status":    "healthy",
		"service":   "goneurotic-learn",
		"timestamp": time.Now().UTC(),
		"version":   "1.0.0",
		"tutorials": len(s.tutorials),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleAPIProxy proxies requests to the API server
func (s *Server) handleAPIProxy(w http.ResponseWriter, r *http.Request) {
	if s.apiProxy == nil {
		http.Error(w, "API server not configured", http.StatusServiceUnavailable)
		return
	}

	// Forward to API server (keep /api prefix)
	s.apiProxy.ServeHTTP(w, r)
}

// handleHome handles the home page
func (s *Server) handleHome(w http.ResponseWriter, r *http.Request) {
	data := &TemplateData{
		Title:       "GoNeurotic-Learn: Neural Networks & Time Series",
		Tutorials:   s.getFeaturedTutorials(),
		Config:      s.config,
		CurrentTime: time.Now().Format("January 2, 2006 15:04:05"),
	}

	s.renderTemplate(w, "home.html", data)
}

// handleTutorialList handles tutorial listing page
func (s *Server) handleTutorialList(w http.ResponseWriter, r *http.Request) {
	data := &TemplateData{
		Title:       "All Tutorials",
		Tutorials:   s.tutorials,
		Config:      s.config,
	}

	s.renderTemplate(w, "tutorial_list.html", data)
}

// handleTutorial handles individual tutorial pages
func (s *Server) handleTutorial(w http.ResponseWriter, r *http.Request) {
	tutorialID := chi.URLParam(r, "id")

	var tutorial *Tutorial
	for i := range s.tutorials {
		if s.tutorials[i].ID == tutorialID {
			tutorial = &s.tutorials[i]
			break
		}
	}

	if tutorial == nil {
		s.handleNotFound(w, r)
		return
	}

	data := &TemplateData{
		Title:       tutorial.Title,
		Tutorial:    tutorial,
		Config:      s.config,
	}

	s.renderTemplate(w, "tutorial.html", data)
}

// handleTutorialsByPart handles tutorials filtered by part
func (s *Server) handleTutorialsByPart(w http.ResponseWriter, r *http.Request) {
	part := chi.URLParam(r, "part")

	var filtered []Tutorial
	for _, t := range s.tutorials {
		if t.Part == part {
			filtered = append(filtered, t)
		}
	}

	data := &TemplateData{
		Title:       fmt.Sprintf("%s Tutorials", strings.Title(part)),
		Tutorials:   filtered,
		Config:      s.config,
		Data:        part,
	}

	s.renderTemplate(w, "tutorial_list.html", data)
}

// handleNetworkVisualization handles neural network visualization
func (s *Server) handleNetworkVisualization(w http.ResponseWriter, r *http.Request) {
	data := &TemplateData{
		Title: "Neural Network Visualizer",
		Config: s.config,
	}

	s.renderTemplate(w, "visualization.html", data)
}

// handleXORVisualization handles XOR problem visualization
func (s *Server) handleXORVisualization(w http.ResponseWriter, r *http.Request) {
	data := &TemplateData{
		Title: "XOR Problem Visualizer",
		Config: s.config,
		Data: map[string]interface{}{
			"type": "xor",
		},
	}

	s.renderTemplate(w, "visualization.html", data)
}

// handleTimeSeriesVisualization handles time series visualization
func (s *Server) handleTimeSeriesVisualization(w http.ResponseWriter, r *http.Request) {
	data := &TemplateData{
		Title: "Time Series Visualizer",
		Config: s.config,
		Data: map[string]interface{}{
			"type": "timeseries",
		},
	}

	s.renderTemplate(w, "visualization.html", data)
}

// handleActivationVisualization handles activation function visualization
func (s *Server) handleActivationVisualization(w http.ResponseWriter, r *http.Request) {
	data := &TemplateData{
		Title: "Activation Function Visualizer",
		Config: s.config,
		Data: map[string]interface{}{
			"type": "activations",
		},
	}

	s.renderTemplate(w, "visualization.html", data)
}

// handleProgress handles user progress page
func (s *Server) handleProgress(w http.ResponseWriter, r *http.Request) {
	// For now, create a demo user progress
	user := &UserProgress{
		ID:         "demo-user",
		Name:       "Demo User",
		Started:    time.Now().Add(-24 * time.Hour),
		LastActive: time.Now(),
		Completed:  make(map[string]time.Time),
		Scores:     make(map[string]float64),
		Badges:     []string{"beginner", "first-tutorial"},
		TimeSpent:  make(map[string]time.Duration),
	}

	data := &TemplateData{
		Title:  "Learning Progress",
		Config: s.config,
		User:   user,
	}

	s.renderTemplate(w, "progress.html", data)
}

// handleProgressComplete marks a tutorial as complete
func (s *Server) handleProgressComplete(w http.ResponseWriter, r *http.Request) {
	tutorialID := chi.URLParam(r, "id")

	// In a real app, this would update user progress in a database
	response := map[string]interface{}{
		"success":    true,
		"tutorial_id": tutorialID,
		"completed_at": time.Now().UTC(),
		"message":    "Tutorial marked as complete!",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleBadges shows user badges
func (s *Server) handleBadges(w http.ResponseWriter, r *http.Request) {
	badges := []map[string]interface{}{
		{
			"id":          "beginner",
			"name":        "Neural Network Beginner",
			"description": "Completed first neural network tutorial",
			"icon":        "🎓",
			"earned":      true,
		},
		{
			"id":          "xor-master",
			"name":        "XOR Master",
			"description": "Solved the XOR problem with 95%+ accuracy",
			"icon":        "🧠",
			"earned":      false,
		},
		{
			"id":          "timeseries-explorer",
			"name":        "Time Series Explorer",
			"description": "Completed first time series tutorial",
			"icon":        "📈",
			"earned":      false,
		},
		{
			"id":          "airpassengers-champion",
			"name":        "AirPassengers Champion",
			"description": "Achieved top performance on AirPassengers challenge",
			"icon":        "✈️",
			"earned":      false,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"badges": badges,
		"count":  len(badges),
		"earned": 1,
	})
}

// handleAbout handles about page
func (s *Server) handleAbout(w http.ResponseWriter, r *http.Request) {
	data := &TemplateData{
		Title: "About GoNeurotic-Learn",
		Config: s.config,
	}

	s.renderTemplate(w, "about.html", data)
}

// handleHelp handles help page
func (s *Server) handleHelp(w http.ResponseWriter, r *http.Request) {
	data := &TemplateData{
		Title: "Help & Documentation",
		Config: s.config,
	}

	s.renderTemplate(w, "help.html", data)
}

// handleSyllabus handles syllabus page
func (s *Server) handleSyllabus(w http.ResponseWriter, r *http.Request) {
	data := &TemplateData{
		Title: "Complete Learning Syllabus",
		Config: s.config,
		Tutorials: s.tutorials,
	}

	s.renderTemplate(w, "syllabus.html", data)
}

// handleNotFound handles 404 errors
func (s *Server) handleNotFound(w http.ResponseWriter, r *http.Request) {
	data := &TemplateData{
		Title: "Page Not Found",
		Config: s.config,
	}

	w.WriteHeader(http.StatusNotFound)
	s.renderTemplate(w, "404.html", data)
}

// getFeaturedTutorials returns featured tutorials for the home page
func (s *Server) getFeaturedTutorials() []Tutorial {
	featuredIDs := []string{
		"intro-neural-networks",
		"xor-problem",
		"intro-time-series",
		"airpassengers-challenge",
	}

	var featured []Tutorial
	for _, id := range featuredIDs {
		for _, t := range s.tutorials {
			if t.ID == id {
				featured = append(featured, t)
				break
			}
		}
	}

	return featured
}

// Run starts the server
func (s *Server) Run() error {
	log.Printf("Starting GoNeurotic-Learn server on :%d", s.config.Port)
	log.Printf("API Server: %s", s.config.APIServerURL)
	log.Printf("Templates: %s", s.config.TemplatesDir)
	log.Printf("Static files: %s", s.config.StaticDir)

	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", s.config.Port),
		Handler:      s.router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	return server.ListenAndServe()
}

func main() {
	// Load configuration
	config := DefaultConfig()

	// Override from environment variables
	if portStr := os.Getenv("PORT"); portStr != "" {
		if port, err := strconv.Atoi(portStr); err == nil {
			config.Port = port
		}
	}

	if apiURL := os.Getenv("API_SERVER_URL"); apiURL != "" {
		config.APIServerURL = apiURL
	}

	if templatesDir := os.Getenv("TEMPLATES_DIR"); templatesDir != "" {
		config.TemplatesDir = templatesDir
	}

	if staticDir := os.Getenv("STATIC_DIR"); staticDir != "" {
		config.StaticDir = staticDir
	}

	// Create and run server
	server := NewServer(config)

	// Ensure static and template directories exist
	if err := os.MkdirAll(config.StaticDir, 0755); err != nil {
		log.Printf("Warning: Could not create static directory: %v", err)
	}

	if err := os.MkdirAll(config.TemplatesDir, 0755); err != nil {
		log.Printf("Warning: Could not create templates directory: %v", err)
	}

	// Start server
	if err := server.Run(); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
