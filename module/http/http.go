//go:build !wasm && !embedded
// +build !wasm,!embedded

package http

import (
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/GoMemPipe/mempipe/module"
)

// HTTPModule provides HTTP client functions (native implementation).
type HTTPModule struct {
	client *http.Client
}

// NewHTTPModule creates a new HTTP module.
func NewHTTPModule() *HTTPModule {
	return &HTTPModule{
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (m *HTTPModule) Name() string { return "http" }
func (m *HTTPModule) Init() error  { return nil }

// --- Typed public methods ---

// Get performs an HTTP GET request.
func (m *HTTPModule) Get(url string) (*Response, error) {
	start := time.Now()
	resp, err := m.client.Get(url)
	elapsed := time.Since(start).Seconds() * 1000

	if err != nil {
		return &Response{
			StatusCode:   0,
			Headers:      make(map[string]string),
			ResponseTime: elapsed,
			Error:        err,
		}, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	headers := make(map[string]string)
	for key, values := range resp.Header {
		headers[key] = strings.Join(values, ", ")
	}

	return &Response{
		StatusCode:   resp.StatusCode,
		Body:         string(body),
		Headers:      headers,
		ResponseTime: elapsed,
	}, nil
}

// Post performs an HTTP POST request.
func (m *HTTPModule) Post(url, body, contentType string) (*Response, error) {
	start := time.Now()
	resp, err := m.client.Post(url, contentType, strings.NewReader(body))
	elapsed := time.Since(start).Seconds() * 1000

	if err != nil {
		return &Response{
			StatusCode:   0,
			Headers:      make(map[string]string),
			ResponseTime: elapsed,
			Error:        err,
		}, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	headers := make(map[string]string)
	for key, values := range resp.Header {
		headers[key] = strings.Join(values, ", ")
	}

	return &Response{
		StatusCode:   resp.StatusCode,
		Body:         string(respBody),
		Headers:      headers,
		ResponseTime: elapsed,
	}, nil
}

// SetTimeout sets the HTTP client timeout.
func (m *HTTPModule) SetTimeout(ms float64) {
	m.client.Timeout = time.Duration(ms) * time.Millisecond
}

// HealthCheck returns true if the URL responds with a 2xx status code.
func (m *HTTPModule) HealthCheck(url string) bool {
	resp, err := m.Get(url)
	if err != nil || resp.Error != nil {
		return false
	}
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}

func init() {
	module.Register(NewHTTPModule())
}
