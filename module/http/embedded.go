//go:build embedded
// +build embedded

package http

import (
	"fmt"

	"github.com/GoMemPipe/mempipe/module"
)

// HTTPModule provides a no-op HTTP implementation for embedded/TinyGo environments.
type HTTPModule struct{}

// NewHTTPModule creates a new embedded HTTP module.
func NewHTTPModule() *HTTPModule { return &HTTPModule{} }

func (m *HTTPModule) Name() string { return "http" }
func (m *HTTPModule) Init() error  { return nil }

var errUnavailable = fmt.Errorf("HTTP not available in embedded mode")

// Get is a no-op in embedded mode.
func (m *HTTPModule) Get(url string) (*Response, error) {
	return &Response{
		Headers: make(map[string]string),
		Error:   errUnavailable,
	}, errUnavailable
}

// Post is a no-op in embedded mode.
func (m *HTTPModule) Post(url, body, contentType string) (*Response, error) {
	return &Response{
		Headers: make(map[string]string),
		Error:   errUnavailable,
	}, errUnavailable
}

// SetTimeout is a no-op in embedded mode.
func (m *HTTPModule) SetTimeout(ms float64) {}

// HealthCheck always returns false in embedded mode.
func (m *HTTPModule) HealthCheck(url string) bool { return false }

func init() {
	module.Register(NewHTTPModule())
}
