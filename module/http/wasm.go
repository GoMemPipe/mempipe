//go:build wasm
// +build wasm

package http

import (
	"fmt"
	"syscall/js"
	"time"

	"github.com/GoMemPipe/mempipe/module"
)

// HTTPModule provides HTTP client functions via JavaScript fetch API (WASM).
type HTTPModule struct {
	timeout float64 // timeout in milliseconds
}

// NewHTTPModule creates a new WASM HTTP module.
func NewHTTPModule() *HTTPModule {
	return &HTTPModule{timeout: 30000}
}

func (m *HTTPModule) Name() string { return "http" }
func (m *HTTPModule) Init() error  { return nil }

// --- Typed public methods ---

// Get performs an HTTP GET request via JavaScript fetch.
func (m *HTTPModule) Get(url string) (*Response, error) {
	start := time.Now()
	done := make(chan *Response, 1)

	fetchFunc := js.Global().Get("fetch")
	promise := fetchFunc.Invoke(url)

	thenFunc := js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		response := args[0]
		statusCode := response.Get("status").Int()
		textPromise := response.Call("text")
		textPromise.Call("then", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
			body := args[0].String()
			elapsed := time.Since(start).Seconds() * 1000
			done <- &Response{
				StatusCode:   statusCode,
				Body:         body,
				Headers:      make(map[string]string),
				ResponseTime: elapsed,
			}
			return nil
		}))
		return nil
	})

	catchFunc := js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		elapsed := time.Since(start).Seconds() * 1000
		errMsg := args[0].Get("message").String()
		done <- &Response{
			Headers:      make(map[string]string),
			ResponseTime: elapsed,
			Error:        fmt.Errorf("fetch error: %s", errMsg),
		}
		return nil
	})

	promise.Call("then", thenFunc).Call("catch", catchFunc)

	select {
	case resp := <-done:
		return resp, resp.Error
	case <-time.After(time.Duration(m.timeout) * time.Millisecond):
		err := fmt.Errorf("request timeout")
		return &Response{Error: err}, err
	}
}

// Post performs an HTTP POST request via JavaScript fetch.
func (m *HTTPModule) Post(url, body, contentType string) (*Response, error) {
	start := time.Now()
	done := make(chan *Response, 1)

	options := js.Global().Get("Object").New()
	options.Set("method", "POST")
	options.Set("body", body)
	headers := js.Global().Get("Object").New()
	headers.Set("Content-Type", contentType)
	options.Set("headers", headers)

	fetchFunc := js.Global().Get("fetch")
	promise := fetchFunc.Invoke(url, options)

	thenFunc := js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		response := args[0]
		statusCode := response.Get("status").Int()
		textPromise := response.Call("text")
		textPromise.Call("then", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
			respBody := args[0].String()
			elapsed := time.Since(start).Seconds() * 1000
			done <- &Response{
				StatusCode:   statusCode,
				Body:         respBody,
				Headers:      make(map[string]string),
				ResponseTime: elapsed,
			}
			return nil
		}))
		return nil
	})

	catchFunc := js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		elapsed := time.Since(start).Seconds() * 1000
		errMsg := args[0].Get("message").String()
		done <- &Response{
			Headers:      make(map[string]string),
			ResponseTime: elapsed,
			Error:        fmt.Errorf("fetch error: %s", errMsg),
		}
		return nil
	})

	promise.Call("then", thenFunc).Call("catch", catchFunc)

	select {
	case resp := <-done:
		return resp, resp.Error
	case <-time.After(time.Duration(m.timeout) * time.Millisecond):
		err := fmt.Errorf("request timeout")
		return &Response{Error: err}, err
	}
}

// SetTimeout sets the HTTP request timeout.
func (m *HTTPModule) SetTimeout(ms float64) {
	m.timeout = ms
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
