# MobileNetV3-Large Image Classification with MemPipe

Run MobileNetV3-Large (ImageNet 1000-class) inference with **zero heap allocations** on the hot path. Works on Linux, WASM, and bare-metal microcontrollers via TinyGo.

## Prerequisites

Convert the MobileNetV3 ONNX model to `.mpmodel`:

```bash
# Install the converter (one-time setup)
pip install -e tools/mempipe-convert

# Convert the ONNX model
mempipe-convert onnx --transformer MobileNet-v3.onnx -o mnv3.mpmodel
```

If `pip install` is restricted on your system, use `PYTHONPATH` directly:

```bash
PYTHONPATH=tools/mempipe-convert python3 -m mempipe_convert.cli \
    onnx --transformer MobileNet-v3.onnx -o mnv3.mpmodel
```

Expected output (0 skipped ops):
```
Converted: MobileNet-v3.onnx → mnv3.mpmodel
  Ops: 142, Tensors: 273
  Weights: 21,883,352 bytes
```

## Run on Linux (x86_64 / ARM64)

### Quick smoke test (random input)

```bash
go run ./examples/mobilenet_v3 -model mnv3.mpmodel -random -v
```

### Classify an image (PNG / JPEG / GIF)

The Go example natively decodes PNG, JPEG, and GIF images, applies center-crop + bilinear resize to 224×224, and runs ImageNet normalization automatically:

```bash
# Classify the included NYC street photo
go run ./examples/mobilenet_v3 -model mnv3.mpmodel \
    -image examples/mobilenet_v3/nyc-complete-street-brooklyn-cropped.png -top 10
```

A result image with classification labels is saved automatically as `<input>_result.png`:

```
Top predictions:
  1. class  906  window shade                    0.2935 (29.3%)
  2. class  487  mobile phone                    0.1168 (11.7%)
  ...

Result saved: examples/mobilenet_v3/nyc-complete-street-brooklyn-cropped_result.png
```

Use `-out` to specify a custom output path, or `-no-save` to skip saving.

### Convert WebP or other formats with the Python helper

Go's stdlib doesn't support WebP. Use the included `convert_image.py` to convert any image format (WebP, BMP, TIFF, etc.) to either PNG or raw float32:

```bash
# Convert WebP → PNG (then use the Go example)
python3 examples/mobilenet_v3/convert_image.py photo.webp -o photo.png --format png

# Or convert to raw float32 binary (for embedded/headless use)
python3 examples/mobilenet_v3/convert_image.py photo.webp -o photo.rgb
```

The Python converter applies the same ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) and center-crop preprocessing.

**Requirements:** `pip install Pillow numpy`

### With raw float32 input

For headless/embedded pipelines, you can also pass pre-processed `.rgb` files:

```bash
# Prepare raw input with convert_image.py
python3 examples/mobilenet_v3/convert_image.py photo.jpg -o photo.rgb

# Run inference
go run ./examples/mobilenet_v3 -model mnv3.mpmodel -image photo.rgb -top 5
```

### Benchmark

```bash
go run ./examples/mobilenet_v3 -model mnv3.mpmodel -random -iter 100 -v
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-model` | `mnv3.mpmodel` | Path to converted `.mpmodel` file |
| `-image` | | Image file: PNG/JPEG/GIF (native) or raw `.rgb` float32 |
| `-out` | `<input>_result.png` | Output result image path |
| `-random` | `false` | Use random input (for benchmarking / smoke test) |
| `-top` | `5` | Number of top predictions to display |
| `-iter` | `1` | Inference iterations (for benchmarking) |
| `-seed` | `42` | Random seed for `-random` mode |
| `-no-save` | `false` | Skip saving the result image |
| `-v` | `false` | Verbose: per-iteration timing |

## Deploy to a Microcontroller (TinyGo)

MobileNetV3-Large (20.9 MB of weights) requires a board with **at least 64 MB RAM**
(weights + activations arena). Suitable embedded targets include:

- **Raspberry Pi Pico W** (with external PSRAM) or **Raspberry Pi 4/5** (ARM64 Linux)
- **ESP32-S3** with PSRAM (8 MB — use INT8 quantized model, see below)
- **STM32H7** series (up to 1 MB internal + external SDRAM)
- **Any ARM Cortex-M/A board** supported by [TinyGo](https://tinygo.org/docs/reference/microcontrollers/)

### Step 1: Quantize the model (optional, for memory-constrained boards)

INT8 quantization reduces weights from ~21 MB to ~5.3 MB:

```bash
mempipe-convert onnx --transformer MobileNet-v3.onnx -o mnv3-int8.mpmodel --quantize int8
```

### Step 2: Embed the model in the firmware

On embedded targets there's no filesystem. Embed the model bytes at compile time:

```go
//go:build embedded

package main

import (
    _ "embed"
    "github.com/GoMemPipe/mempipe/inference"
)

//go:embed mnv3.mpmodel
var modelData []byte

func main() {
    // Load from embedded bytes (no filesystem needed)
    model, err := inference.LoadModelFromBytes(modelData)
    if err != nil {
        // handle error (e.g. blink LED)
        return
    }

    engine, err := inference.NewEngine(model)
    if err != nil {
        return
    }

    // Fill input tensor from camera / sensor
    inputTensor := engine.InputTensors()[0]
    inputSlice := inputTensor.Float32s()
    // ... copy image data into inputSlice ...

    // Run inference (zero-alloc hot path)
    outputs, err := engine.InferTensor()
    if err != nil {
        return
    }

    // Read classification result
    logits := outputs[0].Float32s()
    bestClass := argmax(logits)
    // ... use bestClass (e.g. display on LCD, trigger GPIO) ...
}
```

### Step 3: Build with TinyGo

```bash
# List supported targets
tinygo targets

# Build for a specific board (example: Raspberry Pi Pico)
tinygo build -target=pico -tags embedded -o firmware.uf2 ./examples/mobilenet_v3

# Build for ARM Cortex-M (e.g. STM32 Nucleo)
tinygo build -target=nucleo-f722ze -tags embedded -o firmware.bin ./examples/mobilenet_v3

# Flash to the board
tinygo flash -target=pico ./examples/mobilenet_v3
```

### Step 4: Cross-compile for ARM Linux boards

For boards running Linux (Raspberry Pi, BeagleBone, NVIDIA Jetson):

```bash
# Raspberry Pi 4/5 (ARM64)
GOOS=linux GOARCH=arm64 go build -o mobilenetv3-arm64 ./examples/mobilenet_v3

# Raspberry Pi 3 / Zero 2 W (ARMv7)
GOOS=linux GOARCH=arm GOARM=7 go build -o mobilenetv3-armv7 ./examples/mobilenet_v3

# Copy binary + model to the board
scp mobilenetv3-arm64 mnv3.mpmodel pi@raspberrypi:~/

# Run on the board
ssh pi@raspberrypi './mobilenetv3-arm64 -model mnv3.mpmodel -random -v'
```

## Architecture

The inference pipeline:

1. **Converter** (`mempipe-convert`): Reads the ONNX model, maps all 142 ops
   (Conv2D with padding/strides/groups, HardSwish, HardSigmoid, GlobalAvgPool,
   BatchNorm, ReLU, Flatten, Gemm, etc.) to the `.mpmodel` binary format.

2. **Model loader** (`inference.LoadModel` / `LoadModelFromBytes`): Parses the
   header → metadata → graph → weights from the `.mpmodel` file.

3. **Engine** (`inference.NewEngine`): Allocates a single contiguous arena for all
   weights + activations + I/O tensors. Compiles the 142-op graph with
   pre-resolved tensor pointers.

4. **Inference** (`engine.InferTensor`): Executes the full MobileNetV3 forward pass
   with **zero heap allocations**. All tensor data lives in the pre-allocated arena.

### MobileNetV3-Large key operations

| Operator | Count | Description |
|----------|-------|-------------|
| Conv2D | 62 | Depthwise separable + pointwise convolutions |
| HardSwish | 21 | $x \cdot \text{clip}(x/6 + 0.5, 0, 1)$ — primary activation |
| ReLU | 19 | Standard ReLU (early layers) |
| Add | 10 | Residual connections |
| GlobalAvgPool2D | 9 | Squeeze-excitation spatial reduction |
| HardSigmoid | 8 | $\text{clip}(\alpha x + \beta, 0, 1)$ — SE gate |
| Mul | 8 | SE channel-wise reweighting |
| BatchNorm | 0* | Folded into Conv weights at export time |
| Gemm (Dense) | 2 | Final classifier head (1280 → 1000) |
| Flatten | 1 | Before classifier |
| Div | 1 | Normalization |
| Sub | 1 | Normalization |

## Memory requirements

| Configuration | Weights | Arena (total) | Suitable for |
|--------------|---------|---------------|--------------|
| FP32 (default) | 20.9 MB | ~58 MB | Linux, RPi 4/5, WASM |
| INT8 quantized | ~5.3 MB | ~15 MB | ESP32-S3 + PSRAM, STM32H7 |
