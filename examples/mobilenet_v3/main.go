// Command mobilenet_v3 runs MobileNetV3-Large image classification with zero
// heap allocations on the hot path.
//
// Convert the ONNX model first:
//
//	PYTHONPATH=tools/mempipe-convert python3 -m mempipe_convert.cli onnx --transformer MobileNet-v3.onnx -o mnv3.mpmodel
//
// Then run inference:
//
//	go run ./examples/mobilenet_v3 -model mnv3.mpmodel -image examples/mobilenet_v3/nyc-complete-street-brooklyn-cropped.webp
//	go run ./examples/mobilenet_v3 -model mnv3.mpmodel -random -v
//
// The result image with classification label is saved next to the input.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/GoMemPipe/mempipe/inference"
)

// NOTE: This ONNX model already has Sub/Div ops for ImageNet normalization
// as its first two graph nodes, so we pass raw [0,1]-scaled pixels.

func main() {
	// ── CLI flags ──────────────────────────────────────────────────────
	modelPath := flag.String("model", "mnv3.mpmodel", "Path to converted .mpmodel file")
	imagePath := flag.String("image", "", "Path to image (PNG/JPEG/GIF) or raw .rgb file")
	outPath := flag.String("out", "", "Output result image path (default: <input>_result.png)")
	useRandom := flag.Bool("random", false, "Use random input tensor (for benchmarking)")
	topN := flag.Int("top", 5, "Number of top predictions to show")
	iterations := flag.Int("iter", 1, "Number of inference iterations (for benchmarking)")
	verbose := flag.Bool("v", false, "Verbose: print timing and model details")
	seed := flag.Int64("seed", 42, "Random seed for -random mode")
	noSave := flag.Bool("no-save", false, "Skip saving the result image")
	flag.Parse()

	if *imagePath == "" && !*useRandom {
		fmt.Fprintln(os.Stderr, "error: provide -image <path> or -random")
		flag.Usage()
		os.Exit(1)
	}

	// ── Step 1: Load converted MobileNetV3 model ──────────────────────
	fmt.Fprintf(os.Stderr, "Loading model: %s\n", *modelPath)
	model, err := inference.LoadModel(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load model: %v\n", err)
		os.Exit(1)
	}

	inShape := model.Metadata.InputShapes[0]
	outShape := model.Metadata.OutputShapes[0]
	numClasses := outShape.Dims[len(outShape.Dims)-1]
	imgSize := inShape.Dims[len(inShape.Dims)-1] // 224

	fmt.Fprintf(os.Stderr, "Model:      %s\n", model.Metadata.Name)
	fmt.Fprintf(os.Stderr, "Input:      %v\n", inShape.Dims)
	fmt.Fprintf(os.Stderr, "Output:     %v (classes: %d)\n", outShape.Dims, numClasses)
	fmt.Fprintf(os.Stderr, "Weights:    %.1f MB\n", float64(model.WeightsSize())/(1024*1024))
	fmt.Fprintf(os.Stderr, "Graph ops:  %d\n", len(model.Graph))

	// ── Step 2: Create zero-alloc engine ──────────────────────────────
	engine, err := inference.NewEngine(model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "engine init failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Arena:      %.1f MB used / %.1f MB total\n",
		float64(engine.ArenaUsed())/(1024*1024),
		float64(engine.ArenaTotal())/(1024*1024))
	fmt.Fprintln(os.Stderr)

	// ── Step 3: Prepare input ─────────────────────────────────────────
	inputTensor := engine.InputTensors()[0]
	inputSlice := inputTensor.Float32s()

	var srcImg image.Image // original image for result rendering

	if *useRandom {
		rng := rand.New(rand.NewSource(*seed))
		for i := range inputSlice {
			inputSlice[i] = rng.Float32()
		}
		fmt.Fprintf(os.Stderr, "Input: random tensor (%d floats, seed=%d)\n", len(inputSlice), *seed)
	} else if strings.HasSuffix(strings.ToLower(*imagePath), ".rgb") {
		// Raw float32 binary
		data, err := os.ReadFile(*imagePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to read image: %v\n", err)
			os.Exit(1)
		}
		expectedBytes := len(inputSlice) * 4
		if len(data) != expectedBytes {
			fmt.Fprintf(os.Stderr, "image size mismatch: got %d bytes, expected %d (%v float32s)\n",
				len(data), expectedBytes, inShape.Dims)
			os.Exit(1)
		}
		for i := range inputSlice {
			inputSlice[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		fmt.Fprintf(os.Stderr, "Input: %s (%d floats, raw format)\n", *imagePath, len(inputSlice))
	} else {
		// Decode standard image format (PNG, JPEG, GIF)
		srcImg, err = loadAndPreprocess(*imagePath, inputSlice, imgSize)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to load image: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "Input: %s (%dx%d → %dx%d, [0,1] scaled)\n",
			*imagePath, srcImg.Bounds().Dx(), srcImg.Bounds().Dy(), imgSize, imgSize)
	}
	fmt.Fprintln(os.Stderr)

	// ── Step 4: Inference (zero-alloc hot path) ───────────────────────
	var totalDuration time.Duration
	var finalProbs []float32
	var finalTopIdx []int

	for iter := 0; iter < *iterations; iter++ {
		start := time.Now()

		outputs, err := engine.InferTensor()
		if err != nil {
			fmt.Fprintf(os.Stderr, "inference error: %v\n", err)
			os.Exit(1)
		}

		elapsed := time.Since(start)
		totalDuration += elapsed

		if *verbose || iter == *iterations-1 {
			logits := outputs[0].Float32s()
			probs := softmax(logits[:numClasses])
			topIndices := topKIndices(probs, *topN)

			if iter == *iterations-1 {
				finalProbs = probs
				finalTopIdx = topIndices

				fmt.Fprintf(os.Stderr, "Inference time: %.2f ms\n", float64(elapsed.Microseconds())/1000.0)
				if *iterations > 1 {
					avg := float64(totalDuration.Microseconds()) / float64(*iterations) / 1000.0
					fmt.Fprintf(os.Stderr, "Average over %d iters: %.2f ms\n", *iterations, avg)
				}
				fmt.Fprintln(os.Stderr)

				fmt.Println("Top predictions:")
				for rank, idx := range topIndices {
					label := labelForClass(idx)
					fmt.Printf("  %d. class %4d  %-30s  %.4f (%.1f%%)\n",
						rank+1, idx, label, probs[idx], probs[idx]*100)
				}
			} else if *verbose {
				fmt.Fprintf(os.Stderr, "  iter %d: %.2f ms\n", iter, float64(elapsed.Microseconds())/1000.0)
			}
		}
	}

	// ── Step 5: Save result image ─────────────────────────────────────
	if !*noSave && *imagePath != "" && !*useRandom && srcImg != nil && finalProbs != nil {
		resultPath := *outPath
		if resultPath == "" {
			ext := filepath.Ext(*imagePath)
			base := strings.TrimSuffix(*imagePath, ext)
			resultPath = base + "_result.png"
		}

		err := saveResultImage(srcImg, resultPath, finalTopIdx, finalProbs, imgSize,
			float64(totalDuration.Microseconds())/float64(*iterations)/1000.0)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to save result: %v\n", err)
		} else {
			fmt.Fprintf(os.Stderr, "\nResult saved: %s\n", resultPath)
		}
	}
}

// ────────────────────────────────────────────────────────────────────────────
// Image loading & preprocessing
// ────────────────────────────────────────────────────────────────────────────

// loadAndPreprocess decodes an image file, center-crops to square, resizes to
// imgSize×imgSize with bilinear interpolation, and writes [0,1]-scaled pixels
// into dst in NCHW order. The model's internal Sub/Div ops handle ImageNet
// normalization, so we do NOT normalize here.
// Returns the original image for later result rendering.
func loadAndPreprocess(path string, dst []float32, imgSize int) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decode %s: %w", path, err)
	}
	origImg := img

	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	// Center-crop to square
	cropSize := w
	if h < cropSize {
		cropSize = h
	}
	left := (w - cropSize) / 2
	top := (h - cropSize) / 2

	// Bilinear resize to imgSize × imgSize and fill NCHW tensor
	// dst layout: [1, 3, imgSize, imgSize]
	for c := 0; c < 3; c++ {
		for row := 0; row < imgSize; row++ {
			for col := 0; col < imgSize; col++ {
				// Map (row, col) back to source coordinates
				srcX := float64(col)*float64(cropSize)/float64(imgSize) + float64(left) + float64(bounds.Min.X)
				srcY := float64(row)*float64(cropSize)/float64(imgSize) + float64(top) + float64(bounds.Min.Y)

				// Bilinear interpolation
				x0 := int(srcX)
				y0 := int(srcY)
				x1 := x0 + 1
				y1 := y0 + 1
				fx := float32(srcX) - float32(x0)
				fy := float32(srcY) - float32(y0)

				// Clamp
				if x1 >= bounds.Max.X {
					x1 = bounds.Max.X - 1
				}
				if y1 >= bounds.Max.Y {
					y1 = bounds.Max.Y - 1
				}

				v00 := channelVal(img, x0, y0, c)
				v10 := channelVal(img, x1, y0, c)
				v01 := channelVal(img, x0, y1, c)
				v11 := channelVal(img, x1, y1, c)

				val := v00*(1-fx)*(1-fy) + v10*fx*(1-fy) + v01*(1-fx)*fy + v11*fx*fy

				// Pass raw [0,1] values — the model's Sub/Div ops
				// handle ImageNet normalization internally.
				dst[c*imgSize*imgSize+row*imgSize+col] = val
			}
		}
	}
	return origImg, nil
}

// channelVal returns the [0,1]-scaled value of channel c (0=R, 1=G, 2=B) at pixel (x, y).
func channelVal(img image.Image, x, y, c int) float32 {
	r, g, b, _ := img.At(x, y).RGBA()
	switch c {
	case 0:
		return float32(r) / 65535.0
	case 1:
		return float32(g) / 65535.0
	default:
		return float32(b) / 65535.0
	}
}

// ────────────────────────────────────────────────────────────────────────────
// Result image generation
// ────────────────────────────────────────────────────────────────────────────

// saveResultImage renders the original image with a classification label banner
// at the top and saves as PNG.
func saveResultImage(
	srcImg image.Image,
	outPath string,
	topIdx []int,
	probs []float32,
	modelSize int,
	inferMs float64,
) error {
	bounds := srcImg.Bounds()
	origW, origH := bounds.Dx(), bounds.Dy()

	// Determine result image size: use original or at least 400px wide for readability
	resultW := origW
	if resultW < 400 {
		resultW = 400
	}
	// Scale factor from original
	scale := float64(resultW) / float64(origW)
	resultH := int(float64(origH) * scale)

	// Banner height: one line per prediction + header
	lineH := 22
	bannerLines := len(topIdx) + 1 // +1 for header
	bannerH := lineH*bannerLines + 12
	totalH := resultH + bannerH

	result := image.NewRGBA(image.Rect(0, 0, resultW, totalH))

	// Fill banner background (dark semi-transparent)
	bannerColor := color.RGBA{R: 20, G: 20, B: 30, A: 230}
	draw.Draw(result, image.Rect(0, 0, resultW, bannerH), &image.Uniform{bannerColor}, image.Point{}, draw.Src)

	// Draw header text
	headerText := fmt.Sprintf("MobileNetV3-Large | %.1fms inference", inferMs)
	drawText(result, 8, 6, headerText, color.RGBA{R: 130, G: 200, B: 255, A: 255})

	// Draw predictions
	for i, idx := range topIdx {
		label := labelForClass(idx)
		prob := probs[idx]
		y := 6 + (i+1)*lineH

		// Confidence bar background
		barW := int(float64(resultW-170) * float64(prob))
		if barW < 0 {
			barW = 0
		}
		barColor := confidenceColor(prob)
		draw.Draw(result,
			image.Rect(160, y+2, 160+barW, y+lineH-4),
			&image.Uniform{barColor}, image.Point{}, draw.Src)

		// Prediction label
		text := fmt.Sprintf("#%d %-20s %5.1f%%", i+1, label, prob*100)
		drawText(result, 8, y, text, color.RGBA{R: 240, G: 240, B: 240, A: 255})
	}

	// Draw the original image below the banner (nearest-neighbor scale)
	for dy := 0; dy < resultH; dy++ {
		for dx := 0; dx < resultW; dx++ {
			sx := int(float64(dx)/scale) + bounds.Min.X
			sy := int(float64(dy)/scale) + bounds.Min.Y
			if sx >= bounds.Max.X {
				sx = bounds.Max.X - 1
			}
			if sy >= bounds.Max.Y {
				sy = bounds.Max.Y - 1
			}
			r, g, b, a := srcImg.At(sx, sy).RGBA()
			result.SetRGBA(dx, bannerH+dy, color.RGBA{
				R: uint8(r >> 8), G: uint8(g >> 8), B: uint8(b >> 8), A: uint8(a >> 8),
			})
		}
	}

	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, result)
}

// confidenceColor returns a green-to-red gradient color based on confidence.
func confidenceColor(prob float32) color.RGBA {
	// High confidence = green, low = orange/red
	if prob > 0.5 {
		return color.RGBA{R: 50, G: 180, B: 80, A: 180}
	} else if prob > 0.2 {
		return color.RGBA{R: 200, G: 180, B: 40, A: 180}
	} else if prob > 0.05 {
		return color.RGBA{R: 200, G: 120, B: 40, A: 180}
	}
	return color.RGBA{R: 180, G: 70, B: 50, A: 150}
}

// drawText renders a string onto an RGBA image using a built-in 5×7 bitmap font.
// No external dependencies required.
func drawText(img *image.RGBA, x, y int, text string, col color.RGBA) {
	for _, ch := range text {
		glyph := getGlyph(byte(ch))
		for gy := 0; gy < 7; gy++ {
			row := glyph[gy]
			for gx := 0; gx < 5; gx++ {
				if row&(1<<(4-gx)) != 0 {
					px := x + gx
					py := y + gy
					if px >= 0 && px < img.Bounds().Dx() && py >= 0 && py < img.Bounds().Dy() {
						img.SetRGBA(px, py, col)
						// Bold effect: double-draw shifted right
						if px+1 < img.Bounds().Dx() {
							img.SetRGBA(px+1, py, col)
						}
					}
				}
			}
		}
		x += 7 // character width + spacing
	}
}

// getGlyph returns a 5×7 bitmap for a printable ASCII character.
// Each row is a byte where bits 4–0 represent columns left→right.
func getGlyph(ch byte) [7]byte {
	if ch < 32 || ch > 126 {
		ch = '?'
	}
	idx := int(ch) - 32
	if idx >= len(font5x7) {
		idx = int('?') - 32
	}
	return font5x7[idx]
}

// font5x7 is a minimal 5×7 pixel bitmap font for ASCII 32–126.
var font5x7 = [95][7]byte{
	// 32 ' '
	{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
	// 33 '!'
	{0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04},
	// 34 '"'
	{0x0A, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00},
	// 35 '#'
	{0x0A, 0x1F, 0x0A, 0x0A, 0x1F, 0x0A, 0x00},
	// 36 '$'
	{0x04, 0x0F, 0x14, 0x0E, 0x05, 0x1E, 0x04},
	// 37 '%'
	{0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03},
	// 38 '&'
	{0x08, 0x14, 0x14, 0x08, 0x15, 0x12, 0x0D},
	// 39 '''
	{0x04, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00},
	// 40 '('
	{0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02},
	// 41 ')'
	{0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08},
	// 42 '*'
	{0x00, 0x04, 0x15, 0x0E, 0x15, 0x04, 0x00},
	// 43 '+'
	{0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00},
	// 44 ','
	{0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x08},
	// 45 '-'
	{0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00},
	// 46 '.'
	{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04},
	// 47 '/'
	{0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10},
	// 48 '0'
	{0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E},
	// 49 '1'
	{0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},
	// 50 '2'
	{0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F},
	// 51 '3'
	{0x0E, 0x11, 0x01, 0x0E, 0x01, 0x11, 0x0E},
	// 52 '4'
	{0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},
	// 53 '5'
	{0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},
	// 54 '6'
	{0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E},
	// 55 '7'
	{0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},
	// 56 '8'
	{0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},
	// 57 '9'
	{0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C},
	// 58 ':'
	{0x00, 0x00, 0x04, 0x00, 0x00, 0x04, 0x00},
	// 59 ';'
	{0x00, 0x00, 0x04, 0x00, 0x00, 0x04, 0x08},
	// 60 '<'
	{0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02},
	// 61 '='
	{0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00},
	// 62 '>'
	{0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08},
	// 63 '?'
	{0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04},
	// 64 '@'
	{0x0E, 0x11, 0x17, 0x15, 0x17, 0x10, 0x0E},
	// 65 'A'
	{0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},
	// 66 'B'
	{0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E},
	// 67 'C'
	{0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E},
	// 68 'D'
	{0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E},
	// 69 'E'
	{0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F},
	// 70 'F'
	{0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10},
	// 71 'G'
	{0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F},
	// 72 'H'
	{0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},
	// 73 'I'
	{0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E},
	// 74 'J'
	{0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C},
	// 75 'K'
	{0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11},
	// 76 'L'
	{0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F},
	// 77 'M'
	{0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11},
	// 78 'N'
	{0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11},
	// 79 'O'
	{0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},
	// 80 'P'
	{0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10},
	// 81 'Q'
	{0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D},
	// 82 'R'
	{0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11},
	// 83 'S'
	{0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E},
	// 84 'T'
	{0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},
	// 85 'U'
	{0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},
	// 86 'V'
	{0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04},
	// 87 'W'
	{0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11},
	// 88 'X'
	{0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11},
	// 89 'Y'
	{0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04},
	// 90 'Z'
	{0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F},
	// 91 '['
	{0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E},
	// 92 '\'
	{0x10, 0x10, 0x08, 0x04, 0x02, 0x01, 0x01},
	// 93 ']'
	{0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E},
	// 94 '^'
	{0x04, 0x0A, 0x11, 0x00, 0x00, 0x00, 0x00},
	// 95 '_'
	{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F},
	// 96 '`'
	{0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00},
	// 97 'a'
	{0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F},
	// 98 'b'
	{0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x1E},
	// 99 'c'
	{0x00, 0x00, 0x0E, 0x11, 0x10, 0x11, 0x0E},
	// 100 'd'
	{0x01, 0x01, 0x0F, 0x11, 0x11, 0x11, 0x0F},
	// 101 'e'
	{0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E},
	// 102 'f'
	{0x06, 0x08, 0x08, 0x1E, 0x08, 0x08, 0x08},
	// 103 'g'
	{0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x0E},
	// 104 'h'
	{0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x11},
	// 105 'i'
	{0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E},
	// 106 'j'
	{0x02, 0x00, 0x02, 0x02, 0x02, 0x12, 0x0C},
	// 107 'k'
	{0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12},
	// 108 'l'
	{0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E},
	// 109 'm'
	{0x00, 0x00, 0x1A, 0x15, 0x15, 0x15, 0x15},
	// 110 'n'
	{0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11},
	// 111 'o'
	{0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E},
	// 112 'p'
	{0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10},
	// 113 'q'
	{0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x01},
	// 114 'r'
	{0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10},
	// 115 's'
	{0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E},
	// 116 't'
	{0x08, 0x08, 0x1E, 0x08, 0x08, 0x09, 0x06},
	// 117 'u'
	{0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D},
	// 118 'v'
	{0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04},
	// 119 'w'
	{0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A},
	// 120 'x'
	{0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11},
	// 121 'y'
	{0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E},
	// 122 'z'
	{0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F},
	// 123 '{'
	{0x02, 0x04, 0x04, 0x08, 0x04, 0x04, 0x02},
	// 124 '|'
	{0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},
	// 125 '}'
	{0x08, 0x04, 0x04, 0x02, 0x04, 0x04, 0x08},
	// 126 '~'
	{0x00, 0x00, 0x08, 0x15, 0x02, 0x00, 0x00},
}

// ────────────────────────────────────────────────────────────────────────────
// Sampling helpers
// ────────────────────────────────────────────────────────────────────────────

// softmax computes softmax probabilities from logits.
func softmax(logits []float32) []float32 {
	n := len(logits)
	probs := make([]float32, n)

	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float32
	for i, v := range logits {
		probs[i] = float32(math.Exp(float64(v - maxVal)))
		sum += probs[i]
	}
	invSum := 1.0 / sum
	for i := range probs {
		probs[i] *= invSum
	}
	return probs
}

// topKIndices returns indices of the top-k largest values.
func topKIndices(values []float32, k int) []int {
	if k > len(values) {
		k = len(values)
	}
	indices := make([]int, k)
	used := make([]bool, len(values))

	for i := 0; i < k; i++ {
		best := -1
		bestVal := float32(-math.MaxFloat32)
		for j, v := range values {
			if !used[j] && v > bestVal {
				bestVal = v
				best = j
			}
		}
		indices[i] = best
		if best >= 0 {
			used[best] = true
		}
	}
	return indices
}

// ────────────────────────────────────────────────────────────────────────────
// ImageNet 1000-class labels
// ────────────────────────────────────────────────────────────────────────────

func labelForClass(idx int) string {
	if idx >= 0 && idx < len(imagenetLabels) {
		return imagenetLabels[idx]
	}
	return fmt.Sprintf("class_%d", idx)
}

// imagenetLabels contains the full 1000-class ImageNet label set.
var imagenetLabels = [...]string{
	"tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
	"electric ray", "stingray", "cock", "hen", "ostrich",
	"brambling", "goldfinch", "house finch", "junco", "indigo bunting",
	"American robin", "bulbul", "jay", "magpie", "chickadee",
	"American dipper", "kite", "bald eagle", "vulture", "great grey owl",
	"fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl",
	"American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle",
	"mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana",
	"Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
	"Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
	"American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake",
	"smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake",
	"night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba",
	"sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder", "trilobite",
	"harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider",
	"southern black widow", "tarantula", "wolf spider", "tick", "centipede",
	"black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peacock",
	"quail", "partridge", "grey parrot", "macaw", "sulphur-crested cockatoo",
	"lorikeet", "coucal", "bee eater", "hornbill", "hummingbird",
	"jacamar", "toucan", "duck", "red-breasted merganser", "goose",
	"black swan", "tusker", "echidna", "platypus", "wallaby",
	"koala", "wombat", "jellyfish", "sea anemone", "brain coral",
	"flatworm", "nematode", "conch", "snail", "slug",
	"sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
	"fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish",
	"hermit crab", "isopod", "white stork", "black stork", "spoonbill",
	"flamingo", "little blue heron", "great egret", "bittern", "crane",
	"limpkin", "common crane", "American coot", "bustard", "ruddy turnstone",
	"dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican",
	"king penguin", "albatross", "grey whale", "killer whale", "dugong",
	"sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese",
	"Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback",
	"Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound",
	"Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi",
	"Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound",
	"Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier",
	"American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
	"Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier",
	"Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier",
	"Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier",
	"Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso",
	"Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever",
	"German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter",
	"Brittany", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel",
	"Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael",
	"Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog",
	"Shetland Sheepdog", "Collie", "Border Collie", "Bouvier des Flandres", "Rottweiler",
	"German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
	"Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
	"French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute",
	"Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "Pug",
	"Leonberger", "Newfoundland", "Pyrenean Mountain Dog", "Samoyed", "Pomeranian",
	"Chow Chow", "Keeshond", "Griffon Bruxellois", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi",
	"Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog", "grey wolf",
	"Alaskan tundra wolf", "red wolf", "coyote", "dingo", "dhole",
	"African wild dog", "hyena", "red fox", "kit fox", "Arctic fox",
	"grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
	"Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard",
	"jaguar", "lion", "tiger", "cheetah", "brown bear",
	"American black bear", "polar bear", "sloth bear", "mongoose", "meerkat",
	"tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
	"dung beetle", "rhinoceros beetle", "weevil", "fly", "bee",
	"ant", "grasshopper", "cricket", "stick insect", "cockroach",
	"mantis", "cicada", "leafhopper", "lacewing", "dragonfly",
	"damselfly", "red admiral", "ringlet", "monarch butterfly", "small white",
	"sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber",
	"cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine",
	"fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel",
	"zebra", "pig", "wild boar", "warthog", "hippopotamus",
	"ox", "water buffalo", "bison", "ram", "bighorn sheep",
	"Alpine ibex", "hartebeest", "impala", "gazelle", "dromedary",
	"llama", "weasel", "mink", "European polecat", "black-footed ferret",
	"otter", "skunk", "badger", "armadillo", "three-toed sloth",
	"orangutan", "gorilla", "chimpanzee", "gibbon", "siamang",
	"guenon", "patas monkey", "baboon", "macaque", "langur",
	"black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey",
	"titi", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri",
	"Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek",
	"eel", "coho salmon", "rock beauty", "clownfish", "sturgeon",
	"garfish", "lionfish", "pufferfish", "abacus", "abaya",
	"academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner",
	"airship", "altar", "ambulance", "amphibious vehicle", "analog clock",
	"apiary", "apron", "waste container", "assault rifle", "backpack",
	"bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid",
	"banjo", "baluster", "barbell", "barber chair", "barbershop",
	"barn", "barometer", "barrel", "wheelbarrow", "baseball",
	"basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
	"bathtub", "station wagon", "lighthouse", "beaker", "military cap",
	"beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle",
	"bikini", "ring binder", "binoculars", "birdhouse", "boathouse",
	"bobsled", "bolo tie", "poke bonnet", "bookcase", "bookstore",
	"bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
	"breakwater", "breastplate", "broom", "bucket", "buckle",
	"bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron",
	"candle", "cannon", "canoe", "can opener", "cardigan",
	"car mirror", "carousel", "tool kit", "carton", "car wheel",
	"automated teller machine", "cassette", "cassette player", "castle", "catamaran",
	"CD player", "cello", "mobile phone", "chain", "chain-link fence",
	"chain mail", "chainsaw", "storage chest", "chiffonier", "wind chime",
	"china cabinet", "Christmas stocking", "church", "movie theater", "cleaver",
	"cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug",
	"coffeemaker", "coil", "combination lock", "computer keyboard", "confectionery",
	"container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
	"cowboy hat", "cradle", "crane", "crash helmet", "crate",
	"infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass",
	"dam", "desk", "desktop computer", "rotary dial telephone", "diaper",
	"digital clock", "digital watch", "dining table", "dishcloth", "dishwasher",
	"disc brake", "dock", "dog sled", "dome", "doormat",
	"drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven",
	"electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope",
	"espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat",
	"fire truck", "fire screen", "flagpole", "flute", "folding chair",
	"football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
	"freight car", "French horn", "frying pan", "fur coat", "garbage truck",
	"gas mask", "gas pump", "goblet", "go-kart", "golf ball",
	"golf cart", "gondola", "gong", "gown", "grand piano",
	"greenhouse", "radiator grille", "grocery store", "guillotine", "barrette",
	"hair dryer", "hair spray", "half-track", "hammer", "hamper",
	"hand dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica",
	"harp", "combine harvester", "hatchet", "holster", "home theater",
	"honeycomb", "hook", "hoop skirt", "horizontal bar", "horse-drawn vehicle",
	"hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans",
	"jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick",
	"kimono", "knee pad", "knot", "lab coat", "ladle",
	"lampshade", "laptop computer", "lawn mower", "lens cap", "paper knife",
	"library", "lifeboat", "lighter", "limousine", "ocean liner",
	"lipstick", "slip-on shoe", "lotion", "music speaker", "loupe",
	"sawmill", "magnetic compass", "mail bag", "mailbox", "tights",
	"one-piece swimsuit", "manhole cover", "maraca", "marimba", "mask",
	"match", "maypole", "maze", "measuring cup", "medicine cabinet",
	"megalith", "microphone", "microwave oven", "military uniform", "milk can",
	"minibus", "miniskirt", "minivan", "missile", "mitten",
	"mixing bowl", "mobile home", "Model T", "modem", "monastery",
	"monitor", "moped", "mortar and pestle", "graduation cap", "mosque",
	"mosquito net", "vespa", "mountain bike", "tent", "computer mouse",
	"mousetrap", "moving van", "muzzle", "metal nail", "neck brace",
	"necklace", "nipple", "notebook computer", "obelisk", "oboe",
	"ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope",
	"overskirt", "bullock cart", "oxygen mask", "product packet", "paddle",
	"paddle wheel", "padlock", "paintbrush", "pajamas", "palace",
	"pan flute", "paper towel", "parachute", "parallel bars", "park bench",
	"parking meter", "railroad car", "patio", "payphone", "pedestal",
	"pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier",
	"plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier",
	"piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel",
	"pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
	"plate rack", "farm plow", "plunger", "Polaroid camera", "pole",
	"police van", "poncho", "billiard table", "soda bottle", "pot",
	"potter's wheel", "power drill", "prayer rug", "printer", "prison",
	"missile", "projector", "hockey puck", "punching bag", "purse",
	"quill", "quilt", "race car", "racket", "radiator",
	"radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel",
	"reflex camera", "refrigerator", "remote control", "restaurant", "revolver",
	"rifle", "rocking chair", "rotisserie", "pencil eraser", "rugby ball",
	"ruler", "running shoe", "safe", "safety pin", "salt shaker",
	"sandal", "sarong", "saxophone", "scabbard", "weighing scale",
	"school bus", "schooner", "scoreboard", "CRT monitor", "screw",
	"screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
	"shoji screen", "shopping basket", "shopping cart", "shovel", "shower cap",
	"shower curtain", "ski", "balaclava", "sleeping bag", "slide rule",
	"sliding door", "slot machine", "snorkel", "snowmobile", "snowplow",
	"soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero",
	"soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula",
	"motorboat", "spider web", "spindle", "sports car", "spotlight",
	"stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope",
	"scarf", "stone wall", "stopwatch", "stove", "strainer",
	"tram", "stretcher", "couch", "stupa", "submarine",
	"suit", "sundial", "sunglass", "sunglasses", "sunscreen",
	"suspension bridge", "mop", "sweatshirt", "swimsuit", "swing",
	"electrical switch", "syringe", "table lamp", "tank", "tape player",
	"teapot", "teddy bear", "television", "tennis ball", "thatched roof",
	"front curtain", "thimble", "threshing machine", "throne", "tile roof",
	"toaster", "tobacco shop", "toilet seat", "torch", "totem pole",
	"tow truck", "toy store", "tractor", "semi-trailer truck", "tray",
	"trench coat", "tricycle", "trimaran", "tripod", "triumphal arch",
	"trolleybus", "trombone", "hot tub", "turnstile", "typewriter",
	"umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase",
	"vault", "velvet fabric", "vending machine", "vestment", "viaduct",
	"violin", "volleyball", "waffle iron", "wall clock", "wallet",
	"wardrobe", "military aircraft", "sink", "washing machine", "water bottle",
	"water jug", "water tower", "whiskey jug", "whistle", "hair wig",
	"window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
	"wok", "wooden spoon", "wool", "split-rail fence", "shipwreck",
	"sailboat", "yurt", "website", "comic book", "crossword",
	"traffic sign", "traffic light", "dust jacket", "menu", "plate",
	"guacamole", "consomme", "hot pot", "trifle", "ice cream",
	"popsicle", "baguette", "bagel", "pretzel", "cheeseburger",
	"hotdog", "mashed potatoes", "cabbage", "broccoli", "cauliflower",
	"zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
	"artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith",
	"strawberry", "orange", "lemon", "fig", "pineapple",
	"banana", "jackfruit", "cherimoya", "pomegranate", "hay",
	"carbonara", "chocolate syrup", "dough", "meatloaf", "pizza",
	"pot pie", "burrito", "red wine", "espresso", "teacup",
	"eggnog", "mountain", "bubble", "cliff", "coral reef",
	"geyser", "lakeshore", "promontory", "sandbar", "beach",
	"valley", "volcano", "baseball player", "bridegroom", "scuba diver",
	"rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
	"rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra",
	"stinkhorn mushroom", "earth star fungus", "hen of the woods", "bolete", "ear of corn",
	"toilet paper",
}
