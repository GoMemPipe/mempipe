// Command hft_pipeline demonstrates a high-frequency trading signal pipeline.
//
// Pipeline stages:
//
//	market_data → signal_compute (moving averages, z-scores) → order_gen
//
// Uses ring-buffer style regions for time-series data. All computation
// happens in arena memory with zero allocations on the hot path.
//
// Usage:
//
//	go run ./examples/hft_pipeline
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	mempipe "github.com/GoMemPipe/mempipe"
)

const (
	windowSize = 20 // moving average window
	totalTicks = 10000
	zThreshold = 1.5 // z-score threshold for signals
)

// ── Region structs ──

// MarketTick represents a single market data update.
type MarketTick struct {
	Price  float32 `mempipe:"field:price"`
	Volume float32 `mempipe:"field:volume"`
	BidAsk float32 `mempipe:"field:bid_ask_spread"`
	SeqNum uint64  `mempipe:"field:seq_num"`
}

// SignalState holds computed trading signals.
type SignalState struct {
	SMA        float32 `mempipe:"field:sma"`         // simple moving average
	EMA        float32 `mempipe:"field:ema"`         // exponential moving average
	StdDev     float32 `mempipe:"field:std_dev"`     // rolling standard deviation
	ZScore     float32 `mempipe:"field:z_score"`     // (price - SMA) / stddev
	Momentum   float32 `mempipe:"field:momentum"`    // rate of change
	SignalType uint32  `mempipe:"field:signal_type"` // 0=none, 1=buy, 2=sell
}

// OrderState holds the latest generated order.
type OrderState struct {
	Side       uint32  `mempipe:"field:side"` // 1=buy, 2=sell
	Price      float32 `mempipe:"field:price"`
	Quantity   float32 `mempipe:"field:quantity"`
	OrderCount uint64  `mempipe:"field:order_count"`
}

// PnLState tracks profit and loss.
type PnLState struct {
	Position   float32 `mempipe:"field:position"`
	AvgEntry   float32 `mempipe:"field:avg_entry"`
	RealizedPL float32 `mempipe:"field:realized_pl"`
	UnrealPL   float32 `mempipe:"field:unreal_pl"`
	TradeCount uint64  `mempipe:"field:trade_count"`
}

func main() {
	fmt.Println("── MemPipe HFT Pipeline Example ──")
	fmt.Printf("Window: %d, Z-threshold: %.1f, Ticks: %d\n\n", windowSize, zThreshold, totalTicks)

	pipe := mempipe.NewPipeline()

	market := mempipe.AddRegion[MarketTick](pipe, "market")
	signals := mempipe.AddRegion[SignalState](pipe, "signals")
	orders := mempipe.AddRegion[OrderState](pipe, "orders")
	pnl := mempipe.AddRegion[PnLState](pipe, "pnl")

	// Ring buffer for price history (local, stack-allocated)
	var priceRing [windowSize]float32
	ringIdx := 0
	ringFull := false

	// EMA smoothing factor
	emaAlpha := float32(2.0 / float64(windowSize+1))

	// Simulated market data generator
	rng := rand.New(rand.NewSource(42))
	basePrice := float32(100.0)
	currentTick := 0

	// ── Cell 1: Market data ingestion ──
	pipe.Cell("market_data", func() {
		// Geometric Brownian motion: dS = µ·S·dt + σ·S·dW
		drift := float32(0.0001)
		vol := float32(0.02)
		dW := float32(rng.NormFloat64())
		basePrice *= 1 + drift + vol*dW

		md := MarketTick{
			Price:  basePrice,
			Volume: float32(1000 + rng.Intn(9000)),
			BidAsk: float32(0.01 + rng.Float64()*0.04),
			SeqNum: uint64(currentTick),
		}
		market.Set(md)

		// Update ring buffer
		priceRing[ringIdx] = basePrice
		ringIdx++
		if ringIdx >= windowSize {
			ringIdx = 0
			ringFull = true
		}
	}, nil, []string{"market"})

	// ── Cell 2: Signal computation ──
	pipe.Cell("signal_compute", func() {
		md := market.Get()
		sig := signals.Get()

		// SMA
		count := windowSize
		if !ringFull {
			count = ringIdx
		}
		if count > 0 {
			sum := float32(0)
			for i := 0; i < count; i++ {
				sum += priceRing[i]
			}
			sig.SMA = sum / float32(count)

			// Standard deviation
			varSum := float32(0)
			for i := 0; i < count; i++ {
				d := priceRing[i] - sig.SMA
				varSum += d * d
			}
			sig.StdDev = float32(math.Sqrt(float64(varSum / float32(count))))

			// Z-score
			if sig.StdDev > 0.0001 {
				sig.ZScore = (md.Price - sig.SMA) / sig.StdDev
			}
		}

		// EMA
		if sig.EMA == 0 {
			sig.EMA = md.Price
		} else {
			sig.EMA = emaAlpha*md.Price + (1-emaAlpha)*sig.EMA
		}

		// Momentum (price change from oldest in window)
		if ringFull {
			oldest := priceRing[ringIdx] // oldest value in ring
			sig.Momentum = (md.Price - oldest) / oldest * 100
		}

		// Generate signal
		sig.SignalType = 0
		if sig.ZScore < -zThreshold && ringFull {
			sig.SignalType = 1 // buy: price below mean
		} else if sig.ZScore > zThreshold && ringFull {
			sig.SignalType = 2 // sell: price above mean
		}

		signals.Set(sig)
	}, []string{"market"}, []string{"signals"})

	// ── Cell 3: Order generation ──
	pipe.Cell("order_gen", func() {
		sig := signals.Get()
		md := market.Get()

		if sig.SignalType == 0 {
			return
		}

		ord := orders.Get()
		ord.Side = sig.SignalType
		ord.Price = md.Price
		// Size inversely proportional to spread
		if md.BidAsk > 0 {
			ord.Quantity = float32(math.Min(float64(100/md.BidAsk), 1000))
		}
		ord.OrderCount++
		orders.Set(ord)

		// Update P&L tracking
		p := pnl.Get()
		p.TradeCount++
		if sig.SignalType == 1 { // buy
			avgOldPos := p.Position * p.AvgEntry
			p.Position += ord.Quantity
			if p.Position > 0 {
				p.AvgEntry = (avgOldPos + ord.Quantity*md.Price) / p.Position
			}
		} else { // sell
			if p.Position > 0 {
				sellQty := float32(math.Min(float64(ord.Quantity), float64(p.Position)))
				p.RealizedPL += sellQty * (md.Price - p.AvgEntry)
				p.Position -= sellQty
			}
		}
		// Mark-to-market unrealized P&L
		if p.Position > 0 {
			p.UnrealPL = p.Position * (md.Price - p.AvgEntry)
		} else {
			p.UnrealPL = 0
		}
		pnl.Set(p)
	}, []string{"signals"}, []string{"orders"})

	// ── Run pipeline ──
	pipe.OnIteration(func(iter int) {
		currentTick = iter
		if iter > 0 && iter%(totalTicks/5) == 0 {
			sig := signals.Get()
			md := market.Get()
			ord := orders.Get()
			p := pnl.Get()
			sigStr := "HOLD"
			if sig.SignalType == 1 {
				sigStr = "BUY "
			} else if sig.SignalType == 2 {
				sigStr = "SELL"
			}
			fmt.Printf("  Tick %5d: price=%.2f  SMA=%.2f  z=% .2f  [%s]  orders=%d  P&L=%.2f (unreal=%.2f)\n",
				iter, md.Price, sig.SMA, sig.ZScore, sigStr, ord.OrderCount, p.RealizedPL, p.UnrealPL)
		}
	})

	fmt.Println("Running strategy...")
	start := time.Now()
	if err := pipe.Run(totalTicks); err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	elapsed := time.Since(start)

	// ── Final report ──
	sig := signals.Get()
	md := market.Get()
	ord := orders.Get()
	p := pnl.Get()

	fmt.Println()
	fmt.Println("── Final Market State ──")
	fmt.Printf("Price: %.4f  SMA: %.4f  EMA: %.4f\n", md.Price, sig.SMA, sig.EMA)
	fmt.Printf("Z-score: %.4f  StdDev: %.4f  Momentum: %.4f%%\n", sig.ZScore, sig.StdDev, sig.Momentum)

	fmt.Println()
	fmt.Println("── Trading Results ──")
	fmt.Printf("Total orders: %d\n", ord.OrderCount)
	fmt.Printf("Total trades: %d\n", p.TradeCount)
	fmt.Printf("Position: %.2f @ avg %.4f\n", p.Position, p.AvgEntry)
	fmt.Printf("Realized P&L: %.4f\n", p.RealizedPL)
	fmt.Printf("Unrealized P&L: %.4f\n", p.UnrealPL)
	fmt.Printf("Total P&L: %.4f\n", p.RealizedPL+p.UnrealPL)

	fmt.Println()
	fmt.Println("── Performance ──")
	fmt.Printf("Elapsed: %v\n", elapsed)
	fmt.Printf("%.0f ticks/sec\n", float64(totalTicks)/elapsed.Seconds())
	fmt.Printf("%.2f µs/tick\n", float64(elapsed.Microseconds())/float64(totalTicks))
}
