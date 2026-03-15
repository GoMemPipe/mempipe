// Command game_server demonstrates a tick-based game loop with MemPipe.
//
// The pipeline models a simple 2D game with:
//   - Entity positions and velocities in arena regions
//   - Four cells: input_processor → physics → collision → state_sync
//   - All state lives in typed arena memory (zero allocations per tick)
//
// Usage:
//
//	go run ./examples/game_server
package main

import (
	"fmt"
	"math"
	"time"

	mempipe "github.com/GoMemPipe/mempipe"
)

const (
	maxEntities = 8
	tickRate    = 60    // ticks per second
	totalTicks  = 600   // 10 seconds of simulation
	worldWidth  = 100.0 // world bounds
	worldHeight = 100.0
)

// ── Region structs ──

// InputState represents queued movement commands.
type InputState struct {
	AccelX float32 `mempipe:"field:accel_x"`
	AccelY float32 `mempipe:"field:accel_y"`
	Active uint32  `mempipe:"field:active"`
}

// Physics holds entity positions and velocities.
type Physics struct {
	PosX  float32 `mempipe:"field:pos_x"`
	PosY  float32 `mempipe:"field:pos_y"`
	VelX  float32 `mempipe:"field:vel_x"`
	VelY  float32 `mempipe:"field:vel_y"`
	Alive uint32  `mempipe:"field:alive"`
}

// CollisionResult tracks detected collisions.
type CollisionResult struct {
	Count     uint32 `mempipe:"field:count"`
	LastPairA uint32 `mempipe:"field:last_pair_a"`
	LastPairB uint32 `mempipe:"field:last_pair_b"`
	TotalHits uint32 `mempipe:"field:total_hits"`
}

// GameState holds aggregate server state.
type GameState struct {
	Tick       uint32  `mempipe:"field:tick"`
	AliveCount uint32  `mempipe:"field:alive_count"`
	AvgSpeed   float32 `mempipe:"field:avg_speed"`
}

func main() {
	fmt.Println("── MemPipe Game Server Example ──")
	fmt.Printf("Entities: %d, Tick rate: %d Hz, Duration: %d ticks\n\n", maxEntities, tickRate, totalTicks)

	pipe := mempipe.NewPipeline(mempipe.WithWorkers(1))

	input := mempipe.AddRegion[InputState](pipe, "input")
	physics := mempipe.AddRegion[Physics](pipe, "physics")
	collision := mempipe.AddRegion[CollisionResult](pipe, "collision")
	state := mempipe.AddRegion[GameState](pipe, "state")

	// Entity state stored outside arena for multi-entity simulation.
	// In production, you'd use a ring-buffer region per entity.
	type Entity struct {
		posX, posY float32
		velX, velY float32
		alive      bool
		radius     float32
	}
	entities := make([]Entity, maxEntities)

	// Initialize entities in a circle pattern
	for i := range entities {
		angle := float64(i) * 2.0 * math.Pi / float64(maxEntities)
		entities[i] = Entity{
			posX:   float32(worldWidth/2 + 30*math.Cos(angle)),
			posY:   float32(worldHeight/2 + 30*math.Sin(angle)),
			velX:   float32(2.0 * math.Sin(angle)),
			velY:   float32(-2.0 * math.Cos(angle)),
			alive:  true,
			radius: 3.0,
		}
	}

	currentTick := 0

	// ── Cell 1: Input processor ──
	// Simulates receiving player input; applies sinusoidal acceleration.
	pipe.Cell("input_processor", func() {
		t := float64(currentTick) / float64(tickRate)
		in := InputState{
			AccelX: float32(0.5 * math.Sin(t*2.0)),
			AccelY: float32(0.5 * math.Cos(t*1.5)),
			Active: uint32(maxEntities),
		}
		input.Set(in)
	}, nil, []string{"input"})

	// ── Cell 2: Physics ──
	// Euler integration + wall bouncing.
	pipe.Cell("physics", func() {
		in := input.Get()
		dt := float32(1.0) / float32(tickRate)

		for i := range entities {
			if !entities[i].alive {
				continue
			}
			// Apply acceleration
			entities[i].velX += in.AccelX * dt
			entities[i].velY += in.AccelY * dt

			// Drag
			entities[i].velX *= 0.99
			entities[i].velY *= 0.99

			// Integrate position
			entities[i].posX += entities[i].velX
			entities[i].posY += entities[i].velY

			// Wall bounce
			if entities[i].posX < 0 {
				entities[i].posX = -entities[i].posX
				entities[i].velX = -entities[i].velX
			}
			if entities[i].posX > worldWidth {
				entities[i].posX = 2*worldWidth - entities[i].posX
				entities[i].velX = -entities[i].velX
			}
			if entities[i].posY < 0 {
				entities[i].posY = -entities[i].posY
				entities[i].velY = -entities[i].velY
			}
			if entities[i].posY > worldHeight {
				entities[i].posY = 2*worldHeight - entities[i].posY
				entities[i].velY = -entities[i].velY
			}
		}

		// Write a representative entity to the physics region for inspection
		if len(entities) > 0 {
			e := entities[0]
			physics.Set(Physics{
				PosX:  e.posX,
				PosY:  e.posY,
				VelX:  e.velX,
				VelY:  e.velY,
				Alive: boolToUint(e.alive),
			})
		}
	}, []string{"input"}, []string{"physics"})

	// ── Cell 3: Collision detection ──
	// O(n²) brute-force AABB check (fine for small entity counts).
	pipe.Cell("collision", func() {
		cr := collision.Get()
		cr.Count = 0
		for i := 0; i < maxEntities; i++ {
			if !entities[i].alive {
				continue
			}
			for j := i + 1; j < maxEntities; j++ {
				if !entities[j].alive {
					continue
				}
				dx := entities[i].posX - entities[j].posX
				dy := entities[i].posY - entities[j].posY
				dist := float32(math.Sqrt(float64(dx*dx + dy*dy)))
				minDist := entities[i].radius + entities[j].radius

				if dist < minDist {
					cr.Count++
					cr.TotalHits++
					cr.LastPairA = uint32(i)
					cr.LastPairB = uint32(j)

					// Elastic bounce: swap velocity components
					entities[i].velX, entities[j].velX = entities[j].velX, entities[i].velX
					entities[i].velY, entities[j].velY = entities[j].velY, entities[i].velY

					// Separate overlapping entities
					overlap := (minDist - dist) / 2
					if dist > 0 {
						nx := dx / dist
						ny := dy / dist
						entities[i].posX += nx * overlap
						entities[i].posY += ny * overlap
						entities[j].posX -= nx * overlap
						entities[j].posY -= ny * overlap
					}
				}
			}
		}
		collision.Set(cr)
	}, []string{"physics"}, []string{"collision"})

	// ── Cell 4: State sync ──
	// Aggregates game state for monitoring.
	pipe.Cell("state_sync", func() {
		s := state.Get()
		s.Tick = uint32(currentTick)
		alive := uint32(0)
		totalSpeed := float32(0)
		for _, e := range entities {
			if e.alive {
				alive++
				totalSpeed += float32(math.Sqrt(float64(e.velX*e.velX + e.velY*e.velY)))
			}
		}
		s.AliveCount = alive
		if alive > 0 {
			s.AvgSpeed = totalSpeed / float32(alive)
		}
		state.Set(s)
	}, []string{"collision"}, []string{"state"})

	// ── Run simulation ──
	pipe.OnIteration(func(iter int) {
		currentTick = iter
		if iter > 0 && iter%(totalTicks/10) == 0 {
			s := state.Get()
			cr := collision.Get()
			p := physics.Get()
			fmt.Printf("  Tick %4d: alive=%d  avg_speed=%.2f  collisions=%d (total=%d)  e0_pos=(%.1f, %.1f)\n",
				s.Tick, s.AliveCount, s.AvgSpeed, cr.Count, cr.TotalHits, p.PosX, p.PosY)
		}
	})

	fmt.Println("Running simulation...")
	start := time.Now()
	if err := pipe.Run(totalTicks); err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	elapsed := time.Since(start)

	// ── Final report ──
	s := state.Get()
	cr := collision.Get()
	fmt.Println()
	fmt.Println("── Final State ──")
	fmt.Printf("Ticks: %d\n", s.Tick)
	fmt.Printf("Alive: %d / %d\n", s.AliveCount, maxEntities)
	fmt.Printf("Avg speed: %.3f\n", s.AvgSpeed)
	fmt.Printf("Total collisions: %d\n", cr.TotalHits)
	fmt.Println()
	fmt.Println("── Performance ──")
	fmt.Printf("Elapsed: %v\n", elapsed)
	fmt.Printf("%.0f ticks/sec (target: %d)\n", float64(totalTicks)/elapsed.Seconds(), tickRate)
	fmt.Printf("%.2f µs/tick\n", float64(elapsed.Microseconds())/float64(totalTicks))

	// Print entity positions
	fmt.Println()
	fmt.Println("── Entity Positions ──")
	for i, e := range entities {
		if e.alive {
			fmt.Printf("  [%d] pos=(%.1f, %.1f) vel=(%.2f, %.2f)\n", i, e.posX, e.posY, e.velX, e.velY)
		}
	}
}

func boolToUint(b bool) uint32 {
	if b {
		return 1
	}
	return 0
}
