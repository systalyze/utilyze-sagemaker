package tschart

import "math"

// Braille dot bitmasks indexed by [col%2][row%4], each braille
// character is a 2×4 dot matrix encoded in Unicode block U+2800
var dotBits = [2][4]uint8{
	{0x01, 0x02, 0x04, 0x40}, // left column
	{0x08, 0x10, 0x20, 0x80}, // right column
}

// brailleGrid is a minimal braille dot buffer storing one byte per canvas
// cell (canvasW × canvasH), where each bit corresponds to a dot in the 2×4
// braille matrix. The full graph coordinate space is (canvasW*2) × (canvasH*4)
type brailleGrid struct {
	canvasW, canvasH int
	graphW, graphH   int
	bits             []uint8
}

func newBrailleGrid(canvasW, canvasH int) brailleGrid {
	return brailleGrid{
		canvasW: canvasW,
		canvasH: canvasH,
		graphW:  canvasW * 2,
		graphH:  canvasH * 4,
		bits:    make([]uint8, canvasW*canvasH),
	}
}

func (g *brailleGrid) clear() {
	clear(g.bits)
}

func (g *brailleGrid) setDot(graphX, graphY int) {
	if graphX < 0 || graphY < 0 || graphX >= g.graphW || graphY >= g.graphH {
		return
	}
	cx, cy := graphX/2, graphY/4
	g.bits[cy*g.canvasW+cx] |= dotBits[graphX%2][graphY%4]
}

// mapY converts a normalized value [0,1] to a graph-resolution Y coordinate
// where 0.0 is the top and 1.0 is the bottom.
func (g *brailleGrid) mapY(normalized float64) int {
	return int(math.Round(clamp01(1-normalized) * float64(g.graphH-1)))
}

// drawLine rasterizes a line between two graph points using Bresenham's algorithm
func (g *brailleGrid) drawLine(x0, y0, x1, y1 int) {
	dx := abs(x1 - x0)
	dy := -abs(y1 - y0)
	sx, sy := 1, 1
	if x0 > x1 {
		sx = -1
	}
	if y0 > y1 {
		sy = -1
	}
	err := dx + dy
	for {
		g.setDot(x0, y0)
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := err * 2
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

// drawValues plots raw values, normalizing inline to the [minY, maxY] range
func (g *brailleGrid) drawValues(values []float64, minY, maxY float64) {
	span := maxY - minY
	if span <= 0 || g.graphW <= 0 {
		return
	}
	pad := max(0, g.graphW-len(values))
	srcStart := max(0, len(values)-g.graphW)

	prevX, prevY := -1, -1
	for i := range g.graphW {
		si := srcStart + (i - pad)
		if si < 0 || si >= len(values) {
			prevX, prevY = -1, -1
			continue
		}
		v := values[si]
		if math.IsNaN(v) || math.IsInf(v, 0) {
			prevX, prevY = -1, -1
			continue
		}
		x, y := i, g.mapY(clamp01((v-minY)/span))
		if prevX >= 0 {
			g.drawLine(prevX, prevY, x, y)
		} else {
			g.setDot(x, y)
		}
		prevX, prevY = x, y
	}
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
