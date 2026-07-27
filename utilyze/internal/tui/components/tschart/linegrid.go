package tschart

import "math"

const (
	connN = 1 << iota
	connS
	connE
	connW
)

var connectorRunes = [16]rune{
	0:                             ' ',
	connN:                         '╵',
	connS:                         '╷',
	connE:                         '╶',
	connW:                         '╴',
	connN | connS:                 '│',
	connE | connW:                 '─',
	connS | connE:                 '┌',
	connS | connW:                 '┐',
	connN | connE:                 '└',
	connN | connW:                 '┘',
	connN | connS | connE:         '├',
	connN | connS | connW:         '┤',
	connS | connE | connW:         '┬',
	connN | connE | connW:         '┴',
	connN | connS | connE | connW: '┼',
}

// lineGrid is a 1-cell-per-column grid for plain-line (non-braille) rendering.
// Each cell stores a 4-bit {N,S,E,W} connector mask; the corresponding
// box-drawing rune is derived at render time.
type lineGrid struct {
	canvasW, canvasH int
	masks            []uint8
}

func newLineGrid(canvasW, canvasH int) lineGrid {
	return lineGrid{
		canvasW: canvasW,
		canvasH: canvasH,
		masks:   make([]uint8, canvasW*canvasH),
	}
}

func (g *lineGrid) clear() {
	clear(g.masks)
}

func (g *lineGrid) addMask(x, y int, m uint8) {
	if x < 0 || x >= g.canvasW || y < 0 || y >= g.canvasH {
		return
	}
	g.masks[y*g.canvasW+x] |= m
}

func (g *lineGrid) runeAt(x, y int) rune {
	if x < 0 || x >= g.canvasW || y < 0 || y >= g.canvasH {
		return 0
	}
	m := g.masks[y*g.canvasW+x]
	if m == 0 {
		return 0
	}
	return connectorRunes[m]
}

func (g *lineGrid) mapY(normalized float64) int {
	if g.canvasH <= 1 {
		return 0
	}
	return int(math.Round(clamp01(1-normalized) * float64(g.canvasH-1)))
}

// drawValues rasterizes values as a box-drawing polyline, normalizing inline
// to the [minY, maxY] range. Values are aligned to the right edge of the
// canvas when len(values) < canvasW.
func (g *lineGrid) drawValues(values []float64, minY, maxY float64) {
	span := maxY - minY
	if span <= 0 || g.canvasW <= 0 || g.canvasH <= 0 {
		return
	}
	pad := max(0, g.canvasW-len(values))
	srcStart := max(0, len(values)-g.canvasW)

	prevX, prevY := -1, -1
	for i := range g.canvasW {
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
		if (v-minY) > span*0.005 && g.canvasH >= 3 && y >= g.canvasH-1 {
			y = g.canvasH - 2
		}
		if (maxY-v) > span*0.005 && g.canvasH >= 3 && y <= 0 {
			y = 1
		}
		if prevX < 0 {
			g.addMask(x, y, connE|connW)
			prevX, prevY = x, y
			continue
		}
		g.drawSegment(prevX, prevY, x, y)
		prevX, prevY = x, y
	}
}

// drawSegment connects (x1, y1) to (x2, y2) with x2 > x1, placing the
// horizontal run on row y1 and the vertical step at column x2.
func (g *lineGrid) drawSegment(x1, y1, x2, y2 int) {
	if x2 <= x1 {
		return
	}
	g.addMask(x1, y1, connE)
	for x := x1 + 1; x < x2; x++ {
		g.addMask(x, y1, connW|connE)
	}
	if y1 == y2 {
		g.addMask(x2, y2, connW)
		return
	}
	if y2 < y1 {
		g.addMask(x2, y1, connW|connN)
		for yy := y2 + 1; yy < y1; yy++ {
			g.addMask(x2, yy, connN|connS)
		}
		g.addMask(x2, y2, connS)
	} else {
		g.addMask(x2, y1, connW|connS)
		for yy := y1 + 1; yy < y2; yy++ {
			g.addMask(x2, yy, connN|connS)
		}
		g.addMask(x2, y2, connN)
	}
}

// fillRow sets a solid horizontal run across every column of the given row.
func (g *lineGrid) fillRow(y int) {
	if y < 0 || y >= g.canvasH || g.canvasW <= 0 {
		return
	}
	if g.canvasW == 1 {
		g.addMask(0, y, connE|connW)
		return
	}
	g.addMask(0, y, connE)
	for x := 1; x < g.canvasW-1; x++ {
		g.addMask(x, y, connW|connE)
	}
	g.addMask(g.canvasW-1, y, connW)
}
