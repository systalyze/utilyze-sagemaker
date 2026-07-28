package tschart

import (
	"time"

	"charm.land/lipgloss/v2"
)

type TickFormatter func(chart *Model, v float64, index int, n int) string

type Option func(*Model)

func WithXTicks(ticks int) Option {
	return func(m *Model) {
		m.XTicks = ticks
	}
}

func WithXTickFormatter(formatter TickFormatter) Option {
	return func(m *Model) {
		m.XTickFormatter = formatter
	}
}

func WithYRange(min, max float64) Option {
	return func(m *Model) {
		m.YRange = [2]float64{min, max}
	}
}

func WithYTicks(ticks int) Option {
	return func(m *Model) {
		m.YTicks = ticks
	}
}

func WithYTickFormatter(formatter TickFormatter) Option {
	return func(m *Model) {
		m.YTickFormatter = formatter
	}
}

func WithResolution(d time.Duration) Option {
	return func(m *Model) {
		m.Resolution = d
	}
}

func WithAutoScale() Option {
	return func(m *Model) {
		m.AutoScale = true
	}
}

func WithStyles(borderStyle, axisStyle, panelStyle lipgloss.Style) Option {
	return func(m *Model) {
		m.BorderStyle = borderStyle
		m.AxisStyle = axisStyle
		m.PanelStyle = panelStyle
	}
}

func (m *Model) SetStyles(borderStyle, axisStyle, panelStyle lipgloss.Style) {
	m.BorderStyle = borderStyle
	m.AxisStyle = axisStyle
	m.PanelStyle = panelStyle
	m.buildStyleANSITable()
	m.Resize(m.width, m.height)
}

func WithDetailMode(enabled bool) Option {
	return func(m *Model) {
		m.DetailMode = enabled
	}
}
