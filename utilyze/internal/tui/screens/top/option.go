package top

import "time"

type Option func(*model)

func WithRefreshInterval(interval time.Duration) Option {
	return func(m *model) {
		m.drawInterval = interval
	}
}

func WithResolution(resolution time.Duration) Option {
	return func(m *model) {
		m.resolution = resolution
	}
}

func WithConnectionURL(url string) Option {
	return func(m *model) {
		m.connectionURL = url
	}
}
