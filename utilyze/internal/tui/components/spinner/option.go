package spinner

import "charm.land/lipgloss/v2"

type options struct {
	Spinner spinner

	Style *lipgloss.Style
}

type Option func(*options)
