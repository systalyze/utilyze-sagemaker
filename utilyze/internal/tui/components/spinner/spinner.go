package spinner

import (
	"sync/atomic"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
)

var id int64

func nextId() int64 {
	return atomic.AddInt64(&id, 1)
}

type spinner struct {
	Frames []string
	SPF    time.Duration
}

var LogoSpinner = spinner{
	Frames: []string{
		"⠊⠁  ",
		"⠈⠑  ",
		" ⠑⠄ ",
		" ⠐⠤ ",
		"  ⠤⠂",
		"  ⠨⠂",
		"  ⠉⠂",
		" ⠐⠉ ",
		" ⠔⠁ ",
		"⠠⠔  ",
		"⠢⠄  ",
		"⠪   ",
	},
	SPF: time.Second / 10,
}

type Model struct {
	options *options

	id    int64
	frame int
}

func New(style *lipgloss.Style, opts ...Option) Model {
	o := &options{Spinner: LogoSpinner, Style: style}
	for _, option := range opts {
		option(o)
	}
	return Model{options: o, id: nextId()}
}

func (m Model) Tick() tea.Msg {
	return tickMsg{time: time.Now(), id: m.id}
}

func (m Model) Init() tea.Cmd {
	return m.Tick
}

func (m Model) View() string {
	if m.frame >= len(m.options.Spinner.Frames) {
		return "(error)"
	}
	return m.options.Style.Render(m.options.Spinner.Frames[m.frame])
}

func (m Model) Update(msg tea.Msg) (Model, tea.Cmd) {
	switch typedMsg := msg.(type) {
	case tickMsg:
		if typedMsg.id != m.id {
			return m, nil
		}

		m.frame = (m.frame + 1) % len(m.options.Spinner.Frames)
		return m, tea.Tick(m.options.Spinner.SPF, func(t time.Time) tea.Msg {
			return tickMsg{time: t, id: m.id}
		})
	default:
		return m, nil
	}
}
