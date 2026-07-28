package spinner

import "time"

type tickMsg struct {
	time time.Time
	id   int64
}
