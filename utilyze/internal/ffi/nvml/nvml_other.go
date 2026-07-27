//go:build !linux

package nvml

import "github.com/systalyze/utilyze/internal/ffi"

func load() error {
	return ffi.ErrUnsupportedPlatform
}
