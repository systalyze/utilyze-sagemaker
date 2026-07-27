//go:build !linux

package sampler

import "github.com/systalyze/utilyze/internal/ffi"

func load() (string, error) {
	return "", ffi.ErrUnsupportedPlatform
}

func HasProfilingCapabilities() (bool, error) {
	return false, ffi.ErrUnsupportedPlatform
}
