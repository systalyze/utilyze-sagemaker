//go:build !linux

package cupti

import "github.com/systalyze/utilyze/internal/ffi"

func EnsureLoaded() error {
	return ffi.ErrUnsupportedPlatform
}
