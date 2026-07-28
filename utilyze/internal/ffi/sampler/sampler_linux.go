//go:build linux

package sampler

import (
	_ "embed"
	"errors"
	"fmt"
	"sync"

	"github.com/ebitengine/purego"
	"golang.org/x/sys/unix"
)

//go:embed embedded/libutlz_sampler.so.*
var libsampler []byte

func createMemFd() (string, error) {
	fd, err := unix.MemfdCreate("libsampler", unix.MFD_CLOEXEC|unix.MFD_ALLOW_SEALING)
	if err != nil {
		return "", err
	}

	if _, err := unix.Write(fd, libsampler); err != nil {
		unix.Close(fd)
		return "", err
	}

	if _, err := unix.FcntlInt(uintptr(fd), unix.F_ADD_SEALS, unix.F_SEAL_WRITE|unix.F_SEAL_SHRINK|unix.F_SEAL_GROW); err != nil {
		unix.Close(fd)
		return "", err
	}

	return fmt.Sprintf("/proc/self/fd/%d", fd), nil
}

var loadOnce sync.Once
var loadErr error

var SHA256SUM string

func load() (string, error) {
	loadOnce.Do(func() {
		path, pathErr := createMemFd()
		if pathErr != nil {
			loadErr = pathErr
			return
		}

		lib, openErr := purego.Dlopen(path, purego.RTLD_NOW)
		if openErr != nil {
			loadErr = errors.Join(loadErr, openErr)
			return
		}

		defer func() {
			if err := recover(); err != nil {
				closeErr := purego.Dlclose(lib)
				loadErr = errors.Join(fmt.Errorf("failed to load symbols: %v", err), closeErr)
			}
		}()
		loadSymbols(lib)
	})
	return SHA256SUM, loadErr
}

func loadSymbols(lib uintptr) {
	purego.RegisterLibFunc(&utlzSamplerCreate, lib, sym_utlzSamplerCreate)
	purego.RegisterLibFunc(&utlzSamplerIsInitialized, lib, sym_utlzSamplerIsInitialized)
	purego.RegisterLibFunc(&utlzSamplerGetMetricCount, lib, sym_utlzSamplerGetMetricCount)
	purego.RegisterLibFunc(&utlzSamplerGetMetricName, lib, sym_utlzSamplerGetMetricName)
	purego.RegisterLibFunc(&utlzSamplerGetError, lib, sym_utlzSamplerGetError)
	purego.RegisterLibFunc(&utlzSamplerPoll, lib, sym_utlzSamplerPoll)
	purego.RegisterLibFunc(&utlzSamplerDestroy, lib, sym_utlzSamplerDestroy)
}

func HasProfilingCapabilities() (bool, error) {
	// version 3 capabilities are 64 bit and need to twice-wide for lower/higher bits
	hdr := unix.CapUserHeader{Version: unix.LINUX_CAPABILITY_VERSION_3}
	data := [2]unix.CapUserData{}
	if err := unix.Capget(&hdr, &data[0]); err != nil {
		return false, fmt.Errorf("failed to inspect capabilities: %w", err)
	}

	cap := unix.CAP_SYS_ADMIN
	wordIndex := cap / 32
	bitIndex := cap % 32
	if int(wordIndex) >= len(data) || data[wordIndex].Effective&(1<<bitIndex) == 0 {
		return false, nil
	}
	return true, nil
}
