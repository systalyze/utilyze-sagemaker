//go:build linux

package cupti

import (
	"archive/zip"
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/systalyze/utilyze/internal/ffi"
	"github.com/systalyze/utilyze/internal/pypi"
)

const (
	cuptPypiDistribution = "nvidia-cuda-cupti-cu12"

	downloadTimeout = 60 * time.Second

	dirMode  = 0o755
	fileMode = 0o755
)

func platformTag() (string, error) {
	switch runtime.GOARCH {
	case "amd64":
		return "x86_64", nil
	case "arm64":
		return "aarch64", nil
	default:
		return "", ffi.ErrUnsupportedPlatform
	}
}

func findLatestRelease() (pypi.ProjectReleaseMetadata, string, error) {
	platform, err := platformTag()
	if err != nil {
		return pypi.ProjectReleaseMetadata{}, "", err
	}
	return pypi.FindWheelLatestRelease(cuptPypiDistribution, "", "none", platform)
}

func promptDownload() (bool, error) {
	answer, err := bufio.NewReader(os.Stdin).ReadString('\n')
	if err != nil && !errors.Is(err, io.EOF) {
		return false, err
	}
	answer = strings.ToLower(strings.TrimSpace(answer))
	return answer == "y" || answer == "yes", nil
}

func installWheel(release pypi.ProjectReleaseMetadata, downloadDir string) error {
	if err := os.MkdirAll(downloadDir, dirMode); err != nil {
		return err
	}

	wheelPath, err := downloadWheel(release, downloadDir)
	if err != nil {
		return err
	}
	defer os.Remove(wheelPath)
	return extractWheel(wheelPath, downloadDir)
}

func downloadWheel(release pypi.ProjectReleaseMetadata, downloadDir string) (string, error) {
	resp, err := (&http.Client{Timeout: downloadTimeout}).Get(release.Url)
	if err != nil {
		return "", fmt.Errorf("download wheel: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download wheel: HTTP %d", resp.StatusCode)
	}

	tmpFile, err := os.CreateTemp(downloadDir, "."+release.Filename+".*")
	if err != nil {
		return "", err
	}
	tmpPath := tmpFile.Name()
	wheelPath := filepath.Join(downloadDir, release.Filename)

	hasher := sha256.New()
	_, copyErr := io.Copy(tmpFile, io.TeeReader(resp.Body, hasher))
	closeErr := tmpFile.Close()
	if copyErr != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("download wheel: %w", copyErr)
	}
	if closeErr != nil {
		os.Remove(tmpPath)
		return "", closeErr
	}

	if expected := release.Digests.Sha256; expected != "" {
		got := hex.EncodeToString(hasher.Sum(nil))
		if !strings.EqualFold(got, expected) {
			os.Remove(tmpPath)
			return "", fmt.Errorf("download wheel: checksum mismatch: expected %s, got %s", expected, got)
		}
	}
	if err := os.Rename(tmpPath, wheelPath); err != nil {
		os.Remove(tmpPath)
		return "", err
	}
	return wheelPath, nil
}

func extractWheel(path, targetDir string) error {
	r, err := zip.OpenReader(path)
	if err != nil {
		return fmt.Errorf("open wheel: %w", err)
	}
	defer r.Close()

	files := make(map[string]*zip.File)
	for _, f := range r.File {
		files[filepath.Base(f.Name)] = f
	}
	for _, name := range requiredLibs {
		f := files[name]
		if f == nil {
			return fmt.Errorf("wheel missing: %s", missingLibs(files))
		}
		if err := extractWheelFile(f, targetDir, name); err != nil {
			return err
		}
	}
	return nil
}

func extractWheelFile(f *zip.File, targetDir, name string) error {
	rc, err := f.Open()
	if err != nil {
		return fmt.Errorf("open %s in wheel: %w", name, err)
	}
	defer rc.Close()

	tmpFile, err := os.CreateTemp(targetDir, "."+name+".*")
	if err != nil {
		return fmt.Errorf("create temp file for %s: %w", name, err)
	}
	tmpPath := tmpFile.Name()

	if _, err := io.Copy(tmpFile, rc); err != nil {
		tmpFile.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("extract %s: %w", name, err)
	}
	if err := tmpFile.Close(); err != nil {
		os.Remove(tmpPath)
		return err
	}
	if err := os.Chmod(tmpPath, fileMode); err != nil {
		os.Remove(tmpPath)
		return err
	}
	if err := os.Rename(tmpPath, filepath.Join(targetDir, name)); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("install %s: %w", name, err)
	}
	return nil
}

func missingLibs(files map[string]*zip.File) string {
	missing := make([]string, 0, len(requiredLibs))
	for _, name := range requiredLibs {
		if files[name] == nil {
			missing = append(missing, name)
		}
	}
	sort.Strings(missing)
	return strings.Join(missing, ", ")
}
