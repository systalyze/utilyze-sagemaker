package pypi

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const (
	pypiUrl          = "https://pypi.org"
	defaultPythonTag = "py3"
)

type ProjectReleaseMetadataDigests struct {
	Sha256 string `json:"sha256"`
}

type ProjectReleaseMetadata struct {
	Digests       ProjectReleaseMetadataDigests `json:"digests"`
	Filename      string                        `json:"filename"`
	PythonVersion string                        `json:"python_version"`
	Size          int                           `json:"size"`
	Url           string                        `json:"url"`
}

type ProjectMetadata struct {
	Urls []ProjectReleaseMetadata `json:"urls"`
}

func FindWheelLatestRelease(distribution string, python string, abi string, platform string) (ProjectReleaseMetadata, string, error) {
	metaUrl := fmt.Sprintf("%s/pypi/%s/json", pypiUrl, distribution)
	resp, err := http.Get(metaUrl)
	if err != nil {
		return ProjectReleaseMetadata{}, "", fmt.Errorf("pypi: get project metadata: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return ProjectReleaseMetadata{}, "", fmt.Errorf("pypi: get project metadata: HTTP %d: %s", resp.StatusCode, string(body))
	}

	var metadata ProjectMetadata
	if err := json.NewDecoder(resp.Body).Decode(&metadata); err != nil {
		return ProjectReleaseMetadata{}, "", fmt.Errorf("pypi: decode project metadata: %w", err)
	}

	if python == "" {
		python = defaultPythonTag
	}

	return findMatchingRelease(metadata.Urls, python, abi, platform)
}

// ref: https://packaging.python.org/en/latest/specifications/binary-distribution-format/#file-name-convention
func parseFilename(filename string) (version string, python string, abi string, platform string, err error) {
	parts := strings.Split(filename, "-")
	if len(parts) == 6 {
		// there's a build tag
		version = parts[1]
		python = parts[3]
		abi = parts[4]
		platform = parts[5]
		return
	}
	if len(parts) == 5 {
		version = parts[1]
		python = parts[2]
		abi = parts[3]
		platform = parts[4]
		return
	}
	return "", "", "", "", fmt.Errorf("pypi: parse filename: invalid filename: %s", filename)
}

func findMatchingRelease(releases []ProjectReleaseMetadata, desiredPython string, desiredAbi string, desiredPlatform string) (ProjectReleaseMetadata, string, error) {
	for _, release := range releases {
		version, python, abi, platform, err := parseFilename(release.Filename)
		if err != nil {
			continue
		}
		if strings.HasPrefix(python, desiredPython) && abi == desiredAbi && strings.Contains(platform, desiredPlatform) {
			return release, version, nil
		}
	}
	return ProjectReleaseMetadata{}, "", fmt.Errorf("pypi: find matching release: not found for python=%s, abi=%s, platform=%s", desiredPython, desiredAbi, desiredPlatform)
}
