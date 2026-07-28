package version

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"

	"golang.org/x/mod/semver"
)

var REPO = "systalyze/utilyze"

type Release struct {
	TagName string `json:"tag_name"`
}

func GetLatestReleaseTag(ctx context.Context) (string, error) {
	url := fmt.Sprintf("https://api.github.com/repos/%s/releases/latest", REPO)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("X-GitHub-Api-Version", "2026-03-10")
	if token := os.Getenv("GITHUB_TOKEN"); token != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var release Release
	if err := json.NewDecoder(resp.Body).Decode(&release); err != nil {
		return "", err
	}
	return release.TagName, nil
}

func CheckForUpdates(ctx context.Context, current string) error {
	latestTag, err := GetLatestReleaseTag(ctx)
	if err != nil {
		return err
	}

	current = normalizeVersion(current)
	latestTag = normalizeVersion(latestTag)
	if semver.Compare(latestTag, current) > 0 {
		fmt.Fprintf(os.Stderr, "A new version of utlz is available: %s (current: %s)\n", latestTag, current)
		fmt.Fprintf(os.Stderr, "Run `curl -sSfL https://systalyze.com/utilyze/install.sh | sh` to update\n")
	}
	return nil
}

func normalizeVersion(version string) string {
	if strings.HasPrefix(version, "v") {
		return version
	}
	return "v" + version
}
