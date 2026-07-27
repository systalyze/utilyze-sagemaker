package vllm

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/systalyze/utilyze/internal/inference"
)

type Backend struct {
	client  *http.Client
	timeout time.Duration
}

func NewBackend(timeout time.Duration) *Backend {
	return &Backend{
		client:  &http.Client{Timeout: timeout},
		timeout: timeout,
	}
}

func (b *Backend) Name() string {
	return "vllm"
}

func (b *Backend) Discover(ctx context.Context, cohort inference.ProcessCohort) (inference.Endpoint, string, bool, error) {
	if !cohortLooksVLLMLike(cohort) {
		return inference.Endpoint{}, "", false, nil
	}

	for _, port := range cohortPorts(cohort) {
		url := fmt.Sprintf("http://127.0.0.1:%d", port)
		id, err := b.fetchModelID(ctx, url)
		if err != nil {
			slog.Debug("inference: vllm probe error", "gpu", cohort.GPU, "sid", cohort.SessionID, "url", url, "err", err)
			continue
		}
		if id == "" {
			continue
		}
		return inference.Endpoint{URL: url, Port: port}, id, true, nil
	}

	return inference.Endpoint{}, "", false, nil
}

func cohortLooksVLLMLike(cohort inference.ProcessCohort) bool {
	for _, proc := range cohort.Processes {
		if IsVLLMLike(proc.Comm, proc.Cmdline) {
			return true
		}
	}
	return false
}

func (b *Backend) fetchModelID(ctx context.Context, baseURL string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, b.timeout)
	defer cancel()

	requestURL, err := url.JoinPath(baseURL, "/v1/models")
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL, nil)
	if err != nil {
		return "", err
	}
	resp, err := b.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", nil
	}

	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return "", nil
	}
	if len(body.Data) == 0 {
		return "", nil
	}
	return body.Data[0].ID, nil
}

func cohortPorts(cohort inference.ProcessCohort) []int {
	seen := make(map[int]bool)
	ports := make([]int, 0)
	for _, proc := range cohort.Processes {
		for _, port := range proc.Ports {
			if seen[port] {
				continue
			}
			seen[port] = true
			ports = append(ports, port)
		}
	}
	sort.Ints(ports)
	return ports
}

func IsVLLMLike(comm string, cmdline []string) bool {
	commLower := strings.ToLower(comm)
	if strings.HasPrefix(commLower, "vllm") {
		return true
	}
	for _, arg := range cmdline {
		lower := strings.ToLower(arg)
		if strings.HasSuffix(lower, ".log") || strings.HasSuffix(lower, ".txt") {
			continue
		}
		base := filepath.Base(lower)
		if base == "vllm" || strings.HasPrefix(base, "vllm.") || strings.HasPrefix(lower, "vllm.") {
			return true
		}
	}
	return false
}
