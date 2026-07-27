package metrics

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"
)

// Runs only against a live vLLM server: UTLZ_TEST_VLLM_URL=http://... UTLZ_TEST_VLLM_MODEL=...
func TestVllmScraperLive(t *testing.T) {
	url := os.Getenv("UTLZ_TEST_VLLM_URL")
	model := os.Getenv("UTLZ_TEST_VLLM_MODEL")
	if url == "" || model == "" {
		t.Skip("UTLZ_TEST_VLLM_URL not set")
	}

	s := newVllmScraper(3 * time.Second)
	if _, ok := s.Scrape(context.Background(), url, model); ok {
		t.Fatal("first scrape must not produce a window")
	}
	time.Sleep(2 * time.Second)

	w, ok := s.Scrape(context.Background(), url, model)
	if !ok {
		t.Fatal("no window produced; did requests finish between scrapes?")
	}
	out, _ := json.Marshal(w)
	t.Logf("live workload: %s", out)

	if w.ModelName != model || w.IslMean <= 0 || w.OslMean <= 0 {
		t.Fatalf("bad means: %+v", w)
	}
	if w.IslHistogram.Count <= 0 || len(w.IslHistogram.Bins) == 0 {
		t.Fatalf("bad isl histogram: %+v", w.IslHistogram)
	}
	if w.EngineCount < 1 || w.WindowMs <= 0 {
		t.Fatalf("bad engine/window: %+v", w)
	}
}
