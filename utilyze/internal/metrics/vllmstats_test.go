package metrics

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func promBody(count, sum, running float64) string {
	cum1 := count * 0.3
	return fmt.Sprintf(`# HELP vllm:request_prompt_tokens Number of prefill tokens processed.
vllm:request_prompt_tokens_bucket{engine="0",le="500",model_name="m1"} %g
vllm:request_prompt_tokens_bucket{engine="0",le="1000",model_name="m1"} %g
vllm:request_prompt_tokens_bucket{engine="0",le="+Inf",model_name="m1"} %g
vllm:request_prompt_tokens_count{engine="0",model_name="m1"} %g
vllm:request_prompt_tokens_sum{engine="0",model_name="m1"} %g
vllm:request_generation_tokens_bucket{engine="0",le="500",model_name="m1"} %g
vllm:request_generation_tokens_bucket{engine="0",le="+Inf",model_name="m1"} %g
vllm:request_generation_tokens_count{engine="0",model_name="m1"} %g
vllm:request_generation_tokens_sum{engine="0",model_name="m1"} %g
vllm:num_requests_running{engine="0",model_name="m1"} %g
vllm:num_requests_running{engine="1",model_name="m1"} %g
vllm:request_prompt_tokens_count{engine="0",model_name="other"} 999
`, cum1, count, count, count, sum, count, count, count, sum/4, running, running+2)
}

func TestVllmScraperWindowDiff(t *testing.T) {
	body := promBody(100, 90000, 7)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	s := newVllmScraper(0)
	if _, ok := s.Scrape(context.Background(), srv.URL, "m1"); ok {
		t.Fatal("first scrape must not produce a window")
	}

	body = promBody(140, 130000, 9)
	time.Sleep(5 * time.Millisecond)
	w, ok := s.Scrape(context.Background(), srv.URL, "m1")
	if !ok {
		t.Fatal("second scrape should produce a window")
	}
	if w.ModelName != "m1" || w.IslMean != 1000 || w.OslMean != 250 {
		t.Fatalf("means: %+v", w)
	}
	if w.IslHistogram.Count != 40 || len(w.IslHistogram.Bins) != 2 {
		t.Fatalf("isl histogram: %+v", w.IslHistogram)
	}
	if b := w.IslHistogram.Bins[0]; b.Lower != 0 || b.Upper != 500 || b.Count != 12 {
		t.Fatalf("bin0: %+v", b)
	}
	if b := w.IslHistogram.Bins[1]; b.Lower != 501 || b.Upper != 1000 || b.Count != 28 {
		t.Fatalf("bin1: %+v", b)
	}
	if w.ConcurrencyMean != 20 || w.EngineCount != 2 {
		t.Fatalf("concurrency: %+v", w)
	}

	body = promBody(140, 130000, 9) // no new completions: stale re-send
	if w, ok := s.Scrape(context.Background(), srv.URL, "m1"); !ok || w.IslMean != 1000 {
		t.Fatalf("quiet window should re-send last workload, got %v %v", w, ok)
	}

	body = promBody(150, 140000, 9)
	time.Sleep(5 * time.Millisecond)
	if w, ok := s.Scrape(context.Background(), srv.URL, "m1"); !ok || w.IslHistogram.Count != 10 {
		t.Fatalf("accumulated window should span since last emit: %+v %v", w, ok)
	}

	body = promBody(10, 9000, 1) // counter reset
	if _, ok := s.Scrape(context.Background(), srv.URL, "m1"); ok {
		t.Fatal("counter reset must not produce a window")
	}
}

func TestPromLineParsing(t *testing.T) {
	sm, ok := parsePromLine(`vllm:request_prompt_tokens_bucket{engine="0",le="2000",model_name="org/model-a"} 42`)
	if !ok || sm.name != "vllm:request_prompt_tokens_bucket" || sm.value != 42 {
		t.Fatalf("%+v %v", sm, ok)
	}
	if sm.labels["le"] != "2000" || sm.labels["model_name"] != "org/model-a" {
		t.Fatalf("labels: %+v", sm.labels)
	}

	for _, bad := range []string{
		"garbage",
		`vllm:foo{a="b"}`,
		"vllm:foo ",
		"vllm:foo",
		`vllm:foo{unclosed="x"`,
		`vllm:foo{a="b"} NaN`,
		`vllm:foo{a="b"} +Inf`,
	} {
		if _, ok := parsePromLine(bad); ok {
			t.Fatalf("accepted %q", bad)
		}
	}

	if sm, ok := parsePromLine(`vllm:foo{a="b"} 1.23e+06 1699999999000`); !ok || sm.value != 1230000 {
		t.Fatalf("exponent+timestamp: %+v %v", sm, ok)
	}
	if sm, ok := parsePromLine(`vllm:foo{name="a\"b",le="500"} 7`); !ok || sm.labels["name"] != `a"b` || sm.labels["le"] != "500" {
		t.Fatalf("escaped label: %+v %v", sm, ok)
	}
}

func TestBucketOnlyResetDiscardsWindow(t *testing.T) {
	prev := &vllmCumulative{
		islCount: 100, oslCount: 100, engines: "0,1",
		islBucket: map[float64]float64{500: 90},
		oslBucket: map[float64]float64{500: 100},
	}
	cur := &vllmCumulative{
		islCount: 110, oslCount: 110, engines: "0,1",
		islBucket: map[float64]float64{500: 40}, // engine restart: bucket fell, sum count rose
		oslBucket: map[float64]float64{500: 110},
	}
	if monotonic(prev, cur) {
		t.Fatal("bucket regression not detected")
	}
	cur.islBucket[500] = 95
	cur.engines = "0"
	if monotonic(prev, cur) {
		t.Fatal("engine set change not detected")
	}
}
