package metrics

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	promMaxBody      = 4 << 20
	metricPromptTok  = "vllm:request_prompt_tokens"
	metricGenTok     = "vllm:request_generation_tokens"
	metricNumRunning = "vllm:num_requests_running"
	metricE2ESum     = "vllm:e2e_request_latency_seconds_sum"
)

type vllmCumulative struct {
	at         time.Time
	islSum     float64
	islCount   float64
	islBucket  map[float64]float64
	oslSum     float64
	oslCount   float64
	oslBucket  map[float64]float64
	latencySum float64
	engines    string
	running    float64
}

const staleWorkloadTTL = 60 * time.Second

type lastEmit struct {
	w  MetricsWorkload
	at time.Time
}

type vllmScraper struct {
	client   *http.Client
	mu       sync.Mutex
	prev     map[string]*vllmCumulative
	lastEmit map[string]lastEmit
}

func newVllmScraper(timeout time.Duration) *vllmScraper {
	if timeout <= 0 {
		timeout = 2 * time.Second
	}
	return &vllmScraper{
		client:   &http.Client{Timeout: timeout},
		prev:     make(map[string]*vllmCumulative),
		lastEmit: make(map[string]lastEmit),
	}
}

// Scrape diffs against the last emitted window's baseline. Scrapes with no
// completed requests keep the baseline, so deltas accumulate until a window
// has data; a stale last window is re-sent so slow workloads don't flap.
func (s *vllmScraper) Scrape(ctx context.Context, endpointURL, model string) (*MetricsWorkload, bool) {
	cur, err := s.collect(ctx, endpointURL, model)
	key := endpointURL + "|" + model
	if err != nil || cur == nil {
		return s.lastIfFresh(key)
	}

	s.mu.Lock()
	prev := s.prev[key]
	s.mu.Unlock()

	if prev == nil || !monotonic(prev, cur) {
		s.store(key, cur, nil)
		return nil, false
	}
	dIslCount := cur.islCount - prev.islCount
	dOslCount := cur.oslCount - prev.oslCount
	if dIslCount <= 0 || dOslCount <= 0 {
		return s.lastIfFresh(key)
	}

	windowSec := cur.at.Sub(prev.at).Seconds()
	concMean := cur.running
	if windowSec > 0 && cur.latencySum > prev.latencySum {
		// Little's law: time-averaged concurrency over the window
		concMean = (cur.latencySum - prev.latencySum) / windowSec
	}
	concMax := int(math.Max(cur.running, math.Ceil(concMean)))

	w := &MetricsWorkload{
		ModelName:       model,
		WindowMs:        cur.at.Sub(prev.at).Milliseconds(),
		IslMean:         round2((cur.islSum - prev.islSum) / dIslCount),
		IslHistogram:    diffHistogram(prev.islBucket, cur.islBucket, dIslCount),
		OslMean:         round2((cur.oslSum - prev.oslSum) / dOslCount),
		OslHistogram:    diffHistogram(prev.oslBucket, cur.oslBucket, dOslCount),
		ConcurrencyMean: round2(concMean),
		ConcurrencyMax:  concMax,
		EngineCount:     strings.Count(cur.engines, ",") + 1,
	}
	if !finiteWorkload(w) {
		s.store(key, cur, nil)
		return nil, false
	}
	s.store(key, cur, w)
	return w, true
}

func (s *vllmScraper) store(key string, cum *vllmCumulative, emitted *MetricsWorkload) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.prev[key] = cum
	if emitted != nil {
		s.lastEmit[key] = lastEmit{w: *emitted, at: cum.at}
	}
}

func (s *vllmScraper) lastIfFresh(key string) (*MetricsWorkload, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e, ok := s.lastEmit[key]; ok && time.Since(e.at) < staleWorkloadTTL {
		w := e.w
		return &w, true
	}
	return nil, false
}

// prune drops window state for endpoints no longer attributed.
func (s *vllmScraper) prune(live map[string]bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for k := range s.prev {
		if !live[k] {
			delete(s.prev, k)
			delete(s.lastEmit, k)
		}
	}
}

func (s *vllmScraper) collect(ctx context.Context, endpointURL, model string) (*vllmCumulative, error) {
	samples, err := s.fetch(ctx, endpointURL)
	if err != nil {
		return nil, err
	}

	cur := &vllmCumulative{
		at:        time.Now(),
		islBucket: map[float64]float64{},
		oslBucket: map[float64]float64{},
	}
	engines := map[string]bool{}
	for _, sm := range samples {
		if sm.labels["model_name"] != model {
			continue
		}
		switch sm.name {
		case metricPromptTok + "_sum":
			cur.islSum += sm.value
		case metricPromptTok + "_count":
			cur.islCount += sm.value
		case metricPromptTok + "_bucket":
			if le, ok := parseLe(sm.labels["le"]); ok {
				cur.islBucket[le] += sm.value
			}
		case metricGenTok + "_sum":
			cur.oslSum += sm.value
		case metricGenTok + "_count":
			cur.oslCount += sm.value
		case metricGenTok + "_bucket":
			if le, ok := parseLe(sm.labels["le"]); ok {
				cur.oslBucket[le] += sm.value
			}
		case metricE2ESum:
			cur.latencySum += sm.value
		case metricNumRunning:
			cur.running += sm.value
			engines[sm.labels["engine"]] = true
		}
	}
	if cur.islCount == 0 && cur.oslCount == 0 && len(engines) == 0 {
		return nil, nil
	}
	names := make([]string, 0, len(engines))
	for e := range engines {
		names = append(names, e)
	}
	sort.Strings(names)
	cur.engines = strings.Join(names, ",")
	return cur, nil
}

// monotonic rejects windows spanning a counter reset or engine-set change,
// including per-bucket regressions a summed count can mask under DP.
func monotonic(prev, cur *vllmCumulative) bool {
	if cur.engines != prev.engines {
		return false
	}
	if cur.islCount < prev.islCount || cur.oslCount < prev.oslCount ||
		cur.islSum < prev.islSum || cur.oslSum < prev.oslSum ||
		cur.latencySum < prev.latencySum {
		return false
	}
	if len(cur.islBucket) != len(prev.islBucket) || len(cur.oslBucket) != len(prev.oslBucket) {
		return false
	}
	for le, v := range prev.islBucket {
		if cv, ok := cur.islBucket[le]; !ok || cv < v {
			return false
		}
	}
	for le, v := range prev.oslBucket {
		if cv, ok := cur.oslBucket[le]; !ok || cv < v {
			return false
		}
	}
	return true
}

func diffHistogram(prev, cur map[float64]float64, total float64) MetricsHistogram {
	les := make([]float64, 0, len(cur))
	for le := range cur {
		les = append(les, le)
	}
	sort.Float64s(les)

	h := MetricsHistogram{Count: int(total), Bins: []MetricsHistogramBin{}}
	lower := 0
	var prevCum float64
	for _, le := range les {
		cum := cur[le] - prev[le]
		if n := cum - prevCum; n > 0 {
			h.Bins = append(h.Bins, MetricsHistogramBin{Lower: lower, Upper: int(le), Count: int(n)})
		}
		lower = int(le) + 1
		prevCum = cum
	}
	// +Inf tail: bound it at 2x the last finite edge
	if n := total - prevCum; n > 0 && len(les) > 0 {
		if last := int(les[len(les)-1]); last >= 1 {
			h.Bins = append(h.Bins, MetricsHistogramBin{Lower: last + 1, Upper: last * 2, Count: int(n)})
		}
	}
	return h
}

func finiteWorkload(w *MetricsWorkload) bool {
	for _, v := range []float64{w.IslMean, w.OslMean, w.ConcurrencyMean} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
	}
	return w.ConcurrencyMax >= 0 && w.WindowMs > 0
}

type promSample struct {
	name   string
	labels map[string]string
	value  float64
}

func relevantLine(line string) bool {
	return strings.HasPrefix(line, metricPromptTok) ||
		strings.HasPrefix(line, metricGenTok) ||
		strings.HasPrefix(line, metricNumRunning) ||
		strings.HasPrefix(line, metricE2ESum)
}

func (s *vllmScraper) fetch(ctx context.Context, endpointURL string) ([]promSample, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpointURL+"/metrics", nil)
	if err != nil {
		return nil, err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("metrics: HTTP %d", resp.StatusCode)
	}

	var samples []promSample
	scanner := bufio.NewScanner(io.LimitReader(resp.Body, promMaxBody))
	scanner.Buffer(make([]byte, 64*1024), 1<<20)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, "#") || !relevantLine(line) {
			continue
		}
		if sm, ok := parsePromLine(line); ok {
			samples = append(samples, sm)
		}
	}
	return samples, scanner.Err()
}

func parsePromLine(line string) (promSample, bool) {
	sm := promSample{labels: map[string]string{}}

	nameEnd := strings.IndexAny(line, "{ ")
	if nameEnd < 0 {
		return sm, false
	}
	sm.name = line[:nameEnd]
	rest := line[nameEnd:]

	if strings.HasPrefix(rest, "{") {
		end := labelsEnd(rest)
		if end < 0 {
			return sm, false
		}
		parseLabels(rest[1:end], sm.labels)
		rest = rest[end+1:]
	}

	fields := strings.Fields(rest)
	if len(fields) == 0 {
		return sm, false
	}
	v, err := strconv.ParseFloat(fields[0], 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
		return sm, false
	}
	sm.value = v
	return sm, true
}

func labelsEnd(s string) int {
	inQuote := false
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '\\':
			if inQuote {
				i++
			}
		case '"':
			inQuote = !inQuote
		case '}':
			if !inQuote {
				return i
			}
		}
	}
	return -1
}

func parseLabels(s string, out map[string]string) {
	for len(s) > 0 {
		eq := strings.Index(s, "=")
		if eq < 0 || len(s) < eq+2 || s[eq+1] != '"' {
			return
		}
		key := strings.TrimLeft(s[:eq], ",")
		i := eq + 2
		var val strings.Builder
		for i < len(s) {
			c := s[i]
			if c == '\\' && i+1 < len(s) {
				next := s[i+1]
				switch next {
				case 'n':
					val.WriteByte('\n')
				default:
					val.WriteByte(next)
				}
				i += 2
				continue
			}
			if c == '"' {
				break
			}
			val.WriteByte(c)
			i++
		}
		out[strings.TrimSpace(key)] = val.String()
		s = s[min(i+1, len(s)):]
	}
}

func parseLe(s string) (float64, bool) {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsInf(v, 0) || math.IsNaN(v) {
		return 0, false
	}
	return v, true
}

func round2(v float64) float64 {
	return math.Round(v*100) / 100
}
