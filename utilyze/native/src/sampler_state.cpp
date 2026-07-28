#include "sampler_internal.hpp"

namespace utlz_sampler_internal {

void* state_create(int device_index, const char* metrics_csv,
                   int interval_ms, int start_pass) {
    State* s = new State();
    s->cfg.deviceIndex = device_index;
    s->cfg.intervalMs = interval_ms > 0 ? interval_ms : 200;
    s->cfg.startPass = start_pass;
    s->cfg.counterDataMaxSamples = (uint32_t)ReadEnvInt("UTLZ_PERF_MAX_SAMPLES", 1024, 32, 8192);
    s->cfg.maxNumUndecodedSamples = (size_t)ReadEnvInt("UTLZ_PERF_MAX_UNDECODED", 32768, 16, 65536);
    s->cfg.debug = DebugLogsEnabled();
    s->cfg.triggersPerPass = ReadEnvInt("UTLZ_PERF_TRIGGERS_PER_PASS", 2, 1, 64);
    s->cfg.triggerSpacingMs = ReadEnvInt("UTLZ_PERF_TRIGGER_SPACING_MS", 120, 1, 5000);
    s->cfg.publishWarmupRotations = ReadEnvInt("UTLZ_PERF_PUBLISH_WARMUP_ROTATIONS", 1, 0, 128);

    const char* backend = GetEnv("UTLZ_BACKEND");
    if (backend && *backend && !EqualsIgnoreCase(backend, "perfworks")) {
        SetError(s, std::string("unsupported backend '") + backend + "' (this build supports only 'perfworks')");
        return s;
    }

    if (!metrics_csv || !*metrics_csv) {
        SetError(s, "utlz_sampler_create: metrics_csv is required");
        return s;
    }

    s->cfg.requestedMetricNames = SplitCsv(metrics_csv);
    if (s->cfg.requestedMetricNames.empty()) {
        SetError(s, "utlz_sampler_create: no valid metrics parsed from metrics_csv");
        return s;
    }

    if (!Setup(s)) {
        Cleanup(s);
        return s;
    }

    s->initialized = true;
    return s;
}

int state_is_initialized(void* handle) {
    if (!handle) return 0;
    State* s = (State*)handle;
    return s->initialized ? 1 : 0;
}

int state_metric_count(void* handle) {
    if (!handle) return 0;
    State* s = (State*)handle;
    if (!s->initialized) return 0;
    return (int)s->perf.globalMetricNames.size();
}

const char* state_metric_name(void* handle, int index) {
    static const char* empty = "";
    if (!handle) return empty;
    State* s = (State*)handle;
    if (!s->initialized) return empty;
    if (index < 0 || (size_t)index >= s->perf.globalMetricNames.size()) return empty;
    return s->perf.globalMetricNames[(size_t)index].c_str();
}

const char* state_error(void* handle) {
    static const char* none = "";
    if (!handle) return none;
    State* s = (State*)handle;

    static thread_local std::string threadLocalError;
    {
        std::lock_guard<std::mutex> lock(s->errorMutex);
        threadLocalError = s->lastError;
    }
    return threadLocalError.c_str();
}

int state_poll(void* handle, double* out_values, int max_metrics,
               int* out_pm_samples, int* out_group_id, int* out_group_count) {
    if (!handle) return -1;
    State* s = (State*)handle;
    if (!s->initialized) {
        SetError(s, "utlz_sampler_poll: sampler is not initialized");
        return -1;
    }

    const int metricCount = (int)s->perf.globalMetricNames.size();
    if (!out_values || max_metrics < metricCount) {
        SetError(s, "utlz_sampler_poll: output buffer is too small");
        return -1;
    }

    const int defaultGroupCount = (int)std::max((size_t)1, s->perf.fullRotation.numPasses);

    if (out_pm_samples) *out_pm_samples = 0;
    if (out_group_id) *out_group_id = 0;
    if (out_group_count) *out_group_count = defaultGroupCount;

    for (int i = 0; i < metricCount; ++i) {
        out_values[i] = std::numeric_limits<double>::quiet_NaN();
    }

    std::deque<WindowRecord> records;
    {
        std::lock_guard<std::mutex> lock(s->ringMutex);
        if (s->ring.empty()) return 0;
        records.swap(s->ring);
    }

    const int drained = (int)records.size();

    std::vector<double> sums((size_t)metricCount, 0.0);
    std::vector<int> counts((size_t)metricCount, 0);
    int pmSamplesTotal = 0;

    for (const auto& rec : records) {
        pmSamplesTotal += (int)rec.pmSamples;
        if (out_group_id) *out_group_id = rec.groupId;
        if (out_group_count) *out_group_count = rec.groupCount;

        const size_t n = std::min(rec.metricMeans.size(), (size_t)metricCount);
        for (size_t i = 0; i < n; ++i) {
            const double v = rec.metricMeans[i];
            if (std::isfinite(v)) {
                sums[i] += v;
                counts[i] += 1;
            }
        }
    }

    for (int i = 0; i < metricCount; ++i) {
        if (counts[(size_t)i] > 0) {
            out_values[i] = sums[(size_t)i] / (double)counts[(size_t)i];
        }
    }

    if (out_pm_samples) *out_pm_samples = pmSamplesTotal;

    return drained;
}

void state_destroy(void* handle) {
    if (!handle) return;
    State* s = (State*)handle;
    Cleanup(s);
    delete s;
}

} // namespace utlz_sampler_internal
