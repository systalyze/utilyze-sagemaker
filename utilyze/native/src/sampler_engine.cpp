#include "sampler_internal.hpp"

extern "C" {
typedef struct NVPW_CUDA_LoadDriver_Params
{
    size_t structSize;
    void* pPriv;
} NVPW_CUDA_LoadDriver_Params;
#define NVPW_CUDA_LoadDriver_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_CUDA_LoadDriver_Params, pPriv)

NVPA_Status NVPW_CUDA_LoadDriver(NVPW_CUDA_LoadDriver_Params* pParams);
}

namespace utlz_sampler_internal {

static std::mutex s_globalInitMutex;
static int s_globalInitRefCount = 0;

static bool TraceLogsEnabled() {
    static bool enabled = (GetEnv("UTLZ_PERF_TRACE") != nullptr);
    return enabled;
}

static void Logf(State* s, const char* fmt, ...) {
    if (!DebugLogsEnabled()) return;
    std::fprintf(stderr, "[utlz:%d] ", s ? s->cfg.deviceIndex : -1);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "\n");
}

static const char* CudaErr(CUresult r) {
    const char* e = "unknown";
    cuGetErrorString(r, &e);
    return e ? e : "unknown";
}

static bool CudaOk(State* s, CUresult r, const char* where) {
    if (r == CUDA_SUCCESS) return true;
    std::ostringstream oss;
    oss << where << ": CUDA error " << (int)r << " (" << CudaErr(r) << ")";
    SetError(s, oss.str());
    return false;
}

static bool NvpaOk(State* s, NVPA_Status st, const char* where) {
    if (st == NVPA_STATUS_SUCCESS) return true;
    SetError(s, std::string(where) + ": " + FormatStatus(st));
    return false;
}

static int FindMetricIndex(const std::vector<std::string>& metrics, const char* name) {
    for (size_t i = 0; i < metrics.size(); ++i) {
        if (metrics[i] == name) return (int)i;
    }
    return -1;
}

static bool GlobalInit(State* s) {
    std::lock_guard<std::mutex> lock(s_globalInitMutex);

    if (s_globalInitRefCount == 0) {
        if (!InitializeNvPerf()) {
            SetError(s, "InitializeNvPerf failed");
            return false;
        }

        if (!CudaOk(s, cuInit(0), "cuInit")) return false;

        NVPW_CUDA_LoadDriver_Params loadDriver{NVPW_CUDA_LoadDriver_Params_STRUCT_SIZE};
        if (!NvpaOk(s, NVPW_CUDA_LoadDriver(&loadDriver), "NVPW_CUDA_LoadDriver")) return false;
    }

    s_globalInitRefCount++;
    return true;
}

static void GlobalRelease() {
    std::lock_guard<std::mutex> lock(s_globalInitMutex);
    if (s_globalInitRefCount > 0) s_globalInitRefCount--;
}

static bool InitCounterDataImage(State* s, const CounterConfiguration& cfg, std::vector<uint8_t>& outCounterDataImage) {
    return GpuPeriodicSamplerCreateCounterData(
        s->cfg.deviceIndex,
        cfg.counterDataPrefix.data(),
        cfg.counterDataPrefix.size(),
        s->cfg.counterDataMaxSamples,
        NVPW_PERIODIC_SAMPLER_COUNTER_DATA_APPEND_MODE_CIRCULAR,
        outCounterDataImage);
}

static bool BuildFullRotationConfig(State* s) {
    std::vector<std::string> requested;
    requested.reserve(s->cfg.requestedMetricNames.size());
    for (const auto& m : s->cfg.requestedMetricNames) {
        if (FindMetricIndex(requested, m.c_str()) < 0) requested.push_back(m);
    }

    if (requested.empty()) {
        SetError(s, "metric request list is empty");
        return false;
    }

    s->perf.globalMetricNames = requested;

    auto& full = s->perf.fullRotation;
    full = FullRotationState{};
    full.warmupRotationsRemaining = s->cfg.publishWarmupRotations;

    NVPW_RawCounterConfig* pRawCounterConfig = DeviceCreateRawCounterConfig(s->perf.deviceIds.pChipName);
    if (!pRawCounterConfig) {
        SetError(s, "DeviceCreateRawCounterConfig failed for full rotation config");
        return false;
    }

    MetricsConfigBuilder configBuilder;
    if (!configBuilder.Initialize(s->perf.metricsEvaluator, pRawCounterConfig, s->perf.deviceIds.pChipName)) {
        SetError(s, "MetricsConfigBuilder::Initialize failed for full rotation config");
        return false;
    }

    full.metricEvalRequests.reserve(requested.size());
    for (const auto& m : requested) {
        NVPW_MetricEvalRequest req{};
        if (!s->perf.metricsEvaluator.ToMetricEvalRequest(m.c_str(), req)) {
            SetError(s, "ToMetricEvalRequest failed for metric in full rotation config: " + m);
            return false;
        }
        full.metricEvalRequests.push_back(req);
    }

    if (!configBuilder.AddMetrics(full.metricEvalRequests.data(), full.metricEvalRequests.size(), /*keepInstances=*/false)) {
        SetError(s, "MetricsConfigBuilder::AddMetrics failed for full rotation config");
        return false;
    }

    if (!CreateConfiguration(configBuilder, full.configuration)) {
        SetError(s, "CreateConfiguration failed for full rotation config");
        return false;
    }

    full.numPasses = full.configuration.numPasses;
    if (full.numPasses == 0) {
        SetError(s, "full rotation config reported zero passes");
        return false;
    }

    if (!InitCounterDataImage(s, full.configuration, full.counterDataImage)) {
        SetError(s, "counter data init failed for full rotation config");
        return false;
    }

    // Save a pristine copy so we can reset the image before each rotation cycle.
    // PerfWorks accumulates raw counters into the image across DecodeCounters calls,
    // so without resetting, pct_of_peak_sustained_elapsed metrics show a cumulative
    // average that decays as 1/t after a workload ends instead of dropping to zero.
    full.pristineCounterDataImage = full.counterDataImage;

    if (!s->perf.metricsEvaluator.MetricsEvaluatorSetDeviceAttributes(
            full.counterDataImage.data(),
            full.counterDataImage.size())) {
        SetError(s, "MetricsEvaluatorSetDeviceAttributes failed for full rotation config");
        return false;
    }

    Logf(s, "PerfWorks full rotation config passes: %zu (triggers/pass=%d, spacing=%dms)",
         full.numPasses,
         s->cfg.triggersPerPass,
         s->cfg.triggerSpacingMs);

    return true;
}

static bool BeginSessionFullRotation(State* s) {
    const uint64_t intervalNs = (uint64_t)std::max(1, s->cfg.intervalMs) * 1000000ull;
    s->perf.samplingInterval = s->perf.sampler.GetGpuPulseSamplingInterval((uint32_t)intervalNs);

    size_t rb = 0;
    if (!GpuPeriodicSamplerCalculateRecordBufferSize(
            s->cfg.deviceIndex,
            s->perf.fullRotation.configuration.configImage,
            s->cfg.maxNumUndecodedSamples,
            rb)) {
        SetError(s, "GpuPeriodicSamplerCalculateRecordBufferSize failed for full rotation mode");
        return false;
    }
    if (rb == 0) {
        SetError(s, "recordBufferSize is zero for full rotation mode");
        return false;
    }
    s->perf.recordBufferSize = rb;

    const std::vector<NVPW_GPU_PeriodicSampler_TriggerSource> triggers = {
        NVPW_GPU_PERIODIC_SAMPLER_TRIGGER_SOURCE_CPU_SYSCALL,
    };

    if (!s->perf.sampler.BeginSession(
            s->perf.recordBufferSize,
            1,
            triggers,
            /*samplingInterval=*/0,
            NVPW_GPU_PERIODIC_SAMPLER_RECORD_BUFFER_APPEND_MODE_KEEP_LATEST)) {
        if (!s->perf.sampler.BeginSession(
                s->perf.recordBufferSize,
                1,
                triggers,
                /*samplingInterval=*/0,
                NVPW_GPU_PERIODIC_SAMPLER_RECORD_BUFFER_APPEND_MODE_KEEP_OLDEST)) {
            SetError(s, "GpuPeriodicSampler::BeginSession failed for full rotation mode");
            return false;
        }
    }

    s->perf.inSession = true;
    s->perf.active = false;
    s->perf.fullRotation.activePass = s->perf.fullRotation.numPasses > 0
        ? ((size_t)s->cfg.startPass) % s->perf.fullRotation.numPasses
        : 0;

    Logf(s, "PerfWorks sampling started (full rotation passes=%zu)", s->perf.fullRotation.numPasses);
    return true;
}

static bool StartFullRotationPass(State* s, size_t passIndex) {
    if (!s->perf.inSession) {
        SetError(s, "StartFullRotationPass called without active session");
        return false;
    }

    using clock = std::chrono::steady_clock;
    const bool trace = TraceLogsEnabled();
    auto t0 = clock::now();

    if (!s->perf.sampler.SetConfig(s->perf.fullRotation.configuration.configImage, passIndex)) {
        SetError(s, "GpuPeriodicSampler::SetConfig failed for full rotation pass");
        return false;
    }
    auto t1 = clock::now();

    if (!s->perf.sampler.StartSampling()) {
        SetError(s, "GpuPeriodicSampler::StartSampling failed for full rotation pass");
        return false;
    }
    auto t2 = clock::now();

    s->perf.fullRotation.activePass = passIndex;
    s->perf.active = true;

    if (trace) {
        const auto setUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        const auto startUs = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        Logf(s, "TraceStartFullPass: pass=%zu set=%lldus start=%lldus",
             passIndex,
             (long long)setUs,
             (long long)startUs);
    }

    return true;
}

static bool StopFullRotationPass(State* s) {
    if (!s->perf.active) return true;

    using clock = std::chrono::steady_clock;
    const bool trace = TraceLogsEnabled();
    auto t0 = clock::now();

    if (!s->perf.sampler.StopSampling()) {
        SetError(s, "GpuPeriodicSampler::StopSampling failed for full rotation pass");
        return false;
    }

    auto t1 = clock::now();
    s->perf.active = false;

    if (trace) {
        const auto stopUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        Logf(s, "TraceStopFullPass: pass=%zu stop=%lldus",
             s->perf.fullRotation.activePass,
             (long long)stopUs);
    }

    return true;
}

static void EmitMetricSnapshot(State* s,
                               const std::vector<double>& metricMeans,
                               uint64_t pmSamples,
                               int groupId,
                               int groupCount) {
    WindowRecord rec;
    rec.pmSamples = pmSamples;
    rec.groupId = groupId;
    rec.groupCount = groupCount;
    rec.metricMeans = metricMeans;

    {
        std::lock_guard<std::mutex> lock(s->ringMutex);
        s->ring.push_back(std::move(rec));
        while (s->ring.size() > s->maxRingRecords) {
            s->ring.pop_front();
        }
    }
}

static bool ReadCompletedFullRotationSamples(State* s,
                                             std::vector<FullRotationSample>& outSamples,
                                             uint64_t& outLatestValidTsEnd,
                                             size_t& outValidSamples) {
    outSamples.clear();
    outLatestValidTsEnd = 0;
    outValidSamples = 0;

    CounterDataInfo info{};
    if (!CounterDataGetInfo(
            s->perf.fullRotation.counterDataImage.data(),
            s->perf.fullRotation.counterDataImage.size(),
            info)) {
        SetError(s, "CounterDataGetInfo failed for full rotation data");
        return false;
    }

    outSamples.reserve(info.numCompletedRanges);
    for (uint32_t rangeIndex = 0; rangeIndex < info.numCompletedRanges; ++rangeIndex) {
        FullRotationSample sample;
        SampleTimestamp ts{};
        if (!CounterDataGetSampleTime(s->perf.fullRotation.counterDataImage.data(), rangeIndex, ts)) {
            SetError(s, "CounterDataGetSampleTime failed for full rotation data");
            return false;
        }
        sample.tsStart = ts.start;
        sample.tsEnd = ts.end;
        sample.values.assign(s->perf.globalMetricNames.size(), std::numeric_limits<double>::quiet_NaN());

        if (!s->perf.metricsEvaluator.EvaluateToGpuValues(
                s->perf.fullRotation.counterDataImage.data(),
                s->perf.fullRotation.counterDataImage.size(),
                rangeIndex,
                s->perf.fullRotation.metricEvalRequests.size(),
                s->perf.fullRotation.metricEvalRequests.data(),
                sample.values.data())) {
            SetError(s, "EvaluateToGpuValues failed for full rotation data");
            return false;
        }

        const bool validTs = (sample.tsStart != 0 && sample.tsEnd != 0 && sample.tsEnd > sample.tsStart);
        if (validTs) {
            outValidSamples++;
            outLatestValidTsEnd = std::max(outLatestValidTsEnd, sample.tsEnd);
        }

        outSamples.push_back(std::move(sample));
    }

    return true;
}

static bool DecodeFullRotationLatest(State* s,
                                     size_t& outNumUnreadBytes,
                                     size_t& outNumBytesConsumed,
                                     size_t& outNumSamplesMerged,
                                     uint32_t& outStopReason,
                                     CounterDataInfo& outInfo) {
    outNumUnreadBytes = 0;
    outNumBytesConsumed = 0;
    outNumSamplesMerged = 0;
    outStopReason = 0;
    outInfo = CounterDataInfo{};

    GpuPeriodicSampler::GetRecordBufferStatusParams status{};
    status.queryNumUnreadBytes = true;
    status.queryOverflow = true;
    status.queryWriteOffset = false;
    status.queryReadOffset = false;

    if (!s->perf.sampler.GetRecordBufferStatus(status)) {
        SetError(s, "GetRecordBufferStatus failed for full rotation mode");
        return false;
    }
    if (status.overflow) {
        SetError(s, "PerfWorks record buffer overflow in full rotation mode");
        return false;
    }

    outNumUnreadBytes = status.numUnreadBytes;
    if (status.numUnreadBytes > 0) {
        NVPW_GPU_PeriodicSampler_DecodeStopReason stopReason = NVPW_GPU_PERIODIC_SAMPLER_DECODE_STOP_REASON_OTHER;
        if (!s->perf.sampler.DecodeCounters(
                s->perf.fullRotation.counterDataImage,
                status.numUnreadBytes,
                stopReason,
                outNumSamplesMerged,
                outNumBytesConsumed)) {
            SetError(s, "DecodeCounters failed for full rotation mode");
            return false;
        }
        outStopReason = (uint32_t)stopReason;

        if (outNumBytesConsumed > 0) {
            if (!s->perf.sampler.AcknowledgeRecordBuffer(outNumBytesConsumed)) {
                SetError(s, "AcknowledgeRecordBuffer failed for full rotation mode");
                return false;
            }
        }
    }

    if (!CounterDataGetInfo(
            s->perf.fullRotation.counterDataImage.data(),
            s->perf.fullRotation.counterDataImage.size(),
            outInfo)) {
        SetError(s, "CounterDataGetInfo failed after full rotation decode");
        return false;
    }

    return true;
}

static bool MaybePublishFullRotationSnapshot(State* s) {
    std::vector<FullRotationSample> completedSamples;
    uint64_t latestValidTsEnd = 0;
    size_t validSamples = 0;
    if (!ReadCompletedFullRotationSamples(s, completedSamples, latestValidTsEnd, validSamples)) {
        return false;
    }

    if (latestValidTsEnd == 0 || validSamples == 0) {
        return true;
    }

    if (latestValidTsEnd == s->perf.fullRotation.lastPublishedTsEnd) {
        return true;
    }

    if (s->perf.fullRotation.warmupRotationsRemaining > 0) {
        s->perf.fullRotation.lastPublishedTsEnd = latestValidTsEnd;
        s->perf.fullRotation.warmupRotationsRemaining--;
        Logf(s, "FullRotationWarmupSkip: latestValidTsEnd=%llu remaining=%d",
             (unsigned long long)latestValidTsEnd,
             s->perf.fullRotation.warmupRotationsRemaining);
        return true;
    }

    const FullRotationSample* latestValidSample = nullptr;
    for (const auto& sample : completedSamples) {
        if (!(sample.tsStart != 0 && sample.tsEnd != 0 && sample.tsEnd > sample.tsStart)) continue;
        if (!latestValidSample || sample.tsEnd > latestValidSample->tsEnd) {
            latestValidSample = &sample;
        }
    }

    if (!latestValidSample) {
        return true;
    }

    EmitMetricSnapshot(
        s,
        latestValidSample->values,
        1,
        (int)s->perf.fullRotation.activePass,
        (int)std::max((size_t)1, s->perf.fullRotation.numPasses));

    s->perf.fullRotation.lastPublishedTsEnd = latestValidTsEnd;
    Logf(s, "FullRotationPublish: pass=%zu latestValidTsEnd=%llu validSamples=%zu mode=latest-valid-only",
         s->perf.fullRotation.activePass,
         (unsigned long long)latestValidTsEnd,
         validSamples);
    return true;
}

static void WorkerLoopFullRotation(State* s) {
    using clock = std::chrono::steady_clock;
    const auto triggerSpacing = std::chrono::milliseconds(std::max(1, s->cfg.triggerSpacingMs));
    const auto rotationPeriod = triggerSpacing * std::max(1, s->cfg.triggersPerPass) * (int)std::max((size_t)1, s->perf.fullRotation.numPasses);
    const auto initialDelay = std::chrono::milliseconds(
        std::max(0LL, (long long)((std::max(0, s->cfg.startPass) * std::max(1, s->cfg.triggerSpacingMs)) % std::max(1LL, (long long)rotationPeriod.count()))));

    if (initialDelay.count() > 0) {
        std::this_thread::sleep_for(initialDelay);
    }

    const size_t startPass = s->perf.fullRotation.numPasses > 0
        ? ((size_t)s->cfg.startPass) % s->perf.fullRotation.numPasses
        : 0;

    while (!s->stopWorker.load(std::memory_order_relaxed)) {
        // Reset counter data image to pristine state before each rotation cycle.
        // Without this, PerfWorks accumulates raw counters across rotations, causing
        // pct_of_peak_sustained_elapsed to show a 1/t decay instead of current values.
        std::memcpy(s->perf.fullRotation.counterDataImage.data(),
                     s->perf.fullRotation.pristineCounterDataImage.data(),
                     s->perf.fullRotation.pristineCounterDataImage.size());

        bool ok = true;
        for (size_t passOffset = 0; passOffset < s->perf.fullRotation.numPasses && !s->stopWorker.load(std::memory_order_relaxed); ++passOffset) {
            const size_t passIndex = (startPass + passOffset) % s->perf.fullRotation.numPasses;
            auto iterStart = clock::now();

            if (!StartFullRotationPass(s, passIndex)) {
                ok = false;
                break;
            }

            for (int ti = 0; ti < s->cfg.triggersPerPass && !s->stopWorker.load(std::memory_order_relaxed); ++ti) {
                std::this_thread::sleep_for(triggerSpacing);
                if (!s->perf.sampler.CpuTrigger()) {
                    SetError(s, "GpuPeriodicSampler::CpuTrigger failed for full rotation mode");
                    ok = false;
                    break;
                }
            }
            if (s->perf.active) {
                ok = StopFullRotationPass(s) && ok;
            }
            if (!ok) break;

            size_t numUnreadBytes = 0;
            size_t numBytesConsumed = 0;
            size_t numSamplesMerged = 0;
            uint32_t stopReason = 0;
            CounterDataInfo info{};
            if (!DecodeFullRotationLatest(s, numUnreadBytes, numBytesConsumed, numSamplesMerged, stopReason, info)) {
                ok = false;
                break;
            }

            if (s->cfg.debug) {
                auto iterEnd = clock::now();
                auto iterUs = std::chrono::duration_cast<std::chrono::microseconds>(iterEnd - iterStart).count();
                Logf(s,
                     "FullRotationIter: pass=%zu unreadBytes=%zu bytesAck=%zu merged=%zu completed=%u iter_total=%lldus",
                     passIndex,
                     numUnreadBytes,
                     numBytesConsumed,
                     numSamplesMerged,
                     info.numCompletedRanges,
                     (long long)iterUs);
            }
        }

        if (!ok) break;
        if (s->stopWorker.load(std::memory_order_relaxed)) break;
        if (!MaybePublishFullRotationSnapshot(s)) break;
    }

    if (s->perf.active) {
        (void)StopFullRotationPass(s);
    }
    (void)MaybePublishFullRotationSnapshot(s);
}

bool Setup(State* s) {
    if (!GlobalInit(s)) return false;
    s->globalInitAcquired = true;

    CUdevice dev = 0;
    if (!CudaOk(s, cuDeviceGet(&dev, s->cfg.deviceIndex), "cuDeviceGet")) {
        return false;
    }
    (void)dev;

    if (!s->perf.sampler.Initialize((size_t)s->cfg.deviceIndex)) {
        SetError(s, "GpuPeriodicSampler::Initialize failed for device " + std::to_string(s->cfg.deviceIndex));
        return false;
    }

    s->perf.deviceIds = s->perf.sampler.GetDeviceIdentifiers();
    if (!s->perf.deviceIds.pChipName || !*s->perf.deviceIds.pChipName) {
        SetError(s, "failed to get chip name for device " + std::to_string(s->cfg.deviceIndex));
        return false;
    }

    {
        std::vector<uint8_t> scratch;
        NVPW_MetricsEvaluator* pEvaluator = DeviceCreateMetricsEvaluator(scratch, s->perf.deviceIds.pChipName);
        if (!pEvaluator) {
            SetError(s, "DeviceCreateMetricsEvaluator failed");
            return false;
        }
        s->perf.metricsEvaluator = MetricsEvaluator(pEvaluator, std::move(scratch));
    }

    if (!BuildFullRotationConfig(s)) return false;
    if (!BeginSessionFullRotation(s)) return false;

    s->stopWorker.store(false, std::memory_order_relaxed);
    s->worker = std::thread(WorkerLoopFullRotation, s);
    s->workerStarted = true;

    Logf(s, "sampler active (PerfWorks)");
    return true;
}

void Cleanup(State* s) {
    if (!s) return;

    s->stopWorker.store(true, std::memory_order_relaxed);
    if (s->workerStarted && s->worker.joinable()) {
        s->worker.join();
        s->workerStarted = false;
    }

    if (s->perf.inSession) {
        (void)StopFullRotationPass(s);
        (void)s->perf.sampler.EndSession();
        s->perf.inSession = false;
    }

    s->perf.sampler.Reset();
    s->perf.metricsEvaluator.Reset();
    s->perf.fullRotation.counterDataImage.clear();
    s->perf.fullRotation.pristineCounterDataImage.clear();
    s->perf.fullRotation.metricEvalRequests.clear();
    s->perf.fullRotation.configuration = CounterConfiguration{};
    s->perf.fullRotation.numPasses = 0;
    s->perf.fullRotation.activePass = 0;
    s->perf.fullRotation.lastPublishedTsEnd = 0;
    s->perf.fullRotation.warmupRotationsRemaining = 0;

    {
        std::lock_guard<std::mutex> lock(s->ringMutex);
        s->ring.clear();
    }

    s->initialized = false;
    if (s->globalInitAcquired) {
        GlobalRelease();
        s->globalInitAcquired = false;
    }
}

} // namespace utlz_sampler_internal
