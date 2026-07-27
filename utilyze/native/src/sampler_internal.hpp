#pragma once

#include <cuda.h>

#include "NvPerfCounterConfiguration.h"
#include "NvPerfCounterData.h"
#include "NvPerfDeviceProperties.h"
#include "NvPerfInit.h"
#include "NvPerfMetricsConfigBuilder.h"
#include "NvPerfMetricsEvaluator.h"
#include "NvPerfPeriodicSamplerCommon.h"
#include "NvPerfPeriodicSamplerGpu.h"
#include "nvperf_common.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace utlz_sampler_internal {

using namespace nv::perf;
using namespace nv::perf::sampler;

struct Config {
    int deviceIndex = 0;
    int intervalMs = 200;
    int startPass = 0;
    bool debug = false;

    int triggersPerPass = 2;
    int triggerSpacingMs = 120;
    int publishWarmupRotations = 1;

    uint32_t counterDataMaxSamples = 1024;
    size_t maxNumUndecodedSamples = 32768;

    std::vector<std::string> requestedMetricNames;
};

struct FullRotationSample {
    uint64_t tsStart = 0;
    uint64_t tsEnd = 0;
    std::vector<double> values;
};

struct FullRotationState {
    CounterConfiguration configuration;
    std::vector<NVPW_MetricEvalRequest> metricEvalRequests;
    std::vector<uint8_t> counterDataImage;
    std::vector<uint8_t> pristineCounterDataImage; // fresh copy for resetting each rotation

    size_t numPasses = 0;
    size_t activePass = 0;
    uint64_t lastPublishedTsEnd = 0;
    int warmupRotationsRemaining = 0;
};

struct PerfState {
    bool active = false;
    bool inSession = false;

    DeviceIdentifiers deviceIds{};
    GpuPeriodicSampler sampler;
    MetricsEvaluator metricsEvaluator;

    std::vector<std::string> globalMetricNames;
    FullRotationState fullRotation;

    GpuPeriodicSampler::GpuPulseSamplingInterval samplingInterval{};
    size_t recordBufferSize = 0;
};

struct WindowRecord {
    uint64_t pmSamples = 0;
    int groupId = 0;
    int groupCount = 0;
    std::vector<double> metricMeans;
};

struct State {
    Config cfg;

    bool initialized = false;
    bool globalInitAcquired = false;
    std::mutex errorMutex;
    std::string lastError;

    PerfState perf;

    std::mutex ringMutex;
    std::deque<WindowRecord> ring;
    size_t maxRingRecords = 4096;

    std::thread worker;
    std::atomic<bool> stopWorker{false};
    bool workerStarted = false;
};

const char* GetEnv(const char* name);
bool DebugLogsEnabled();
void SetError(State* s, const std::string& msg);
std::vector<std::string> SplitCsv(const std::string& csv);
int ReadEnvInt(const char* name, int fallback, int minVal, int maxVal);
bool EqualsIgnoreCase(const char* a, const char* b);
bool Setup(State* s);
void Cleanup(State* s);

void* state_create(int device_index, const char* metrics_csv,
                   int interval_ms, int start_pass);
int state_is_initialized(void* handle);
int state_metric_count(void* handle);
const char* state_metric_name(void* handle, int index);
const char* state_error(void* handle);
int state_poll(void* handle, double* out_values, int max_metrics,
               int* out_pm_samples, int* out_group_id, int* out_group_count);
void state_destroy(void* handle);

void* multi_create(const int* device_indices, int num_devices,
                   const char* metrics_csv, int interval_ms);
int multi_is_initialized(void* handle, int device_index);
int multi_get_metric_count(void* handle);
const char* multi_get_metric_name(void* handle, int index);
const char* multi_get_error(void* handle, int device_index);
int multi_poll(void* handle, int device_index,
               double* out_values, int max_metrics,
               int* out_pm_samples, int* out_group_id, int* out_group_count);
void multi_destroy(void* handle);

} // namespace utlz_sampler_internal
