#include "sampler_internal.hpp"

namespace {

struct MultiState {
    std::vector<void*> handles;
    std::vector<int> gpuDeviceIds;

    int metricCount = 0;
    std::vector<std::string> metricNames;
    std::string fallbackError;

    void* findHandle(int deviceIndex) const {
        for (size_t i = 0; i < gpuDeviceIds.size(); ++i) {
            if (gpuDeviceIds[i] == deviceIndex) return handles[i];
        }
        return nullptr;
    }
};

} // namespace

namespace utlz_sampler_internal {

void* multi_create(const int* device_indices, int num_devices,
                   const char* metrics_csv, int interval_ms) {
    if (!device_indices || num_devices <= 0) return nullptr;

    auto* ms = new MultiState();
    ms->handles.resize((size_t)num_devices, nullptr);
    ms->gpuDeviceIds.assign(device_indices, device_indices + num_devices);

    for (int i = 0; i < num_devices; ++i) {
        ms->handles[(size_t)i] = state_create(
            device_indices[i], metrics_csv, interval_ms,
            /*start_pass=*/i);
    }

    for (int i = 0; i < num_devices; ++i) {
        if (state_is_initialized(ms->handles[(size_t)i])) {
            ms->metricCount = state_metric_count(ms->handles[(size_t)i]);
            ms->metricNames.reserve((size_t)ms->metricCount);
            for (int m = 0; m < ms->metricCount; ++m) {
                ms->metricNames.emplace_back(state_metric_name(ms->handles[(size_t)i], m));
            }
            break;
        }
    }

    if (ms->metricCount == 0) {
        ms->fallbackError = "no GPUs initialized successfully";
        if (num_devices > 0 && ms->handles[0]) {
            const char* e = state_error(ms->handles[0]);
            if (e && *e) ms->fallbackError = e;
        }
    }

    return ms;
}

int multi_is_initialized(void* handle, int device_index) {
    if (!handle) return 0;
    auto* ms = (MultiState*)handle;
    void* h = ms->findHandle(device_index);
    return h ? state_is_initialized(h) : 0;
}

int multi_get_metric_count(void* handle) {
    if (!handle) return 0;
    return ((MultiState*)handle)->metricCount;
}

const char* multi_get_metric_name(void* handle, int index) {
    static const char* empty = "";
    if (!handle) return empty;
    auto* ms = (MultiState*)handle;
    if (index < 0 || (size_t)index >= ms->metricNames.size()) return empty;
    return ms->metricNames[(size_t)index].c_str();
}

const char* multi_get_error(void* handle, int device_index) {
    static const char* none = "";
    if (!handle) return none;
    auto* ms = (MultiState*)handle;
    void* h = ms->findHandle(device_index);
    if (h) return state_error(h);
    return ms->fallbackError.c_str();
}

int multi_poll(void* handle, int device_index,
               double* out_values, int max_metrics,
               int* out_pm_samples, int* out_group_id, int* out_group_count) {
    if (!handle) return -1;
    auto* ms = (MultiState*)handle;
    void* h = ms->findHandle(device_index);
    if (!h) return -1;
    return state_poll(h, out_values, max_metrics, out_pm_samples, out_group_id, out_group_count);
}

void multi_destroy(void* handle) {
    if (!handle) return;
    auto* ms = (MultiState*)handle;
    for (auto* h : ms->handles) {
        if (h) state_destroy(h);
    }
    delete ms;
}

} // namespace utlz_sampler_internal
