#include "utlz_sampler.h"
#include "sampler_internal.hpp"

extern "C" {

sampler_handle_t utlz_sampler_create(const int* device_indices, int num_devices,
                                      const char* metrics_csv, int interval_ms) {
    return utlz_sampler_internal::multi_create(device_indices, num_devices, metrics_csv, interval_ms);
}

int utlz_sampler_is_initialized(sampler_handle_t handle, int device_index) {
    return utlz_sampler_internal::multi_is_initialized(handle, device_index);
}

int utlz_sampler_get_metric_count(sampler_handle_t handle) {
    return utlz_sampler_internal::multi_get_metric_count(handle);
}

const char* utlz_sampler_get_metric_name(sampler_handle_t handle, int index) {
    return utlz_sampler_internal::multi_get_metric_name(handle, index);
}

const char* utlz_sampler_get_error(sampler_handle_t handle, int device_index) {
    return utlz_sampler_internal::multi_get_error(handle, device_index);
}

int utlz_sampler_poll(sampler_handle_t handle, int device_index,
                       double* out_values, int max_metrics,
                       int* out_pm_samples, int* out_group_id, int* out_group_count) {
    return utlz_sampler_internal::multi_poll(handle, device_index, out_values, max_metrics,
                                            out_pm_samples, out_group_id, out_group_count);
}

void utlz_sampler_destroy(sampler_handle_t handle) {
    utlz_sampler_internal::multi_destroy(handle);
}

} // extern "C"
