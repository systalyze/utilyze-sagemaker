#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void* sampler_handle_t;

sampler_handle_t utlz_sampler_create(const int* device_indices, int num_devices,
                                      const char* metrics_csv, int interval_ms);
int utlz_sampler_is_initialized(sampler_handle_t handle, int device_index);
int utlz_sampler_get_metric_count(sampler_handle_t handle);
const char* utlz_sampler_get_metric_name(sampler_handle_t handle, int index);
const char* utlz_sampler_get_error(sampler_handle_t handle, int device_index);
int utlz_sampler_poll(sampler_handle_t handle, int device_index,
                       double* out_values, int max_metrics,
                       int* out_pm_samples, int* out_group_id, int* out_group_count);
void utlz_sampler_destroy(sampler_handle_t handle);

#ifdef __cplusplus
}
#endif
