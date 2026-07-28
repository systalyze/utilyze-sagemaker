package sampler

import (
	"unsafe"
)

type samplerHandle unsafe.Pointer

const (
	sym_utlzSamplerCreate         = "utlz_sampler_create"
	sym_utlzSamplerIsInitialized  = "utlz_sampler_is_initialized"
	sym_utlzSamplerGetMetricCount = "utlz_sampler_get_metric_count"
	sym_utlzSamplerGetMetricName  = "utlz_sampler_get_metric_name"
	sym_utlzSamplerGetError       = "utlz_sampler_get_error"
	sym_utlzSamplerPoll           = "utlz_sampler_poll"
	sym_utlzSamplerDestroy        = "utlz_sampler_destroy"
)

var (
	utlzSamplerCreate         func(deviceIndices []int32, numDevices int32, metricsCsv string, intervalMs int32) samplerHandle
	utlzSamplerIsInitialized  func(handle samplerHandle, deviceIndex int32) int32
	utlzSamplerGetMetricCount func(handle samplerHandle) int32
	utlzSamplerGetMetricName  func(handle samplerHandle, index int32) string
	utlzSamplerGetError       func(handle samplerHandle, deviceIndex int32) string
	utlzSamplerPoll           func(handle samplerHandle, deviceIndex int32, outValues []float64, maxMetrics int32, outPmSamples *int32, outGroupId *int32, outGroupCount *int32) int32
	utlzSamplerDestroy        func(handle samplerHandle)
)
