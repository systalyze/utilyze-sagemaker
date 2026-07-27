package metrics

type MetricsPayload struct {
	SchemaVersion int               `json:"schema_version"`
	HostID        string            `json:"host_id,omitempty"`
	ClientIDs     []string          `json:"client_ids,omitempty"`
	SampledAtMs   int64             `json:"sampled_at_ms"`
	Mode          string            `json:"mode"`
	GpuCount      int               `json:"gpu_count"`
	GPUs          []MetricsGpu      `json:"gpus"`
	Workloads     []MetricsWorkload `json:"workloads,omitempty"`
}

type MetricsWorkload struct {
	ModelName       string           `json:"model_name"`
	WindowMs        int64            `json:"window_ms"`
	IslMean         float64          `json:"isl_mean"`
	IslHistogram    MetricsHistogram `json:"isl_histogram"`
	OslMean         float64          `json:"osl_mean"`
	OslHistogram    MetricsHistogram `json:"osl_histogram"`
	ConcurrencyMean float64          `json:"concurrency_mean"`
	ConcurrencyMax  int              `json:"concurrency_max"`
	EngineCount     int              `json:"engine_count"`
}

type MetricsHistogram struct {
	Count int                   `json:"count"`
	Bins  []MetricsHistogramBin `json:"bins"`
}

type MetricsHistogramBin struct {
	Lower int `json:"lower"`
	Upper int `json:"upper"`
	Count int `json:"count"`
}

type MetricsGpu struct {
	Index      int     `json:"index"`
	GpuID      string  `json:"gpu_id"`
	GpuModel   string  `json:"gpu_model"`
	ModelName  *string `json:"model_name,omitempty"`
	ComputePct float64 `json:"compute_pct"`
	MemoryPct  float64 `json:"memory_pct"`
	PcieGBs    float64 `json:"pcie_gbs"`
	NvlinkGBs  float64 `json:"nvlink_gbs"`
}

type MetricsResponse struct {
	GpuCeilings []GpuCeilingResponse `json:"gpu_ceilings"`
}

type GpuCeilingResponse struct {
	Index             int      `json:"index"`
	ModelName         *string  `json:"model_name,omitempty"`
	ComputeSolCeiling *float64 `json:"compute_sol_ceiling"`
}
