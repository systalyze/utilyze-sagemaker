# utilyze-sagemaker

Scripts to reproduce the SOL headroom results from the Utilyze report:
five workloads, Gemma-4-31B FP4, vLLM 0.21.0, 500 requests per workload.
Each run measures throughput and SOL with utlz and records a live
Attainable-SOL ceiling per workload.

## Requirements

- 5 free NVIDIA GPUs (A100, H100, or H200; with fewer, run workloads one at
  a time with `run_workload.py`)
- Docker with the NVIDIA container runtime
- Python 3.10+ with `pip install websockets`
- utlz built from `utilyze/` (see below)
- `nvidia/Gemma-4-31B-IT-NVFP4` at `$MODELS_DIR/Gemma-4-31B-IT-NVFP4-bf16kv`
  - for A100: delete `quantization_config.kv_cache_scheme` from
    `config.json` and `quantization.kv_cache_quant_algo` from
    `hf_quant_config.json` (FP8 KV is unsupported there; without the keys
    vLLM selects bf16 KV)

## Run

```sh
cd scripts && MODELS_DIR=~/models UTLZ_BIN=/path/to/utlz ./run_all.sh
cat results/summary.csv
```

Compare `compute_sol` and `attainable_live` per workload against
`reference/fig1_reference.json`.

## Scripts

- `run_workload.py` one workload: vLLM server in Docker, utlz SOL capture,
  `vllm bench serve` load generation; the report's server flags live here
- `run_all.sh` the five workloads in parallel, one GPU and one utlz each
- `analyze.py` throughput, steady-window SOL, and live ceiling per workload
- `throughput.py` measured vs attainable tok/s per workload, converting the
  live ceiling from utilization
- `utlz_collector.py` internal, records the utlz feed

## Building utlz

Needs Go 1.25+, gcc/g++ with C++17, and a CUDA toolkit with CUPTI under
`/usr/local/cuda-<ver>` (the Nsight Perf SDK is vendored).

```sh
cd utilyze
make native CUDA_VERSION=12.4   # match your /usr/local/cuda-<ver>
cp dist/libutlz_sampler.so.0.* internal/ffi/sampler/embedded/libutlz_sampler.so.0
go build -o utlz ./cmd
```

Run with `sudo` (GPU counters need CAP_SYS_ADMIN). `sudo ./utlz` is the
interactive TUI; the harness uses server mode, which `run_all.sh` starts
itself. Point `UTLZ_BACKEND_URL` at an Attainable-SOL backend to get live
ceilings; without it utlz is a pure SOL profiler.
