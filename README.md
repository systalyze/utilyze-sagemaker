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

## Sweep pruning reproduction (Attainable SOL)

`scripts/sweep/sweep.py` reproduces the sweep-economics result: a tuning sweep
of 200 sampled server configs x a 10-step concurrency ladder, where Attainable
SOL stops each config's ladder once measured compute-SOL reaches 90% of the
attainable ceiling.

```bash
# GPU-free harness check (CI)
python3 scripts/sweep/test_dry.py

# the paper run: ~3.5 days wall on 4x H200 (~330 GPU-hours, ~$2.6k of GPU
# time at on-demand p5en pricing; a full 8-GPU instance bills ~2x that)
sudo $UTLZ_BIN -mode server -devices 0,1,2,3 -port 9080 &   # needs UTLZ_BACKEND_URL
python3 scripts/sweep/sweep.py --configs 200 --mode replay --rule auto \
    --gpus 0,1,2,3 --utlz-url ws://127.0.0.1:9080/live \
    --model-nvfp4 /path/to/Gemma-4-31B-IT-NVFP4-bf16kv \
    --model-bf16 /path/to/gemma-4-31b-it
```

- `--mode replay` measures the full ladder and evaluates the stop rule offline:
  one run proves both the savings and that pruning leaves the winner unchanged
  (`summary.json`: `saving_pct`, `winner_unchanged`, `winner_regret_pct`).
- `--mode live` actually stops each ladder when the rule fires (the production
  behavior; its cost should match replay's prediction).
- `--rule auto` (default) uses the Attainable-SOL rule per config where ceiling
  data exists and falls back to a 10% marginal-gain plateau rule where it does
  not; `configs.csv:rule_used` and `summary.json:rules_used` record which rule
  made each stop decision. If no config ever gets usable ceilings the run
  completes but prints a warning: that means backend coverage or attribution is
  missing, not that pruning saves nothing.
- Before the grid, a preflight boots one server per risky feature (spec decode,
  fp8 KV, BF16 weights) and drops configs using a feature that cannot start,
  with counts in `summary.json:configs_dropped_preflight`
  (`--skip-preflight` disables this).
- Both checkpoints must be downloaded first; `google/gemma-4-31b-it` is a gated
  HuggingFace repo (accept the license, use an authenticated download).
  Budget ~120 GB of disk for the pair.
- Outputs are rewritten after every config, so an interrupted run keeps
  everything measured so far.
- The sweep scripts pin `vllm/vllm-openai:v0.23.0` (the version the deck's
  numbers were measured on); the fig-1 kit above pins v0.21.0 to match the
  original report. Override with `VLLM_IMAGE`.
- EAGLE-3 combos are excluded from the grid: no trained draft head exists for
  Gemma-4, which is why the space is 2,304 valid configs rather than 3,072.

Outputs land in `scripts/sweep/results-sweep/<run>/`: `sweep.csv` (one row per
measurement, with SOL, ceiling, and pruned flags), `configs.csv` (per-config
stop point and timing), `summary.json` (cost with and without pruning).
