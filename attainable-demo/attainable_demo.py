#!/usr/bin/env python3
"""Query the Utilyze Attainable-SOL endpoint for a spread of real models,
GPUs, sequence lengths and concurrencies.

    python3 attainable_demo.py
"""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UTLZ_URL = "https://api.systalyze.com/v1/utilyze"

H200 = "NVIDIA H200"
H100 = "NVIDIA H100 80GB HBM3"
A100 = "NVIDIA A100-SXM4-80GB"

# Short labels for display; the full NVML product name is what gets sent.
GPU_LABEL = {H200: "H200", H100: "H100", A100: "A100"}

# (model, gpu, gpu_count, input_len, output_len, concurrency)
EXAMPLES = [
    ("Qwen/Qwen3.5-4B",                  A100, 1,   128, 1024,  32),
    ("Qwen/Qwen3.5-9B",                  H100, 1,   256,   64, 128),
    ("Qwen/Qwen3.5-9B",                  H200, 1,  1024,  256,  64),
    ("Qwen/Qwen3.5-9B",                  H200, 1, 16384, 1024,   4),
    ("google/gemma-4-12B-it",            H200, 1,  1024,  256,  64),
    ("google/gemma-4-12B-it",            H100, 1,  2048,  512,  32),
    ("google/gemma-4-26B-A4B-it",        H200, 2,  2048,  512,  32),
    ("google/gemma-4-26B-A4B-it",        H200, 2,  1024,  256,  64),
    ("Qwen/Qwen3.5-35B-A3B",             H200, 2,   512,  128, 128),
    ("Qwen/Qwen3.5-35B-A3B",             H200, 2,  1024,  256,  64),
    ("Qwen/Qwen3.5-35B-A3B",             A100, 2,  4096,  256,  32),
    ("google/gemma-4-31b-it",            H200, 1,  1024,  256,  64),
    ("google/gemma-4-31b-it",            H100, 2,  8192,  512,  16),
    ("Qwen/Qwen3.5-122B-A10B",           H200, 4,  1024,  256,  64),
    ("Qwen/Qwen3.5-122B-A10B",           H200, 4, 16384, 1024,  16),
    ("Qwen/Qwen3.5-397B-A17B",           H200, 8,   512,  128, 128),
    ("Qwen/Qwen3.5-397B-A17B",           H200, 8,  1024,  256,  64),
    ("openai/gpt-oss-20b",               H200, 1,  1024,  256,  64),
    ("openai/gpt-oss-20b",               H100, 1,  8192,  512,  16),
    ("openai/gpt-oss-120b",              H200, 4,  1024,  256,  64),
    ("openai/gpt-oss-120b",              H200, 4, 16384, 1024,  16),
    ("meta-llama/Llama-3.1-8B-Instruct", H200, 1,  2048,  512,  32),
    ("microsoft/phi-4",                  H200, 1,  1024,  256,  32),
    ("deepseek-ai/DeepSeek-R1",          H200, 8,   512,  128, 128),
]


def build_payload(model, gpu, gpu_count, isl, osl, concurrency):
    """One utlz telemetry tick (schema_version 2).

    The observed per-GPU counters are sent as zero: the attainable ceiling is a
    property of the deployment and the traffic shape, not of current
    utilisation.
    """
    return {
        "schema_version": 2,
        "host_id": "attainable-demo",
        "sampled_at_ms": int(time.time() * 1000),
        "mode": "native",
        "gpu_count": gpu_count,
        "gpus": [
            {
                "index": i,
                "gpu_id": f"GPU-demo-{i:04d}",
                "gpu_model": gpu,
                "model_name": model,
                "compute_pct": 0.0,
                "memory_pct": 0.0,
                "pcie_gbs": 0.0,
                "nvlink_gbs": 0.0,
            }
            for i in range(gpu_count)
        ],
        "workloads": [
            {
                "model_name": model,
                "window_ms": 5000,
                "isl_mean": float(isl),
                "isl_histogram": {"count": 500, "bins": [
                    {"lower": isl // 2, "upper": isl * 3 // 2, "count": 500}]},
                "osl_mean": float(osl),
                "osl_histogram": {"count": 500, "bins": [
                    {"lower": osl // 2, "upper": osl * 3 // 2, "count": 500}]},
                "concurrency_mean": float(concurrency),
                "concurrency_max": concurrency,
                "engine_count": 1,
            }
        ],
    }


def attainable_sol(payload, timeout=45):
    """POST a tick, return the attainable compute-SOL ceiling (or None)."""
    request = urllib.request.Request(
        f"{UTLZ_URL}/metrics",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        ceilings = json.loads(response.read().decode()).get("gpu_ceilings", [])
    return ceilings[0]["compute_sol_ceiling"] if ceilings else None


def run(example):
    model, gpu, gpu_count, isl, osl, concurrency = example
    try:
        ceiling = attainable_sol(build_payload(*example))
    except Exception as exc:
        ceiling = None
        print(f"  ! {model}: {type(exc).__name__}: {exc}")
    return {
        "model": model,
        "gpu": GPU_LABEL.get(gpu, gpu.replace("NVIDIA ", "")),
        "gpus": gpu_count,
        "isl": isl,
        "osl": osl,
        "concurrency": concurrency,
        "attainable_sol": None if ceiling is None else round(ceiling, 2),
    }


def main():
    print(f"endpoint: {UTLZ_URL}\n")
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(run, EXAMPLES))

    hdr = (f"{'model':<38}{'gpu':<7}{'n':>2}{'isl':>8}{'osl':>7}"
           f"{'conc':>7}{'attainable':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        attn = "-" if r["attainable_sol"] is None else f"{r['attainable_sol']:.2f}%"
        print(f"{r['model']:<38}{r['gpu']:<7}{r['gpus']:>2}{r['isl']:>8}{r['osl']:>7}"
              f"{r['concurrency']:>7}{attn:>12}")

    done = [r["attainable_sol"] for r in rows if r["attainable_sol"] is not None]
    if done:
        print(f"\nattainable range: {min(done):.2f}% – {max(done):.2f}%")
    return rows


if __name__ == "__main__":
    main()
