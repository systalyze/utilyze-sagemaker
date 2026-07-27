#!/usr/bin/env python3
"""Run one workload: vLLM server (docker) + utlz SOL capture + load gen.

Load generation uses `vllm bench serve` (random dataset, fixed seed) inside the
same vllm/vllm-openai:v0.21.0 image, an established tool rather than a
hand-rolled loop. The utlz WebSocket feed is captured to JSONL for the whole
workload; analyze.py later averages Valid frames within the steady window recorded
in the result JSON.

Usage:
  run_workload.py --workload nvfp4-base-c32 --gpu 0 \
      --input-tokens 2048 --output-tokens 128 --concurrency 32 --requests 500 \
      [--server-arg=--max-num-seqs=64 ...] [--model nvidia/Gemma-4-31B-IT-NVFP4]

Writes results/<workload-id>/: server.log, bench.json, sol.jsonl, workload.json
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

IMAGE = os.environ.get("VLLM_IMAGE", "vllm/vllm-openai:v0.21.0")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/models"))
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
COLLECTOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utlz_collector.py")
PYTHON = sys.executable


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def wait_healthy(port, timeout, container):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = subprocess.run(
            ["docker", "inspect", container, "--format", "{{.State.Status}}"],
            capture_output=True, text=True).stdout.strip()
        if status != "running":
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3):
                return True
        except Exception:
            time.sleep(3)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True)
    p.add_argument("--gpu", required=True, help="device id or comma list for TP")
    # default: local symlink dir with the ckpt's FP8-KV declaration stripped;
    # Ampere triton lacks fp8e4nv, and explicit bf16 trips an attn-backend assert,
    # so 'auto' over an edited config is the only route to the report's bf16-KV base
    p.add_argument("--model", default="/models/Gemma-4-31B-IT-NVFP4-bf16kv")
    p.add_argument("--served-model-name", default="nvidia/Gemma-4-31B-IT-NVFP4")
    p.add_argument("--input-tokens", type=int, required=True)
    p.add_argument("--output-tokens", type=int, required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--requests", type=int, default=500)
    p.add_argument("--warmups", type=int, default=50)
    p.add_argument("--prefix-len", type=int, default=0,
                   help="random-prefix-len (shared prefix tokens); report's shared-prefix mode")
    p.add_argument("--server-arg", action="append", default=[],
                   help="extra vllm serve arg, repeatable (e.g. --server-arg=--max-num-seqs=64)")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--utlz-url", default="ws://127.0.0.1:8079/live")
    p.add_argument("--load-timeout", type=float, default=1500)
    args = p.parse_args()

    gpu0 = int(str(args.gpu).split(",")[0])
    port = args.port or (8100 + gpu0)
    outdir = os.path.abspath(os.path.join(RESULTS_ROOT, args.workload))
    os.makedirs(outdir, exist_ok=True)
    container = f"workload-{args.workload}"

    server_cmd = [
        "docker", "run", "-d", "--name", container,
        "--gpus", f'"device={args.gpu}"', "--network", "host",
        "-v", f"{MODELS_DIR}:/models",
        "--shm-size", "16g", IMAGE,
        "--model", args.model, "--port", str(port),
        "--served-model-name", args.served_model_name,
        "--gpu-memory-utilization", "0.90",
    ] + args.server_arg
    # defaults matching the report's base config unless overridden per-workload
    joined = " ".join(args.server_arg)
    if "--max-model-len" not in joined:
        server_cmd += ["--max-model-len", "8192"]
    if "--max-num-batched-tokens" not in joined:
        server_cmd += ["--max-num-batched-tokens", "4096"]  # multimodal floor (2496) exceeds vllm 0.21 default
    if "prefix-caching" not in joined:
        server_cmd += ["--no-enable-prefix-caching"]
    if len(str(args.gpu).split(",")) > 1 and "--tensor-parallel-size" not in joined:
        server_cmd += ["--tensor-parallel-size", str(len(str(args.gpu).split(",")))]

    subprocess.run(["docker", "rm", "-f", container], capture_output=True)
    wait_gpu_free(args.gpu)
    sh(server_cmd)
    result = {"workload": args.workload, "gpu": args.gpu, "port": port,
              "server_cmd": server_cmd, "started": time.time()}
    collector = None
    try:
        if not wait_healthy(port, args.load_timeout, container):
            result["status"] = "server_failed"
            return finish(result, outdir, container)

        collector = subprocess.Popen(
            [PYTHON, COLLECTOR, "--url", args.utlz_url,
             "--out", os.path.join(outdir, "sol.jsonl")],
            stderr=open(os.path.join(outdir, "collector.log"), "w"))

        bench_cmd = [
            "docker", "run", "--rm", "--network", "host",
            "-v", f"{MODELS_DIR}:/models",
            "-v", f"{outdir}:/out", "--entrypoint", "vllm", IMAGE,
            "bench", "serve",
            "--backend", "vllm",
            "--base-url", f"http://127.0.0.1:{port}",
            "--model", args.served_model_name,
            "--tokenizer", args.model,  # tokenizer files live in the local model dir
            "--dataset-name", "random",
            "--random-input-len", str(args.input_tokens),
            "--random-output-len", str(args.output_tokens),
            "--random-range-ratio", "0",
            "--random-prefix-len", str(args.prefix_len),
            "--num-prompts", str(args.requests),
            "--num-warmups", str(args.warmups),
            "--max-concurrency", str(args.concurrency),
            "--ignore-eos",
            "--seed", "42",
            "--save-result", "--result-dir", "/out", "--result-filename", "bench.json",
        ]
        result["bench_started"] = time.time()
        bench = subprocess.run(bench_cmd, capture_output=True, text=True)
        result["bench_ended"] = time.time()
        result["bench_rc"] = bench.returncode
        with open(os.path.join(outdir, "bench.stdout"), "w") as f:
            f.write(bench.stdout + "\n--- stderr ---\n" + bench.stderr)
        completed = 0
        try:
            with open(os.path.join(outdir, "bench.json")) as f:
                completed = json.load(f).get("completed", 0)
        except (OSError, json.JSONDecodeError):
            pass
        result["status"] = ("ok" if bench.returncode == 0 and completed > 0
                            else "bench_failed")
    finally:
        if collector:
            collector.send_signal(signal.SIGTERM)
            try:
                collector.wait(15)
            except subprocess.TimeoutExpired:
                collector.kill()
        finish(result, outdir, container)


def wait_gpu_free(gpu, timeout=90):
    """Block until the GPU's memory is drained (prior workload's container may
    still be releasing VRAM when we're scheduled onto the device)."""
    gpu0 = str(gpu).split(",")[0]
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", gpu0], capture_output=True, text=True).stdout.strip()
        try:
            if int(out) < 2000:
                return
        except ValueError:
            return
        time.sleep(3)


def finish(result, outdir, container):
    if result.get("_finished"):  # called from both error path and finally
        return
    result["_finished"] = True
    with open(os.path.join(outdir, "server.log"), "w") as f:
        subprocess.run(["docker", "logs", container], stdout=f, stderr=subprocess.STDOUT)
    subprocess.run(["docker", "rm", "-f", container], capture_output=True)
    result["ended"] = time.time()
    with open(os.path.join(outdir, "workload.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: result[k] for k in ("workload", "status") if k in result}))
    if result.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
