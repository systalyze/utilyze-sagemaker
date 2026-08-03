#!/usr/bin/env python3
"""Sweep-pruning reproduction: N server configs x a concurrency ladder,
with Attainable SOL stopping each ladder at 90% of the attainable ceiling.

  --mode replay   measure the full ladder, evaluate the stop rule offline
                  (one run shows the savings and whether the winner changes)
  --mode live     actually stop each ladder when the rule fires
  --rule          auto (sol90 where ceilings exist, else plateau) | sol90 |
                  plateau | off (control)

Paper run (~3.5 days on 4x H200, ~330 GPU-hours, ~$2.6k on-demand):
  sweep.py --configs 200 --mode replay --gpus 0,1,2,3 \
      --utlz-url ws://127.0.0.1:9080/live \
      --model-nvfp4 /path/to/Gemma-4-31B-IT-NVFP4-bf16kv \
      --model-bf16 /path/to/gemma-4-31b-it

--dry-run exercises grid, rules, and accounting without GPUs.
Writes results-sweep/<run>/: sweep.csv, configs.csv, summary.json
(rewritten after every config, so a crash keeps everything measured so far).
"""
import argparse
import csv
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.request

IMAGE = os.environ.get("VLLM_IMAGE", "vllm/vllm-openai:v0.23.0")
HERE = os.path.dirname(os.path.abspath(__file__))
COLLECTOR = os.path.join(HERE, "..", "utlz_collector.py")
PYTHON = sys.executable

LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
MAX_MODEL_LEN = 8192
SOL_STOP_FRACTION = 0.90    # sol90: stop once sol >= this fraction of ceiling
PLATEAU_MIN_GAIN = 0.10     # plateau: stop once marginal gain drops below this
REQS_PER_CONC = 8           # bench sizing: n = max(REQS_PER_CONC*c, MIN_REQS)
MIN_REQS = 64
HEALTH_TIMEOUT_S = 900
BENCH_TIMEOUT_S = 3600

# EAGLE-3 excluded: no trained draft head for Gemma-4 (3,072 raw -> 2,304 valid)
DIMS = {
    "topology": ["4xTP1", "2xTP2", "1xTP4"],
    "weights": ["NVFP4", "BF16"],
    "kv_dtype": ["auto", "fp8_e4m3"],
    "max_num_batched_tokens": [2560, 4096, 8192, 16384],
    "max_num_seqs": [32, 64, 128, 256],
    "spec_decode": ["off", "mtp2", "mtp4"],
    "gpu_mem_util": [0.85, 0.95],
    "cuda_graphs": ["on", "eager"],
}
TOPO = {"4xTP1": dict(tp=1, replicas=4), "2xTP2": dict(tp=2, replicas=2),
        "1xTP4": dict(tp=4, replicas=1)}

CONFIG_FIELDS = (["config", "status"] + list(DIMS) +
                 ["rule_used", "stop_c", "points_measured", "points_pruned",
                  "startup_s", "bench_s", "prunable_s", "best_c", "best_tok_s"])
ROW_FIELDS = ["config", "c", "whole_tok_s", "otps", "tok_s_user", "dur_s",
              "sol", "ceiling", "n_ceil", "pruned"]


def build_grid(seed, n):
    grid = [dict(zip(DIMS, combo)) for combo in itertools.product(*DIMS.values())]
    return random.Random(seed).sample(grid, n)


def config_id(i, cf):
    return (f"c{i:03d}-{cf['topology']}-{cf['weights']}-kv{cf['kv_dtype']}"
            f"-mnbt{cf['max_num_batched_tokens']}-mns{cf['max_num_seqs']}"
            f"-{cf['spec_decode']}-mu{cf['gpu_mem_util']}-{cf['cuda_graphs']}")


def server_args(cf, model, served):
    args = ["--model", model, "--served-model-name", served,
            "--host", "0.0.0.0",
            "--max-model-len", str(MAX_MODEL_LEN),
            "--max-num-batched-tokens", str(cf["max_num_batched_tokens"]),
            "--max-num-seqs", str(cf["max_num_seqs"]),
            "--gpu-memory-utilization", str(cf["gpu_mem_util"]),
            "--no-enable-prefix-caching",
            "--tensor-parallel-size", str(TOPO[cf["topology"]]["tp"])]
    if cf["kv_dtype"] != "auto":
        args += ["--kv-cache-dtype", cf["kv_dtype"]]
    if cf["cuda_graphs"] == "eager":
        args += ["--enforce-eager"]
    if cf["spec_decode"] != "off":
        k = cf["spec_decode"][-1]
        args += ["--speculative-config",
                 json.dumps({"method": "mtp", "num_speculative_tokens": int(k)})]
    return args


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def container_running(name):
    r = subprocess.run(["docker", "inspect", name, "--format", "{{.State.Status}}"],
                       capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "running"


def wait_healthy(name, port, timeout=HEALTH_TIMEOUT_S):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not container_running(name):
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3):
                return True
        except Exception:
            time.sleep(5)
    return False


class Replica:
    def __init__(self, name, port, devices):
        self.name, self.port, self.devices = name, port, devices


def dump_logs(reps, outdir):
    for rep in reps:
        subprocess.run(f"docker logs {rep.name} > {outdir}/server-{rep.name}.log 2>&1",
                       shell=True)


def stop_servers(reps):
    for rep in reps:
        subprocess.run(["docker", "rm", "-f", rep.name], capture_output=True)


def start_servers(cid, cf, gpus, base_port, model, served, outdir):
    topo = TOPO[cf["topology"]]
    reps = []
    try:
        for r in range(topo["replicas"]):
            devs = gpus[r * topo["tp"]:(r + 1) * topo["tp"]]
            name = f"sweep-{cid}-r{r}"
            port = base_port + r
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            sh(["docker", "run", "-d", "--rm", "--name", name,
                "--gpus", f'"device={",".join(map(str, devs))}"',
                "--ipc=host", "--shm-size", "16g", "--network", "host",
                "-v", f"{model}:{model}:ro",
                IMAGE, *server_args(cf, model, served), "--port", str(port)])
            reps.append(Replica(name, port, devs))
        for rep in reps:
            if not wait_healthy(rep.name, rep.port):
                dump_logs(reps, outdir)
                raise RuntimeError(f"replica {rep.name} failed to become healthy")
        dump_logs(reps, outdir)
        return reps
    except Exception:
        stop_servers(reps)
        raise


def bench_point(reps, c, isl, osl, model, served, outdir):
    if c < len(reps):   # total c below replica count is not a real operating point
        return None
    per = c // len(reps)
    n = max(REQS_PER_CONC * per, MIN_REQS)
    procs, t0 = [], time.time()
    for rep in reps:
        log = open(os.path.join(outdir, f"bench-c{c}-{rep.name}.log"), "w")
        cmd = ["docker", "exec", rep.name, "vllm", "bench", "serve",
               "--host", "127.0.0.1", "--port", str(rep.port),
               "--model", served, "--tokenizer", model, "--backend", "vllm",
               "--dataset-name", "random",
               "--random-input-len", str(isl), "--random-output-len", str(osl),
               "--random-range-ratio", "0", "--seed", "42",
               "--num-prompts", str(n), "--max-concurrency", str(per),
               "--num-warmups", str(per), "--ignore-eos", "--no-oversample",
               "--save-result", "--result-filename",
               f"/tmp/bench-{rep.name}-c{c}.json"]
        procs.append((rep, subprocess.Popen(cmd, stdout=log, stderr=log), log))
    agg, tpots, total_out = 0.0, [], 0
    try:
        for rep, pr, log in procs:
            rc = pr.wait(timeout=BENCH_TIMEOUT_S)
            log.close()
            if rc != 0:
                raise RuntimeError(
                    f"bench c={c} failed on {rep.name} (rc={rc}), "
                    f"see bench-c{c}-{rep.name}.log")
            raw = sh(["docker", "exec", rep.name, "cat",
                      f"/tmp/bench-{rep.name}-c{c}.json"]).stdout
            d = json.loads(raw)
            with open(os.path.join(outdir, f"bench-c{c}-{rep.name}.json"), "w") as f:
                f.write(raw)
            agg += d["output_throughput"]
            total_out += d.get("total_output_tokens", 0)
            tpot = d.get("median_tpot_ms") or d.get("mean_tpot_ms")
            if tpot:
                tpots.append(tpot)
    except subprocess.TimeoutExpired:
        for rep, pr, log in procs:
            pr.kill()
            log.close()
        raise RuntimeError(f"bench c={c} exceeded {BENCH_TIMEOUT_S}s")
    dur = time.time() - t0
    return dict(otps=agg,
                whole_tok_s=total_out / dur if total_out else agg,
                tok_s_user=1000.0 / statistics.median(tpots) if tpots else None,
                dur_s=dur)


def sol_window(sol_path, devices, t0, t1):
    sols, ceils = [], []
    try:
        with open(sol_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = rec.get("recv_ts", 0)
                if not (t0 <= ts <= t1):
                    continue
                ev = rec.get("event", {})
                for g in ev.get("snapshot", {}).get("GPUs", []):
                    if g.get("DeviceID") in devices and g.get("SOL", {}).get("Valid"):
                        sols.append(g["SOL"]["ComputePct"])
                # utlz /live ceilings: {"type":"ceilings","ceilings":{"<idx>":{"Index":..,"ComputeSolCeiling":..}}}
                for k, g in (ev.get("ceilings") or {}).items():
                    if int(k) in devices and g.get("ComputeSolCeiling") is not None:
                        ceils.append(g["ComputeSolCeiling"])
    except FileNotFoundError:
        pass
    return dict(
        sol=statistics.median(sols) if sols else None,
        ceiling=statistics.median(ceils) if ceils else None,
        n_sol=len(sols), n_ceil=len(ceils))


def dry_measure(cf, rng):
    """Seeded synthetic ladder: saturating throughput with rollover."""
    topo = TOPO[cf["topology"]]
    peak = rng.uniform(150, 450) * (1.15 if cf["weights"] == "BF16" else 1.0)
    knee_c = rng.choice([16, 32, 64])
    ceiling = rng.uniform(20, 60)
    pts = []
    for c in LADDER:
        if c < topo["replicas"]:
            pts.append(None)
            continue
        frac = min(1.0, (c / knee_c) ** 0.6)
        roll = 1.0 if c <= 4 * knee_c else max(0.55, 1 - 0.15 * math.log2(c / (4 * knee_c)))
        tps = peak * frac * roll * rng.uniform(0.98, 1.02)
        pts.append(dict(
            otps=tps, whole_tok_s=tps * rng.uniform(0.97, 1.0),
            tok_s_user=tps / c * topo["replicas"] * rng.uniform(0.9, 1.1),
            dur_s=rng.uniform(90, 150),
            sol=ceiling * min(0.99, frac * roll) * rng.uniform(0.97, 1.03),
            ceiling=ceiling, n_sol=200, n_ceil=12))
    return dict(startup_s=rng.uniform(240, 480), points=pts)


def metric(pt):
    return pt["whole_tok_s"]


def stop_index(points, rule, min_ceilings):
    """Returns (first index where the rule fires or None, rule actually used)."""
    if rule == "off":
        return None, "off"
    if rule == "auto":
        has_ceilings = any(pt and pt.get("n_ceil", 0) >= min_ceilings for pt in points)
        rule = "sol90" if has_ceilings else "plateau"
    prev = None
    for i, pt in enumerate(points):
        if pt is None:
            continue
        if rule == "sol90":
            if (pt.get("ceiling") is not None and pt.get("n_ceil", 0) >= min_ceilings
                    and pt.get("sol") is not None
                    and pt["sol"] >= SOL_STOP_FRACTION * pt["ceiling"]):
                return i, rule
        elif rule == "plateau":
            if prev and metric(prev) > 0 and \
                    (metric(pt) - metric(prev)) / metric(prev) < PLATEAU_MIN_GAIN:
                return i, rule
        prev = pt
    return None, rule


def summarize(rows, config_rows, args, ngpu, rules_used, dropped):
    ok = [r for r in config_rows if r["status"] == "ok"]
    full_h = sum((r["startup_s"] + r["bench_s"]) for r in ok) / 3600
    if args.mode == "live":
        full_h += sum(r["prunable_s"] for r in ok) / 3600
    pruned_h = full_h - sum(r["prunable_s"] for r in ok) / 3600
    winner_full = max(ok, key=lambda r: r["best_tok_s"] or 0, default=None)
    best_pruned = {}
    for r in rows:
        if not r["pruned"]:
            best_pruned[r["config"]] = max(best_pruned.get(r["config"], 0),
                                           r["whole_tok_s"])
    winner_pruned = max(best_pruned, key=best_pruned.get, default=None)
    full_best = (winner_full or {}).get("best_tok_s") or 0
    return dict(
        mode=args.mode, rule=args.rule, rules_used=rules_used,
        configs=len(ok), configs_failed=len(config_rows) - len(ok),
        configs_dropped_preflight=dropped,
        ladder=LADDER, gpu_count=ngpu, gpu_hr_usd=args.gpu_hr_usd,
        points_measured=sum(r["points_measured"] for r in ok),
        points_pruned=sum(r["points_pruned"] for r in ok),
        full_cost_usd=round(full_h * ngpu * args.gpu_hr_usd, 2),
        pruned_cost_usd=round(pruned_h * ngpu * args.gpu_hr_usd, 2),
        saving_pct=round(100 * (1 - pruned_h / full_h), 1) if full_h else None,
        winner_full=(winner_full or {}).get("config"),
        winner_pruned=winner_pruned,
        winner_unchanged=((winner_full or {}).get("config") == winner_pruned
                          if args.mode == "replay" else None),
        winner_regret_pct=(round(100 * (1 - best_pruned.get(winner_pruned, 0) / full_best), 2)
                           if args.mode == "replay" and full_best else None),
    )


def write_outputs(outroot, rows, config_rows, summary):
    with open(os.path.join(outroot, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(outroot, "configs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONFIG_FIELDS, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(config_rows)
    with open(os.path.join(outroot, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


def preflight(grid, gpus, base_port, models, outroot):
    """Boot one server per risky feature; drop configs whose feature fails."""
    checks = []
    for feature, pick in (("spec_decode", lambda cf: cf["spec_decode"] != "off"),
                          ("kv_dtype", lambda cf: cf["kv_dtype"] != "auto"),
                          ("weights", lambda cf: cf["weights"] == "BF16")):
        cf = next((c for c in grid if pick(c)), None)
        if cf:
            checks.append((feature, pick, cf))
    dropped = {}
    for feature, pick, cf in checks:
        cid = f"preflight-{feature}"
        cdir = os.path.join(outroot, cid)
        os.makedirs(cdir, exist_ok=True)
        model, served = models[cf["weights"]]
        try:
            reps = start_servers(cid, cf, gpus, base_port, model, served, cdir)
            stop_servers(reps)
            print(f"[preflight] {feature}={cf[feature]}: ok", flush=True)
        except (RuntimeError, subprocess.CalledProcessError) as e:
            n_before = len(grid)
            grid = [c for c in grid if not pick(c)]
            dropped[feature] = n_before - len(grid)
            print(f"[preflight] {feature}: FAILED ({e}); dropping "
                  f"{dropped[feature]} configs that use it", flush=True)
    return grid, dropped


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", type=int, default=200)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--mode", choices=["replay", "live"], default="replay")
    p.add_argument("--rule", choices=["auto", "sol90", "plateau", "off"], default="auto")
    p.add_argument("--min-ceilings", type=int, default=3,
                   help="ceiling samples (5s cadence) required before sol90 may fire")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--model-nvfp4", default=os.environ.get("SWEEP_MODEL_NVFP4", ""),
                   help="host path of the NVFP4 checkpoint dir")
    p.add_argument("--model-bf16", default=os.environ.get("SWEEP_MODEL_BF16", ""),
                   help="host path of the BF16 checkpoint dir")
    p.add_argument("--served-nvfp4", default="nvidia/Gemma-4-31B-IT-NVFP4",
                   help="served model name for the NVFP4 weights (ceiling lookups key on it)")
    p.add_argument("--served-bf16", default="google/gemma-4-31b-it")
    p.add_argument("--isl", type=int, default=2048)
    p.add_argument("--osl", type=int, default=128)
    p.add_argument("--gpu-hr-usd", type=float, default=7.912,
                   help="p5en.48xlarge on-demand / 8")
    p.add_argument("--utlz-url", default=os.environ.get("UTLZ_URL", ""),
                   help="ws://host:port/live of a running utlz covering --gpus; "
                        "without it sol90 cannot fire and rule auto uses plateau")
    p.add_argument("--base-port", type=int, default=8300)
    p.add_argument("--out", default="")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    if len(gpus) != 4:
        p.error("the config grid assumes a 4-GPU budget; pass exactly 4 GPU ids")
    models = {"NVFP4": (args.model_nvfp4, args.served_nvfp4),
              "BF16": (args.model_bf16, args.served_bf16)}
    if not args.dry_run:
        for w, (path, _) in models.items():
            if not path or not os.path.isdir(path):
                p.error(f"--model-{w.lower()} must point at a checkpoint dir (got {path!r})")
        if args.rule == "sol90" and not args.utlz_url:
            p.error("--rule sol90 requires --utlz-url (use --rule auto or plateau otherwise)")

    run_name = args.out or f"{'dry' if args.dry_run else 'run'}-{args.mode}-{args.rule}-n{args.configs}"
    outroot = os.path.join(HERE, "results-sweep", run_name)
    os.makedirs(outroot, exist_ok=True)
    grid = build_grid(args.seed, args.configs)
    rng = random.Random(args.seed + 1)

    dropped = {}
    if not args.dry_run and not args.skip_preflight:
        grid, dropped = preflight(grid, gpus, args.base_port, models, outroot)

    rows, config_rows = [], []
    rules_used = {}
    for i, cf in enumerate(grid):
        cid = config_id(i, cf)
        cdir = os.path.join(outroot, cid)
        os.makedirs(cdir, exist_ok=True)
        status = "ok"

        if args.dry_run:
            meas = dry_measure(cf, rng)
            startup_s, points = meas["startup_s"], meas["points"]
            if args.mode == "live" and args.rule != "off":
                si, _ = stop_index(points, args.rule, args.min_ceilings)
                if si is not None:
                    points = points[:si + 1] + [None] * (len(LADDER) - si - 1)
        else:
            model, served = models[cf["weights"]]
            t0 = time.time()
            collector = None
            sol_path = os.path.join(cdir, "sol.jsonl")
            if args.utlz_url:
                collector = subprocess.Popen(
                    [PYTHON, COLLECTOR, "--url", args.utlz_url, "--out", sol_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                reps = start_servers(cid, cf, gpus, args.base_port, model, served, cdir)
            except (RuntimeError, subprocess.CalledProcessError) as e:
                print(f"[{cid}] SKIP config: {e}", flush=True)
                config_rows.append(dict(config=cid, status="server_failed", **cf))
                if collector:
                    collector.terminate()
                    collector.wait()
                continue
            startup_s = time.time() - t0
            points = []
            try:
                for c in LADDER:
                    pt0 = time.time()
                    try:
                        r = bench_point(reps, c, args.isl, args.osl, model, served, cdir)
                    except (RuntimeError, subprocess.CalledProcessError,
                            json.JSONDecodeError) as e:
                        print(f"[{cid}] bench failed at c={c}: {e}", flush=True)
                        status = "bench_failed"
                        break
                    if r is not None:
                        r.update(sol_window(sol_path, set(gpus), pt0, time.time()))
                    points.append(r)
                    if args.mode == "live" and args.rule != "off" and r is not None:
                        si, _ = stop_index(points, args.rule, args.min_ceilings)
                        if si is not None:
                            break
            finally:
                dump_logs(reps, cdir)
                stop_servers(reps)
                if collector:
                    collector.terminate()
                    collector.wait()
            points += [None] * (len(LADDER) - len(points))

        si, rule_used = stop_index(points, args.rule, args.min_ceilings)
        rules_used[rule_used] = rules_used.get(rule_used, 0) + 1
        measured = [(c, pt) for c, pt in zip(LADDER, points) if pt is not None]
        best = max(measured, key=lambda t: metric(t[1]), default=(None, None))
        kept_s = sum(pt["dur_s"] for _, pt in measured)
        if args.mode == "replay":
            pruned_idx = [j for j in range(len(LADDER))
                          if si is not None and j > si and points[j] is not None]
            prunable_s = sum(points[j]["dur_s"] for j in pruned_idx)
            n_pruned = len(pruned_idx)
        else:
            n_pruned = (len(LADDER) - (si + 1)) if si is not None else 0
            last_dur = measured[-1][1]["dur_s"] if measured else 0
            prunable_s = n_pruned * last_dur   # lower-bound estimate: unrun tail
        for j, (c, pt) in enumerate(zip(LADDER, points)):
            if pt is None:
                continue
            rows.append(dict(
                config=cid, c=c, whole_tok_s=round(metric(pt), 1),
                otps=round(pt["otps"], 1),
                tok_s_user=round(pt["tok_s_user"], 2) if pt.get("tok_s_user") else "",
                dur_s=round(pt["dur_s"], 1),
                sol=round(pt["sol"], 1) if pt.get("sol") is not None else "",
                ceiling=round(pt["ceiling"], 1) if pt.get("ceiling") else "",
                n_ceil=pt.get("n_ceil", 0),
                pruned=(si is not None and j > si)))
        config_rows.append(dict(
            config=cid, status=status, **cf,
            rule_used=rule_used,
            stop_c=(LADDER[si] if si is not None else ""),
            points_measured=len(measured), points_pruned=n_pruned,
            startup_s=round(startup_s, 1), bench_s=round(kept_s, 1),
            prunable_s=round(prunable_s, 1),
            best_c=best[0], best_tok_s=round(metric(best[1]), 1) if best[1] else ""))
        print(f"[{cid}] status={status} rule={rule_used} measured={len(measured)} "
              f"stop_c={LADDER[si] if si is not None else '-'} best_c={best[0]}",
              flush=True)

        summary = summarize(rows, config_rows, args, len(gpus), rules_used, dropped)
        write_outputs(outroot, rows, config_rows, summary)

    if args.rule in ("auto", "sol90") and not args.dry_run and \
            rules_used.get("sol90", 0) == 0:
        print("WARNING: no config had usable Attainable-SOL ceilings; every stop "
              "decision fell back to the plateau rule. Check --utlz-url, backend "
              "coverage for the served model names, and utlz attribution.",
              flush=True)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
