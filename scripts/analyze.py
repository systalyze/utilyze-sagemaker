#!/usr/bin/env python3
"""Join bench.json throughput with steady-state SOL averages for workloads.

Usage: analyze.py results/<workload-id> [more workloads...] [--csv out.csv]

For each workload: reads workload.json (steady window = bench start+settle .. bench end),
bench.json (vllm bench serve result: output_throughput = OTPS), and sol.jsonl
(utlz frames). Averages only Valid==true SOL frames inside the steady window
for the workload's GPU(s). Prints a table and optional CSV.
"""
import argparse
import csv
import json
import os
import statistics as st
import sys

SETTLE_S = 20  # skip first N s of bench (warmup requests + ramp)


def load_workload(workload_dir):
    with open(os.path.join(workload_dir, "workload.json")) as f:
        workload = json.load(f)
    row = {"workload": workload["workload"], "status": workload.get("status"),
           "gpu": str(workload["gpu"])}
    bench_path = os.path.join(workload_dir, "bench.json")
    if os.path.exists(bench_path):
        with open(bench_path) as f:
            bench = json.load(f)
        row.update(
            otps=round(bench.get("output_throughput", 0), 1),
            total_tps=round(bench.get("total_token_throughput", 0), 1),
            ttft_p50_ms=round(bench.get("median_ttft_ms", 0), 1),
            itl_p50_ms=round(bench.get("median_itl_ms", 0), 1),
            duration_s=round(bench.get("duration", 0), 1),
        )
    t0 = workload.get("bench_started", 0) + SETTLE_S
    t1 = workload.get("bench_ended", float("inf"))
    gpus = [int(g) for g in str(workload["gpu"]).split(",")]
    comp, mem, sma, nvml = [], [], [], []
    ceils = []
    ceil_seen = 0
    sol_path = os.path.join(workload_dir, "sol.jsonl")
    if os.path.exists(sol_path):
        with open(sol_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not t0 <= rec["recv_ts"] <= t1:
                    continue
                ev = rec["event"]
                if ev.get("type") == "ceilings" and ev.get("ceilings"):
                    ceil_seen += 1
                    for idx, g in ev["ceilings"].items():
                        if int(idx) in gpus and g.get("ComputeSolCeiling") is not None:
                            ceils.append(g["ComputeSolCeiling"])
                if ev.get("type") != "metrics":
                    continue
                for g in ev["snapshot"]["GPUs"]:
                    if g["DeviceID"] in gpus and g["SOL"]["Valid"]:
                        comp.append(g["SOL"]["ComputePct"])
                        mem.append(g["SOL"]["MemoryPct"])
                        if g["DCGMUtilization"]["Valid"]:
                            sma.append(g["DCGMUtilization"]["SMActivePct"])
                        if g["NVMLUtilization"]["Valid"]:
                            nvml.append(g["NVMLUtilization"]["UtilPct"])
    if comp:
        row.update(
            n_sol_frames=len(comp),
            compute_sol=round(st.mean(comp), 1),
            memory_sol=round(st.mean(mem), 1),
            compute_sol_p10=round(sorted(comp)[len(comp) // 10], 1),
            compute_sol_stdev=round(st.pstdev(comp), 2),
            memory_sol_stdev=round(st.pstdev(mem), 2),
            sm_active=round(st.mean(sma), 1) if sma else None,
            nvml_util=round(st.mean(nvml), 1) if nvml else None,
            mc_ratio=round(st.mean(mem) / st.mean(comp), 2) if st.mean(comp) > 0 else None,
            ceilings_nonempty=ceil_seen,
            attainable_live=round(st.median(ceils), 1) if ceils else None,
        )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workloads", nargs="+")
    ap.add_argument("--csv")
    args = ap.parse_args()
    rows = [load_workload(c) for c in args.workloads]
    cols = ["workload", "status", "otps", "compute_sol", "memory_sol", "mc_ratio",
            "sm_active", "nvml_util", "compute_sol_stdev", "compute_sol_p10",
            "n_sol_frames", "ttft_p50_ms", "itl_p50_ms", "ceilings_nonempty", "attainable_live"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r.get(c, "")) for c in cols))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
