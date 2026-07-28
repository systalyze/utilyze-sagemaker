#!/usr/bin/env python3
"""Measured vs attainable throughput per workload (the slide-8 numbers).

Attainable tok/s is converted from the live Attainable-SOL utilization:
util * peak_flops * osl / flops_per_request, with per-request flops derived
from the model config (text parameters only, lm head charged to generated
tokens, sliding-window attention). Matches the engine's op graph within 1%.

Usage: throughput.py results/summary.csv ../reference/fig1_reference.json \
           <model_dir>/config.json --gpu H200
"""
import argparse
import csv
import json

PEAK_TFLOPS = {"A100": 312, "H100": 989, "H200": 989}


def flops_per_request(cfg_path, isl, osl):
    tc = json.load(open(cfg_path))["text_config"]
    d, ff, layers = tc["hidden_size"], tc["intermediate_size"], tc["num_hidden_layers"]
    qd = tc["num_attention_heads"] * tc["head_dim"]
    kvd = tc["num_key_value_heads"] * tc["head_dim"]
    win = tc["sliding_window"]
    sliding = sum(1 for t in tc["layer_types"] if t == "sliding_attention")
    linear = 2 * layers * (d * (qd + 2 * kvd) + qd * d + 3 * d * ff)
    lm_head = 2 * d * tc["vocab_size"]

    def attn(ctx):
        return 4 * qd * (sliding * min(ctx, win) + (layers - sliding) * ctx)

    return isl * (linear + attn(isl / 2)) + osl * (linear + lm_head + attn(isl + osl / 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("summary")
    p.add_argument("reference")
    p.add_argument("model_config")
    p.add_argument("--gpu", choices=PEAK_TFLOPS, default="A100")
    args = p.parse_args()

    peak = PEAK_TFLOPS[args.gpu] * 1e12
    ref = json.load(open(args.reference))["cells"]
    print(f"{'workload':<14}{'measured_otps':>14}{'attainable_otps':>16}")
    for row in csv.DictReader(open(args.summary)):
        name = row["workload"]
        shape = ref.get(name)
        live = row.get("attainable_live")
        if not shape or live in (None, "", "None"):
            continue
        flops = flops_per_request(args.model_config, shape["isl"], shape["osl"])
        attainable = float(live) / 100 * peak * shape["osl"] / flops
        print(f"{name:<14}{float(row['otps']):>14.0f}{attainable:>16.0f}")


if __name__ == "__main__":
    main()
