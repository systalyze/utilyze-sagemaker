#!/usr/bin/env python3
"""GPU-free sanity check: both modes, all rules, accounting invariants."""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "sweep.py")


def run(*extra):
    out = f"citest-{'-'.join(extra).replace('--', '')}"
    subprocess.run([sys.executable, SWEEP, "--dry-run", "--configs", "24",
                    "--out", out, *extra], check=True, capture_output=True)
    path = os.path.join(HERE, "results-sweep", out)
    with open(os.path.join(path, "summary.json")) as f:
        return json.load(f), path


def main():
    s, p1 = run("--mode", "replay", "--rule", "sol90")
    assert s["configs"] == 24, s
    assert s["points_pruned"] > 0, "sol90 replay pruned nothing"
    assert 0 < s["saving_pct"] < 100, s["saving_pct"]
    assert s["pruned_cost_usd"] < s["full_cost_usd"]
    assert s["winner_regret_pct"] is not None and s["winner_regret_pct"] < 15, \
        f"pruning regret suspiciously high: {s['winner_regret_pct']}%"

    s2, p2 = run("--mode", "live", "--rule", "plateau")
    assert s2["points_measured"] < 24 * 10, "live mode measured the full grid"
    assert s2["points_pruned"] > 0 and s2["saving_pct"] > 0, s2
    assert s2["winner_unchanged"] is None, "live mode cannot test the winner"

    s3, p3 = run("--mode", "replay", "--rule", "off")
    assert s3["points_pruned"] == 0 and s3["saving_pct"] == 0.0, s3

    s4, p4 = run("--mode", "replay", "--rule", "auto")
    assert s4["rules_used"].get("sol90", 0) > 0, s4["rules_used"]
    assert s4["points_pruned"] > 0, "auto rule pruned nothing"

    for p in (p1, p2, p3, p4):
        shutil.rmtree(p)
    print("sweep dry-run checks passed")


if __name__ == "__main__":
    main()
