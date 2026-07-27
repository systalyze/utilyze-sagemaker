#!/usr/bin/env python3
"""Collect utlz WebSocket events (metrics + ceilings) to JSONL.

Usage: utlz_collector.py [--url ws://127.0.0.1:8079/live] --out sol.jsonl [--duration SECS]

Each line: {"recv_ts": <unix float>, "event": <raw utlz event>}
Event types observed from utlz v0.1.3: "metrics" (snapshot per ~250ms) and
"ceilings" (attainable-SOL, may be empty/absent when unsupported).
"""
import argparse
import asyncio
import json
import signal
import sys
import time

import websockets


async def collect(url: str, out_path: str, duration: float | None) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    if duration:
        loop.call_later(duration, stop.set)

    n_metrics = n_ceilings = 0
    with open(out_path, "a", buffering=1) as out:
        while not stop.is_set():
            try:
                async with websockets.connect(url, max_size=None) as ws:
                    while not stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        except asyncio.TimeoutError:
                            continue
                        event = json.loads(raw)
                        out.write(json.dumps({"recv_ts": time.time(), "event": event}) + "\n")
                        etype = event.get("type")
                        if etype == "metrics":
                            n_metrics += 1
                        elif etype == "ceilings":
                            n_ceilings += 1
            except (OSError, websockets.WebSocketException) as e:
                if stop.is_set():
                    break
                print(f"collector: reconnecting after {type(e).__name__}: {e}", file=sys.stderr)
                await asyncio.sleep(1.0)
    print(f"collector: wrote {n_metrics} metrics + {n_ceilings} ceilings events", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8079/live")
    p.add_argument("--out", required=True)
    p.add_argument("--duration", type=float, default=None)
    args = p.parse_args()
    url = args.url
    if "client_id=" not in url:  # server 400s without one (service.go)
        url += ("&" if "?" in url else "?") + "client_id=harness-collector"
    asyncio.run(collect(url, args.out, args.duration))


if __name__ == "__main__":
    main()
