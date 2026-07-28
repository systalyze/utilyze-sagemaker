#!/usr/bin/env bash
# Run the five report workloads and produce results/summary.csv.
# One GPU and one utlz instance per workload, so each workload gets its own live ceiling.
set -u
cd "$(dirname "$0")"

GPUS=(${GPUS:-0 1 2 3 4})
UTLZ=${UTLZ_BIN:-utlz}
BASE_PORT=9080

WORKLOADS=(
  "decode-b16  128  1024 16"
  "decode-b64  128  1024 64"
  "bal512-b32  512  512  32"
  "bal2048-b32 2048 128  32"
  "prefill-b32 4800 32   32"
)

if [ "${#GPUS[@]}" -lt "${#WORKLOADS[@]}" ]; then
  echo "need ${#WORKLOADS[@]} GPUs in \$GPUS (got: ${GPUS[*]}); run workloads one at a time with run_workload.py instead" >&2
  exit 1
fi

utlz_pids=()
for i in "${!WORKLOADS[@]}"; do
  sudo "$UTLZ" -mode server -devices "${GPUS[$i]}" -port "$((BASE_PORT + i))" \
    > "results-utlz-gpu${GPUS[$i]}.log" 2>&1 &
  utlz_pids+=($!)
done
sleep 10

wl_pids=()
workloads=()
for i in "${!WORKLOADS[@]}"; do
  set -- ${WORKLOADS[$i]}
  python3 run_workload.py --workload "$1" --gpu "${GPUS[$i]}" \
    --utlz-url "ws://127.0.0.1:$((BASE_PORT + i))/live" \
    --input-tokens "$2" --output-tokens "$3" --concurrency "$4" --requests 500 \
    > "results-$1.log" 2>&1 &
  wl_pids+=($!)
  workloads+=("$1")
done

rc=0
for i in "${!wl_pids[@]}"; do
  wait "${wl_pids[$i]}" || { echo "workload ${workloads[$i]} failed, see results-${workloads[$i]}.log" >&2; rc=1; }
done

sudo kill "${utlz_pids[@]}" 2>/dev/null

python3 analyze.py results/decode-b16 results/decode-b64 results/bal512-b32 results/bal2048-b32 results/prefill-b32 --csv results/summary.csv
exit $rc
