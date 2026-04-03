#!/bin/bash
# Run the LR+UV winner combos after current batch finishes.
# Usage: nohup bash imt_gpt/run_combos.sh WAIT_PID >> imt_gpt/results/stairclimb.log 2>&1 &
cd "$(dirname "$0")/.."
PYTHON=imt_gpt/.venv/bin/python

if [ -n "$1" ]; then
    echo "=== Waiting for PID $1 to finish ==="
    while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi

echo "=== LR+UV combo batch starting at $(date) ==="

COMBOS=(
    lr_2x_UV_stable
    lr_2x_UV
    lr_3x
    lr_3x_UV_stable
    unfold_seed
    unfold_aligned
)

for name in "${COMBOS[@]}"; do
    echo ""
    echo "=== Queuing: $name at $(date) ==="
    $PYTHON -m imt_gpt.stairclimb --run "$name"
    echo "=== Finished: $name at $(date) ==="
done

echo ""
echo "=== Combo batch complete at $(date) ==="
$PYTHON -m imt_gpt.stairclimb --list
