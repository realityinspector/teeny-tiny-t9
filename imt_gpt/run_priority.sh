#!/bin/bash
# Unfold experiments first, then LR+UV combos.
cd "$(dirname "$0")/.."
PYTHON=imt_gpt/.venv/bin/python

if [ -n "$1" ]; then
    echo "=== Waiting for PID $1 to finish ==="
    while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi

echo "=== Priority batch starting at $(date) ==="

PRIORITY=(
    # Unfold: the new approach
    unfold_seed
    unfold_aligned
    # LR+UV combos: stacking known winners
    lr_2x_UV_stable
    lr_2x_UV
    lr_3x
    lr_3x_UV_stable
)

for name in "${PRIORITY[@]}"; do
    echo ""
    echo "=== Queuing: $name at $(date) ==="
    $PYTHON -m imt_gpt.stairclimb --run "$name"
    echo "=== Finished: $name at $(date) ==="
done

echo ""
echo "=== Priority batch complete at $(date) ==="
$PYTHON -m imt_gpt.stairclimb --list
