#!/bin/bash
# Run priority stairclimb hypotheses sequentially.
# Usage: nohup bash imt_gpt/run_stairclimb_batch.sh >> imt_gpt/results/stairclimb.log 2>&1 &
cd "$(dirname "$0")/.."
PYTHON=imt_gpt/.venv/bin/python

# Wait for any currently running stairclimb to finish
echo "=== Batch stairclimb starting at $(date) ==="

# Priority order: hybrid alignment sweep, then training knobs, then controls
PRIORITY=(
    hybrid_V_1.0
    hybrid_UV_0.5
    unfold_seed
    unfold_aligned
    warmup_200
    warmup_50
    lr_2x
    spike_skip_50x
    noise_0.1
    noise_0.3
    medium_spectra
    shuffled_layers
    spike_skip_50x_warmup_200
)

for name in "${PRIORITY[@]}"; do
    echo ""
    echo "=== Queuing: $name at $(date) ==="
    $PYTHON -m imt_gpt.stairclimb --run "$name"
    echo "=== Finished: $name at $(date) ==="
done

echo ""
echo "=== Batch complete at $(date) ==="
$PYTHON -m imt_gpt.stairclimb --list
