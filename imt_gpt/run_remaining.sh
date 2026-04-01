#!/bin/bash
# Run remaining diagnostics + spike-skip test, survives session close.
# Usage: nohup bash imt_gpt/run_remaining.sh >> imt_gpt/results/diagnostics.log 2>&1 &

cd "$(dirname "$0")/.."
PYTHON=imt_gpt/.venv/bin/python

echo "=== Waiting for diagnostic suite (PID $1) to finish ==="
if [ -n "$1" ]; then
    while kill -0 "$1" 2>/dev/null; do sleep 60; done
    echo "=== Diagnostic suite finished ==="
fi

# Run spike-skip test
echo ""
echo "============================================================"
echo "  SPIKE-SKIP TEST: extracted_s137 with spike_skip_mult=50"
echo "============================================================"
$PYTHON imt_gpt/test_spike_skip.py

echo ""
echo "=== ALL RUNS COMPLETE at $(date) ==="
