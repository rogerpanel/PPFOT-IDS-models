#!/usr/bin/env bash
# ============================================================
# PPFOT-IDS: Full Reproducibility Script
# Byzantine-Robust Federated IDS via Rényi-Private OT
# Anaedevha, Trofimov & Borodachev — IEEE Access 2026
# ============================================================
#
# Usage:
#   bash scripts/reproduce_results.sh [--data-path /path/to/ics3d] [--device cuda]
#
# This script:
#   1. Installs dependencies
#   2. Downloads the ICS3D dataset (if no --data-path given)
#   3. Runs all 5 experiment suites
#   4. Generates figures and summary tables
#

set -euo pipefail
cd "$(dirname "$0")/.."

# Parse arguments
DATA_PATH=""
DEVICE="auto"
ROUNDS=50

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path) DATA_PATH="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "PPFOT-IDS Reproducibility Suite"
echo "============================================================"
echo "  Device:     $DEVICE"
echo "  Rounds:     $ROUNDS"
echo "  Data path:  ${DATA_PATH:-'(will download)'}"
echo "============================================================"

# Step 1: Install dependencies
echo ""
echo "[1/3] Installing dependencies..."
pip install -q -r requirements.txt

# Step 2: Download data if needed
if [ -z "$DATA_PATH" ]; then
    echo ""
    echo "[2/3] Downloading ICS3D dataset..."
    DATA_PATH=$(python -c "
import kagglehub
path = kagglehub.dataset_download('rogernickanaedevha/integrated-cloud-security-3datasets-ics3d')
print(path)
")
    echo "  Dataset at: $DATA_PATH"
else
    echo ""
    echo "[2/3] Using provided data path: $DATA_PATH"
fi

# Step 3: Run all experiments
echo ""
echo "[3/3] Running experiments..."
python experiments/run_all.py \
    --data-path "$DATA_PATH" \
    --device "$DEVICE" \
    --rounds "$ROUNDS"

echo ""
echo "============================================================"
echo "Reproduction complete!"
echo "Results saved in: results/"
echo "============================================================"
