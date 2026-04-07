#!/bin/bash
# run_all.sh
# ----------
# End-to-end script to reproduce all experiments and figures from the paper:
#   "Mechanistic Evidence of Layer-Dependent Positional Bias in In-Context Learning"
#
# Usage:
#   bash experiments/run_all.sh [--device cpu|cuda]
#
# Requirements:
#   pip install -r requirements.txt

set -e

DEVICE="cpu"
OUTPUT_DIR="figures"
RESULTS_DIR="results"

# Parse optional --device argument
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --device) DEVICE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "============================================================"
echo " PAC-Mech-Interp: Reproducing all paper results"
echo " Device: $DEVICE"
echo " Output: $OUTPUT_DIR/"
echo "============================================================"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR"

# Step 1: Mechanistic attention analysis (Figure 1)
echo "[1/2] Running mechanistic attention analysis (Figure 1)..."
python experiments/run_attention_analysis.py \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR"
echo ""

# Step 2: PAC experiments (Figures 2 & 3)
echo "[2/2] Running PAC experiments (Figures 2 & 3)..."
python experiments/run_pac_experiments.py \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --results_dir "$RESULTS_DIR"
echo ""

echo "============================================================"
echo " All experiments complete!"
echo " Figures saved to: $OUTPUT_DIR/"
echo " Raw results saved to: $RESULTS_DIR/"
echo "============================================================"
