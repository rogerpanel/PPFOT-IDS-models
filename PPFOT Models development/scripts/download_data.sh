#!/usr/bin/env bash
# Download ICS3D dataset from Kaggle
# Requires: pip install kagglehub  (or kaggle CLI with API token)

set -euo pipefail

echo "============================================================"
echo "Downloading Integrated Cloud Security 3Datasets (ICS3D)"
echo "Source: kaggle.com/datasets/rogernickanaedevha/integrated-cloud-security-3datasets-ics3d"
echo "License: CC BY-NC-SA 4.0"
echo "============================================================"

python -c "
import kagglehub
path = kagglehub.dataset_download('rogernickanaedevha/integrated-cloud-security-3datasets-ics3d')
print(f'Dataset downloaded to: {path}')
"

echo ""
echo "Download complete. Pass the printed path as --data-path to experiment scripts."
