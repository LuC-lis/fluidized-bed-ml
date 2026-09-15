#!/usr/bin/env bash
# Run the complete analysis pipeline, in order, from raw data to verified claims.
#
#   ./run_all.sh                 # everything (about 10 minutes, 4.6 GB read)
#   ./run_all.sh --fast          # skip the two slowest stages (04, 07 caches)
#
# The raw recordings are never written to; all outputs land in results/.

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
FAST=0
[[ "${1:-}" == "--fast" ]] && FAST=1

echo "== 01 data inventory =="
"$PYTHON" scripts/01_data_inventory.py

echo "== 02 pressure regime =="
"$PYTHON" scripts/02_pressure_regime.py

echo "== 03 wavelet validation (the key regression test) =="
"$PYTHON" scripts/03_wavelet_validation.py

if [[ $FAST -eq 0 ]]; then
  echo "== 04 optical probe field (slow: parses 3 GB of .pvd) =="
  "$PYTHON" scripts/04_ofp_concentration.py
fi

echo "== 05 vibration features =="
"$PYTHON" scripts/05_vibration_features.py

echo "== 06 surrogate (needs the group's dataset; runs on a demo otherwise) =="
"$PYTHON" scripts/06_train_surrogate.py --demo

if [[ $FAST -eq 0 ]]; then
  echo "== 07 regime classification (slow: fits 18 model/protocol combinations) =="
  "$PYTHON" scripts/07_regime_classification.py
fi

echo "== 08 verify every headline claim =="
"$PYTHON" scripts/08_verify_claims.py

echo "== tests =="
"$PYTHON" -m pytest tests -q

echo
echo "Pipeline complete. Figures in results/figures, tables in results/tables."
