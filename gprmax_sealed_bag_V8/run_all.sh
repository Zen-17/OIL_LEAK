#!/usr/bin/env bash
# Run all 49 GPRMax simulation files (48 bag scenes + 1 baseline).
# Each scan uses -n 21 for 21 antenna positions.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Collect all .in files: bag files first, then baseline
mapfile -t FILES < <(ls bag_*.in 2>/dev/null | sort; echo "baseline.in")

TOTAL=${#FILES[@]}
COUNT=0

for INFILE in "${FILES[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[${COUNT}/${TOTAL}] Running: ${INFILE} ..."
    python -m gprMax "${INFILE}" -n 21
    echo "[${COUNT}/${TOTAL}] Done: ${INFILE}"
done

echo ""
echo "All ${TOTAL} simulations complete."
