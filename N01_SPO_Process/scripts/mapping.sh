#!/bin/bash

#SBATCH -N 1
#SBATCH -n 124
#SBATCH -c 1

MAX_JOBS=124
FAILED_FILE=$2

cat $1 | xargs -P $MAX_JOBS -I {} bash -c 'if python ../../scripts/map_QM_to_CG.py {} ../refer_CG 2>&1; then
        echo "[succeed] {}"
    else
        echo "[failed] {}"
        echo "{}" >> "$0"
    fi
' "$FAILED_FILE"
