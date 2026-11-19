#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Activate the desired Python environment before invoking/submitting this script.
# Arguments match silicate_analysis.py: TYPE [WORKERS] [TRAJECTORY] [options].
export MPLBACKEND=${MPLBACKEND:-Agg}
exec "${PYTHON:-python}" "$script_dir/silicate_analysis.py" "$@"
