#!/usr/bin/env bash
# Run from any directory. Task-list entries are relative to qm_source in workflow.json.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python}" "$script_dir/map_dataset.py" "$@"
