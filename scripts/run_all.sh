#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT_DIR/.venv/bin/python"

"$PY" "$ROOT_DIR/scripts/problem1_pipeline.py"
"$PY" "$ROOT_DIR/scripts/problem2_pipeline.py"

echo "All pipelines completed."
