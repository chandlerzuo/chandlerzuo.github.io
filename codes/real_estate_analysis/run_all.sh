#!/usr/bin/env bash
# Reproduce the post end to end: download -> simulate -> figures.
#   ./run_all.sh            # reuse cached downloads in data/raw/
#   ./run_all.sh --refresh  # re-download everything
set -euo pipefail
cd "$(dirname "$0")/code"
python3 download_data.py "$@"
python3 history_pipeline.py
python3 make_history_charts.py
