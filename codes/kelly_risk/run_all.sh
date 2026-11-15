#!/usr/bin/env bash
# Reproduce the post end to end: simulate -> figures. Needs numpy, pandas, scipy, matplotlib.
# Takes about 5 minutes; all random seeds are fixed.
set -euo pipefail
cd "$(dirname "$0")/code"
python3 sim.py
python3 make_charts.py
