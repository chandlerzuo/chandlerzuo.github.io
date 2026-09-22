#!/usr/bin/env bash
# Reproduce the whole study from a clean checkout.
#
# No data is committed. The fetch steps below download every input from public
# keyless sources into data/raw/, build_panel.py turns them into the analysis
# panel in data/processed/, and the rest write to outputs/. Fetches are cached
# and skipped on re-run, so this is safe to invoke repeatedly; delete data/raw/
# to force a clean re-download. Roughly 35 minutes end to end on a laptop.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "== creating venv =="
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q numpy pandas scipy statsmodels matplotlib \
    scikit-learn arch hmmlearn pyarrow openpyxl
fi

echo "== 1/10 fetch: macro series (FRED) =="        ; $PY fetch_fred.py
echo "== 2/10 fetch: Fama-French factors =="        ; $PY fetch_factors.py
echo "== 3/10 fetch: bitcoin, 3 sources + QA =="    ; $PY fetch_btc.py
echo "== 4/10 fetch: metals, ETFs, OECD, JST =="    ; $PY fetch_assets.py
echo "== 5/10 fetch: altcoins, funding =="          ; $PY fetch_crypto.py
echo "== 6/10 build analysis panel =="              ; $PY build_panel.py
echo "== 7/10 static betas + horse race =="         ; $PY betas_static.py
echo "== 8/10 time-varying betas =="                ; $PY betas_tvp.py
echo "== 8/10 regimes =="                           ; $PY regimes.py
echo "== 8/10 correlation + beta breaks =="          ; $PY breaks_correlation.py
echo "== 8/10 crypto-native controls =="            ; $PY crypto_controls.py
echo "== 9/10 analogues (~10 min) =="               ; $PY analogues.py
echo "== 9/10 analogue out-of-sample (~8 min) =="   ; $PY validate_analogues.py
echo "== 9/10 analogues incl. emerging markets ==" ; $PY analogues_em.py
echo "== 10/10 figures =="                          ; $PY figures.py
                                                      $PY fig_essay.py
                                                      $PY fig_essay_analogues.py

echo
echo "done. see REPORT.md, outputs/tables/ and outputs/figures/"
