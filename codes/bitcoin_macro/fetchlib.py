"""Provenance-tracked keyless data fetching.

Every raw payload lands immutably in data/raw/ and gets a line in
data/provenance.jsonl recording url, retrieval time, byte count and sha256.
Nothing downstream is allowed to touch data/raw/; cleaning happens in build_*.py.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

def _project_root() -> Path:
    """Locate the project root whether scripts live in src/ or at the top level.

    In the research checkout the modules sit in <root>/src/; in the published
    blog bundle they sit directly in <root>/. Anchoring on "the directory that
    contains (or will contain) data/" keeps every path correct in both layouts
    instead of silently writing data/ one level too high.
    """
    here = Path(__file__).resolve().parent
    if (here / "data").exists() or here.name != "src":
        return here
    return here.parent


ROOT = _project_root()
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROV = ROOT / "data" / "provenance.jsonl"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)
# Output directories too, so a clean checkout with no outputs/ still runs.
(ROOT / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs" / "tables").mkdir(parents=True, exist_ok=True)


def _log(record: dict) -> None:
    with PROV.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def fetch(url: str, name: str, *, ua: bool = False, force: bool = False,
          min_bytes: int = 200, retries: int = 3, pause: float = 1.0) -> Path:
    """Download `url` to data/raw/`name`, with caching and provenance.

    Raises RuntimeError if the payload is missing or implausibly small, so a
    silent failure can never masquerade as data further down the pipeline.
    """
    dest = RAW / name
    if dest.exists() and dest.stat().st_size >= min_bytes and not force:
        return dest

    cmd = ["curl", "-sSL", "--fail", "--max-time", "120"]
    if ua:
        cmd += ["-A", UA]
    cmd += [url, "-o", str(dest)]

    last = ""
    for attempt in range(retries):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and dest.exists() and dest.stat().st_size >= min_bytes:
            size = dest.stat().st_size
            _log({
                "name": name,
                "url": url,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "bytes": size,
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            })
            return dest
        last = (proc.stderr or "").strip() or f"size={dest.stat().st_size if dest.exists() else 0}"
        time.sleep(pause * (attempt + 1))

    raise RuntimeError(f"fetch failed: {name} <- {url} ({last})")


def fred(series: str, *, force: bool = False) -> Path:
    """One FRED series per request.

    Deliberately NOT batched: the multi-series fredgraph.csv endpoint returns a
    ZIP whose members are split by observation frequency, with commas in the
    member names. One series per call keeps parsing trivial.
    """
    return fetch(
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}",
        f"fred_{series}.csv", force=force,
    )
