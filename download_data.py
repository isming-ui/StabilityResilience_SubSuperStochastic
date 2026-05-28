"""
Download the Cedar Creek BioDIV (e120) aboveground-biomass dataset used by
biodiv_validation.py.

The data are hosted by the Environmental Data Initiative (EDI) under
  Tilman, D. (2024) "Plant Aboveground Biomass Data" (knb-lter-cdr.273.11)
  https://doi.org/10.6073/pasta/27ddb5d8aebe24db99caa3933e9bc8e2
and distributed under CC BY 4.0. They are NOT redistributed with this code;
this script fetches them from the authoritative source on first run.

Cite the original dataset whenever the downloaded CSV is used.
"""
from __future__ import annotations
import os
import ssl
import sys
import urllib.request

DATA_URL = (
    "https://pasta.lternet.edu/package/data/eml/"
    "knb-lter-cdr/273/11/27ddb5d8aebe24db99caa3933e9bc8e2"
)
DEST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "ref", "biodiv_e120_biomass.csv",
)


def _make_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def main() -> None:
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    if os.path.exists(DEST):
        print(f"Already present: {DEST}")
        return
    print(f"Downloading BioDIV (e120) biomass CSV from EDI ...")
    print(f"  source: {DATA_URL}")
    print(f"  target: {DEST}")
    ctx = _make_ssl_context()
    with urllib.request.urlopen(DATA_URL, context=ctx) as resp, \
         open(DEST, "wb") as f:
        f.write(resp.read())
    size_mb = os.path.getsize(DEST) / (1024 * 1024)
    print(f"Done. {size_mb:.2f} MB written.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"download failed: {e}")
