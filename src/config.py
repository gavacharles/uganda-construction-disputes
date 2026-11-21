"""Project paths and constants.

Working folder (code, derived data, outputs): this repository, outside OneDrive.
Raw inputs (read-only): the OneDrive project folder SOURCE, linked in as data/raw/<period>.
"""
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path.home() / "Library/CloudStorage/OneDrive-Personal(2)/PhD/UJ/Research/Papers/Temporal Analysis"
RAW = ROOT / "data" / "raw"          # read-only symlinks -> SOURCE/2015-2025, SOURCE/2026
DOWNLOADS = SOURCE / "Downloaded judgements"   # user's zips + gold_download_list.xlsx
JUDGMENTS = Path.home() / "Research" / "ulii_judgments"   # extracted full-text PDFs (1.5 GB)
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
LOGS = OUT / "logs"
FIGS = OUT / "figures"

PERIODS = ["2015-2025", "2026"]
HEADER_ROW = 3  # zero-based; rows 1-3 are title/source notes

for p in (INTERIM, PROCESSED, LOGS, FIGS):
    p.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
for period in PERIODS:  # link raw period folders from OneDrive (never copied or modified)
    link = RAW / period
    if not link.exists() and (SOURCE / period).exists():
        link.symlink_to(SOURCE / period)

# Gold labels used by src.code.kappa / src.code.validate. PROVISIONAL: AI-coded (Claude, codebook v1.0,
# 21 Nov 2026; see the sheet README). Swap for the human-coded sheet (outputs/coding/gold_coding_sheet.xlsx) when available.
GOLD_SHEET = OUT / "coding" / "gold_coding_sheet_AI.xlsx"
GOLD_SOURCE = "AI-coded (provisional)"
