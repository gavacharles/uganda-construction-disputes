"""Project paths and constants."""
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"          # read-only (symlinks to period folders)
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
LOGS = OUT / "logs"
FIGS = OUT / "figures"

PERIODS = ["2015-2025", "2026"]
HEADER_ROW = 3  # zero-based; rows 1-3 are title/source notes

for p in (INTERIM, PROCESSED, LOGS, FIGS):
    p.mkdir(parents=True, exist_ok=True)

# full-text judgments live outside OneDrive (1.5 GB); originals zipped in 'Downloaded judgements/'
JUDGMENTS = Path.home() / "Research" / "ulii_judgments"
