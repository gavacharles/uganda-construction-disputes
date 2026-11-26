"""Sector report 2026: refresh everything in order (fetch -> parse -> panel -> indices -> scorecard -> figures),
re-render the disputes figures from the analysis stages in the Hove & Quill theme, and build the Word report.
Usage: python -m src.report2026.run [--refresh] [--no-docx]   (--refresh re-downloads all sources)
"""
import os
import shutil
import subprocess
import sys

from src.config import OUT, ROOT
from src.report2026 import build_panel, fetch, figures, indices, parse_ubos, parse_ura, scorecard

REPORT = OUT / "writing" / "sector_report_2026"
STAGE_FIGS = {  # analysis stage -> figures reused in chapters 6-8 (rendered with UGCD_THEME=hq into outputs/analysis_hq)
    "report_2026": ["report_2026/fig_2026_vs_decade.png"],
    "s3_temporal_legislation": ["03_temporal_legislation/fig2_legislation_timeline_uptake.png"],
    "s4_outcomes_econ": ["04_outcomes_econ/fig1_km_suits_appeals.png"],
    "s5_arbitration": ["05_arbitration/fig1_deference_by_year.png"],
    "s9_reform": ["09_reform/fig1_priority_map.png"],
    "s10_spatial": ["10_spatial/fig1_maps.png"],
    "s11_litigation_filter": ["11_litigation_filter/fig1_family_shares.png"],
}


def stage_figures():
    env = {**os.environ, "UGCD_THEME": "hq"}
    for stage, figs in STAGE_FIGS.items():
        subprocess.run([sys.executable, "-m", f"src.analysis.{stage}"], cwd=ROOT, env=env, check=True,
                       stdout=subprocess.DEVNULL)
        for f in figs:
            shutil.copy(OUT / "analysis_hq" / f, REPORT / "figures" / os.path.basename(f))
    print(f"stage figures: {sum(map(len, STAGE_FIGS.values()))} re-rendered in the H&Q theme")


def main(refresh=False, docx=True):
    fetch.main(refresh=refresh)
    for mod in (parse_ura, parse_ubos, build_panel, indices, scorecard, figures):
        print(f"\n== {mod.__name__}")
        mod.main()
    stage_figures()
    if docx:
        subprocess.run(["node", "scripts/build_sector_report.js", str(REPORT / "sector_report_2026_full.md"),
                        str(REPORT / "HoveQuill_Uganda_Construction_Sector_Report_2026.docx")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv, docx="--no-docx" not in sys.argv)
