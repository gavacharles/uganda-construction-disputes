"""Stage 1.6 — fetch the external layers (spatial, budget, rainfall, OAG, The Observer) from the author's GitHub repos.

Only the specific processed files are downloaded (about 17 MB), never whole repositories: each file is fetched through
the GitHub contents API (`gh api`, raw media type) at the repo's current commit and recorded in a manifest
(repo, path, commit, bytes, sha256) so analyses are reproducible. Files land in data/external/<repo>/<path> (git-ignored).
Usage: python -m src.ingest.external [--refresh]
"""
import hashlib
import subprocess
import sys

import pandas as pd

from src.config import ROOT

DEST = ROOT / "data" / "external"
OWNER = "gavacharles"
FILES = {
    "procurement-risk-uganda": ["docs/data/districts.geojson", "docs/data/district_metrics.json", "docs/data/roads.json",
                                "docs/data/summary.json"],
    "uganda-rainy-season-access-study": ["outputs/access_districts.csv"],
    "uganda-rainfall-climate": ["results/season_trend_zone.csv", "results/season_epochs_zone.csv",
                                "results/rain_annual_cycle_epochs.csv"],
    "oag-reports": [f"government_Auditor_General/{f}" for f in [
        "audit_infrastructure_classifications_improved.csv", "national_dispute_risk_taxonomy_2017_2025_expanded.csv",
        "driver_trend_by_year_2017_2025_expanded.csv", "oag_target_mentions_2017_2025.csv",
        "oag_infrastructure_sentence_corpus_2017_2025_expanded.csv", "consolidated_report_index_2017_2025.csv"]],
    "the-observer": [f"paper2_online_discourse/outputs_observer_full_corpus/{f}" for f in [
        "extended_analysis/articles_extended.csv", "extended_analysis/dispute_cooccurrence_matrix.csv",
        "extended_analysis/sector_driver_crosstab.csv", "extended_analysis/actor_type_counts.csv"]]
        + ["paper2_online_discourse/outputs_observer_full/monthly_dispute_signals.csv"],
}


def gh(*args, binary=False):
    out = subprocess.run(["gh", "api", *args], check=True, capture_output=True)
    return out.stdout if binary else out.stdout.decode().strip()


def main(refresh=False):
    rows = []
    for repo, paths in FILES.items():
        branch = gh(f"repos/{OWNER}/{repo}", "--jq", ".default_branch")
        sha = gh(f"repos/{OWNER}/{repo}/commits/{branch}", "--jq", ".sha")
        for path in paths:
            out = DEST / repo / path
            if out.exists() and not refresh:
                data = out.read_bytes()
            else:
                data = gh(f"repos/{OWNER}/{repo}/contents/{path}?ref={sha}", "-H", "Accept: application/vnd.github.raw", binary=True)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)
            rows.append({"repo": repo, "path": path, "commit": sha[:12], "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()[:16]})
    m = pd.DataFrame(rows)
    m.to_csv(DEST / "manifest.csv", index=False)
    print(m.groupby("repo").bytes.sum().div(1e6).round(2).rename("MB").to_string())
    print(f"total {m.bytes.sum() / 1e6:.1f} MB in {len(m)} files -> {DEST}")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
