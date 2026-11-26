"""Sector report 2026: fetch macro, price, trade and tax data.

Three sources, each recorded in data/external/sector/manifest.csv (source, key, url, bytes, sha256, fetched_utc):
  1. GitHub: processed panels from the author's repos (CIPI + macro panel, weekly FX/Brent, commodities, fiscal,
     observed prices, rainfall, escalation transmission results, seasonal delay outputs).
  2. UBOS (ubos.org/datasets): the latest release of each series the report needs (QGDP, IIP, CPI, PPI, RPPI,
     imports, labour market survey, CIPI). The latest file per series is chosen by the "NN_YYYY" upload prefix.
  3. URA statistics downloads (annual data books FY2018/19-FY2024/25, sector revenue, trade report).
Downloaded files are untrusted data: they are only parsed (openpyxl/xlrd/pdftotext), never executed.
Usage: python -m src.report2026.fetch [--refresh]
"""
import datetime as dt
import hashlib
import re
import subprocess
import sys
import urllib.request

import pandas as pd

from src.config import ROOT

DEST = ROOT / "data" / "external" / "sector"
UA = {"User-Agent": "Mozilla/5.0 (research data download)"}
OWNER = "gavacharles"

GITHUB = {
    "cio-construction-costs": [f"data/shared/{f}" for f in [
        "panel_v1.0_extended.csv", "weekly_drivers.csv", "commodities_monthly.csv", "government_funding_panel.csv",
        "observed_market_prices.csv", "macro_extension_monthly.csv", "fx_imf_monthly.csv", "rainfall_era5_monthly.csv"]],
    "brent-crude-uganda-escalation": [
        "results/transmission_v2/tables/diesel_to_cipi_lag_coeffs.csv",
        "results/transmission_v2/tables/brent_to_diesel_lag_coeffs.csv",
        "results/transmission/tables/material_brent_pass_through_rank.csv",
        "results/case_study_ipc/tables/case_formula_weights.csv",
        "results/tables/scenario_shock_impacts.csv"],
    "uganda-seasons-construction-delay": [
        "outputs/monthly_calendar.csv", "outputs/region_year_metrics.csv", "outputs/trends.csv"],
    "uganda-climate-disputes": ["results/rq2_dispute_risk_regions.csv", "results/rq4_projections_regions.csv"],
}

UBOS_PAGE = "https://www.ubos.org/datasets/"
# key -> regex on the file name (case-insensitive); latest upload wins
UBOS = {
    "qgdp_current": r"QGDP[_-]Current[_-]Prices",
    "qgdp_constant": r"QGDP[_-]Constant[_-]Prices",
    "agdp": r"AGDP_Publication_Tables",
    "iip": r"Index_of_Industrial_Production.*Excel_Tables",
    "cpi": r"CPI.*(Excel|excel).*Tables|CPI_Excel_Tables",
    "ppi": r"PPI",
    "rppi": r"(RPPI|Residential_Property_Price_Index)",
    "imports_composition": r"Composition_of_Imports",
    "trade_monthly": r"Total-Monthly-Merchandise-trade",
    "labour_market": r"Labour_Market_Survey.*Extracted_Tables\.xlsx",
}
UBOS_CIPI = r"CIPI"   # every CIPI workbook from Sept 2025 is kept (to document the publication gap)

URA_PAGES = ["https://ura.go.ug/download-category/statistics/",
             "https://ura.go.ug/download-category/statistics/?cp=2"]
URA_KEEP = r"data-book|economic-sector|imports-trade-report|collection-ratios"


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read(), r.headers.get("Content-Disposition", "")


def _gh(*args, binary=False):
    out = subprocess.run(["gh", "api", *args], check=True, capture_output=True)
    return out.stdout if binary else out.stdout.decode().strip()


def _save(out, data, refresh, getter):
    if out.exists() and not refresh:
        return out.read_bytes()
    data = getter() if data is None else data
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return data


def _row(source, key, url, path, data):
    return {"source": source, "key": key, "url": url, "path": str(path.relative_to(ROOT)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()[:16],
            "fetched_utc": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")}


def upload_rank(name):
    """Order UBOS files by upload prefix 'NN_YYYY' (month_year); files without one rank first (older naming)."""
    m = re.match(r"(\d{2})_(\d{4})", name)
    return (int(m.group(2)), int(m.group(1))) if m else (0, 0)


def fetch_github(refresh):
    rows = []
    for repo, paths in GITHUB.items():
        sha = _gh(f"repos/{OWNER}/{repo}/commits/HEAD", "--jq", ".sha")
        for p in paths:
            out = DEST / "github" / repo / p
            data = _save(out, None, refresh, lambda: _gh(f"repos/{OWNER}/{repo}/contents/{p}?ref={sha}", "-H",
                                                         "Accept: application/vnd.github.raw", binary=True))
            rows.append(_row("github", repo, f"{repo}@{sha[:12]}:{p}", out, data))
    return rows


def fetch_ubos(refresh):
    html = _get(UBOS_PAGE)[0].decode("utf-8", "ignore")
    links = sorted(set(re.findall(r"https://www\.ubos\.org/wp-content/uploads/statistics/[^\"' <>]+\.xlsx?", html)))
    rows = []
    picks = {}
    for key, pat in UBOS.items():
        hits = [u for u in links if re.search(pat, u.rsplit("/", 1)[1], re.I) and "CIPI" not in u.upper()]
        if hits:
            picks[key] = [max(hits, key=lambda u: upload_rank(u.rsplit("/", 1)[1]))]
    picks["cipi"] = [u for u in links if re.search(UBOS_CIPI, u, re.I) and upload_rank(u.rsplit("/", 1)[1]) >= (2025, 1)
                     or re.search(r"CIPI-(September|OCTOBER|NOVEMBER)-2025", u, re.I)]
    for key, urls in picks.items():
        for u in urls:
            out = DEST / "ubos" / u.rsplit("/", 1)[1]
            data = _save(out, None, refresh, lambda: _get(u)[0])
            rows.append(_row("ubos", key, u, out, data))
    return rows


def fetch_ura(refresh):
    rows, seen = [], set()
    for page in URA_PAGES:
        html = _get(page)[0].decode("utf-8", "ignore")
        for u in re.findall(r'data-downloadurl="([^"]+)"', html):
            u = re.sub(r"&(amp;)?refresh=[0-9a-f]+", "", u.replace("&amp;", "&"))
            slug = re.search(r"/download/([^/]+)/", u).group(1)
            if slug in seen or not re.search(URA_KEEP, slug):
                continue
            seen.add(slug)
            hits = list((DEST / "ura").glob(f"{slug}.*"))
            if hits and not refresh:
                out, data = hits[0], hits[0].read_bytes()
            else:
                data, disp = _get(u)
                ext = "pdf" if data[:4] == b"%PDF" else "xls" if data[:4] == b"\xd0\xcf\x11\xe0" else "xlsx" \
                    if data[:2] == b"PK" else "bin"
                out = DEST / "ura" / f"{slug}.{ext}"
                _save(out, data, True, None)
            rows.append(_row("ura", slug, u, out, data))
    return rows


def main(refresh=False):
    rows = []
    for name, fn in [("github", fetch_github), ("ubos", fetch_ubos), ("ura", fetch_ura)]:
        try:
            r = fn(refresh)
            rows += r
            print(f"{name}: {len(r)} files")
        except Exception as e:  # keep going; the manifest shows what is missing
            print(f"{name}: FAILED ({e})")
    m = pd.DataFrame(rows)
    m.to_csv(DEST / "manifest.csv", index=False)
    print(f"total {m.bytes.sum() / 1e6:.1f} MB in {len(m)} files -> {DEST}")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
