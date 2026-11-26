"""PhD (stage 13 input): MoFPED budget data portal downloads (national approved budgets, releases, expenditure).

Source: https://www.budget.finance.go.ug/dataportal (Excel files by financial year). Only national (central government)
files are fetched. Files are saved unchanged to data/external/mofped/portal/ with a manifest (url, bytes, sha256 prefix,
date); existing files are not re-downloaded unless --refresh.
Harmonised output: data/processed/fiscal_triangulation/mofped_budget_long.csv.gz (+ mofped_coverage.csv).
Usage: python -m src.ingest.mofped [--refresh] | --harmonise (skip the fetch)
"""
import datetime as dt
import hashlib
import io
import re
import subprocess
import sys
import time
import urllib.parse
import zipfile

import pandas as pd

from src.config import ROOT

PORTAL = "https://www.budget.finance.go.ug/dataportal"
DEST = ROOT / "data" / "external" / "mofped" / "portal"
KEEP = re.compile(r"(Approved\s*_?Budget_Nat|Releases(Quarterly|Q1)?_Nat|Expenditure(Quarterly|Q1)?_Nat)", re.I)
UA = {"User-Agent": "Mozilla/5.0 (academic research data client)"}


def get(url, tries=4, binary=True):
    """curl with retries (the portal drops chunked transfers intermittently); binary files must be valid xlsx (zip)."""
    for k in range(tries):
        r = subprocess.run(["curl", "-sSfL", "--retry", "3", "--max-time", "300", "-A", UA["User-Agent"], url],
                           capture_output=True)
        if r.returncode == 0 and r.stdout and (not binary or zipfile.is_zipfile(io.BytesIO(r.stdout))):   # truncated downloads are
            return r.stdout                                                                    # not valid zip archives
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"download failed: {url} ({r.stderr.decode()[:200]})")


def links(html):
    urls = re.findall(r'href="([^"]+\.xlsx?)"', html)
    seen, out = set(), []
    for u in urls:
        name = urllib.parse.unquote(u.rsplit("/", 1)[-1])
        if KEEP.search(name.replace(" ", "")) and u not in seen:
            seen.add(u)
            out.append((u, name))
    return out


def main(refresh=False):
    DEST.mkdir(parents=True, exist_ok=True)
    rows = []
    for url, name in links(get(PORTAL, binary=False).decode("utf-8", "ignore")):
        out = DEST / name.replace(" ", "_")
        if out.exists() and not refresh and zipfile.is_zipfile(out):
            data = out.read_bytes()
        else:
            data = get(url)
            out.write_bytes(data)
            time.sleep(1)
        rows.append({"file": out.name, "url": url, "bytes": len(data),
                     "sha256_16": hashlib.sha256(data).hexdigest()[:16], "fetched": dt.date.today().isoformat()})
    m = pd.DataFrame(rows).sort_values("file")
    m.to_csv(DEST / "manifest.csv", index=False)
    print(m[["file", "bytes"]].to_string(index=False))




# ---------- harmonise ----------
OUT = ROOT / "data" / "processed" / "fiscal_triangulation"
FUND = {"wagerecurrent": "wage", "wagenonrecurrent": "wage", "wage": "wage", "nonwagerecurrent": "nonwage",
        "nonwage": "nonwage", "goudevelopment": "gou_dev", "goudev": "gou_dev", "donor": "ext_fin", "extfin": "ext_fin", "externalfin": "ext_fin",
        "arrears": "arrears", "arearsrec": "arrears", "arrearsdev": "arrears", "ntr": "aia", "aia": "aia",
        "aiarec": "aia", "aiadev": "aia"}
WIDE_FUNDS = ["Wage", "NonWage", "AIARec", "Arears_Rec", "GoUDev", "ExtFin", "AIADev", "Arrears_Dev"]


def _n(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def _pick(cols, *names):
    for c in cols:
        if _n(c) in names:
            return c
    return None


def read_file(path):
    """One portal file -> long rows: fy, stage, quarter, fund, vote_code, vote_name, item_code, item_desc, amount (UGX)."""
    name = path.name
    fy = f"20{name[:2]}/{name[3:5]}"
    stage = "approved" if "Approved" in name else "released" if "Release" in name else "spent"
    best = None
    for sh in pd.ExcelFile(path).sheet_names:
        head = pd.read_excel(path, sheet_name=sh, nrows=2)
        cols = list(head.columns)
        item = _pick(cols, "itemcode")
        amt = _pick(cols, "amount", "approvedbudget")
        wide = [c for c in cols if c in WIDE_FUNDS]
        if item and (amt or len(wide) >= 4) and _pick(cols, "votecode"):
            score = (1 if amt else 0, -len(cols))         # prefer long tables
            if best is None or score > best[0]:
                best = (score, sh, amt, wide)
    if best is None:
        raise ValueError(f"no data sheet in {name}")
    _, sh, amt, wide = best
    d = pd.read_excel(path, sheet_name=sh, dtype={c: str for c in ("VoteCode", "Vote_Code", "ItemCode", "Item_Code")})
    cols = list(d.columns)
    vc, vn = _pick(cols, "votecode"), _pick(cols, "votename", "vote")
    ic, idesc = _pick(cols, "itemcode"), _pick(cols, "itemdescription", "description")
    if amt:
        fund = _pick(cols, "fundtype", "sourceoffunding")
        qcol, tcol = _pick(cols, "quarter"), _pick(cols, "typeofinfo")
        out = pd.DataFrame({"vote_code": d[vc], "vote_name": d[vn], "item_code": d[ic], "item_desc": d[idesc],
                            "fund": d[fund].map(lambda v: FUND.get(_n(v), _n(v))),
                            "amount": pd.to_numeric(d[amt], errors="coerce")})
        if stage == "approved":
            out["quarter"] = "FY"
        elif qcol is not None:
            out["quarter"] = d[qcol].astype(str).str.extract(r"(Q[1-4])", expand=False)
        else:
            out["quarter"] = d[tcol].astype(str).str.extract(r"(Q[1-4])", expand=False)
    else:
        m = d.melt(id_vars=[vc, vn, ic, idesc], value_vars=wide, var_name="fund_col", value_name="amount")
        out = pd.DataFrame({"vote_code": m[vc], "vote_name": m[vn], "item_code": m[ic], "item_desc": m[idesc],
                            "fund": m.fund_col.map(lambda v: FUND.get(_n(v), _n(v))),
                            "amount": pd.to_numeric(m.amount, errors="coerce"), "quarter": "FY"})
    out["vote_code"] = out.vote_code.astype(str).str.strip().str.zfill(3)
    out["item_code"] = out.item_code.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    out = out.dropna(subset=["amount"])
    out = out[out.amount != 0]
    g = (out.groupby(["quarter", "fund", "vote_code", "item_code"], dropna=False)
            .agg(vote_name=("vote_name", "first"), item_desc=("item_desc", "first"), amount=("amount", "sum")).reset_index())
    g.insert(0, "stage", stage)
    g.insert(0, "fy", fy)
    g["file"], g["sheet"] = name, sh
    return g


def harmonise():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = [read_file(p) for p in sorted(DEST.glob("*.xlsx"))]
    d = pd.concat(parts, ignore_index=True)
    d.to_csv(OUT / "mofped_budget_long.csv.gz", index=False)
    cov = (d.groupby(["fy", "stage"]).agg(quarters=("quarter", lambda q: ",".join(sorted(set(q)))),
                                           total_ugx_tn=("amount", lambda a: a.sum() / 1e12),
                                           construction_3121_ugx_tn=("amount", lambda a: a[d.loc[a.index, "item_code"]
                                                                     .str.startswith("3121")].sum() / 1e12))
              .reset_index())
    cov.to_csv(OUT / "mofped_coverage.csv", index=False)
    print(cov.round(2).to_string(index=False))


if __name__ == "__main__":
    if "--harmonise" not in sys.argv or "--fetch" in sys.argv:
        main(refresh="--refresh" in sys.argv)
    harmonise()
