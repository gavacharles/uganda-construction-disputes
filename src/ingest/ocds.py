"""Flatten Uganda's PPDA OCDS bulk file (Open Contracting Data Registry, publication 130) into a contracts table.

One row per awarded contract: ocid, buyer, title, procurement method, tender estimate, award value and date,
supplier(s). Works contracts are flagged from the title (the feed has no procurement category).
Source file (24 MB, git-ignored): data/external/ocds/uganda_ppda_full.jsonl.gz
Output: data/interim/ocds_contracts.parquet
"""
import gzip
import json
import re

import pandas as pd

from src.config import INTERIM, ROOT

SRC = ROOT / "data" / "external" / "ocds" / "uganda_ppda_full.jsonl.gz"
FX = {"UGX": 1.0, None: 1.0, "USD": 3700.0, "EUR": 4000.0, "GBP": 4700.0, "KES": 28.0}
WORKS = re.compile(r"\b(construct|rehabilitat|renovat|refurbish|upgrad|civil works?|works? (?:on|at|for|of)|road|bridge|culvert|"
                   r"borehole|drainage|tarmac|bitumen|paving|building|block|classroom|latrine|toilet|dormitor|staff houses?|"
                   r"fenc|water supply|piped water|sewer|market|health cent|maintenance of .* (?:road|building)|periodic maintenance|"
                   r"routine maintenance|extension of|completion of|erection|expansion of)", re.I)
NOT_WORKS = re.compile(r"\b(consultan|supervision|design review|feasibility|supply of|supply and deliver|procurement of|"
                       r"provision of (?:security|cleaning|catering)|hire of|insurance|stationery|fuel|airtime|training)\b", re.I)


def val(v):
    if not v or v.get("amount") is None:
        return None
    r = FX.get(v.get("currency"))
    return float(v["amount"]) * r if r else None


def main():
    rows = []
    with gzip.open(SRC, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            t, buyer = r.get("tender") or {}, r.get("buyer") or {}
            awards = {a.get("id"): a for a in r.get("awards") or []}
            for c in r.get("contracts") or []:
                a = awards.get(c.get("awardID"), {})
                title = c.get("title") or t.get("title") or ""
                rows.append({"ocid": r["ocid"], "contract_id": c.get("id"), "buyer_id": buyer.get("id"), "buyer": buyer.get("name"),
                             "title": title, "method": t.get("procurementMethod"), "estimate_ugx": val(t.get("value")),
                             "value_ugx": val(c.get("value")) or val(a.get("value")), "award_date": a.get("date") or r.get("date"),
                             "suppliers": [s.get("name") for s in a.get("suppliers") or [] if s.get("name")],
                             "tender_start": (t.get("tenderPeriod") or {}).get("startDate"), "status": c.get("status")})
    df = pd.DataFrame(rows).drop_duplicates("contract_id")
    df["award_date"] = pd.to_datetime(df.award_date, errors="coerce", utc=True).dt.tz_localize(None)
    df["year"] = df.award_date.dt.year
    df["works"] = df.title.str.contains(WORKS) & ~df.title.str.contains(NOT_WORKS)
    df.to_parquet(INTERIM / "ocds_contracts.parquet", index=False)
    w = df[df.works]
    print(f"contracts: {len(df)}; works (by title): {len(w)} ({w.value_ugx.sum() / 1e12:.1f} trillion UGX); "
          f"years {int(df.year.min())}–{int(df.year.max())}; buyers {df.buyer.nunique()}; works suppliers {w.suppliers.explode().nunique()}")
    print(w.year.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
