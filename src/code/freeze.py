"""Stage 2.6 — freeze coding v1: one analysis table combining gold labels (where coded) with rule codes.

For each codebook field, gold values override rule values on gold cases; '<field>__src' records gold/rule.
Also snapshots the rule-vs-gold accuracy table that is reported as the coding caveat.
Outputs: data/processed/analysis_cases.parquet, outputs/coding/coding_v1_accuracy.csv
"""
import shutil

import pandas as pd

from src.code.gold_sheet import FIELDS
from src.config import GOLD_SHEET, GOLD_SOURCE, OUT, PROCESSED

NUMERIC = ["contract_value_ugx", "quantum_claimed", "quantum_awarded", "interest_rate_pct"]
SKIP = {"include", "exclusion_reason", "confidence", "notes"}


def main():
    cases = pd.read_parquet(PROCESSED / "cases.parquet").set_index("case_id")
    rule = pd.read_parquet(PROCESSED / "rule_codes.parquet").set_index("case_id")
    gold = pd.read_excel(GOLD_SHEET, sheet_name="coder1", dtype=str).set_index("case_id")
    gold = gold[gold.include.eq("Y") & gold.index.isin(rule.index)]
    out = rule.copy()
    for col, _ in FIELDS:
        if col in SKIP or col not in gold.columns:
            continue
        g = gold[col]
        if col in NUMERIC:
            g = pd.to_numeric(g, errors="coerce")
        if col not in out.columns:
            out[col] = None
        out[f"{col}__src"] = "rule"
        has = g.dropna().index
        out.loc[has, col] = g.loc[has]
        out.loc[has, f"{col}__src"] = "gold"
    out["in_gold"] = out.index.isin(gold.index)
    meta = ["citation", "court", "judge", "date_decided", "year", "period", "parties", "source", "ocr_conf",
            "case_number", "outcome_register", "sector", "leg_ids_cited", "statutes_cited", "precedents_cited",
            "date_filed", "dispute_causes", "arb_grounds"]
    a = cases[[m for m in meta if m in cases.columns]].join(out.drop(columns=[c for c in meta if c in out.columns], errors="ignore"))
    a["gold_source"] = GOLD_SOURCE
    a.reset_index().to_parquet(PROCESSED / "analysis_cases.parquet", index=False)
    shutil.copy(OUT / "coding" / "rule_validation.csv", OUT / "coding" / "coding_v1_accuracy.csv")
    print(f"analysis_cases: {len(a)} cases ({a.in_gold.sum()} with gold labels [{GOLD_SOURCE}])")


if __name__ == "__main__":
    main()
