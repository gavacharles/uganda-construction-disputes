"""Stage 2.4 — score the rule-based codes against the manual gold set (coder1), field by field.

Categorical fields: accuracy, Cohen's kappa, macro-F1; numeric fields: share within 5% of the gold value.
Only rows coder1 has filled are scored (the gold set can be validated as coding progresses).
Output: outputs/coding/rule_validation.csv (+ confusion tables for weak fields in rule_confusion.xlsx)
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from src.code.gold_sheet import FIELDS
from src.config import OUT, PROCESSED

NUMERIC = ["contract_value_ugx", "quantum_claimed", "quantum_awarded", "interest_rate_pct"]
SKIP = {"include", "exclusion_reason", "clauses_cited", "filing_date", "confidence", "notes"}


def main():
    gold = pd.read_excel(OUT / "coding" / "gold_coding_sheet.xlsx", sheet_name="coder1", dtype=str).set_index("case_id")
    rule = pd.read_parquet(PROCESSED / "rule_codes.parquet").set_index("case_id")
    coded = gold.drop(columns=["citation", "court", "date_decided", "parties", "pages", "pdf", "issues_as_framed"],
                      errors="ignore").notna().sum(axis=1) >= 5
    gold = gold[coded & gold.index.isin(rule.index)]
    print(f"gold rows coded: {len(gold)} of 150")
    if gold.empty:
        return
    rows, weak = [], {}
    for col, codes in FIELDS:
        if col in SKIP or col not in rule.columns:
            continue
        g = gold[col]
        r = rule.loc[gold.index, col]
        ok = g.notna()
        if col in NUMERIC:
            gv, rv = pd.to_numeric(g[ok], errors="coerce"), pd.to_numeric(r[ok], errors="coerce")
            close = ((rv - gv).abs() <= 0.05 * gv.abs().clip(lower=1)) | (gv.isna() & rv.isna())
            rows.append({"field": col, "n": int(ok.sum()), "accuracy": round(close.mean(), 2), "kappa": None, "macro_f1": None})
            continue
        gs, rs = g[ok].astype(str), r[ok].fillna("missing").astype(str)
        if len(gs) == 0:
            continue
        k = cohen_kappa_score(gs, rs) if gs.nunique() + rs.nunique() > 2 else float("nan")
        rows.append({"field": col, "n": len(gs), "accuracy": round((gs == rs).mean(), 2), "kappa": round(k, 2),
                     "macro_f1": round(f1_score(gs, rs, average="macro", zero_division=0), 2)})
        if (gs == rs).mean() < 0.8:
            weak[col[:31]] = pd.crosstab(gs.rename("gold"), rs.rename("rule"))
    v = pd.DataFrame(rows)
    v["tune"] = (v.kappa < 0.6) | (v.accuracy < 0.8)
    v.to_csv(OUT / "coding" / "rule_validation.csv", index=False)
    if weak:
        with pd.ExcelWriter(OUT / "coding" / "rule_confusion.xlsx") as xw:
            for name, t in weak.items():
                t.to_excel(xw, sheet_name=name)
    print(v.to_string(index=False))


if __name__ == "__main__":
    main()
