"""Stage 2.2 — inter-coder agreement (Cohen's kappa) per codebook field on the coder2 subset.

Reads outputs/coding/gold_coding_sheet.xlsx; writes outputs/coding/kappa.csv. Fields with kappa < 0.6 are flagged.
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.code.gold_sheet import FIELDS
from src.config import GOLD_SHEET, GOLD_SOURCE, OUT


def main():
    fp = GOLD_SHEET
    c1 = pd.read_excel(fp, sheet_name="coder1", dtype=str).set_index("case_id")
    c2 = pd.read_excel(fp, sheet_name="coder2", dtype=str).set_index("case_id")
    rows = []
    for col, codes in FIELDS:
        if not codes or col in ("confidence",):
            continue
        both = pd.concat([c1[col], c2[col]], axis=1, keys=["a", "b"]).loc[c2.index].dropna()
        if len(both) < 5:
            rows.append({"field": col, "n": len(both), "kappa": None, "agreement": None})
            continue
        k = cohen_kappa_score(both.a, both.b) if both.a.nunique() + both.b.nunique() > 2 else float("nan")
        rows.append({"field": col, "n": len(both), "kappa": round(k, 2), "agreement": round((both.a == both.b).mean(), 2)})
    r = pd.DataFrame(rows)
    r["review"] = r.kappa < 0.6
    r["gold_source"] = GOLD_SOURCE  # AI vs AI: self-consistency, not human inter-coder reliability
    r.to_csv(OUT / "coding" / "kappa.csv", index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
