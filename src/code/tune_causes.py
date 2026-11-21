"""Stage 2.5 — choose per-cause thresholds on the gold TUNING split; the coder2 cases are held out.

Rule for each cause: Y if found in the framed issues (optional) or >= k mentions in the body.
Picks (issues on/off, k in 1..10) maximising Cohen's kappa on gold rows not in the coder2 subset.
Output: outputs/coding/cause_thresholds.json (read by src.code.rule_coder). Re-run the coder afterwards.
"""
import json
import warnings

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.code.rule_coder import CAUSES
from src.config import GOLD_SHEET, OUT, PROCESSED


def main():
    warnings.filterwarnings("ignore")  # rare causes: single-label kappa warnings
    g1 = pd.read_excel(GOLD_SHEET, sheet_name="coder1", dtype=str).set_index("case_id")
    hold = set(pd.read_excel(GOLD_SHEET, sheet_name="coder2", dtype=str).case_id)
    rc = pd.read_parquet(PROCESSED / "rule_codes.parquet").set_index("case_id")
    g1 = g1[g1.include.eq("Y") & g1.index.isin(rc.index)]
    tune = g1[~g1.index.isin(hold)]
    out, rows = {}, []
    for c in CAUSES:
        y = tune[c].eq("Y")
        ni, nb = rc.loc[tune.index, f"n_issues:{c}"], rc.loc[tune.index, f"n_body:{c}"]
        best = max(((cohen_kappa_score(y, (ni.gt(0) & iss) | nb.ge(k)), iss, k) for iss in (True, False) for k in range(1, 11)),
                   key=lambda t: t[0])
        out[c] = {"issues": bool(best[1]), "body": int(best[2])}
        rows.append({"cause": c, "kappa_tune": round(best[0], 2), **out[c], "gold_Y_share": round(y.mean(), 2)})
    (OUT / "coding" / "cause_thresholds.json").write_text(json.dumps(out, indent=1))
    print(f"tuning rows: {len(tune)} | held out (coder2): {len(g1) - len(tune)}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
