"""Model stage 13 — dispute early warning for procuring entities (entity-year), from public procurement data.

Unit: procuring entity (OCDS buyer) x year t, 2019–2025. Target: the entity is a named party in at least one
construction-related or arbitration decision in year t+1 (tax/banking-only matters excluded). Features from the
entity's OCDS works contracts in year t (count, value, non-open share, above-estimate share, supplier concentration,
entity type) and its dispute history up to t. Models: base rate, a history-only baseline, penalised logistic
regression and gradient boosting; temporal validation (train outcomes up to 2024, test 2025–2026); metrics ROC-AUC,
PR-AUC, Brier. Associational, small N; the ranking for 2027 stays internal until reviewed.
Outputs: outputs/model/entity_risk.md, outputs/model/entity_risk_2027.csv (internal)
"""
import re
import warnings

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import INTERIM, OUT, PROCESSED, SEED
from src.analysis.common import CAUSES
from src.model.link_contracts import EXTRA_NOT_WORKS, core, sides

# dissolved entities and their successors (UNRA (Repeal) Act 2024, effective 23 Dec 2024)
SUCCESSOR = {"Uganda National Roads Authority": ("Ministry of Works and Transport", 2025)}

warnings.filterwarnings("ignore")


def etype(name):
    n = str(name).lower()
    return ("local government" if re.search(r"district|municipal|city council|town council|local government", n) else
            "ministry" if "ministry" in n else "university" if re.search(r"universit|college|institute", n) else
            "authority or agency" if re.search(r"authority|agency|commission|board|bureau|council|fund", n) else "other public body")


def main():
    (OUT / "model").mkdir(parents=True, exist_ok=True)
    k = pd.read_parquet(INTERIM / "ocds_contracts.parquet")
    k = k[k.award_date.notna() & k.year.between(2019, 2026)]
    w = k[k.works & ~k.title.str.contains(EXTRA_NOT_WORKS)].copy()
    w["open"] = w.method.eq("open")
    w["above_est"] = (w.value_ugx > 1.05 * w.estimate_ugx) & w.estimate_ugx.gt(0)
    w["sup"] = w.suppliers.map(lambda s: s[0] if len(s) else None)
    feats = []
    for (b, y), g in w.groupby(["buyer", "year"]):
        v = g.value_ugx.fillna(0)
        top = g.groupby("sup").value_ugx.sum().max() / v.sum() if v.sum() > 0 else np.nan
        feats.append({"buyer": b, "year": int(y), "n_works": len(g), "log_value": np.log10(v.sum() + 1), "nonopen_share": 1 - g.open.mean(),
                      "above_est_share": g.above_est.mean(), "top_supplier_share": top, "n_suppliers": g.sup.nunique()})
    F = pd.DataFrame(feats)
    buyers = sorted(k.buyer.dropna().unique())
    grid = pd.MultiIndex.from_product([buyers, range(2019, 2027)], names=["buyer", "year"]).to_frame(index=False)
    F = grid.merge(F, on=["buyer", "year"], how="left").fillna({"n_works": 0, "log_value": 0, "n_suppliers": 0})
    F["etype"] = F.buyer.map(etype)

    # disputes: decisions naming the entity (construction-related or arbitration streams)
    a = pd.read_parquet(PROCESSED / "analysis_cases.parquet")
    bc = {b: core(b) for b in buyers}
    hits = []
    # target excludes tax and other non-procurement matters (adjacent-risk decisions)
    a = a[~a.court.str.contains("Tax", na=False)]
    for r in a.itertuples():
        if re.search(r"revenue authority", str(r.parties), re.I) and not any(getattr(r, c) == "Y" for c in CAUSES):
            continue
        for sd in sides(r.parties):
            c = core(sd)
            if len(c) < 6 or "attorney general" in c:
                continue
            best = max(buyers, key=lambda b: fuzz.token_set_ratio(c, bc[b]))
            if fuzz.token_set_ratio(c, bc[best]) >= 92 and abs(len(c) - len(bc[best])) < 40:
                hits.append({"buyer": best, "year": int(r.year), "case_id": r.case_id})
    H = pd.DataFrame(hits).drop_duplicates(["buyer", "case_id"])
    for old, (new, y0) in SUCCESSOR.items():   # disputes after dissolution count against the successor
        H.loc[(H.buyer == old) & (H.year >= y0), "buyer"] = new
    cnt = H.groupby(["buyer", "year"]).size()
    F["disputes_t"] = [cnt.get((b, y), 0) for b, y in zip(F.buyer, F.year)]
    F["disputes_hist"] = [sum(cnt.get((b, yy), 0) for yy in range(2015, y + 1)) for b, y in zip(F.buyer, F.year)]
    for old, (new, y0) in SUCCESSOR.items():   # successor inherits the predecessor's history from dissolution
        inh = sum(cnt.get((old, yy), 0) for yy in range(2015, y0))
        F.loc[(F.buyer == new) & (F.year >= y0), "disputes_hist"] += inh
        F = F[~((F.buyer == old) & (F.year >= y0))]
    F["y_next"] = [int(cnt.get((b, y + 1), 0) > 0) for b, y in zip(F.buyer, F.year)]
    D = F[F.year <= 2025].copy()
    for c in ["nonopen_share", "above_est_share", "top_supplier_share"]:
        D[c + "_missing"] = D[c].isna().astype(int)
        D[c] = D[c].fillna(D[c].median())
    D = pd.get_dummies(D, columns=["etype"], drop_first=False)
    X_cols = [c for c in D.columns if c not in ("buyer", "year", "y_next")]
    tr, te = D[D.year <= 2023], D[D.year >= 2024]   # outcomes: train <= 2024, test 2025–2026

    res = []
    def score(name, p):
        res.append({"model": name, "ROC-AUC": round(roc_auc_score(te.y_next, p), 3), "PR-AUC": round(average_precision_score(te.y_next, p), 3),
                    "Brier": round(brier_score_loss(te.y_next, np.clip(p, 0, 1)), 4)})
    score("Base rate (no information)", np.full(len(te), tr.y_next.mean()))
    hist_cols = ["disputes_t", "disputes_hist"]
    lr_h = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr[hist_cols], tr.y_next)
    score("Dispute history only", lr_h.predict_proba(te[hist_cols])[:, 1])
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=4000, class_weight=None)).fit(tr[X_cols].astype(float), tr.y_next)
    score("Logistic regression (procurement + history)", lr.predict_proba(te[X_cols].astype(float))[:, 1])
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=SEED).fit(tr[X_cols].astype(float), tr.y_next)
    score("Gradient boosting (procurement + history)", gb.predict_proba(te[X_cols].astype(float))[:, 1])
    R = pd.DataFrame(res).set_index("model")

    coef = pd.Series(lr.named_steps["logisticregression"].coef_[0], index=X_cols).sort_values()
    # ranking for 2027 from 2026 features (refit on all years)
    lr_all = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=4000)).fit(D[X_cols].astype(float), D.y_next)
    T = F[F.year == 2026].copy()
    for c in ["nonopen_share", "above_est_share", "top_supplier_share"]:
        T[c + "_missing"] = T[c].isna().astype(int)
        T[c] = T[c].fillna(D[c].median())
    T = pd.get_dummies(T, columns=["etype"], drop_first=False).reindex(columns=D.columns, fill_value=0)
    T["risk_2027"] = lr_all.predict_proba(T[X_cols].astype(float))[:, 1]
    T.sort_values("risk_2027", ascending=False)[["buyer", "n_works", "log_value", "disputes_hist", "risk_2027"]].to_csv(OUT / "model" / "entity_risk_2027.csv", index=False)

    md = [f"# Dispute early warning for procuring entities (entity-year model)\n",
          f"Entity-years: {len(D)} ({D.buyer.nunique()} entities, 2019–2025); positives (entity named in a decision the next year): "
          f"{int(D.y_next.sum())} ({100 * D.y_next.mean():.1f}%). Train: outcomes up to 2024 ({len(tr)} rows, {int(tr.y_next.sum())} positives); "
          f"test: outcomes 2025–2026 ({len(te)} rows, {int(te.y_next.sum())} positives).\n",
          "## Test performance\n", R.to_markdown(),
          "\nPR-AUC should be read against the test base rate of " + f"{te.y_next.mean():.3f}.\n",
          "## Logistic regression coefficients (standardised; positive = higher risk)\n", coef.round(3).to_frame("coef").to_markdown(),
          "\n## Highest-risk entities for 2027 (internal; review before any publication)\n",
          T.sort_values("risk_2027", ascending=False).head(15)[["buyer", "n_works", "disputes_hist", "risk_2027"]].round(3).to_markdown(index=False),
          "\n**Caveats.** Associational, small N, litigated disputes only (a filtered sample); entity names matched automatically; "
          "OCDS coverage is incomplete before 2019 and for local governments. Not a finding about any entity's conduct."]
    (OUT / "model" / "entity_risk.md").write_text("\n".join(md))
    print("\n".join(md[:5]))


if __name__ == "__main__":
    main()
