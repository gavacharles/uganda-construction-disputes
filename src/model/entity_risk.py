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
    # v2: add multi-source signals (court, tribunal, audit, press) and bid-level red flags
    from src.model.entity_signals import REGULATORS
    S = pd.read_parquet(INTERIM / "entity_signals.parquet").rename(columns={"entity": "buyer"})
    B = pd.read_parquet(INTERIM / "entity_bids.parquet")
    F = F.merge(S, on=["buyer", "year"], how="left").merge(B, on=["buyer", "year"], how="left")
    for c in ["court", "tribunal", "oag", "press"]:
        F[c] = F[c].fillna(0)
    F = F[~F.buyer.str.contains(REGULATORS)].sort_values(["buyer", "year"])
    nxt = F.groupby("buyer")[["court", "tribunal", "oag", "press"]].shift(-1)
    F["y_court"] = ((nxt.court + nxt.tribunal) > 0).astype(float).where(nxt.court.notna())
    F["y_oag"] = (nxt.oag > 0).astype(float).where(nxt.oag.notna())
    F["y_any"] = (nxt.sum(axis=1) > 0).astype(float).where(nxt.court.notna())
    for c in ["court", "tribunal", "oag", "press"]:
        F[f"{c}_hist"] = F.groupby("buyer")[c].cumsum()
    bidc = ["bid_procs", "mean_bidders", "single_bid_share", "bid_cv", "not_lowest_share"]
    shares = ["nonopen_share", "above_est_share", "top_supplier_share"] + bidc
    D = F.copy()
    for c in shares:
        D[c + "_missing"] = D[c].isna().astype(int)
        D[c] = D[c].fillna(D[c].median())
    D = pd.get_dummies(D, columns=["etype"], drop_first=False)
    hist = ["court", "tribunal", "oag", "press", "court_hist", "tribunal_hist", "oag_hist", "press_hist", "disputes_hist"]
    X_cols = [c for c in D.columns if c not in ("buyer", "year", "y_next", "y_court", "y_oag", "y_any", "disputes_t")]
    TARGETS = {"y_court": ("Court or PPDA Tribunal decision next year", 2023, (2024, 2025)),
               "y_oag": ("Auditor-General finding next year", 2022, (2023, 2024)),
               "y_any": ("Any dispute signal next year (court, tribunal, audit or press)", 2022, (2023, 2024))}
    res, coefs = [], {}
    def ev(target, name, p, te):
        res.append({"target": TARGETS[target][0], "model": name, "test rows": len(te), "test positives": int(te[target].sum()),
                    "ROC-AUC": round(roc_auc_score(te[target], p), 3), "PR-AUC": round(average_precision_score(te[target], p), 3),
                    "base rate": round(te[target].mean(), 3), "Brier": round(brier_score_loss(te[target], np.clip(p, 0, 1)), 4)})
    for tg, (label, tmax, (t0, t1)) in TARGETS.items():
        d = D[D[tg].notna() & D.year.between(2019, t1)]
        tr, te = d[d.year <= tmax], d[d.year.between(t0, t1)]
        if te[tg].sum() < 3 or tr[tg].sum() < 5:
            continue
        ev(tg, "Base rate", np.full(len(te), tr[tg].mean()), te)
        h = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(tr[hist].astype(float), tr[tg])
        ev(tg, "History only (all sources)", h.predict_proba(te[hist].astype(float))[:, 1], te)
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)).fit(tr[X_cols].astype(float), tr[tg])
        ev(tg, "Logistic regression (history + procurement + bids)", lr.predict_proba(te[X_cols].astype(float))[:, 1], te)
        gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=SEED).fit(tr[X_cols].astype(float), tr[tg])
        ev(tg, "Gradient boosting", gb.predict_proba(te[X_cols].astype(float))[:, 1], te)
        coefs[label] = pd.Series(lr.named_steps["logisticregression"].coef_[0], index=X_cols)
    R = pd.DataFrame(res)
    coef = pd.DataFrame(coefs).round(3)
    coef = coef.reindex(coef.abs().max(axis=1).sort_values(ascending=False).index)
    # internal ranking for 2027: any-signal model refitted on all years with known outcomes
    d = D[D.y_any.notna()]
    lr_all = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)).fit(d[X_cols].astype(float), d.y_any)
    T = D[D.year == 2026].copy()
    T["risk_2027"] = lr_all.predict_proba(T[X_cols].astype(float))[:, 1]
    T.sort_values("risk_2027", ascending=False)[["buyer", "n_works", "court_hist", "oag_hist", "press_hist", "risk_2027"]].to_csv(OUT / "model" / "entity_risk_2027.csv", index=False)
    md = [f"# Dispute early warning for procuring entities (v2: multi-source target, bid red flags)\n",
          f"Entity-years: {len(D)} ({D.buyer.nunique()} entities, regulators excluded). Positives (outcome known): "
          f"court/tribunal {int(D.y_court.sum())}, audit {int(D.y_oag.sum())}, any signal {int(D.y_any.sum())}.\n",
          "## Test performance (temporal split)\n", R.to_markdown(index=False),
          "\nPR-AUC should be read against the base rate in the same row.\n",
          "## Logistic regression coefficients by target (standardised; positive = higher risk; largest first)\n", coef.head(20).to_markdown(),
          "\n## Highest-risk entities for 2027, any signal (internal; review before any publication)\n",
          T.sort_values("risk_2027", ascending=False).head(15)[["buyer", "n_works", "court_hist", "oag_hist", "press_hist", "risk_2027"]].round(3).to_markdown(index=False),
          "\n**Caveats.** Associational, small N; press and audit mentions are matched by name and include coverage that is not a dispute "
          "involving the entity; court links are automatic until checked (outputs/review/entity_matches.xlsx). Not a finding about any entity's conduct."]
    (OUT / "model" / "entity_risk.md").write_text("\n".join(md))
    print("\n".join(md[:6]))


if __name__ == "__main__":
    main()
