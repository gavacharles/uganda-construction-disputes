"""Analysis stage 7 (RQ1, RQ4) — can the outcome be predicted from the facts, issues and case features?

Target: initiator success (succeeded / partly) vs failure (failed / struck out). Inputs exclude the reasoning and
orders to avoid leakage: TF-IDF of facts + framed issues, plus metadata (stream, proceeding, forum, party types,
state party, coded causes). Models: linear SVM (baseline), logistic regression, naive Bayes (text), random forest,
XGBoost. Nested CV (outer 5-fold, inner 3-fold grid search), seed 42; macro-F1 and ROC-AUC. SHAP on XGBoost.
Output: outputs/analysis/07_classifier/
"""
import re
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.analysis.common import C, CAUSE_LABEL, CAUSES, SEED, Stage, load, plt
from src.code.rule_coder import load_texts
from src.code.segment import segment

warnings.filterwarnings("ignore")
META = ["stream", "proceeding", "court_g", "initiator_type", "respondent_type", "state_party"]


def court_group(c):
    c = str(c)
    return ("appellate" if any(k in c for k in ["Supreme", "Appeal"]) else "commercial" if "Commercial" in c else
            "tribunal" if any(k in c for k in ["PPDA", "CADER", "Tribunal"]) else "high court other")


def main():
    s = Stage("07_classifier", "Stage 7 — Predicting outcomes from facts, issues and case features")
    a = load()
    d = a[a.outcome.isin(["initiator succeeded", "partly succeeded", "initiator failed", "struck out (procedural)"])].copy()
    d["y"] = d.outcome.isin(["initiator succeeded", "partly succeeded"]).astype(int)
    d["court_g"] = d.court.map(court_group)
    for c in META:
        d[c] = d[c].fillna("missing").astype(str)
    for c in CAUSES:
        d[c] = d[c].fillna("N")
    texts = load_texts(a)
    docs = []
    for cid in d.case_id:
        seg, issues, _ = segment(texts[cid])
        t = " ".join([seg["facts"][:15000], seg["issues"], " ".join(issues)])
        docs.append(re.sub(r"[^a-z\s]", " ", t.lower()))
    d["text"] = docs
    X, y = d[["text"] + META + CAUSES], d.y.values
    s.text(f"Decisions with a merits/procedural outcome: **{len(d)}** (initiator succeeded in {y.mean():.0%}). "
           "Inputs: facts and framed issues only (no reasoning or orders) plus case features.")

    tfidf = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS), ngram_range=(1, 2), min_df=5, max_df=0.6, max_features=4000, sublinear_tf=True)
    meta = OneHotEncoder(handle_unknown="ignore", min_frequency=5)
    full = ColumnTransformer([("txt", tfidf, "text"), ("meta", meta, META + CAUSES)])
    metaonly = ColumnTransformer([("meta", OneHotEncoder(handle_unknown="ignore", min_frequency=5), META + CAUSES)])
    textonly = ColumnTransformer([("txt", tfidf, "text")])
    models = {
        "Linear SVM (baseline)": (full, LinearSVC(class_weight="balanced", random_state=SEED), {"clf__C": [0.01, 0.1, 1]}),
        "Logistic regression": (full, LogisticRegression(class_weight="balanced", max_iter=3000), {"clf__C": [0.1, 1, 10]}),
        "Naive Bayes (text)": (textonly, ComplementNB(), {"clf__alpha": [0.1, 0.5, 1.0]}),
        "Random forest": (full, RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=SEED, n_jobs=-1),
                          {"clf__max_depth": [None, 10]}),
        "XGBoost": (full, XGBClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                                        random_state=SEED, n_jobs=4, eval_metric="logloss"), {"clf__max_depth": [2, 4]}),
        "Logistic regression (case features only)": (metaonly, LogisticRegression(class_weight="balanced", max_iter=3000), {"clf__C": [0.1, 1]}),
    }
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    rows = []
    for name, (prep, clf, grid) in models.items():
        gs = GridSearchCV(Pipeline([("prep", prep), ("clf", clf)]), grid, cv=inner, scoring="f1_macro")
        r = cross_validate(gs, X, y, cv=outer, scoring=["f1_macro", "roc_auc", "accuracy"])
        rows.append({"model": name, "macro-F1": r["test_f1_macro"].mean(), "macro-F1 sd": r["test_f1_macro"].std(),
                     "ROC-AUC": r["test_roc_auc"].mean(), "AUC sd": r["test_roc_auc"].std(), "accuracy": r["test_accuracy"].mean()})
    res = pd.DataFrame(rows).set_index("model").round(3)
    s.table(res, "t1_nested_cv", "Table 1. Nested cross-validated performance (outer 5-fold; mean and SD)",
            note=f"Majority-class accuracy = {max(y.mean(), 1 - y.mean()):.2f}; chance AUC = 0.50.")

    # interpretable view: logistic regression coefficients (case features) and SHAP for XGBoost
    lr = Pipeline([("prep", metaonly), ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, C=1))]).fit(X, y)
    names = lr.named_steps["prep"].get_feature_names_out()
    coef = pd.Series(lr.named_steps["clf"].coef_[0], index=[n.replace("meta__", "") for n in names])
    coef = coef[~coef.index.str.endswith("_N")]
    coef.index = [next((CAUSE_LABEL[c] + " in issue" for c in CAUSES if i == f"{c}_Y"), i) for i in coef.index]
    top = pd.concat([coef.nlargest(8), coef.nsmallest(8)]).round(2).rename("log-odds").to_frame()
    s.table(top, "t2_lr_coefficients", "Table 2. Case features most associated with initiator success (logistic regression, full data)",
            note="Positive = higher odds the initiator succeeds. Descriptive associations, not causal effects.")

    xgb = Pipeline([("prep", full), ("clf", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                                                           colsample_bytree=0.5, random_state=SEED, n_jobs=4))]).fit(X, y)
    Xt = xgb.named_steps["prep"].transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    fn = [n.replace("txt__", "").replace("meta__", "") for n in xgb.named_steps["prep"].get_feature_names_out()]
    sv = shap.TreeExplainer(xgb.named_steps["clf"]).shap_values(Xt)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=fn).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.barh(imp.index[::-1], imp.values[::-1], color=C["accent"])
    ax.set_xlabel("mean |SHAP value| (log-odds)")
    ax.tick_params(axis="y", labelsize=7.5)
    s.style(ax, grid="x")
    s.fig(fig, "fig1_shap_importance.png", "Twenty most influential inputs in the XGBoost outcome model (mean absolute SHAP value). "
          "Words come from the facts and framed issues; 'X_Y' denotes a coded case feature.")
    s.text("**Reading.** Outcome prediction from pre-decision information is modest; the value of the model is in which "
           "features carry signal (Table 2, Figure 1), not in forecasting individual cases. Report judge effects only in aggregate.")
    s.write()


if __name__ == "__main__":
    main()
