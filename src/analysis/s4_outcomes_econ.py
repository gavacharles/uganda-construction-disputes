"""Analysis stage 4 (RQ4) — time to judgment, recovery ratio, interest and costs.

A. Duration: filing (exact filing date, else mid-year of the case-number year) to decision. Every case is decided,
   so there is no censoring; the bias is right truncation (recent long cases are missing). Kaplan-Meier curves and
   a Cox model, with a sensitivity run on cases filed by 2019.
B. Recovery ratio = principal awarded / principal claimed (same currency, 0 when the claim failed): fractional
   logit, bootstrap CIs (seed 42), isolation forest for anomalous awards. Sensitivity: gold-labelled cases only.
C. Interest rates and basis; costs allocation.
Output: outputs/analysis/04_outcomes_econ/
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.ensemble import IsolationForest

from src.analysis.common import C, SEED, STREAMS, Stage, load, plt

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(SEED)


def court_group(c):
    c = str(c)
    return ("appellate" if any(k in c for k in ["Supreme", "Appeal", "Constitutional"]) else
            "commercial" if "Commercial" in c else "tribunal" if any(k in c for k in ["PPDA", "CADER", "Tribunal"]) else "high court other")


def boot_ci(x, f=np.median, n=2000):
    x = np.asarray(x)
    b = [f(RNG.choice(x, len(x))) for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def main():
    s = Stage("04_outcomes_econ", "Stage 4 — Time to judgment, recovery, interest and costs (RQ4)")
    a = load()
    a["court_g"] = a.court.map(court_group)
    a["state"] = a.state_party.eq("Y").astype(int)
    a["n_causes"] = a.dispute_causes.str.len().fillna(0)

    # ---- A. duration
    fd = pd.to_datetime(a.filing_date, format="%d/%m/%Y", errors="coerce")
    yf = pd.to_numeric(a.case_number.str.extract(r"(?i)of\s*((?:19|20)\d{2})")[0], errors="coerce")
    filed = fd.fillna(pd.to_datetime(yf.astype("Int64").astype(str) + "-07-01", errors="coerce"))
    a["years"] = (a.date_decided - filed).dt.days / 365.25
    a["filed_year"] = filed.dt.year
    a["filing_basis"] = np.where(fd.notna(), "exact date", np.where(yf.notna(), "case-number year", None))
    d = a[a.years.between(0, 25)].copy()
    s.text(f"Durations available for **{len(d)}** decisions ({(d.filing_basis == 'exact date').sum()} from an exact filing date, "
           f"the rest from the case-number year, so ±6 months). Proceeding-level: an application's clock starts at the "
           "application, not the originating suit.")
    tab = d.groupby("proceeding").years.agg(n="size", median="median", p75=lambda x: x.quantile(.75)).round(2)
    tab["median 95% CI"] = [("%.2f–%.2f" % tuple(boot_ci(d[d.proceeding == p].years))) for p in tab.index]
    s.table(tab.sort_values("n", ascending=False), "t1_duration_by_proceeding", "Table 1. Years from filing to decision by type of proceeding")

    main_ = d[d.proceeding.isin(["suit", "appeal"])].copy()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    km = KaplanMeierFitter()
    for st in STREAMS:
        x = main_[main_.stream == st]
        if len(x) >= 10:
            km.fit(x.years, event_observed=np.ones(len(x)), label=f"{st} (n={len(x)})")
            km.plot_survival_function(ax=ax, ci_show=True, color=C[st])
    ax.set_xlabel("Years since filing")
    ax.set_ylabel("Share still undecided")
    ax.set_xlim(0, 15)
    s.style(ax)
    s.fig(fig, "fig1_km_suits_appeals.png", "Kaplan–Meier curves of time from filing to decision for suits and appeals, by "
          "stream (95% bands). All cases are decided (no censoring); curves describe decided cases only and understate "
          "durations of recent filings (right truncation).")

    rows = []
    for label, sub in [("all suits & appeals", main_), ("filed by 2019 (truncation check)", main_[main_.filed_year <= 2019])]:
        x = sub.assign(appeal=sub.proceeding.eq("appeal").astype(int), commercial=sub.court_g.eq("commercial").astype(int),
                       arbitration=sub.stream.ne("construction").astype(int))[["years", "state", "appeal", "commercial", "arbitration", "n_causes"]].dropna()
        x["event"] = 1
        cph = CoxPHFitter(penalizer=0.01).fit(x, "years", "event")
        r = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(3)
        r.columns = ["HR", "low", "high", "p"]
        r.insert(0, "model", f"{label} (n={len(x)})")
        rows.append(r)
    s.table(pd.concat(rows), "t2_cox_duration", "Table 2. Cox model of the rate of decision (hazard ratio > 1 = decided faster)",
            note="Covariates: state party, appeal (vs suit), Commercial Division, arbitration-related stream, number of causes in issue. "
                 "Proportional-hazards assumption to be checked before publication (cph.check_assumptions).")

    # ---- B. recovery ratio
    q = a[a.quantum_claimed.notna() & a.quantum_awarded.notna() & (a.quantum_claimed > 0)].copy()
    q = q[q.outcome.isin(["initiator succeeded", "partly succeeded", "initiator failed"])]
    q["recovery"] = (q.quantum_awarded / q.quantum_claimed).clip(0, 1)
    over = int((q.quantum_awarded > q.quantum_claimed * 1.05).sum())
    s.text(f"Recovery ratio available for **{len(q)}** decisions with a claimed and an awarded principal "
           f"({int(q.quantum_awarded__src.eq('gold').sum())} gold-labelled). {over} awards exceed the sum claimed "
           "(usually an extraction mismatch, e.g. claimed figure taken from one head of claim); ratios are capped at 1.")
    rs = q.groupby("stream").recovery.agg(n="size", mean="mean", median="median").round(2).reindex(STREAMS).dropna()
    rs["mean 95% CI"] = ["%.2f–%.2f" % tuple(boot_ci(q[q.stream == st].recovery, np.mean)) for st in rs.index]
    s.table(rs, "t3_recovery_by_stream", "Table 3. Recovery ratio (awarded ÷ claimed) by stream")
    q["log_claim"] = np.log10(q.quantum_claimed.where(q.currency.eq("UGX"), q.quantum_claimed * 3700))
    q["usd"] = q.currency.eq("USD").astype(int)
    rows = []
    for label, sub in [("all", q), ("gold-labelled only", q[q.quantum_awarded__src.eq("gold")])]:
        if len(sub) < 25:
            continue
        m = smf.glm("recovery ~ state + log_claim + n_causes + court_g", data=sub,
                    family=sm.families.Binomial()).fit(cov_type="HC1")
        r = pd.DataFrame({"OR": np.exp(m.params), "p": m.pvalues}).round(3)
        r.insert(0, "sample", f"{label} (n={len(sub)})")
        rows.append(r.drop(index="Intercept"))
    s.table(pd.concat(rows), "t4_recovery_model", "Table 4. Fractional logit of the recovery ratio",
            note="OR > 1 = higher share of the claim recovered. log_claim = log10 claimed principal (UGX; USD at 3,700). Robust (HC1) SEs.")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for st in STREAMS:
        x = q[q.stream == st]
        if len(x):
            ax.scatter(x.log_claim, x.recovery, s=16, alpha=0.7, color=C[st], label=st)
    iso = IsolationForest(random_state=SEED, contamination=0.05).fit(q[["log_claim", "recovery"]])
    q["anomaly"] = iso.predict(q[["log_claim", "recovery"]]) == -1
    ax.scatter(q[q.anomaly].log_claim, q[q.anomaly].recovery, s=60, facecolors="none", edgecolors=C["ink"], label="isolation-forest outlier")
    ax.set_xlabel("Claimed principal (log10 UGX)")
    ax.set_ylabel("Recovery ratio")
    ax.legend(frameon=False, fontsize=7.5)
    s.style(ax)
    s.fig(fig, "fig2_recovery_vs_claim.png", "Recovery ratio against size of claim. Circled points are flagged as anomalous "
          "awards by an isolation forest (5% contamination); they are candidates for manual verification.")
    s.table(q[q.anomaly][["citation", "stream", "currency", "quantum_claimed", "quantum_awarded", "recovery"]].round(3).set_index("citation"),
            "t5_anomalous_awards", "Table 5. Awards flagged as anomalous (verify against the judgment)")

    # ---- C. interest and costs
    r = a[a.interest_rate_pct.notna()]
    it = r.groupby("interest_basis").interest_rate_pct.agg(n="size", median="median", min="min", max="max").round(1)
    s.table(it, "t6_interest", f"Table 6. Interest rates awarded (% p.a.) by stated basis (n = {len(r)})",
            note="Interest basis has low agreement with the gold labels (κ≈0.3); rates themselves are extracted from the orders.")
    ct = pd.crosstab(a.outcome, a.costs_to, normalize="index").mul(100).round(1)
    s.table(ct, "t7_costs_by_outcome", "Table 7. Costs allocation by outcome (% of decisions)")
    s.write()


if __name__ == "__main__":
    main()
