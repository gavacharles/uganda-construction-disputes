"""Analysis stage 5 (RQ6) — how Ugandan courts treat arbitration.

Application types and outcomes; set-aside grounds raised and their success (Fisher, BH-corrected); court
deference by application type, forum, state party and period (logistic); time to decision; underlying dispute
(construction 'both' vs other arbitration).
Output: outputs/analysis/05_arbitration/
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from src.analysis.common import C, Stage, load, plt

warnings.filterwarnings("ignore")
GROUNDS = {"g_public_policy": "Public policy", "g_jurisdiction": "Excess of jurisdiction", "g_natural_justice": "Natural justice",
           "g_out_of_time": "Out of time", "g_arbitrability": "Arbitrability", "g_invalid_agreement": "Invalid agreement"}
WIN = {"set-aside": ["set aside", "partly set aside", "remitted"], "enforcement or recognition": ["enforced"]}


def main():
    s = Stage("05_arbitration", "Stage 5 — The arbitration interface (RQ6)")
    a = load()
    arb = a[a.stream.isin(["arbitration", "both"])].copy()
    arb["underlying"] = np.where(arb.stream == "both", "construction", "other commercial")
    arb["success"] = (arb.outcome.isin(["initiator succeeded", "partly succeeded"])
                      | (arb.arb_application_type.eq("stay or referral") & arb.outcome.eq("referred to arbitration"))).astype(int)
    s.text(f"Arbitration-related decisions: **{len(arb)}** ({(arb.stream == 'both').sum()} arising from construction contracts). "
           "Application type agreement with the gold labels: κ≈0.5; treat sub-type shares as indicative.")

    t = arb.groupby("arb_application_type").agg(n=("case_id", "size"), initiator_success=("success", "mean"),
                                                 referred=("outcome", lambda x: (x == "referred to arbitration").mean()),
                                                 construction_share=("underlying", lambda x: (x == "construction").mean()))
    t[["initiator_success", "referred", "construction_share"]] = (100 * t[["initiator_success", "referred", "construction_share"]]).round(1)
    s.table(t.sort_values("n", ascending=False), "t1_application_types", "Table 1. Arbitration applications: type, success and underlying dispute (%)",
            note="Initiator success = applicant obtained what it sought (e.g. award set aside for a set-aside application; stay granted for a stay application).")

    sa = arb[arb.arb_application_type == "set-aside"].copy()
    sa["award_set_aside"] = sa.award_result.isin(WIN["set-aside"]).astype(int)
    rows = []
    for g, lab in GROUNDS.items():
        raised = sa[g] == "Y"
        if raised.sum() < 3:
            rows.append({"ground": lab, "raised": int(raised.sum())})
            continue
        tab = pd.crosstab(raised, sa.award_set_aside).reindex(index=[True, False], columns=[1, 0], fill_value=0)
        orr, p = fisher_exact(tab.values)
        rows.append({"ground": lab, "raised": int(raised.sum()), "raised %": round(100 * raised.mean(), 1),
                     "set aside when raised %": round(100 * sa[raised].award_set_aside.mean(), 1),
                     "set aside when not raised %": round(100 * sa[~raised].award_set_aside.mean(), 1), "OR": round(orr, 2), "p": p})
    g = pd.DataFrame(rows).set_index("ground")
    ok = g.p.notna()
    g.loc[ok, "p_BH"] = multipletests(g.loc[ok, "p"], method="fdr_bh")[1].round(3)
    g["p"] = g.p.round(3)
    s.table(g, "t2_setaside_grounds", f"Table 2. Grounds raised in set-aside applications and their success (n = {len(sa)}; "
            f"awards set aside or remitted in {100 * sa.award_set_aside.mean():.0f}%)",
            note="Grounds raised (Y/N) are coded from the framed issues and reasoning; several may be raised. Fisher exact tests, BH-adjusted.")

    dfr = arb[arb.deference.isin(["Y", "N"])].copy()
    dfr["defer"] = dfr.deference.eq("Y").astype(int)
    dfr["commercial"] = dfr.court.str.contains("Commercial", na=False).astype(int)
    dfr["state"] = dfr.state_party.eq("Y").astype(int)
    dfr["post2024"] = (dfr.date_decided >= "2024-12-23").astype(int)
    dfr["construction"] = (dfr.underlying == "construction").astype(int)
    m = smf.logit("defer ~ construction + state + commercial + post2024 + I(year - 2020)", data=dfr).fit(disp=0)
    r = pd.DataFrame({"OR": np.exp(m.params), "low": np.exp(m.conf_int()[0]), "high": np.exp(m.conf_int()[1]), "p": m.pvalues}).round(3)
    s.table(r.drop(index="Intercept"), "t3_deference_model", f"Table 3. Logistic model of court deference to arbitration (n = {len(dfr)}; "
            f"deferred in {100 * dfr.defer.mean():.0f}%)",
            note="Deference = court referred, stayed, upheld or enforced (Y) vs set aside, refused a stay or refused enforcement (N). "
                 "post2024 = decided after the Arbitration and Conciliation (Amendment) Act 2024; year trend included.")

    yr = dfr.groupby("year").defer.agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(yr.index, 100 * yr["mean"], color=C["arbitration"], width=0.7)
    for x, (mv, n) in yr.iterrows():
        ax.text(x, 100 * mv + 1.5, f"n={int(n)}", ha="center", fontsize=7, color=C["ink2"])
    ax.axvline(2024.98, color=C["ink2"], ls=":", lw=0.9)
    ax.text(2024.98, 104, " Amendment Act 2024", fontsize=7, color=C["ink2"])
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of decisions deferring to arbitration")
    ax.set_xticks(yr.index)
    s.style(ax)
    s.fig(fig, "fig1_deference_by_year.png", "Share of arbitration-related decisions in which the court deferred to the arbitral "
          "process (referral, stay, award upheld or enforced), by year; n = decisions with a coded deference outcome.")

    yf = pd.to_numeric(arb.case_number.str.extract(r"(?i)of\s*((?:19|20)\d{2})")[0], errors="coerce")
    arb["years"] = (arb.year - yf).where(lambda x: x.between(0, 15))
    du = arb.groupby("arb_application_type").years.agg(n="count", median="median", mean="mean").round(2)
    s.table(du[du.n >= 5], "t4_time_to_decision", "Table 4. Years from filing of the application to decision (case-number year to decision year)")
    und = pd.crosstab(arb.underlying, arb.outcome, normalize="index").mul(100).round(1)
    s.table(und, "t5_outcome_by_underlying", "Table 5. Outcomes of arbitration-related decisions by underlying dispute (%)")
    s.write()


if __name__ == "__main__":
    main()
