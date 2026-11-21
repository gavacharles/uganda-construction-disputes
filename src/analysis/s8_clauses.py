"""Analysis stage 8 (RQ3) — clauses, contract forms and procedural requirements.

A. Clause numbers cited, mapped to topics by form family (PPDA/World Bank SBD GCC vs FIDIC 1999). Indicative:
   the same number means different things in different forms, and the family is inferred (form named, else state
   employer -> SBD).
B. Procedural requirements: contractual notices/time bars, skipped dispute tiers, statutory limitation — how often
   raised and what happens when they are (Fisher, BH).
C. Contract form and outcome: adjusted logistic model. Propensity-score matching on form (planned) is not feasible:
   FIDIC appears in ~10 decisions; reported as a limitation.
Output: outputs/analysis/08_clauses/
"""
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from src.analysis.common import C as COL, Stage, load, plt

warnings.filterwarnings("ignore")
SBD = {1: "definitions & interpretation", 7: "subcontracting", 13: "insurance", 24: "disputes", 25: "dispute procedure / adjudicator",
       26: "replacement of adjudicator", 27: "programme", 28: "extension of time", 32: "early warning", 34: "defects",
       35: "correction of defects", 40: "variations", 42: "payment certificates", 43: "payments", 44: "compensation events",
       47: "price adjustment", 48: "retention", 49: "liquidated damages", 51: "advance payment", 52: "securities",
       55: "completion", 59: "termination", 60: "payment upon termination"}
FIDIC = {1: "general provisions", 2: "the employer", 3: "the engineer", 4: "the contractor (incl. 4.2 performance security)",
         8: "commencement, delays & EOT", 10: "taking over", 11: "defects liability", 13: "variations & adjustments",
         14: "contract price & payment", 15: "termination by employer", 16: "suspension & termination by contractor",
         17: "risk & responsibility", 19: "force majeure", 20: "claims, disputes & arbitration", 63: "(1987) contractor default",
         67: "(1987) settlement of disputes", 69: "(1987) employer default"}


def family(form, state):
    if str(form).startswith("FIDIC"):
        return "FIDIC"
    if form == "PPDA SBD" or state == "Y":
        return "PPDA SBD (GCC)"
    return "other / bespoke"


def topics(clauses, fam):
    if pd.isna(clauses):
        return []
    nums = re.findall(r"(?<![\w.])(\d{1,2})(?:\.\d{1,2})*(?![\w])", re.sub(r"ACA s\.\s?\d+|O\.\d+ r\.\d+|\b(?:19|20)\d{2}\b", " ", str(clauses)))
    table = FIDIC if fam == "FIDIC" else SBD if fam.startswith("PPDA") else {}
    return sorted({table[int(n)] for n in nums if int(n) in table})


def fisher_row(label, flag, outcome):
    t = pd.crosstab(flag, outcome).reindex(index=[True, False], columns=[True, False], fill_value=0)
    orr, p = fisher_exact(t.values)
    return {"requirement": label, "raised": int(flag.sum()), "outcome when raised %": round(100 * outcome[flag].mean(), 1),
            "when not raised %": round(100 * outcome[~flag].mean(), 1), "OR": round(orr, 2), "p": p}


def main():
    s = Stage("08_clauses", "Stage 8 — Clauses, contract forms and procedural requirements (RQ3)")
    a = load()
    con = a[a.stream.isin(["construction", "both"])].copy()
    con["family"] = [family(f, st) for f, st in zip(con.contract_form, con.state_party)]
    con["topics"] = [topics(c, f) for c, f in zip(con.clauses_cited, con.family)]
    has = con.clauses_cited.notna()
    s.text(f"Construction-related decisions: **{len(con)}**; {has.sum()} ({has.mean():.0%}) cite at least one contract clause "
           f"by number. Form family: {con.family.value_counts().to_dict()}.")

    tp = con.explode("topics").dropna(subset=["topics"])
    tt = tp.groupby(["family", "topics"]).case_id.nunique().rename("decisions").reset_index()
    tt["% of family decisions citing clauses"] = [round(100 * n / max(1, (con.family.eq(f) & has).sum()), 1) for f, n in zip(tt.family, tt.decisions)]
    tt = tt.sort_values(["family", "decisions"], ascending=[True, False]).groupby("family").head(10)
    s.table(tt.set_index(["family", "topics"]), "t1_clause_topics", "Table 1. Contract clauses cited, by topic and form family (top 10 per family)",
            note="Indicative mapping of clause numbers to topics: PPDA/World Bank SBD General Conditions numbering for public works "
                 "contracts, FIDIC 1999 Red Book numbering (1987 4th edition where noted) for FIDIC contracts. Verify against the contract "
                 "documents quoted in each judgment before relying on a specific clause.")
    if len(tt):
        top = tp.groupby("topics").case_id.nunique().sort_values().tail(12)
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.barh(top.index, top.values, color=COL["construction"])
        ax.set_xlabel("Decisions citing a clause on this topic")
        ax.tick_params(axis="y", labelsize=8)
        s.style(ax, grid="x")
        s.fig(fig, "fig1_clause_topics.png", "Contract clause topics most often cited in construction-related decisions "
              "(clause numbers mapped to PPDA SBD or FIDIC topics; indicative).")

    # ---- B. procedural requirements
    ok = con.outcome.notna()
    d = con[ok]
    fail = d.outcome.isin(["initiator failed", "struck out (procedural)"])
    referred = d.outcome.eq("referred to arbitration") | d.outcome.eq("struck out (procedural)")
    rows = [fisher_row("dispute tier skipped (engineer/adjudicator/amicable settlement) — initiator failed or struck out",
                       d.tier_skipped_raised.eq("Y"), fail),
            fisher_row("dispute tier skipped — case referred to arbitration or struck out",
                       d.tier_skipped_raised.eq("Y"), referred),
            fisher_row("statutory limitation raised — initiator failed or struck out", d.limitation_raised.eq("Y"), fail)]
    pr = pd.DataFrame(rows).set_index("requirement")
    pr["p_BH"] = multipletests(pr.p, method="fdr_bh")[1].round(3)
    pr["p"] = pr.p.round(3)
    s.table(pr, "t2_procedural_requirements", "Table 2. Procedural requirements raised and their association with outcomes",
            note=f"Contractual notice / time-bar defences were raised in only {int(con.notice_timebar_raised.eq('Y').sum())} of {len(con)} "
                 "construction-related decisions (upheld in " + str(int(con.notice_timebar_upheld.eq('Y').sum())) + "): unlike in common-law "
                 "jurisdictions using FIDIC 1999/2017 sub-clause 20.1, notice-based time bars are almost never litigated in Uganda.")
    tier = d[d.tier_skipped_raised.eq("Y")]
    cons = tier.groupby("year").outcome.apply(lambda x: x.isin(["referred to arbitration", "struck out (procedural)"]).mean())
    s.text(f"**Consistency.** Where a skipped dispute tier was argued (n = {len(tier)}), courts enforced the tier (referral or "
           f"striking out) in {100 * tier.outcome.isin(['referred to arbitration', 'struck out (procedural)']).mean():.0f}% of decisions; "
           f"the annual share ranged {100 * cons.min():.0f}–{100 * cons.max():.0f}% (years with ≥1 such decision).")

    # ---- C. form and outcome
    m = d[d.outcome.isin(["initiator succeeded", "partly succeeded", "initiator failed", "struck out (procedural)"])].copy()
    m["success"] = m.outcome.isin(["initiator succeeded", "partly succeeded"]).astype(int)
    m["form_g"] = np.select([m.contract_form.str.startswith("FIDIC") | m.contract_form.isin(["PPDA SBD", "JBC or PAM"]),
                             m.contract_form.eq("bespoke written")], ["standard form", "bespoke written"], "oral / not stated")
    m["state"] = m.state_party.eq("Y").astype(int)
    m["suit"] = m.proceeding.isin(["suit", "appeal"]).astype(int)
    ct = m.groupby("form_g").agg(n=("success", "size"), success=("success", "mean"))
    ct["success"] = (100 * ct.success).round(1)
    fit = smf.logit("success ~ C(form_g, Treatment('bespoke written')) + state + suit", data=m).fit(disp=0)
    r = pd.DataFrame({"OR": np.exp(fit.params), "low": np.exp(fit.conf_int()[0]), "high": np.exp(fit.conf_int()[1]), "p": fit.pvalues}).round(3)
    r.index = [i.replace("C(form_g, Treatment('bespoke written'))[T.", "form: ").rstrip("]") for i in r.index]
    s.table(ct, "t3_form_success", "Table 3. Initiator success by contract form group (%)")
    s.table(r.drop(index="Intercept"), "t4_form_model", "Table 4. Logistic model of initiator success on contract form (reference: bespoke written)",
            note=f"FIDIC appears in {int(con.contract_form.str.startswith('FIDIC').sum())} decisions, so the planned propensity-score matching "
                 "on FIDIC vs PPDA SBD vs bespoke is not feasible; standard forms are pooled and the model adjusts for state party and "
                 "proceeding type only. Contract form agreement with the gold labels is low (κ≈0.3): treat as exploratory.")
    s.write()


if __name__ == "__main__":
    main()
