"""Analysis stage 9 (RQ9) — cause -> clause -> cost chain and reform priorities.

Per dispute cause (construction-related decisions): prevalence, trend (Stage 3), claim size, recovery shortfall,
time to decision, initiator failure, procedural friction (dispute tier skipped, limitation), state involvement and
legislative coverage (register enactments targeting the cause, and whether courts cite them). A transparent priority
score (mean of min-max scaled indicators) ranks causes; rank stability is checked under alternative weights
(Kendall's tau). Each priority cause is linked to the PPDA SBD (GCC) and FIDIC 2017 clauses that govern it and to a
candidate reform. Clause anchors and reforms are drafted proposals to verify against the forms and validate by Delphi.
Output: outputs/analysis/09_reform/
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.analysis.common import C, CAUSE_LABEL, CAUSES, OUT, SEQ_SCORE, Stage, load, plt

warnings.filterwarnings("ignore")
USD_UGX = 3700
# enactments in the legislation register whose subject matter targets the cause (by register area / title)
LEG_TARGET = {"c_procurement": ["LEG017", "LEG019", "LEG001", "LEG009"], "c_site_land": ["LEG012", "LEG022", "LEG031"],
              "c_defects": ["LEG005", "LEG010", "LEG011", "LEG014", "LEG026"], "c_prof_fees": ["LEG007", "LEG008", "LEG027"],
              "c_payment": [], "c_guarantees": [], "c_termination": [], "c_delay_eot": [], "c_lds": [], "c_variations": [],
              "c_final_account": [], "c_price_adjust": [], "c_force_majeure": [], "c_fraud": ["LEG017"]}
LEVER = {  # (PPDA SBD GCC / FIDIC 2017 anchor, candidate reform) — drafted proposals, to verify and validate (Delphi)
    "c_payment": ("SBD GCC 42–43 (payment certificates, payments); FIDIC 2017 cl. 14.6–14.8",
                  "Fixed certification and payment periods with deemed certification on default; automatic financing charges on late "
                  "payment at a stated rate; consider statutory security-of-payment adjudication for public works."),
    "c_guarantees": ("SBD GCC 51–52 (advance payment, securities); FIDIC 2017 cl. 4.2, 14.2",
                     "Prescribed bond wording (URDG 758) with a notice-and-cure period before any call and reasons stated in the demand; "
                     "release dates tied to taking-over and reduction of advance-payment security as it is recovered."),
    "c_procurement": ("PPDA Act ss. 89–91 (administrative review) and Regulations 2023; SBD ITB",
                      "Standstill period and written reasons before award; publish evaluation reports; tighter review timelines so "
                      "challenges are resolved before contract signature."),
    "c_fraud": ("SBD fraud & corruption provisions; FIDIC 2017 cl. 15.6",
                "Integrity pact and audit rights in the SBD; explicit consequences and evidential standard for termination for corruption."),
    "c_termination": ("SBD GCC 59–60 (termination, payment on termination); FIDIC 2017 cl. 15–16",
                      "Closed list of fundamental breaches, mandatory notice-to-correct with cure period, and a valuation formula at termination."),
    "c_final_account": ("SBD GCC 43, 55 (payments, completion); FIDIC 2017 cl. 14.10–14.13",
                        "Deadlines for the statement at completion and final statement with deemed acceptance if the engineer does not respond."),
    "c_delay_eot": ("SBD GCC 28, 44 (EOT, compensation events); FIDIC 2017 cl. 8.5, 20.2",
                    "Programme updates as a contract obligation; EOT determination deadline for the project manager; compensation events list."),
    "c_defects": ("SBD GCC 34–35 (defects); FIDIC 2017 cl. 11; Building Control Act 2013 / Regs 2020",
                  "Joint defects inspection at taking-over; link building-control inspection records to the contract's defects regime."),
    "c_site_land": ("SBD GCC (possession of site; compensation events); FIDIC 2017 cl. 2.1; Roads Act 2019",
                    "Land acquisition and resettlement completed before award; late site possession as an automatic compensation event."),
    "c_variations": ("SBD GCC 40 (variations); FIDIC 2017 cl. 13",
                     "Written variation orders with valuation before execution; cap on oral instructions."),
    "c_lds": ("SBD GCC 49 (liquidated damages); FIDIC 2017 cl. 8.8",
              "Cap on liquidated damages and a stated daily rate; set-off only after EOT determination."),
    "c_prof_fees": ("Consultancy SBD; FIDIC White Book", "Fee scales and payment milestones for consultants; professional indemnity requirements."),
    "c_price_adjust": ("SBD GCC 47 (price adjustment); FIDIC 2017 cl. 13.7", "Index-based price adjustment for contracts over 18 months."),
    "c_force_majeure": ("FIDIC 2017 cl. 18 (exceptional events)", "Pandemic and government-measure events listed expressly, with cost/time consequences."),
}
DISPUTE_TIER = ("SBD GCC 24–26 (adjudicator); FIDIC 2017 cl. 21 (DAAB)",
                "Standing adjudicator named at award with fixed decision times and interim-binding decisions; clear 'condition precedent' "
                "wording so courts apply the tier consistently (Stage 8: tiers argued in 15% of decisions, enforced in 17%).")


def minmax(x):
    x = pd.Series(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0


def main():
    s = Stage("09_reform", "Stage 9 — Cause → clause → cost: reform priorities (RQ9)")
    a = load()
    con = a[a.stream.isin(["construction", "both"])].copy()
    yf = pd.to_numeric(con.case_number.str.extract(r"(?i)of\s*((?:19|20)\d{2})")[0], errors="coerce")
    con["years"] = (con.year - yf).where(lambda x: x.between(0, 25) & con.proceeding.isin(["suit", "appeal"]))
    con["claim_ugx"] = np.where(con.currency.eq("USD"), con.quantum_claimed * USD_UGX, con.quantum_claimed)
    ok = con.quantum_claimed.gt(0) & con.quantum_awarded.notna()
    con["recovery"] = (con.quantum_awarded / con.quantum_claimed).clip(0, 1).where(ok)
    con["failed"] = con.outcome.isin(["initiator failed", "struck out (procedural)"]).where(
        con.outcome.isin(["initiator succeeded", "partly succeeded", "initiator failed", "struck out (procedural)"]))
    trend = pd.read_csv(OUT / "analysis" / "03_temporal_legislation" / "t2_cause_trends.csv").set_index("cause")["Sen slope (pp/yr)"]
    cited = set(a.leg_ids_cited.explode().dropna())

    rows = []
    for c in CAUSES:
        x = con[con[c] == "Y"]
        if len(x) < 10:
            continue
        rows.append({"cause": CAUSE_LABEL[c], "key": c, "decisions": len(x), "prevalence %": 100 * len(x) / len(con),
                     "trend (pp/yr)": trend.get(CAUSE_LABEL[c], np.nan), "median claim (UGX m)": x.claim_ugx.median() / 1e6,
                     "unrecovered share": 1 - x.recovery.mean() if x.recovery.notna().sum() >= 5 else np.nan,
                     "median years (suits/appeals)": x.years.median(), "initiator failure %": 100 * x.failed.mean(),
                     "tier/limitation friction %": 100 * (x.tier_skipped_raised.eq("Y") | x.limitation_raised.eq("Y")).mean(),
                     "state party %": 100 * x.state_party.eq("Y").mean(),
                     "targeted enactments": len(LEG_TARGET.get(c, [])),
                     "of which cited": sum(l in cited for l in LEG_TARGET.get(c, []))})
    m = pd.DataFrame(rows).set_index("cause")
    ind = {"frequency": minmax(m["prevalence %"]), "growth": minmax(m["trend (pp/yr)"].fillna(0)),
           "cost": minmax(np.log10(m["median claim (UGX m)"].fillna(m["median claim (UGX m)"].median()))),
           "unrecovered": minmax(m["unrecovered share"].fillna(m["unrecovered share"].median())),
           "delay": minmax(m["median years (suits/appeals)"].fillna(m["median years (suits/appeals)"].median())),
           "uncertainty": minmax(m["initiator failure %"]), "friction": minmax(m["tier/limitation friction %"]),
           "legislative gap": minmax(-m["targeted enactments"])}
    I = pd.DataFrame(ind)
    weights = {"equal": {k: 1 for k in I},
               "frequency-led": {**{k: 1 for k in I}, "frequency": 3, "growth": 2},
               "cost-led": {**{k: 1 for k in I}, "cost": 3, "unrecovered": 2, "delay": 2}}
    for w, wt in weights.items():
        m[f"score ({w})"] = sum(I[k] * v for k, v in wt.items()) / sum(wt.values())
    m["rank"] = m["score (equal)"].rank(ascending=False).astype(int)
    m = m.sort_values("rank")
    taus = {w: kendalltau(m["score (equal)"], m[f"score ({w})"])[0] for w in weights if w != "equal"}
    show = m.drop(columns=["key"]).round(2)
    s.table(show, "t1_cause_clause_cost", "Table 1. Cause → cost evidence and reform priority score by dispute cause",
            note="Score = mean of min-max scaled indicators (frequency, growth, claim size, unrecovered share, delay, initiator "
                 "failure, procedural friction, legislative gap); higher = higher reform priority. Rank stability under alternative "
                 "weights (Kendall's τ vs equal weights): " + ", ".join(f"{k} {v:.2f}" for k, v in taus.items()) + ". "
                 "Claim and recovery figures rest on low-agreement quantum fields; use for ranking, not point estimates.")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    sc = ax.scatter(m["prevalence %"], m["unrecovered share"], s=40 + 8 * m["median claim (UGX m)"].clip(upper=60),
                    c=m["score (equal)"], cmap=SEQ_SCORE, edgecolors=C["ink2"], linewidths=0.5)
    for name, r in m.iterrows():
        ax.annotate(name, (r["prevalence %"], r["unrecovered share"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    fig.colorbar(sc, ax=ax, label="priority score (equal weights)")
    ax.set_xlabel("% of construction-related decisions with the cause in issue")
    ax.set_ylabel("Unrecovered share of claims")
    s.style(ax, grid="both")
    s.fig(fig, "fig1_priority_map.png", "Reform priority map: frequency of each dispute cause against the share of claims left "
          "unrecovered; bubble size = median claim, colour = composite priority score.")

    lev = pd.DataFrame([{"priority": int(r["rank"]), "cause": name, "contract anchor (PPDA SBD / FIDIC 2017)": LEVER[r.key][0],
                         "candidate reform": LEVER[r.key][1],
                         "legislation (register)": ", ".join(LEG_TARGET.get(r.key, [])) or "none in 2015–2026 register"}
                        for name, r in m.iterrows() if r.key in LEVER])
    lev.loc[len(lev)] = {"priority": "cross-cutting", "cause": "Dispute-resolution tier", "contract anchor (PPDA SBD / FIDIC 2017)": DISPUTE_TIER[0],
                         "candidate reform": DISPUTE_TIER[1], "legislation (register)": "LEG023, LEG028"}
    lev.loc[len(lev)] = {"priority": "cross-cutting", "cause": "Arbitration with state parties",
                         "contract anchor (PPDA SBD / FIDIC 2017)": "SBD GCC 25 (arbitration); FIDIC 2017 cl. 21.6",
                         "candidate reform": "Express authority and consent to arbitrate for public employers, named seat and rules, so "
                                             "referral and enforcement do not turn on state-specific objections (Stage 5: courts defer "
                                             "markedly less when the state is a party, OR 0.35).",
                         "legislation (register)": "LEG023"}
    s.table(lev.set_index("priority"), "t2_reform_levers", "Table 2. Clause levers and candidate reforms for the priority causes",
            note="Drafted proposals linking the evidence to the governing clauses. Clause numbers follow the PPDA/World Bank SBD "
                 "General Conditions and FIDIC 2017 Red Book; verify against the editions in use and validate with practitioners "
                 "(Delphi round) before adoption (PhD link: SEM of reform acceptance).")
    s.write()


if __name__ == "__main__":
    main()
