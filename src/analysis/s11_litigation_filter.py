"""Analysis stage 11 (RQ8) — the litigation filter: what reaches court compared with audit findings and press coverage.

Three corpora mapped to seven common dispute families (crosswalk below):
  court   — construction-related decisions (coded causes; a decision counts once per family it raises)
  OAG     — infrastructure sentences in Auditor-General reports 2017–2025 (driver labels; oag-reports repo)
  Observer— The Observer articles 2016–2025 (main dispute driver; the-observer repo)
Filter ratio = family share in court ÷ share in the comparison corpus (>1: over-represented in litigation).
Chi-square tests of the court mix against each corpus; annual co-movement (Spearman) over the overlapping years;
sector comparison with The Observer.
Output: outputs/analysis/11_litigation_filter/
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chisquare, spearmanr

from src.analysis.common import C, Stage, load, plt
from src.config import ROOT

warnings.filterwarnings("ignore")
EXT = ROOT / "data" / "external"
FAMILIES = ["Payment & funding", "Delay & time", "Procurement", "Quality & defects", "Land & site",
            "Contract management", "Governance & fraud"]
COURT = {"c_payment": "Payment & funding", "c_final_account": "Payment & funding", "c_delay_eot": "Delay & time",
         "c_lds": "Delay & time", "c_procurement": "Procurement", "c_defects": "Quality & defects", "c_site_land": "Land & site",
         "c_variations": "Contract management", "c_termination": "Contract management", "c_guarantees": "Contract management",
         "c_price_adjust": "Contract management", "c_force_majeure": "Contract management", "c_prof_fees": "Contract management",
         "c_fraud": "Governance & fraud"}
OAG = {"delayed_payments": "Payment & funding", "budget_and_funding_issues": "Payment & funding",
       "delays_and_time_overruns": "Delay & time", "procurement_irregularities": "Procurement",
       "infrastructure_and_equipment": "Quality & defects", "land_and_right_of_way": "Land & site",
       "contract_management": "Contract management", "claims_and_liabilities": "Contract management",
       "governance_and_controls": "Governance & fraud", "compliance_and_regulatory": "Governance & fraud"}
OBS = {"Payment / Financial Dispute": "Payment & funding", "Delay / Time Overrun": "Delay & time",
       "Procurement Irregularity": "Procurement", "Quality / Technical Defect": "Quality & defects",
       "Land / RoW Dispute": "Land & site", "Contract Management Failure": "Contract management",
       "Governance / Oversight Failure": "Governance & fraud"}
SECTOR = {"road": "Roads/Transport", "bridge or other civil": "Roads/Transport", "building": "Buildings/Housing",
          "energy and power": "Energy", "water and sanitation": "Water/Sanitation", "supply-and-install": "Other"}


def main():
    s = Stage("11_litigation_filter", "Stage 11 — The litigation filter: court vs audit vs press (RQ8)")
    a = load()
    con = a[a.stream.isin(["construction", "both"])]
    court = pd.DataFrame([{"case_id": r.case_id, "year": r.year, "family": COURT[c]} for r in con.itertuples() for c in COURT
                          if getattr(r, c) == "Y"]).drop_duplicates(["case_id", "family"]).drop(columns="case_id").assign(source="court")
    oag = pd.read_csv(EXT / "oag-reports/government_Auditor_General/oag_infrastructure_sentence_corpus_2017_2025_expanded.csv")
    lab = oag.driver_label_expanded.fillna(oag.driver_label)
    oag_all = len(oag)
    oag = pd.DataFrame({"year": oag.year, "family": lab.map(OAG)}).dropna().assign(source="OAG")
    obs = pd.read_csv(EXT / "the-observer/paper2_online_discourse/outputs_observer_full_corpus/extended_analysis/articles_extended.csv",
                      usecols=["year", "sector", "main_driver_nice"])
    obs_f = pd.DataFrame({"year": obs.year, "family": obs.main_driver_nice.map(OBS)}).dropna().assign(source="Observer")
    s.text(f"Court: {len(con)} construction-related decisions ({len(court)} decision–family mentions). OAG: {oag_all} infrastructure "
           f"sentences, of which {len(oag)} ({len(oag) / oag_all:.0%}) carry a dispute-relevant driver (administrative, staffing and "
           f"general findings have no litigable counterpart and are excluded). The Observer: {len(obs_f)} articles 2016–2025.")

    sh = pd.concat([court, oag, obs_f]).groupby(["source", "family"]).size().unstack(0).reindex(FAMILIES).fillna(0)
    pct = 100 * sh / sh.sum()
    t = pct.round(1)
    t["filter ratio vs OAG"] = (pct.court / pct.OAG).round(2)
    t["filter ratio vs Observer"] = (pct.court / pct.Observer).round(2)
    s.table(t[["court", "OAG", "Observer", "filter ratio vs OAG", "filter ratio vs Observer"]], "t1_family_shares",
            "Table 1. Dispute family shares (%) in court decisions, audit findings and press coverage",
            note="Filter ratio > 1: the family is over-represented in litigation relative to how often it is flagged by auditors or "
                 "reported by the press; < 1: problems that rarely become litigated disputes.")
    rows = []
    for other in ["OAG", "Observer"]:
        exp = sh[other] / sh[other].sum() * sh.court.sum()
        chi, p = chisquare(sh.court, exp)
        rows.append({"comparison": f"court vs {other}", "chi-square": round(chi, 1), "df": len(FAMILIES) - 1, "p": f"{p:.2g}"})
    s.table(pd.DataFrame(rows).set_index("comparison"), "t2_chisq", "Table 2. Goodness of fit of the court mix to each corpus")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(FAMILIES))
    for i, (src, col) in enumerate([("court", C["construction"]), ("OAG", C["accent"]), ("Observer", C["arbitration"])]):
        ax.bar(x + (i - 1) * 0.27, pct[src], 0.27, color=col, label={"court": "Court decisions", "OAG": "Auditor-General findings",
                                                                       "Observer": "The Observer articles"}[src])
    ax.set_xticks(x, FAMILIES, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("% of classified items")
    ax.legend(frameon=False, fontsize=8)
    s.style(ax)
    s.fig(fig, "fig1_family_shares.png", "Dispute families in court decisions (construction-related), Auditor-General infrastructure "
          "findings (2017–2025) and The Observer articles (2016–2025). Press coverage is dominated by land and right-of-way "
          "conflicts; audits by procurement, delay and funding; courts by payment and contract-management disputes.")

    yrs = range(2017, 2026)
    allf = pd.concat([court, oag, obs_f])
    allf["year"] = pd.to_numeric(allf.year, errors="coerce")
    ann = allf.groupby(["year", "source", "family"]).size().unstack(["source", "family"], fill_value=0).reindex(list(yrs), fill_value=0)
    rows = []
    for fam in FAMILIES:
        c_ = ann.get(("court", fam), pd.Series(0, index=ann.index)).values
        for other in ["OAG", "Observer"]:
            o_ = ann.get((other, fam), pd.Series(0, index=ann.index)).values
            if c_.sum() >= 10 and o_.sum() >= 10:
                r, p = spearmanr(c_, o_)
                rows.append({"family": fam, "vs": other, "rho": round(r, 2), "p": round(p, 3)})
    s.table(pd.DataFrame(rows).set_index(["family", "vs"]), "t3_comovement",
            "Table 3. Annual co-movement of court decisions with audit findings and press coverage, 2017–2025 (Spearman)",
            note="Nine annual points per series: low power; decisions also lag the underlying disputes by years (Stage 4), so "
                 "contemporaneous co-movement is not expected.")

    cs = con.project_type.map(SECTOR).dropna().value_counts(normalize=True).mul(100)
    os_ = obs.sector.value_counts(normalize=True).mul(100)
    sec = pd.DataFrame({"court %": cs, "Observer %": os_}).fillna(0).round(1)
    sec["ratio"] = (sec["court %"] / sec["Observer %"]).replace(np.inf, np.nan).round(2)
    s.table(sec.sort_values("Observer %", ascending=False), "t4_sector", "Table 4. Sector mix: court decisions (with a stated project type) vs The Observer")
    s.write()


if __name__ == "__main__":
    main()
