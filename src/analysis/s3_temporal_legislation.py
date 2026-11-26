"""Analysis stage 3 (RQ2, RQ10) — temporal change and the legislation layer, on the coded corpus.

A. Volume: counts and rates per 1,000 ULII judgments by stream; Mann-Kendall, Sen's slope, Pettitt change point.
B. Mix: annual cause shares (Mann-Kendall, BH-corrected); outcome trend (logistic, success ~ year).
C. Events (from the legislation register, effective = verified commencement date else enactment date; plus COVID):
   segmented Poisson ITS on quarterly counts with ULII exposure; segmented logistic ITS on procurement share
   (PPDA Amendment 2021); arbitration-interface outcomes before/after the Dec 2024 Arbitration Amendment.
D. Legislation timeline and judicial uptake (citations of register enactments and of principal Acts by year).
Output: outputs/analysis/03_temporal_legislation/
"""
import warnings

import numpy as np
import pandas as pd
import pymannkendall as mk
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from src.analysis.common import C, CAUSE_LABEL, CAUSES, NAME, SERIES6, STREAMS, Stage, load, plt
from src.config import INTERIM
from src.temporal.register_temporal import F26, LAST, pettitt, ulii_totals

warnings.filterwarnings("ignore")
PRINCIPAL = ["Arbitration and Conciliation Act", "PPDA Act", "PPDA Regulations", "Contracts Act", "Limitation Act",
             "Government Proceedings Act", "Building Control Act", "Roads Act", "Judicature (Court Annexed Mediation) Rules"]


def events(leg):
    leg = leg.assign(effective=leg.commencement_date.fillna(leg.date))
    pick = {"LEG017": "PPDA (Amendment) Act 2021", "LEG023": "Arbitration Amdt Act & UNRA Repeal 2024"}
    ev = {"COVID-19 lockdown": pd.Timestamp("2020-03-20")}
    for lid, label in pick.items():
        ev[label] = leg.set_index("leg_id").effective[lid]
    return ev


def quarterly(a, tot):
    q = pd.period_range("2015Q1", pd.Period(LAST, "Q"), freq="Q")
    d = pd.DataFrame(index=q)
    for name, sel in [("construction-related", ["construction", "both"]), ("arbitration-related", ["arbitration", "both"])]:
        d[name] = a[a.stream.isin(sel)].groupby(a.date_decided.dt.to_period("Q")).size()
    d = d.fillna(0)
    frac = np.array([min(1.0, max(0.0, ((min(p.end_time, LAST) - p.start_time).days + 1) / ((p.end_time - p.start_time).days + 1))) for p in q])
    d["exposure"] = np.array([tot[p.year] / 4 for p in q]) * frac
    d = d[d.exposure > 0]
    d["t"] = np.arange(len(d))
    return d


def seg_terms(d, date):
    e = pd.Period(date, "Q")
    post = (d.index >= e).astype(int)
    t_post = np.maximum(0, d.t - d.t[post.astype(bool)].min()) * post if post.any() else 0 * post
    return post, t_post


def main():
    s = Stage("03_temporal_legislation", "Stage 3 — Temporal change and legislation (RQ2, RQ10)")
    a = load()
    a = a[a.date_decided.notna()]
    tot = ulii_totals()
    leg = pd.read_parquet(INTERIM / "legislation.parquet")
    EV = events(leg)

    # ---- A. volume
    yr = pd.crosstab(a.year, a.stream).reindex(columns=STREAMS, fill_value=0)
    yr["construction-related"] = yr.construction + yr.both
    yr["arbitration-related"] = yr.arbitration + yr.both
    yr["ULII judgments"] = tot
    rows = []
    for col in ["construction-related", "arbitration-related"]:
        for name, ser in {"count (2026 annualised)": yr[col].where(yr.index < 2026, yr[col] / F26),
                          "rate per 1,000 ULII judgments": yr[col] / yr["ULII judgments"] * 1000}.items():
            r = mk.original_test(ser.values)
            k, _, pp = pettitt(ser.values)
            rows.append({"series": col, "measure": name, "trend": r.trend, "tau": round(r.Tau, 2), "p": round(r.p, 4),
                         "Sen slope / yr": round(r.slope, 3), "change after": int(ser.index[k]), "Pettitt p": round(pp, 3)})
    s.table(pd.DataFrame(rows).set_index(["series", "measure"]), "t1_volume_trends",
            "Table 1. Trends in decision volume, 2015–2026",
            note="Counts grow with ULII's coverage; the rate per 1,000 ULII judgments is the comparable measure.")
    fig, ax = plt.subplots(figsize=(7, 3.3))
    for col, colr in [("construction-related", C["construction"]), ("arbitration-related", C["arbitration"])]:
        ax.plot(yr.index, yr[col] / yr["ULII judgments"] * 1000, marker="o", color=colr, label=col)
    for lab, dt in EV.items():
        ax.axvline(dt.year + dt.dayofyear / 365, color=C["ink2"], ls=":", lw=0.9)
        ax.text(dt.year + dt.dayofyear / 365, ax.get_ylim()[1] * 0.97, " " + lab, fontsize=7, rotation=90, va="top", color=C["ink2"])
    ax.set_ylabel("Decisions per 1,000 ULII judgments")
    ax.set_xticks(yr.index)
    ax.legend(frameon=False, loc="lower left")
    s.style(ax)
    s.fig(fig, "fig1_rate_per_1000.png", "Construction- and arbitration-related decisions per 1,000 judgments published on "
          "ULII, 2015–2026, with the reform events tested in Table 4 (dotted lines). 'Both' decisions count in each series.")

    # ---- B. mix
    con = a[a.stream.isin(["construction", "both"])]
    Y = (con[CAUSES] == "Y").groupby(con.year).mean()
    n_yr = con.groupby("year").size()
    rows = []
    for c in CAUSES:
        if (con[c] == "Y").sum() < 15:
            continue
        r = mk.original_test(Y[c].values)
        early, late = con[con.year <= 2020][c].eq("Y").mean(), con[con.year >= 2021][c].eq("Y").mean()
        rows.append({"cause": CAUSE_LABEL[c], "2015–20 %": round(100 * early, 1), "2021–26 %": round(100 * late, 1),
                     "tau": round(r.Tau, 2), "p": r.p, "Sen slope (pp/yr)": round(100 * r.slope, 2)})
    mt = pd.DataFrame(rows)
    mt["p_BH"] = multipletests(mt.p, method="fdr_bh")[1].round(3)
    mt["p"] = mt.p.round(3)
    s.table(mt.set_index("cause").sort_values("p"), "t2_cause_trends",
            f"Table 2. Trends in cause shares, construction-related decisions (annual n = {n_yr.min()}–{n_yr.max()})",
            note="Mann–Kendall on annual shares; p_BH = Benjamini–Hochberg adjusted. Causes in issue in fewer than 15 decisions omitted.")

    rows = []
    a2 = a[a.outcome.isin(["initiator succeeded", "partly succeeded", "initiator failed", "struck out (procedural)"])].copy()
    a2["success"] = a2.outcome.isin(["initiator succeeded", "partly succeeded"]).astype(int)
    for name, sel in [("construction", ["construction"]), ("both", ["both"]), ("arbitration", ["arbitration"])]:
        d = a2[a2.stream.isin(sel)]
        m = smf.logit("success ~ I(year - 2020)", data=d).fit(disp=0)
        rows.append({"stream": name, "n": len(d), "success %": round(100 * d.success.mean(), 1),
                     "OR per year": round(np.exp(m.params.iloc[1]), 3),
                     "95% CI": f"{np.exp(m.conf_int().iloc[1, 0]):.3f}–{np.exp(m.conf_int().iloc[1, 1]):.3f}",
                     "p": round(m.pvalues.iloc[1], 3)})
    s.table(pd.DataFrame(rows).set_index("stream"), "t3_success_trend",
            "Table 3. Initiator success over time (logistic regression on year of decision)",
            note="Success = initiator succeeded or partly succeeded; denominator excludes referrals, remittals and consent orders.")

    # ---- C. events (ITS)
    q = quarterly(a, tot)
    rows = []
    for lab, dt in EV.items():
        post, tp = seg_terms(q, dt)
        for y in ["construction-related", "arbitration-related"]:
            d = q.assign(y=q[y], post=post, t_post=tp)
            m = smf.glm("y ~ t + post + t_post", data=d, family=sm.families.Poisson(), offset=np.log(d.exposure)).fit(scale="X2")
            ci = np.exp(m.conf_int())
            rows.append({"event": lab, "effective": dt.date(), "series": y, "quarters post": int(post.sum()),
                         "level IRR": round(np.exp(m.params["post"]), 2), "95% CI": f"{ci.loc['post', 0]:.2f}–{ci.loc['post', 1]:.2f}",
                         "p": round(m.pvalues["post"], 3), "slope IRR/qtr": round(np.exp(m.params["t_post"]), 3),
                         "slope p": round(m.pvalues["t_post"], 3)})
    s.table(pd.DataFrame(rows).set_index(["event", "series"]), "t4_its_volume",
            "Table 4. Interrupted time series: quarterly decision volume around reform events",
            note="Quasi-Poisson segmented regression with log(ULII judgments) offset. Associational only: few events, short "
                 "post-periods (the Dec 2024 event has under two years), and decisions lag the disputes they resolve.")

    # procurement share around the PPDA Amendment 2021 (decision level)
    dt = EV["PPDA (Amendment) Act 2021"]
    d = con.assign(proc=con.c_procurement.eq("Y").astype(int), t=(con.date_decided - pd.Timestamp("2015-01-01")).dt.days / 365.25)
    d["post"] = (d.date_decided >= dt).astype(int)
    d["t_post"] = np.maximum(0, d.t - (dt - pd.Timestamp("2015-01-01")).days / 365.25) * d.post
    m = smf.logit("proc ~ t + post + t_post", data=d).fit(disp=0)
    ppda = pd.DataFrame({"OR": np.exp(m.params), "p": m.pvalues}).round(3).loc[["post", "t_post"]]
    ppda.index = ["level change at commencement", "slope change per year after"]
    s.table(ppda, "t5_its_procurement", "Table 5. Procurement disputes' share around the PPDA (Amendment) Act 2021 (segmented logistic ITS)",
            note=f"Construction-related decisions, n = {len(d)}; pre {int((d.post == 0).sum())}, post {int(d.post.sum())}. "
                 f"Share before {100 * d[d.post == 0].proc.mean():.1f}%, after {100 * d[d.post == 1].proc.mean():.1f}%.")

    arb = a[a.stream.isin(["arbitration", "both"])]
    dt = EV["Arbitration Amdt Act & UNRA Repeal 2024"]
    pre, post = arb[arb.date_decided < dt], arb[arb.date_decided >= dt]
    rows = []
    for lab, f in [("set-aside applications (% of arbitration decisions)", lambda x: (x.arb_application_type == "set-aside").mean()),
                   ("awards set aside (% of set-aside decisions)", lambda x: x[x.arb_application_type == "set-aside"].award_result.isin(["set aside", "partly set aside"]).mean()),
                   ("court deferred to arbitration (% of coded)", lambda x: (x.deference == "Y").sum() / max(1, x.deference.isin(["Y", "N"]).sum())),
                   ("referred to arbitration (% of decisions)", lambda x: (x.outcome == "referred to arbitration").mean())]:
        rows.append({"measure": lab, f"before (n={len(pre)})": round(100 * f(pre), 1), f"after (n={len(post)})": round(100 * f(post), 1)})
    s.table(pd.DataFrame(rows).set_index("measure"), "t6_arbitration_before_after",
            f"Table 6. Arbitration interface before and after the Arbitration and Conciliation (Amendment) Act 2024 (effective {dt.date()})",
            note="Descriptive only: the post-period is under two years and coincides with the UNRA repeal.")

    # ---- D. legislation timeline and uptake
    leg = leg.assign(effective=leg.commencement_date.fillna(leg.date), group=leg.area.str.extract(r"^([^(]+)")[0].str.strip())
    leg["subarea"] = leg.area.str.extract(r"\(([^)]+)\)")[0].fillna(leg.group)
    cites = a[["case_id", "year", "leg_ids_cited"]].explode("leg_ids_cited").dropna()
    up = leg.set_index("leg_id")[["title", "effective"]].join(cites.groupby("leg_ids_cited").agg(
        citing_decisions=("case_id", "nunique"), first_cited=("year", "min")))
    up["citing_decisions"] = up.citing_decisions.fillna(0).astype(int)
    up["lag to first citation (yrs)"] = (up.first_cited - up.effective.dt.year).where(up.first_cited.notna())
    s.table(up.sort_values("citing_decisions", ascending=False).head(12).assign(effective=lambda x: x.effective.dt.date),
            "t7_enactment_uptake", "Table 7. Judicial uptake of 2015–2026 enactments (register) in the corpus",
            note=f"{int((up.citing_decisions > 0).sum())} of {len(up)} register enactments are cited at all "
                 f"({int(up.citing_decisions.sum())} citing decisions in total). Matching requires the short title and year; "
                 "citations by short title alone are not counted.")
    stat = a[["year", "statutes_cited"]].explode("statutes_cited")
    share = pd.crosstab(stat.year, stat.statutes_cited).div(a.groupby("year").size(), axis=0)
    share = share[[p for p in PRINCIPAL if p in share.columns]]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.6), gridspec_kw={"height_ratios": [1, 1.4]}, sharex=True)
    groups = leg.subarea.unique()
    ymap = {g: i for i, g in enumerate(sorted(groups))}
    for _, r in leg.iterrows():
        ax1.scatter(r.effective.year + r.effective.dayofyear / 365, ymap[r["subarea"]], s=22,
                    color=C["arbitration"] if r.group.startswith("Arbitration") else C["construction"], zorder=3)
    ax1.set_yticks(list(ymap.values()), list(ymap.keys()), fontsize=6.5)
    ax1.set_title(f"Enactments in force (register; {NAME['arbitration']} = arbitration/ADR, {NAME['construction']} = construction)", fontsize=9, loc="left")
    s.style(ax1, grid="x")
    for col, colr in zip(share.columns[:6], SERIES6):
        ax2.plot(share.index, 100 * share[col], marker="o", ms=3, color=colr, label=col)
    ax2.set_ylabel("% of decisions citing")
    ax2.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax2.set_xticks(share.index)
    s.style(ax2)
    s.fig(fig, "fig2_legislation_timeline_uptake.png",
          "Top: construction and arbitration/ADR enactments 2015–2026 by area and effective date (legislation register). "
          "Bottom: share of decisions citing the principal Acts each year. Courts rely on principal Acts; recent amending "
          "instruments are rarely cited (Table 7).")
    s.table((100 * share).round(1), "t8_principal_act_share", "Table 8. Share of decisions citing principal Acts (%), by year")
    s.write()


if __name__ == "__main__":
    main()
