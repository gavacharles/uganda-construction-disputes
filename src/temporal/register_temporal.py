"""Stage 6 (register-based) — temporal analysis of the 482 register cases, 2015–2026.

RQ2: frequency, mix, outcomes and duration over time. RQ10: shifts around enactments.
Methods: Mann-Kendall + Sen's slope, Pettitt change point, quarterly interrupted time series
(quasi-Poisson, offset = ULII judgments), logistic trends with BH correction.
Outputs: outputs/temporal/*.csv, fig*.png (300 dpi), summary.md
"""
import numpy as np
import pandas as pd
import pymannkendall as mk
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.code.rules import ISSUES, code_registers  # noqa: E402
from src.config import INTERIM, OUT, PROCESSED, RAW, SEED  # noqa: E402

DEST = OUT / "temporal"
LAST = pd.Timestamp("2026-11-16")              # last decision date in the 2026 registers
F26 = LAST.dayofyear / 365                      # 2026 coverage fraction
ULII_2026 = 3560                                # judgments on ULII in 2026 (register notes, approx.)
BINS = {"2015–2018": (2015, 2018), "2019–2021": (2019, 2021), "2022–2024": (2022, 2024), "2025–2026": (2025, 2026)}
EVENTS = {  # date, label, legislation register id (None = non-legislative shock)
    "covid": ("2020-03-20", "COVID-19 lockdown", None),
    "ppda21": ("2021-07-01", "PPDA (Amendment) Act 2021", "Public Procurement and Disposal of Public Assets (Amendment) Act, 2021"),
    "dec24": ("2024-12-23", "Arbitration Amdt & UNRA Repeal 2024", "Arbitration and Conciliation (Amendment) Act, 2024"),
}
C = {"construction": "#2a78d6", "both": "#1baf7a", "arbitration": "#eb6834",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e5e5e2"}
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=C["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(DEST / name, dpi=300)
    plt.close(fig)


def ulii_totals():
    s = pd.read_excel(RAW / "2015-2025" / "Construction_Cases_Register_ULII_2015-2025.xlsx",
                      sheet_name="Summary by year", header=3, dtype=str)
    s = s[pd.to_numeric(s["Year"], errors="coerce").notna()]
    col = next(c for c in s.columns if c.startswith("Judgments on ULII"))
    tot = s.set_index(s["Year"].astype(int))[col].astype(int)
    assert tot.sum() == 20192, tot.sum()  # register notes: 20,192 judgments 2015–2025
    tot.loc[2026] = ULII_2026
    return tot.rename("ulii_judgments")


def pettitt(x):
    x = np.asarray(x, float)
    n = len(x)
    U = np.array([sum(np.sign(x[j] - x[i]) for i in range(t + 1) for j in range(t + 1, n)) for t in range(n - 1)])
    k = int(np.argmax(np.abs(U)))
    K = abs(U[k])
    p = min(1.0, 2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2)))
    return k, K, p


def annual(c, tot):
    a = pd.DataFrame({
        "all": c.groupby("year").size(),
        "construction-related": c[c.stream.isin(["construction", "both"])].groupby("year").size(),
        "arbitration-related": c[c.stream.isin(["arbitration", "both"])].groupby("year").size(),
        "construction only": c[c.stream == "construction"].groupby("year").size(),
        "both": c[c.stream == "both"].groupby("year").size(),
        "arbitration only": c[c.stream == "arbitration"].groupby("year").size(),
    }).fillna(0).astype(int).join(tot)
    return a


def trend_tests(a):
    rows = []
    for col in ["all", "construction-related", "arbitration-related"]:
        series = {
            "count 2015–2025": a.loc[2015:2025, col],
            "count 2015–2026 (2026 annualised)": a[col].where(a.index < 2026, a[col] / F26),
            "rate per 1,000 ULII judgments 2015–2026": a[col] / a["ulii_judgments"] * 1000,
        }
        for name, s in series.items():
            r = mk.original_test(s.values)
            k, K, pp = pettitt(s.values)
            rows.append({"series": col, "measure": name, "trend": r.trend, "tau": round(r.Tau, 2),
                         "p": round(r.p, 4), "sen_slope_per_year": round(r.slope, 3),
                         "pettitt_change_after": int(s.index[k]), "pettitt_p": round(pp, 3)})
    return pd.DataFrame(rows)


def quarterly(c, tot):
    q = pd.period_range("2015Q1", "2026Q4", freq="Q")
    df = pd.DataFrame(index=q)
    df["construction-related"] = c[c.stream.isin(["construction", "both"])].groupby(c.date_decided.dt.to_period("Q")).size()
    df["arbitration-related"] = c[c.stream.isin(["arbitration", "both"])].groupby(c.date_decided.dt.to_period("Q")).size()
    df = df.fillna(0)
    days = pd.Series([p.end_time.normalize() for p in q], index=q)
    frac = ((np.minimum(days, LAST) - pd.Series([p.start_time for p in q], index=q)).dt.days + 1) / \
        pd.Series([p.days_in_month * 0 + (p.end_time - p.start_time).days + 1 for p in q], index=q)
    df["exposure"] = [tot[p.year] / 4 for p in q] * frac.clip(lower=0).values
    df = df[df.exposure > 0]
    df["t"] = np.arange(len(df))
    return df


def its(qdf):
    rows = []
    for key, (date, label, _) in EVENTS.items():
        e = pd.Period(pd.Timestamp(date), freq="Q")
        d = qdf.copy()
        d["post"] = (d.index >= e).astype(int)
        d["t_post"] = np.maximum(0, d["t"] - d.loc[d.index >= e, "t"].min()) * d["post"]
        for y in ["construction-related", "arbitration-related"]:
            d["y"] = d[y]
            m = smf.glm("y ~ t + post + t_post", data=d, family=sm.families.Poisson(),
                        offset=np.log(d["exposure"])).fit(scale="X2")
            ci = np.exp(m.conf_int())
            rows.append({"event": label.replace("\n", " "), "series": y,
                         "quarters_post": int(d.post.sum()),
                         "level_IRR": round(np.exp(m.params["post"]), 2),
                         "level_CI": f"{ci.loc['post', 0]:.2f}–{ci.loc['post', 1]:.2f}",
                         "level_p": round(m.pvalues["post"], 3),
                         "slope_IRR_per_qtr": round(np.exp(m.params["t_post"]), 3),
                         "slope_p": round(m.pvalues["t_post"], 3),
                         "dispersion": round(m.scale, 2)})
    return pd.DataFrame(rows)


def bin_of(y):
    for k, (a, b) in BINS.items():
        if a <= y <= b:
            return k


def issue_trends(c):
    c = c[c.source == "register"]  # issue coding comparable only on register summaries
    rows = []
    cols = [f"issue:{k}" for k in ISSUES]
    for stream_name, sub in [("construction-related", c[c.stream.isin(["construction", "both"])]),
                             ("arbitration-related", c[c.stream.isin(["arbitration", "both"])])]:
        for col in cols:
            y = sub[col].astype(int)
            if y.sum() < 5:
                continue
            m = smf.glm("y ~ I(year - 2020)", data=sub.assign(y=y), family=sm.families.Binomial()).fit()
            rows.append({"stream": stream_name, "issue": col[6:], "n": int(y.sum()),
                         "share_%": round(y.mean() * 100, 1),
                         "OR_per_year": round(np.exp(m.params.iloc[1]), 2), "p": m.pvalues.iloc[1]})
    t = pd.DataFrame(rows)
    t["p_BH"] = multipletests(t.p, method="fdr_bh")[1]
    t[["p", "p_BH"]] = t[["p", "p_BH"]].round(4)
    shares = {s: c[c.stream.isin(m)].groupby("bin")[cols].mean().mul(100).round(1).rename(columns=lambda x: x[6:]).T
              for s, m in [("construction-related", ["construction", "both"]), ("arbitration-related", ["arbitration", "both"])]}
    return t, shares


def outcomes(c):
    d = c[c.outcome_class.isin(["successful", "unsuccessful", "partial"])].copy()
    d["success"] = d.outcome_class.ne("unsuccessful").astype(int)
    out = {}
    out["success_by_bin"] = d.groupby(["bin", "stream"])["success"].agg(["mean", "size"]).round(2).unstack()
    rows = []
    for s, m in [("construction-related", ["construction", "both"]), ("arbitration-related", ["arbitration", "both"])]:
        sub = d[d.stream.isin(m)]
        g = smf.glm("success ~ I(year - 2020)", data=sub, family=sm.families.Binomial()).fit()
        rows.append({"series": s, "n": len(sub), "success_%": round(sub.success.mean() * 100, 1),
                     "OR_per_year": round(np.exp(g.params.iloc[1]), 2), "p": round(g.pvalues.iloc[1], 4)})
    out["success_trend"] = pd.DataFrame(rows)
    arb = c[c.arb_application_type.notna()].copy()
    arb["applicant_success"] = arb.outcome_class.map({"successful": 1, "partial": 1, "unsuccessful": 0})
    out["arb_by_type"] = arb.groupby("arb_application_type").agg(
        n=("case_id", "size"), applicant_success=("applicant_success", "mean"),
        referred=("outcome_class", lambda s: (s == "referred to arbitration").sum())).round(2)
    sa = arb[arb.arb_application_type == "set-aside"]
    out["setaside_by_bin"] = sa.groupby("bin")["applicant_success"].agg(["mean", "count"]).round(2)
    out["state_party_by_bin"] = c.groupby(["bin", "stream"])["state_party"].mean().mul(100).round(1).unstack()
    return out, d


def duration(c):
    d = c[c.years_to_decision.notna()].copy()
    tab = d.groupby(["bin", "proceeding"])["years_to_decision"].describe()[["count", "50%", "mean"]].round(1)
    med = d.groupby("year")["years_to_decision"].median()
    r = mk.original_test(med.values)
    pre = d[d.date_decided < "2021-07-01"].years_to_decision
    post = d[d.date_decided >= "2021-07-01"].years_to_decision
    mw = stats.mannwhitneyu(pre, post)
    test = pd.DataFrame([{"MK_trend_median": r.trend, "tau": round(r.Tau, 2), "p": round(r.p, 4),
                          "sen_slope": round(r.slope, 3),
                          "median_pre_Jul2021": pre.median(), "median_post": post.median(),
                          "n_pre": len(pre), "n_post": len(post), "MannWhitney_p": round(mw.pvalue, 4)}])
    return tab, test, d


# ---------- figures ----------
def fig1(a):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    yrs, bottom = a.index.values, np.zeros(len(a))
    for col, key in [("construction only", "construction"), ("both", "both"), ("arbitration only", "arbitration")]:
        ax.bar(yrs, a[col], bottom=bottom, color=C[key], width=0.72, edgecolor="white", linewidth=1.5, label=col)
        bottom += a[col].values
    ann = a.loc[2026, "all"] / F26
    ax.plot([2026], [ann], marker="_", ms=22, mew=2, color=C["ink2"])
    ax.annotate(f"annualised ≈ {ann:.0f}", (2026, ann), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=8, color=C["ink2"])
    ax.set_ylim(0, ann * 1.12)
    for key, (date, label, _) in EVENTS.items():
        x = pd.Timestamp(date).year + (pd.Timestamp(date).dayofyear - 1) / 365 - 0.5
        ax.axvline(x, color=C["ink2"], ls=(0, (3, 3)), lw=0.8)
        ax.text(x - 0.07, ann * 1.1, label.replace("\n", " "), fontsize=7, va="top", ha="right",
                rotation=90, color=C["ink2"])
    ax.set_xticks(yrs)
    ax.set_ylabel("Judgments (register cases)")
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.1))
    style(ax)
    save(fig, "fig1_annual_by_stream.png")


def fig2(a):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for col, key in [("construction-related", "construction"), ("arbitration-related", "arbitration")]:
        r = a[col] / a["ulii_judgments"] * 1000
        ax.plot(r.index, r.values, color=C[key], lw=2, marker="o", ms=5, label=col)
        ax.annotate(col, (r.index[-1], r.values[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=C["ink"])
    ax.set_xticks(a.index)
    ax.set_xlim(2014.6, 2027.8)
    ax.set_ylabel("Per 1,000 ULII judgments")
    ax.legend(frameon=False, loc="upper left")
    style(ax)
    save(fig, "fig2_rate_per_1000.png")


def fig3(shares):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
    for ax, (name, s) in zip(axes, shares.items()):
        s = s.loc[[f[6:] for f in [f"issue:{k}" for k in ISSUES]]]
        im = ax.imshow(s.values, cmap="Blues", vmin=0, vmax=60, aspect="auto")
        ax.set_xticks(range(s.shape[1]), s.columns, rotation=0, fontsize=8)
        ax.set_yticks(range(s.shape[0]), s.index, fontsize=8)
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                v = s.values[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 35 else C["ink"])
        ax.set_title(name, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.6, label="% of judgments")
    fig.savefig(DEST / "fig3_issue_mix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4(d):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for s, m, key in [("construction-related", ["construction", "both"], "construction"),
                      ("arbitration-related", ["arbitration", "both"], "arbitration")]:
        g = d[d.stream.isin(m)].groupby("bin")["success"].agg(["mean", "size"]).reindex(BINS)
        se = np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"])
        x = np.arange(len(g)) + (-0.06 if key == "construction" else 0.06)
        ax.errorbar(x, g["mean"] * 100, yerr=1.96 * se * 100, color=C[key], lw=2, marker="o", ms=6,
                    capsize=3, label=s)
    ax.set_xticks(range(len(BINS)), BINS.keys())
    ax.set_ylim(0, 100)
    ax.set_ylabel("Initiating party successful (%)")
    ax.legend(frameon=False, loc="lower left")
    style(ax)
    save(fig, "fig4_success_by_period.png")


def fig5(d):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    data = [d.loc[d.bin == b, "years_to_decision"].values for b in BINS]
    bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=True,
                    medianprops=dict(color=C["ink"], lw=1.5), flierprops=dict(ms=3, mfc=C["ink2"], mec="none"))
    for p in bp["boxes"]:
        p.set(facecolor="#cfe0f6", edgecolor=C["construction"])
    ax.set_xticks(range(1, len(BINS) + 1), [f"{b}\n(n={len(x)})" for b, x in zip(BINS, data)])
    ax.set_ylabel("Years, filing to decision")
    style(ax)
    save(fig, "fig5_duration.png")


def main():
    np.random.seed(SEED)
    DEST.mkdir(parents=True, exist_ok=True)
    c = code_registers(pd.read_parquet(PROCESSED / "cases.parquet"))  # registers + included scanned judgments
    c["bin"] = c["year"].map(bin_of)
    c.to_parquet(INTERIM / "registers_coded.parquet", index=False)
    leg = pd.read_parquet(INTERIM / "legislation.parquet")
    for _, (_, _, title) in EVENTS.items():
        assert title is None or title in set(leg.title), title
    tot = ulii_totals()

    a = annual(c, tot)
    tt = pd.concat([trend_tests(a).assign(sample="all cases"),
                    trend_tests(annual(c[~c.subject_unconfirmed], tot)).assign(
                        sample="excl. 'subject matter not stated'")])
    q = quarterly(c, tot)
    it = its(q)
    it_issue, shares = issue_trends(c)
    oc, d_out = outcomes(c)
    dur_tab, dur_test, d_dur = duration(c)

    tables = {"annual_counts": a, "trend_tests": tt, "its_events": it, "issue_trends": it_issue,
              "issue_share_construction": shares["construction-related"],
              "issue_share_arbitration": shares["arbitration-related"],
              **oc, "duration_by_period": dur_tab, "duration_tests": dur_test,
              "proceeding_mix": c.groupby(["bin", "proceeding"]).size().unstack(fill_value=0)}
    for k, v in tables.items():
        v.to_csv(DEST / f"{k}.csv")
    fig1(a); fig2(a); fig3(shares); fig4(d_out); fig5(d_dur)

    caps = {
        "fig1_annual_by_stream.png": "Figure 1. Construction and arbitration judgments in the ULII registers by year of decision and stream, 2015–2026, with key reform and shock dates (dashed). 2026 runs to 16 November; the bar marker shows the annualised 2026 total.",
        "fig2_rate_per_1000.png": "Figure 2. Construction-related and arbitration-related judgments per 1,000 judgments published on ULII, 2015–2026.",
        "fig3_issue_mix.png": "Figure 3. Share of judgments (%) mentioning each issue family by period, from rule-based coding of register key issues and holdings. A judgment may fall in several families.",
        "fig4_success_by_period.png": "Figure 4. Share of judgments in which the initiating party (claimant, applicant or appellant) wholly or partly succeeded, by period, with 95% confidence intervals. Unclear outcomes and referrals to arbitration are excluded.",
        "fig5_duration.png": "Figure 5. Years from filing to decision for suits and appeals, by period of decision (filing year taken from the case number).",
    }
    md = ["# Register-based temporal analysis (2015–2026)\n",
          f"Cases: {len(c)} ({(c.source == 'ocr').sum()} from OCR'd scanned judgments; issue shares use register cases only) (construction-related {c.stream.isin(['construction', 'both']).sum()}, "
          f"arbitration-related {c.stream.isin(['arbitration', 'both']).sum()}). 2026 coverage to {LAST.date()} "
          f"(fraction {F26:.2f}); 2026 ULII total ≈ {ULII_2026}.\n"]
    for k, v in tables.items():
        md += [f"## {k.replace('_', ' ').capitalize()}\n", v.to_markdown(), ""]
    md += ["## Figures\n"] + [f"![{k}]({k})\n\n**{v}**\n" for k, v in caps.items()]
    md += ["## Caveats\n",
           "- Register cases are a filtered sample: only judgments published on ULII, with ULII coverage thinner before 2020.",
           "- Issues, outcomes, party types and arbitration application types are rule-coded from short register fields; validate against the gold set.",
           "- 2026 registers merge same-day rulings in one matter into one row; 2015–2025 registers keep them separate.",
           "- ITS estimates rest on few post-event quarters (Dec 2024 reforms: 8) and are associational."]
    (DEST / "summary.md").write_text("\n".join(md))
    print(tt.to_string(), "\n", it.to_string(), "\n", it_issue.sort_values("p").head(12).to_string())
    print(oc["success_trend"].to_string(), "\n", oc["arb_by_type"].to_string(), "\n", oc["setaside_by_bin"].to_string())
    print(dur_test.to_string(), "\n", dur_tab.to_string())
    print(a.to_string())


if __name__ == "__main__":
    main()
