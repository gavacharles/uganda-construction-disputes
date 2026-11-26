"""Sector report 2026: chapter figures, the Cost-to-Court chain and the one-page scorecard (300 dpi PNG, for Word).

House style: Times New Roman (matches the Word body), white surface, hairline grid, thin 1.6 pt lines, a bold
takeaway headline and a muted unit line in each figure; the formal caption lives in captions.md for Word.
Palette (validated with the dataviz validator, light mode): slot 1 blue #2a78d6, slot 2 orange #eb6834, slot 3 aqua
#1baf7a (sub-3:1 contrast, so it is always direct-labelled); context grey; status green/red only where the colour means
good/bad, always with an arrow. To re-brand for Hove & Quill, change PAL and INK only.
Output: outputs/writing/sector_report_2026/figures/*.png + captions.md
Usage: python -m src.report2026.figures
"""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from src.config import OUT as OUTPUTS, PROCESSED
from src.report2026.indices import BASE, complete_quarters
from src.report2026.scorecard import build as build_scorecard

D = PROCESSED / "sector2026"
FIG = OUTPUTS / "writing" / "sector_report_2026" / "figures"
# Hove & Quill brand (hovequill.com CSS): gold #B08A33, ink #14120F, gray #57534B, stone #E7E3D8, ivory #F6F3EC.
# Chart slots: s1 brand gold, s2 navy, s3 oxblood (validated all-pairs, light mode, >= 3:1 contrast).
PAL = {"s1": "#B08A33", "s2": "#2B67A8", "s3": "#9E3F3F", "grey": "#B3AC9F", "s1_light": "#EFE3C4",
       "rise": "#9E3F3F", "fall": "#2B67A8", "good": "#0ca30c", "bad": "#d03b3b"}
INK = {"primary": "#14120F", "secondary": "#57534B", "muted": "#7d776c", "grid": "#E7E3D8", "surface": "#ffffff",
       "band": "#F6F3EC", "accent": "#8A6B22"}
W = 6.3   # Word text width, inches
CAPTIONS = []

plt.rcParams.update({
    "text.parse_math": False, "font.family": "Times New Roman", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK["grid"], "axes.linewidth": 0.6, "axes.labelcolor": INK["secondary"],
    "xtick.color": INK["secondary"], "ytick.color": INK["secondary"], "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"], "savefig.facecolor": INK["surface"],
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 8.5})


# ---------- helpers ----------
def load():
    rd = lambda f: pd.read_csv(D / f, parse_dates=[0], index_col=0)
    m, q = rd("panel_monthly.csv"), complete_quarters(rd("panel_quarterly.csv"))
    ix = {k: rd(f"{k}.csv") for k in ("idx_ccpi", "idx_procurement", "idx_payment", "idx_dispute", "cipi_nowcast")}
    gap = pd.read_csv(D / "idx_escalation_gap.csv", parse_dates=["base_month"])
    ura = pd.read_csv(D / "ura_series.csv")
    wx = pd.read_csv(D / "weather_outlook.csv")
    return m, q, ix, gap, ura, wx


def headline(fig, title, sub, y=0.985):
    fig.text(0.012, y, title, ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK["primary"])
    fig.text(0.012, y - 0.075 * (3.2 / fig.get_figheight()), sub, ha="left", va="top", fontsize=9,
             color=INK["secondary"])


def save(fig, name, caption, top=0.80):
    fig.subplots_adjust(top=top)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=300)
    plt.close(fig)
    CAPTIONS.append((name, caption))


def base_band(ax):
    ax.axhspan(90, 110, color=INK["band"], zorder=0, lw=0)
    ax.axhline(100, color=INK["muted"], lw=0.6, zorder=1)
    ax.text(1.0, 0.0, "shaded: base average ±1 s.d. ", transform=ax.transAxes, fontsize=7, color=INK["muted"],
            ha="right", va="bottom")


def end_label(ax, x, y, text, color, dx=8, dy=0, **kw):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK["primary"],
                va="center", ha="left", **kw)
    ax.plot([x], [y], "o", ms=4.5, color=color, mec=INK["surface"], mew=1.2, zorder=5)


def yfmt(ax, fmt="{x:,.0f}"):
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter(fmt))


def date_axis(ax, years=1, fmt="%Y"):
    ax.xaxis.set_major_locator(mdates.YearLocator(years))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    ax.grid(axis="x", visible=False)


def qlabel(t):
    return f"{t.year} Q{t.quarter}"


# ---------- chapter 1: economy ----------
def fig_growth(q):
    g = (100 * (q.gdp_construction_va_const / q.gdp_construction_va_const.shift(4) - 1)).dropna()
    sh = q.gdp_construction_share_cur.rolling(4).mean().dropna()
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.2), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.28})
    cols = [PAL["s1"] if v >= 0 else PAL["rise"] for v in g]
    a.bar(g.index, g, width=70, color=cols)
    a.axhline(0, color=INK["muted"], lw=0.6)
    date_axis(a, 2)
    a.set_xlim(g.index[0] - pd.Timedelta(days=60), g.index[-1] + pd.Timedelta(days=420))
    yfmt(a, "{x:.0f}%")
    a.set_title("Real value added, growth on a year earlier", loc="left", color=INK["secondary"], fontsize=9)
    end_label(a, g.index[-1], g.iloc[-1], f"{g.iloc[-1]:.1f}%\n{qlabel(g.index[-1])}", PAL["s1"], dx=6, dy=10)
    b.plot(sh.index, sh, color=PAL["s1"])
    date_axis(b, 3)
    yfmt(b, "{x:.1f}%")
    b.set_xlim(sh.index[0], sh.index[-1] + pd.Timedelta(days=500))
    b.set_title("Share of GDP, current prices (4-qtr avg)", loc="left", color=INK["secondary"], fontsize=9)
    end_label(b, sh.index[-1], sh.iloc[-1], f"{sh.iloc[-1]:.1f}%", PAL["s1"])
    headline(fig, "Construction is still growing, but more slowly than in 2025",
             "Uganda, construction activity, quarterly (fiscal-year quarters shown as calendar quarters)")
    save(fig, "fig_1_1_construction_growth.png",
         "Construction real value added (constant 2016/17 prices), growth on the same quarter a year earlier, and the "
         "sector's share of GDP at current prices (four-quarter average), 2016–2026. Source: UBOS Quarterly GDP "
         f"(release Q4 2025/26). Latest: {qlabel(g.index[-1])}.", top=0.76)


def fig_shock(m):
    s = m.loc["2024-01-01":, ["exchange_rate", "brent_crude_usd"]].dropna()
    r = 100 * s / s.loc["2025-01-01"]
    fig, ax = plt.subplots(figsize=(W, 3.2))
    ax.plot(r.index, r.brent_crude_usd, color=PAL["s2"])
    ax.plot(r.index, r.exchange_rate, color=PAL["s1"])
    ax.axhline(100, color=INK["muted"], lw=0.6)
    date_axis(ax, 1)
    end_label(ax, r.index[-1], r.brent_crude_usd.iloc[-1],
              f"Brent  US${s.brent_crude_usd.iloc[-1]:.0f}/bbl", PAL["s2"])
    end_label(ax, r.index[-1], r.exchange_rate.iloc[-1], f"UGX {s.exchange_rate.iloc[-1]:,.0f} per US$",
              PAL["s1"])
    ax.axvspan(pd.Timestamp("2026-03-01"), r.index[-1], color=INK["band"], zorder=0, lw=0)
    ax.text(pd.Timestamp("2026-03-10"), ax.get_ylim()[1] * 0.985, "2026 oil and\ncurrency shock", fontsize=8,
            color=INK["secondary"], va="top")
    ax.set_ylabel("index, Jan 2025 = 100")
    ax.set_xlim(r.index[0], r.index[-1] + pd.Timedelta(days=210))
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in (2024, 2025, 2026)])
    headline(fig, "Oil and the shilling moved against contractors twice in 2026",
             "Brent crude (US$) and the UGX/US$ exchange rate, monthly averages, indexed to January 2025")
    save(fig, "fig_1_2_oil_fx_shock.png",
         "Brent crude price (US$ per barrel) and the UGX/US$ exchange rate (UGX per dollar; a rise is a weaker "
         "shilling), monthly averages indexed to January 2025 = 100, January 2024 to October 2026. Sources: EIA/FRED "
         "(Brent), Bank of Uganda via the MoFPED macro portal and the CiO macro extension (exchange rate).")


def fig_inflation(m, ix):
    now = ix["cipi_nowcast"]
    cipi = pd.concat([m.CIPI_ALL.dropna(), now.cipi_nowcast])
    cy = (100 * (cipi / cipi.shift(12) - 1)).loc["2021-01-01":]
    cpi = m.cpi_headline_inflation.loc["2021-01-01":].dropna()
    ppi = (100 * (m.ppi_materials / m.ppi_materials.shift(12) - 1)).loc["2021-01-01":].dropna()
    last_off = m.CIPI_ALL.last_valid_index()
    fig, ax = plt.subplots(figsize=(W, 3.2))
    ax.plot(cpi.index, cpi, color=PAL["grey"], lw=1.3)
    ax.plot(ppi.index, ppi, color=PAL["s3"])
    ax.plot(cy.loc[:last_off].index, cy.loc[:last_off], color=PAL["s1"])
    ax.plot(cy.loc[last_off:].index, cy.loc[last_off:], color=PAL["s1"], ls=(0, (2, 1.5)))
    ax.axhline(0, color=INK["muted"], lw=0.6)
    date_axis(ax)
    yfmt(ax, "{x:.0f}%")
    end_label(ax, cpi.index[-1], cpi.iloc[-1], f"Consumer prices {cpi.iloc[-1]:.1f}%", PAL["grey"], dy=6)
    end_label(ax, cy.index[-1], cy.iloc[-1], f"CIPI nowcast {cy.iloc[-1]:.1f}%", PAL["s1"], dy=4)
    end_label(ax, ppi.index[-1], ppi.iloc[-1], f"Materials PPI {ppi.iloc[-1]:.1f}%", PAL["s3"], dy=-9)
    ax.axvline(last_off, color=INK["muted"], lw=0.6)
    ax.text(last_off, ax.get_ylim()[1], " last UBOS CIPI\n release (Apr 2026)", fontsize=7.5, color=INK["muted"],
            va="top")
    ax.set_xlim(cy.index[0], cy.index[-1] + pd.Timedelta(days=640))
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(2021, 2027)])
    headline(fig, "Construction input inflation is turning up, but remains below consumer inflation",
             "Annual inflation, %: CIPI all items (dotted after April 2026 = nowcast), PPI building materials, headline CPI")
    save(fig, "fig_1_3_inflation.png",
         "Annual inflation in construction input prices (UBOS CIPI, all items; May–October 2026 is the Hove & Quill "
         "nowcast), producer prices of building materials (UBOS PPI classes 2392, 2394, 2395 and 2410, simple average) "
         "and headline consumer prices (UBOS CPI), January 2021 to October 2026.")


# ---------- chapter 2: money ----------
def fig_payment(ix, ura):
    p = ix["idx_payment"]["index"]
    t = ura[ura.series == "construction_sector_revenue"].set_index("fy").value
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.2), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.3})
    a.plot(p.index, p, color=PAL["s1"])
    date_axis(a, 2)
    base_band(a)
    a.set_title("Payment Stress Index (higher = more stress)", loc="left", color=INK["secondary"], fontsize=9)
    end_label(a, p.index[-1], p.iloc[-1], f"{p.iloc[-1]:.0f}\n{qlabel(p.index[-1])}", PAL["s1"], dx=5)
    a.set_xlim(p.index[0], p.index[-1] + pd.Timedelta(days=500))
    b.bar(range(len(t)), t.values, color=PAL["s1"], width=0.66)
    b.set_xticks(range(len(t)), [f"{y[2:4]}/{y[-2:]}" for y in t.index], fontsize=7.5, rotation=0)
    b.grid(axis="x", visible=False)
    b.set_title("Tax collected from construction, UGX bn", loc="left", color=INK["secondary"], fontsize=9)
    b.annotate(f"{t.iloc[-1]:,.0f}", (len(t) - 1, t.iloc[-1]), xytext=(0, 3), textcoords="offset points",
               ha="center", fontsize=8.5)
    for lbl in b.get_xticklabels()[::2]:
        lbl.set_visible(False)
    headline(fig, "Fiscal and credit conditions eased into 2025; the sector's tax base kept growing",
             "Left: H&Q index, quarterly (100 = FY2017/18–2024/25 average). Right: URA gross collections by fiscal year")
    save(fig, "fig_2_1_payment_stress_tax.png",
         "Left: Hove & Quill Payment Stress Index (fiscal deficit relative to expenditure, lending rate, and the "
         "inverse of growth in private credit and public expenditure; 100 = FY2017/18–FY2024/25 average, shaded band "
         "= ±1 standard deviation), to the last complete quarter of fiscal data. Right: gross tax collected from the "
         "construction sector (ISIC F), FY2015/16–FY2024/25. Sources: MoFPED, Bank of Uganda, URA annual data books.",
         top=0.76)


# ---------- chapter 3: prices ----------
def fig_cipi_nowcast(m, ix):
    now = ix["cipi_nowcast"]
    c = m.CIPI_ALL.loc["2023-01-01":].dropna()
    fig, ax = plt.subplots(figsize=(W, 3.2))
    x0 = c.index[-1]
    fan_x = [x0] + list(now.index)
    ax.fill_between(fan_x, [c.iloc[-1]] + list(now.lo80), [c.iloc[-1]] + list(now.hi80), color=PAL["s1_light"],
                    lw=0)
    ax.plot(c.index, c, color=PAL["s1"])
    ax.plot(fan_x, [c.iloc[-1]] + list(now.cipi_nowcast), color=PAL["s1"], ls=(0, (2, 1.5)))
    date_axis(ax, 1)
    end_label(ax, c.index[-1], c.iloc[-1], f"UBOS {c.iloc[-1]:.1f}\n(Apr 2026)", PAL["s1"], dx=-58, dy=14)
    end_label(ax, now.index[-1], now.cipi_nowcast.iloc[-1],
              f"Nowcast {now.cipi_nowcast.iloc[-1]:.1f}\n80% range {now.lo80.iloc[-1]:.0f}–{now.hi80.iloc[-1]:.0f}",
              PAL["s1"])
    ax.set_xlim(c.index[0], now.index[-1] + pd.Timedelta(days=190))
    ax.set_ylabel("index")
    headline(fig, "Six months without an official input price index: the nowcast",
             "Construction Input Price Index, all items, monthly; shaded = 80% range of the Hove & Quill nowcast")
    save(fig, "fig_3_1_cipi_nowcast.png",
         "UBOS Construction Input Price Index (all items) to the last published month (April 2026), and the Hove & Quill "
         "nowcast for May–October 2026 with its 80% range. The nowcast regresses the monthly change in the index on "
         "recent changes in Brent crude in shillings, the exchange rate and building-material producer prices; it is "
         "an estimate, not an official statistic, and will be replaced by UBOS releases when they appear.")


SUB = {"AGG": "Aggregates", "ALU": "Aluminium", "BRICK": "Bricks", "CEM": "Cement", "CLAY": "Clay products",
       "DIESEL": "Diesel", "IRON": "Iron", "IRONSTEEL": "Iron & steel", "LABOUR": "Labour", "LIME": "Lime",
       "MURRAM": "Murram", "NAILS": "Nails", "PAINT": "Paint", "PIPES": "Pipes", "SAND": "Sand", "STEEL": "Steel",
       "TILES": "Tiles", "TIMBER": "Timber"}


def fig_subindices(m):
    last = m.CIPI_ALL.last_valid_index()
    y = (100 * (m[list(SUB)].loc[last] / m[list(SUB)].loc[last - pd.DateOffset(years=1)] - 1)).rename(SUB)
    y = y.sort_values()
    fig, ax = plt.subplots(figsize=(W, 4.2))
    cols = [PAL["rise"] if v > 0 else PAL["fall"] for v in y]
    ax.barh(range(len(y)), y.values, color=cols, height=0.62)
    ax.set_yticks(range(len(y)), y.index)
    ax.axvline(0, color=INK["muted"], lw=0.6)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.0f}%"))
    for i, v in enumerate(y.values):
        if abs(v) >= 3 or i in (0, len(y) - 1):
            ax.annotate(f"{v:+.1f}%", (max(v, 0), i), xytext=(4, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8)
    t = list(y.index[::-1][:3])
    headline(fig, f"{t[0]}, {t[1].lower()} and {t[2].lower()} led input-cost rises in the year to April",
             f"Change in CIPI sub-indices, {last:%B %Y} on a year earlier (red = rise, blue = fall)", y=0.99)
    save(fig, "fig_3_2_subindices.png",
         f"Annual change in the 18 UBOS CIPI sub-indices, {last:%B %Y} against {last - pd.DateOffset(years=1):%B %Y}. "
         "Source: UBOS CIPI.", top=0.86)


def fig_ccpi(ix):
    c = ix["idx_ccpi"]
    s = c["index"]
    fig, ax = plt.subplots(figsize=(W, 3.2))
    ax.plot(s.index, s, color=PAL["s1"], lw=1.4)
    date_axis(ax, 1)
    base_band(ax)
    pk = s.loc["2021-06-01":"2023-06-01"].idxmax()
    ax.annotate(f"2022 peak {s[pk]:.0f}", (pk, s[pk]), xytext=(6, 4), textcoords="offset points", fontsize=8.5)
    pk2 = s.loc["2026-01-01":].idxmax()
    ax.annotate(f"May 2026 {s[pk2]:.0f}", (pk2, s[pk2]), xytext=(-70, 2), textcoords="offset points", fontsize=8.5)
    end_label(ax, s.index[-1], s.iloc[-1], f"{s.index[-1]:%b %Y} {s.iloc[-1]:.0f}\n(provisional)", PAL["s1"])
    ax.set_xlim(s.index[0], s.index[-1] + pd.Timedelta(days=330))
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(2018, 2027)])
    ax.set_ylabel("index, 100 = base average")
    headline(fig, "Cost pressure is back near its 2022 extremes",
             "Hove & Quill Construction Cost Pressure Index (CCPI-U), monthly; band = ±1 standard deviation of the base")
    save(fig, "fig_3_3_ccpi.png",
         "Hove & Quill Construction Cost Pressure Index (CCPI-U), October 2017 to September 2026: the average of "
         "standardised three-month changes in the CIPI, the UGX/US$ rate, Brent crude in shillings and building-material "
         "producer prices, the 12-month change in the lending rate and the New York Fed Global Supply Chain Pressure "
         "Index. 100 = FY2017/18–FY2024/25 average; 10 points = one standard deviation. Months after April 2026 use "
         "fewer components and are provisional.")


def fig_escalation(gap):
    g = gap[(gap.duration_m == 18) & gap.reportable & (gap.base_month >= "2021-01-01")].set_index("base_month")
    done, run = g[g.complete], g[~g.complete]
    fig, ax = plt.subplots(figsize=(W, 3.2))
    for col, color, lbl, dy in [("gap_fixed_price_pct", PAL["s2"], "Fixed price", 6),
                                ("gap_formula_pct", PAL["s1"], "With formula", -6)]:
        ax.plot(done.index, done[col], color=color)
        ax.plot(run.index, run[col], color=color, ls=(0, (2, 1.5)))
        end_label(ax, run.index[-1], run[col].iloc[-1], f"{lbl} {run[col].iloc[-1]:+.1f}%", color, dy=dy)
    ax.axhline(0, color=INK["muted"], lw=0.6)
    if len(run):
        ax.axvspan(run.index[0], run.index[-1], color=INK["band"], zorder=0, lw=0)
        ax.text(run.index[0], ax.get_ylim()[1], " contracts still\n running: gap to date", fontsize=7.5,
                color=INK["secondary"], va="top")
    date_axis(ax, 1)
    yfmt(ax, "{x:.0f}%")
    ax.set_xlabel("contract base month (prices fixed at tender)")
    ax.set_xlim(g.index[0], g.index[-1] + pd.Timedelta(days=560))
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(2021, 2026)])
    headline(fig, "Contracts priced in late 2025 are already carrying unrecovered escalation",
             "Escalation not recovered, % of contract value, 18-month contracts with equal monthly valuations")
    save(fig, "fig_3_4_escalation_gap.png",
         "Hove & Quill Escalation Gap: cost escalation not recovered by the contractor, as a percentage of contract "
         "value, for an 18-month contract by base month. The cost basket follows the formula weights of a Ugandan road "
         "contract (fuel, bitumen proxied by Brent in shillings, steel, cement, equipment proxied by the exchange rate, "
         "labour, metals). Fixed price: no adjustment. Formula: FIDIC 13.7-style adjustment with a 20% non-adjustable "
         "element and a one-month index lag. Dotted: contracts still running, measured on months valued to April 2026 "
         "(at least six). Negative values mean prices fell and the contractor gained.")


# ---------- chapter 4: pipeline ----------
def fig_procurement(q, ix):
    a_ = q.works_awards.loc["2019-01-01":].dropna()
    p = ix["idx_procurement"]["index"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.2), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.28})
    a.bar(a_.index, a_, width=70, color=PAL["s1"])
    date_axis(a, 2)
    a.set_title("Public works contracts awarded per quarter", loc="left", color=INK["secondary"], fontsize=9)
    a.annotate(f"{a_.iloc[-1]:.0f}", (a_.index[-1], a_.iloc[-1]), xytext=(0, 3), textcoords="offset points",
               ha="center", fontsize=8.5)
    b.plot(p.index, p, color=PAL["s1"])
    date_axis(b, 2)
    base_band(b)
    b.set_title("Procurement Health Index (higher = healthier)", loc="left", color=INK["secondary"], fontsize=9)
    end_label(b, p.index[-1], p.iloc[-1], f"{p.iloc[-1]:.0f}", PAL["s1"])
    b.set_xlim(p.index[0], p.index[-1] + pd.Timedelta(days=300))
    headline(fig, "Award volumes recovered in 2026; competition did not",
             "Government Procurement Portal (OCDS), works contracts, by quarter of award")
    save(fig, "fig_4_1_procurement.png",
         "Left: public works contracts awarded per quarter on the PPDA Government Procurement Portal. Right: Hove & "
         "Quill Procurement Health Index (mean bidders, single-bid share, share of non-open methods and median days "
         "from tender to award; 100 = base average). Portal coverage has changed over time, so levels before 2021 are "
         "not strictly comparable. Source: PPDA GPP via the Open Contracting Data Registry; last complete quarter shown.",
         top=0.76)


# ---------- chapter 5: delivery and supply ----------
def fig_supply(m, q):
    cem = q.iip_cement.dropna()
    st = m.imp_iron_steel.rolling(3).sum().dropna().loc["2017-01-01":] / 1e3
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.2), gridspec_kw={"wspace": 0.28})
    a.plot(cem.index, cem, color=PAL["s1"])
    date_axis(a, 2)
    a.set_title("Cement, lime and plaster output (index)", loc="left", color=INK["secondary"], fontsize=9)
    end_label(a, cem.index[-1], cem.iloc[-1], f"{cem.iloc[-1]:.0f}", PAL["s1"])
    a.set_xlim(cem.index[0], cem.index[-1] + pd.Timedelta(days=420))
    b.plot(st.index, st, color=PAL["s1"], lw=1.3)
    date_axis(b, 2)
    b.set_title("Iron and steel imports, US$ m (3-month total)", loc="left", color=INK["secondary"], fontsize=9)
    end_label(b, st.index[-1], st.iloc[-1], f"{st.iloc[-1]:.0f}", PAL["s1"])
    b.set_xlim(st.index[0], st.index[-1] + pd.Timedelta(days=420))
    headline(fig, "Materials supply is at record levels",
             "Left: UBOS Index of Industrial Production, ISIC 2394, 2016/17 = 100. Right: UBOS merchandise imports, SITC 67")
    save(fig, "fig_5_1_materials_supply.png",
         "Left: index of industrial production for the manufacture of cement, lime and plaster (ISIC 2394; 2016/17 = "
         "100), quarterly to Q4 FY2025/26. Right: imports of iron and steel (SITC 67), US$ million, rolling three-month "
         "total, to August 2026. Sources: UBOS Index of Industrial Production; UBOS composition of imports.", top=0.76)


def fig_weather(wx):
    regions = [r for r in wx.region.unique() if r != "region"]
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.0), sharey=True, sharex=True,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.12})
    for ax, r in zip(axes.flat, regions):
        w = wx[wx.region == r].sort_values("month")
        ax.bar(w.month - 0.18, w["1981-2000"], width=0.36, color=PAL["grey"], label="1981–2000")
        ax.bar(w.month + 0.18, w["2006-2025"], width=0.36, color=PAL["s1"], label="2006–2025")
        ax.set_title(r, loc="left", fontsize=8.5, color=INK["secondary"])
        ax.set_xticks([1, 4, 7, 10], ["Jan", "Apr", "Jul", "Oct"], fontsize=7.5)
        ax.grid(axis="x", visible=False)
    axes.flat[0].legend(loc="upper left", fontsize=7.5, handlelength=1, borderaxespad=0.1)
    fig.text(0.005, 0.45, "earthworks days lost per month", rotation=90, fontsize=8, color=INK["secondary"], va="center")
    headline(fig, "Rain costs most days in April–May and October–November; the north loses mid-year too",
             "Average earthworks days lost to rain per month, "
             "by region: 1981–2000 against 2006–2025", y=0.99)
    save(fig, "fig_5_2_weather_working_days.png",
         "Average working days lost to rainfall per month for earthworks, by region, for the 1981–2000 and 2006–2025 "
         "normals (a day is lost when rainfall exceeds the earthworks threshold). Source: CHIRPS/TAMSAT analysis in the "
         "uganda-seasons-construction-delay study.", top=0.84)


# ---------- chapter 6: disputes ----------
def fig_disputes(ix):
    d = ix["idx_dispute"]
    c4, sh = d.court_4q.dropna(), 100 * d.payment_share_4q.dropna()
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.2), gridspec_kw={"wspace": 0.28})
    a.plot(c4.index, c4, color=PAL["s1"])
    date_axis(a, 2)
    a.set_title("Construction decisions, rolling 4 quarters", loc="left", color=INK["secondary"], fontsize=9)
    end_label(a, c4.index[-1], c4.iloc[-1], f"{c4.iloc[-1]:.0f}", PAL["s1"])
    a.set_xlim(c4.index[0], c4.index[-1] + pd.Timedelta(days=420))
    b.plot(sh.index, sh, color=PAL["s2"])
    date_axis(b, 2)
    yfmt(b, "{x:.0f}%")
    b.set_title("Payment disputes, share of decisions", loc="left", color=INK["secondary"], fontsize=9)
    end_label(b, sh.index[-1], sh.iloc[-1], f"{sh.iloc[-1]:.0f}%", PAL["s2"])
    b.set_xlim(sh.index[0], sh.index[-1] + pd.Timedelta(days=420))
    headline(fig, "More decisions, and payment is the fight",
             "Courts of Judicature and PPDA Appeals Tribunal, construction-related decisions published on ULII")
    save(fig, "fig_6_1_dispute_climate.png",
         "Left: construction-related court and tribunal decisions published on ULII, rolling four-quarter total. Right: "
         "share of those decisions in which payment or certification was in issue. Counts partly reflect the wider "
         "publication of judgments on ULII. Source: Hove & Quill corpus of ULII decisions (to 16 November 2026).",
         top=0.76)


# ---------- signature pages ----------
LAW_CITED = (8, 34)   # enactments since 2015 cited in any judgment / all (stage 3; sector_report_2026.md section 2)


def fig_chain(sc, gap):
    v = sc.set_index("indicator")
    get = lambda k: v.loc[[i for i in v.index if i.startswith(k)][0]]
    g = gap[(gap.duration_m == 18) & gap.reportable & ~gap.complete].sort_values("base_month").iloc[-1]
    links = [
        ("Economy", f"{get('Construction real value added').value:.1f}%",
         f"real construction growth, {get('Construction real value added').period}"),
        ("Money", f"{get('Central Bank Rate').value:.1f}%", "Central Bank Rate, raised from 9.75% on 14 Oct 2026"),
        ("Prices", f"{get('Construction Cost Pressure').value:.0f}", "cost pressure index (100 = normal)"),
        ("Pipeline", f"{get('Public works awards').value:.0f}", f"public works awards, {get('Public works awards').period}"),
        ("Delivery", f"+{g.gap_fixed_price_pct:.0f}%",
         f"unrecovered escalation, fixed-price contracts priced {g.base_month:%b %Y}"),
        ("Disputes", f"{get('Construction court decisions').value:.0f}", "construction decisions in the last 4 quarters"),
        ("Law", f"{LAW_CITED[0]} of {LAW_CITED[1]}", "enactments since 2015 ever cited by a court"),
        ("Contracts", "5", "reform priorities: bonds, procurement, delay, payment, dispute clauses"),
    ]
    fig = plt.figure(figsize=(W, 3.6))
    ax = fig.add_axes([0, 0, 1, 0.84])
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2.1)
    ax.axis("off")
    bw, bh = 0.84, 0.92
    arrow = dict(arrowstyle="-|>", color=INK["muted"], lw=0.9, shrinkA=0, shrinkB=0)
    for i, (name, num, txt) in enumerate(links):
        row, col = divmod(i, 4)
        col = col if row == 0 else 3 - col           # snake: right along the top, back along the bottom
        x, y = col + 0.08, 1.12 - row * 1.08
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.05", fc=INK["band"], ec="none"))
        ax.text(x + 0.06, y + bh - 0.08, f"{i + 1}  {name.upper()}", fontsize=7.5, color=INK["accent"],
                fontweight="bold", va="top")
        ax.text(x + 0.06, y + bh - 0.27, num, fontsize=16, color=INK["primary"], fontweight="bold", va="top")
        ax.text(x + 0.06, y + 0.36, "\n".join(textwrap.wrap(txt, 23)), fontsize=7.3, color=INK["secondary"], va="top",
                linespacing=1.15)
        if i == 3:
            ax.annotate("", (x + bw / 2, y - 0.16), (x + bw / 2, y), arrowprops=arrow)
        elif i < 7:
            ax.annotate("", (x + bw + 0.16, y + bh / 2) if row == 0 else (x - 0.16, y + bh / 2),
                        (x + bw, y + bh / 2) if row == 0 else (x, y + bh / 2), arrowprops=arrow)
    headline(fig, "From cost to court: the 2026 chain in eight numbers",
             "Each link hands its pressure to the next; disputes follow shocks with a lag of about three years")
    save(fig, "fig_0_1_cost_to_court_chain.png",
         "The Cost-to-Court chain, 2026: one headline number for each link from the economy to contract reform. Values "
         "are the latest available in each source (see the scorecard for periods and sources). The legislative count is "
         "from the Hove & Quill legislation register (2015–2026).", top=1.0)


def half_up(x, nd=1):
    """Round half away from zero (2.25 -> 2.3), not to even."""
    f = 10 ** nd
    return float(np.sign(x) * np.floor(abs(x) * f + 0.5) / f)


def _spark(ax, s, color):
    s = s.dropna()
    s = s.loc[s.index >= s.index[-1] - pd.DateOffset(years=5)] if isinstance(s.index, pd.DatetimeIndex) else s
    ax.plot(range(len(s)), s.values, color=color, lw=0.9)
    ax.plot([len(s) - 1], [s.iloc[-1]], "o", ms=2.6, color=color)
    ax.axis("off")


def scorecard_page():
    sc = build_scorecard()
    fig = plt.figure(figsize=(8.27, 11.69))
    top, rowh = 0.865, 0.0312
    fig.text(0.06, 0.955, "Uganda Construction Sector Scorecard 2026", fontsize=20, fontweight="bold",
             color=INK["primary"])
    fig.text(0.06, 0.933, "Hove & Quill · Construction Practice · latest data to November 2026", fontsize=10.5,
             color=INK["secondary"])
    fig.text(0.06, 0.912, "Arrows show the change on a year earlier: green = good for the sector, red = adverse, "
             "grey = neutral.\nThe dot shows where the latest value sits within its FY2017/18–2024/25 range "
             "(0 = lowest, 100 = highest; tick = median).", fontsize=8.5, color=INK["muted"], va="center",
             linespacing=1.3)
    cols = {"ind": 0.06, "spark": 0.44, "val": 0.60, "chg": 0.715, "pct": 0.82}
    for k, t in [("ind", "Indicator"), ("spark", "Last 5 years"), ("val", "Latest"), ("chg", "On a year"),
                 ("pct", "Position vs base (0–100)")]:
        fig.text(cols[k], top + 0.012, t, fontsize=8, color=INK["muted"], fontweight="bold")
    y, notes = top, []
    for g, block in sc.groupby("group", sort=False):
        y -= 0.008
        fig.text(cols["ind"], y, g.upper(), fontsize=8.5, color=INK["accent"], fontweight="bold", va="center")
        fig.add_artist(plt.Line2D([0.06, 0.95], [y - 0.011, y - 0.011], color=INK["grid"], lw=0.6))
        y -= rowh * 0.72
        for _, r in block.iterrows():
            fig.text(cols["ind"], y, "\n".join(textwrap.wrap(r.indicator, 46)), fontsize=8.8, color=INK["primary"],
                     va="center", linespacing=1.05)
            mark = ""
            if r.note:
                notes.append(r.note)
                mark = f" · note {chr(96 + len(notes))}"
            fig.text(cols["ind"], y - 0.0125, f"{r.period} · {r.source}{mark}", fontsize=6.8, color=INK["muted"],
                     va="center")
            _spark(fig.add_axes([cols["spark"], y - 0.009, 0.13, 0.019]), r._series, PAL["grey"])
            val = f"{r.value:,.0f}" if abs(r.value) >= 100 or float(r.value).is_integer() else f"{r.value:,.1f}"
            fig.text(cols["val"], y + 0.003, val, fontsize=11, fontweight="bold", color=INK["primary"], va="center")
            fig.text(cols["val"], y - 0.0115, r.unit, fontsize=6.8, color=INK["muted"], va="center")
            if pd.notna(r.change_on_year):
                up = r.change_on_year > 0
                color = (PAL["good"] if (up == (r.good == "up")) else PAL["bad"]) if r.good else INK["muted"]
                unit = "%" if r.change_type == "%" else (" pp" if "%" in r.unit else " pts")
                fig.text(cols["chg"], y, f"{r.arrow} {half_up(r.change_on_year):+.1f}{unit}", fontsize=9, color=color,
                         va="center", fontweight="bold")
            if pd.notna(r.pctl_vs_base):
                ax = fig.add_axes([cols["pct"], y - 0.006, 0.12, 0.012])
                ax.set_xlim(-4, 104)
                ax.set_ylim(-1, 1)
                ax.axis("off")
                ax.plot([0, 100], [0, 0], color=INK["grid"], lw=2.2, solid_capstyle="round")
                ax.plot([50, 50], [-0.6, 0.6], color=INK["muted"], lw=0.5)
                ax.plot([r.pctl_vs_base], [0], "o", ms=5, color=PAL["s1"], mec=INK["surface"], mew=1)
            y -= rowh
    fig.text(0.06, y + 0.006, "\n".join(textwrap.wrap(
        "Notes: " + "; ".join(f"({chr(97 + i)}) {n}" for i, n in enumerate(notes)) + ".", 150)),
             fontsize=7, color=INK["muted"], va="top")
    fig.text(0.06, 0.035, "\n".join(textwrap.wrap(
        "Sources: UBOS (QGDP, CIPI, PPI, CPI, IIP, trade), Bank of Uganda and MoFPED macro portal, URA annual data books, "
        "PPDA Government Procurement Portal (OCDS), ULII decisions (Hove & Quill corpus), EIA/FRED. H&Q indices are "
        "standardised on FY2017/18–2024/25 (100 = average; 10 points = one standard deviation). The CIPI nowcast is a "
        "model estimate pending UBOS releases. Court counts partly reflect wider publication of judgments.", 150)),
        fontsize=7, color=INK["muted"], va="bottom")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_0_2_scorecard_page.png", dpi=300)
    fig.savefig(FIG / "fig_0_2_scorecard_page.pdf")
    plt.close(fig)
    CAPTIONS.append(("fig_0_2_scorecard_page.png", "Uganda Construction Sector Scorecard 2026 (full page)."))
    return sc


def main():
    m, q, ix, gap, ura, wx = load()
    sc = scorecard_page()
    fig_chain(sc, gap)
    fig_growth(q)
    fig_shock(m)
    fig_inflation(m, ix)
    fig_payment(ix, ura)
    fig_cipi_nowcast(m, ix)
    fig_subindices(m)
    fig_ccpi(ix)
    fig_escalation(gap)
    fig_procurement(q, ix)
    fig_supply(m, q)
    fig_weather(wx)
    fig_disputes(ix)
    lines = ["# Figure captions — sector report 2026", ""]
    for i, (n, c) in enumerate(sorted(CAPTIONS)):
        lines += [f"**{n}** — {c}", ""]
    (FIG / "captions.md").write_text("\n".join(lines))
    print(f"{len(CAPTIONS)} figures -> {FIG}")


if __name__ == "__main__":
    main()
