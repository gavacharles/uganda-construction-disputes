"""Stage 13 motion suite (PhD): smooth animations and two further static charts.

Animations are rendered frame by frame at 1920x1080, 30 fps (H.264 MP4, crf 18), with eased transitions and data
interpolated between observations; a lighter GIF preview (15 fps, 960 px wide, palette-optimised) is made from each MP4.
  m1  flow map      import routes into Uganda (2021) draw in, staggered by size, then particles flow along them
  m2  cost shock    Brent in shillings, UGX/US$, CIPI diesel and material PPI, July 2017 - October 2026, drawn continuously
  m3  budget Sankey approved -> released -> spent for public construction, morphing from year to year
  m4  who pays tax  every ISIC section: share of GDP vs share of tax, FY2017/18 - FY2024/25 (bubble size = tax)
  m5  dispute pulse construction- and arbitration-related decisions by district, 2015-2026, with pulses for each year
Static: v8 CIPI heat stripes (18 sub-indices, y/y), v9 tax vs output dumbbell by sector (FY2024/25).
Every frame carries the credit line (src.viz.credit).
Outputs: outputs/analysis/13_fiscal_triangulation/visuals/
Usage: python -m src.analysis.s13_motion [m1,m2,...]
"""
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.dates  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from src.analysis import s13_fiscal_extensions as ext  # noqa: E402
from src.analysis import s13_visuals as sv  # noqa: E402
from src.config import PROCESSED, ROOT  # noqa: E402
from src.viz.credit import credit  # noqa: E402

V = sv.V
INK, INK2, GRID, SURF = sv.INK, sv.INK2, sv.GRID, sv.SURF
BLUE, ORANGE, AQUA, VIOLET, GOLD, GREY = sv.BLUE, sv.ORANGE, sv.AQUA, sv.VIOLET, sv.GOLD, sv.GREY
FPS, DPI, SIZE = 30, 150, (12.8, 7.2)          # 1920 x 1080
CAPTIONS = []


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)                    # smoothstep


def ease_out(t):
    t = np.clip(t, 0, 1)
    return 1 - (1 - t) ** 3


def render(fig, draw, n, name, caption):
    """Draw frames 0..n-1 into an MP4 (30 fps) and derive a GIF preview."""
    V.mkdir(parents=True, exist_ok=True)
    mp4, gif = V / f"{name}.mp4", V / f"{name}.gif"
    w = animation.FFMpegWriter(fps=FPS, codec="libx264",
                               extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow"])
    with w.saving(fig, str(mp4), dpi=DPI):
        for i in range(n):
            draw(i)
            w.grab_frame()
    plt.close(fig)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4), "-vf",
                    "fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=sierra2_4a", str(gif)], check=True)
    CAPTIONS.append((f"{name}.mp4 / .gif", caption))
    print(f"{name}: {n} frames, {n / FPS:.1f} s")


def frame_title(fig, head, sub):
    fig.text(0.035, 0.955, head, fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.035, 0.895, sub, fontsize=12, color=INK2, va="top")
    credit(fig)


# ---------- m1: flow map with particles ----------
def m1_flowmap():
    t = sv.trade_flows()
    w = gpd.read_file(sv.NE)
    pts = {r.NAME: (r.LABEL_X, r.LABEL_Y) for r in w.itertuples()} | {r.ADMIN: (r.LABEL_X, r.LABEL_Y) for r in w.itertuples()}
    t = t.assign(ne_name=t.country.replace(sv.NAME_FIX))
    t = t[t.ne_name.isin(pts) & (t.ne_name != "Uganda")]
    by = t.groupby(["ne_name", "group"]).usd_m.sum().reset_index()
    by = by[by.usd_m >= 1].sort_values("usd_m", ascending=False).reset_index(drop=True)
    gcol = {"Steel": BLUE, "Cement and clinker": ORANGE, "Machinery and trucks": VIOLET, "Other inputs": AQUA}
    routes = [sv.arc(pts[r.ne_name], sv.KAMPALA, n=120, bend=0.18 if pts[r.ne_name][0] < sv.KAMPALA[0] else -0.18)
              for r in by.itertuples()]
    vmax = by.usd_m.max()
    totals = t.groupby("ne_name").usd_m.sum().nlargest(10)

    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0.0, 0.03, 1.0, 0.82])
    w.plot(ax=ax, color="#efede8", edgecolor="#d9d6cf", linewidth=0.4)
    w[w.NAME == "Uganda"].plot(ax=ax, color=INK, edgecolor="none")
    ax.set_xlim(-20, 150)
    ax.set_ylim(-38, 62)
    ax.set_aspect("equal")
    ax.axis("off")
    frame_title(fig, "Construction inputs flowing into Uganda, 2021",
                "Recorded imports by country of origin and input group; line width proportional to US$ value")
    for g, c in gcol.items():
        ax.plot([], [], color=c, lw=3, label=g)
    ax.legend(loc="lower left", frameon=False, fontsize=11)
    lines, dots = [], []
    rng = np.random.default_rng(42)
    for (r, p) in zip(by.itertuples(), routes):
        lw = 0.6 + 6.0 * np.sqrt(r.usd_m / vmax)
        ln, = ax.plot([], [], color=gcol[r.group], lw=lw, alpha=0.7, solid_capstyle="round", zorder=3)
        lines.append(ln)
        k = int(np.clip(np.round(r.usd_m / vmax * 6), 1, 6))
        d, = ax.plot([], [], "o", color="white", mec=gcol[r.group], mew=1.0, ms=2.2 + lw * 0.45, zorder=4)
        dots.append((d, rng.random(k)))
    ring, = ax.plot([], [], "o", ms=0, mfc="none", mec=INK, mew=1.4, zorder=5)
    ax.plot(*sv.KAMPALA, "o", ms=8, color=INK, mec=SURF, zorder=6)
    labels = [ax.text(pts[n][0] + (7 if n == "Kenya" else 0), pts[n][1] + 2.4,
                      f"{n.replace('United Arab Emirates', 'UAE')}  {v:,.0f}", fontsize=10, ha="center", color=INK,
                      alpha=0, zorder=7) for n, v in totals.items()]
    total_lab = ax.text(sv.KAMPALA[0] - 3, sv.KAMPALA[1] - 4, f"Kampala\nUS$ {t.usd_m.sum():,.0f} m", ha="right",
                        va="top", fontsize=11, fontweight="bold", color=INK, alpha=0, zorder=7)
    n = 13 * FPS
    stagger, grow = 4.5, 1.4                       # seconds

    def draw(i):
        s = i / FPS
        for j, (ln, p) in enumerate(zip(lines, routes)):
            start = stagger * j / max(1, len(lines) - 1)
            f = ease_out((s - start) / grow)
            k = int(f * (len(p) - 1)) + 1 if f > 0 else 0
            ln.set_data(p[:k, 0], p[:k, 1])
            d, phases = dots[j]
            if f >= 1:
                pos = ((s - start - grow) * 0.35 + phases) % 1.0
                idx = (pos * (len(p) - 1)).astype(int)
                d.set_data(p[idx, 0], p[idx, 1])
            else:
                d.set_data([], [])
        a = ease((s - 7.0) / 1.2)
        for lab in labels:
            lab.set_alpha(a)
        total_lab.set_alpha(a)
        ph = (s % 1.6) / 1.6
        ring.set_data([sv.KAMPALA[0]], [sv.KAMPALA[1]])
        ring.set_markersize(8 + 26 * ph)
        ring.set_alpha(0.6 * (1 - ph))

    render(fig, draw, n, "m1_flowmap_imports_2021",
           "Animated flow map of recorded construction-input imports into Uganda in 2021: routes draw in from the largest, "
           "then particles flow toward Kampala (number of particles and line width scale with value). Sources: URA (2022); "
           "Natural Earth; author's calculations.")


# ---------- m2: continuous cost shock ----------
def m2_cost_shock():
    m = pd.read_csv(PROCESSED / "sector2026" / "panel_monthly.csv", parse_dates=["date"], index_col="date")
    base = "2017-07-01"
    series = {"Brent crude in shillings": ("brent_ugx", ORANGE), "CIPI diesel (UBOS)": ("DIESEL", VIOLET),
              "Material producer prices": ("ppi_materials", AQUA), "UGX per US$": ("exchange_rate", BLUE)}
    d = pd.DataFrame({k: 100 * m[c] / m[c].loc[base] for k, (c, _) in series.items()}).loc[base:].dropna(how="all")
    x = matplotlib.dates.date2num(d.index.to_pydatetime())
    events = [("2020-04-01", "COVID-19\nlockdown"), ("2022-06-01", "Ukraine war\noil spike"),
              ("2026-04-01", "2026 oil and\nshilling shock")]
    ymax = np.nanmax(d.values) * 1.08
    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0.07, 0.1, 0.7, 0.72])
    frame_title(fig, "The cost of building in Uganda follows oil and the shilling",
                "Monthly indices, July 2017 = 100. Brent in shillings is what a road contractor ultimately pays for "
                "fuel and bitumen")
    fig.text(0.07, 0.025, "Sources: EIA/FRED (Brent), Bank of Uganda (exchange rate), UBOS CIPI (to April 2026) and PPI.",
             fontsize=9.5, color=INK2)
    ax.set_xlim(x[0], x[-1] + 20)
    ax.set_ylim(40, ymax)
    ax.axhline(100, color=GREY, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID)
    ax.xaxis_date()
    ax.tick_params(labelsize=11)
    art = {}
    for name, (_, c) in series.items():
        ln, = ax.plot([], [], color=c, lw=2.4, solid_capstyle="round")
        dot, = ax.plot([], [], "o", color=c, ms=7, mec=SURF, mew=1.2)
        lab = ax.text(0, 0, "", color=c, fontsize=11.5, va="center", clip_on=False, fontweight="bold")
        art[name] = (ln, dot, lab, d[name].values)
    ev = []
    for dt, lab in events:
        xv = matplotlib.dates.date2num(pd.Timestamp(dt))
        vl = ax.axvline(xv, color=GRID, lw=1.2, alpha=0, zorder=0)
        tx = ax.text(xv, ymax * 0.97, lab, fontsize=10.5, color=INK2, ha="center", va="top", alpha=0)
        ev.append((xv, vl, tx))
    date_lab = ax.text(0.015, 0.05, "", transform=ax.transAxes, fontsize=24, fontweight="bold", color=INK)
    lab_y = {}
    run, hold = 11 * FPS, 3 * FPS
    n = run + hold

    def draw(i):
        tpos = (len(x) - 1) * min(1.0, i / run)       # fractional month index, constant speed
        k = int(np.floor(tpos))
        f = tpos - k
        xc = x[k] + (x[min(k + 1, len(x) - 1)] - x[k]) * f
        targets = []
        for name, (ln, dot, lab, vals) in art.items():
            ok = ~np.isnan(vals[: k + 1])
            xs, ys = list(x[: k + 1][ok]), list(vals[: k + 1][ok])
            if k + 1 < len(x) and not np.isnan(vals[k + 1]) and not np.isnan(vals[k]):
                xs.append(xc)
                ys.append(vals[k] + (vals[k + 1] - vals[k]) * f)
            ln.set_data(xs, ys)
            if xs:
                dot.set_data([xs[-1]], [ys[-1]])
                targets.append([ys[-1], name, ys[-1]])
        targets.sort(key=lambda e: e[0])
        for j in range(1, len(targets)):
            targets[j][0] = max(targets[j][0], targets[j - 1][0] + 0.06 * (ymax - 40))
        for yv, name, val in targets:
            lab_y[name] = yv if name not in lab_y else lab_y[name] + 0.25 * (yv - lab_y[name])   # glide, no jumps
            ln, dot, lab, _ = art[name]
            lab.set_position((x[-1] + 70, lab_y[name]))
            lab.set_text(f"{name}  {val:.0f}")
        for xv, vl, tx in ev:
            a = ease((xc - xv) / 60)
            vl.set_alpha(a)
            tx.set_alpha(a)
        date_lab.set_text(matplotlib.dates.num2date(xc).strftime("%B %Y"))

    render(fig, draw, n, "m2_cost_shock",
           "Animated monthly indices (July 2017 = 100) of Brent crude in shillings, the UBOS CIPI diesel sub-index (to "
           "April 2026), building-material producer prices and the UGX/US$ rate, drawn continuously to October 2026.")


# ---------- m3: morphing budget Sankey ----------
def m3_budget_sankey():
    _, _, y = sv.budget_flows()
    yrs = [fy for fy in ext.FULL_YEARS if fy in y.index.get_level_values(0)]
    vals = []
    for fy in yrs:
        r = y.loc[fy]
        vals.append({f: (r.loc[f, "approved"], r.loc[f, "released"], r.loc[f, "spent"]) for f in ("gou_dev", "ext_fin")})
    scale_total = max(sum(v[f][0] for f in v) for v in vals) * 1.02
    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.76])
    frame_title(fig, "Public construction funds leak between approval and spending",
                "Central government, buildings and structures (items 3121xx), UGX trillion, year by year")
    year_lab = fig.text(0.965, 0.955, "", fontsize=22, fontweight="bold", color=INK, ha="right", va="top")
    rate_lab = fig.text(0.965, 0.895, "", fontsize=13, color=INK2, ha="right", va="top")
    hold, move = int(1.6 * FPS), int(1.0 * FPS)
    per = hold + move
    n = per * (len(yrs) - 1) + hold + FPS

    def draw(i):
        k, j = divmod(i, per)
        if k >= len(yrs) - 1:
            k, tt = len(yrs) - 1, 0.0
        else:
            tt = ease((j - hold) / move) if j > hold else 0.0
        a, b = vals[k], vals[min(k + 1, len(yrs) - 1)]
        v = {f: [a[f][s] + (b[f][s] - a[f][s]) * tt for s in range(3)] for f in a}
        ax.clear()
        links = []
        for f, name in (("gou_dev", "Government of Uganda"), ("ext_fin", "External finance")):
            links += [(name, "Released", v[f][1], None), (name, "Not released", max(v[f][0] - v[f][1], 1e-6), None)]
        rel, sp = v["gou_dev"][1] + v["ext_fin"][1], v["gou_dev"][2] + v["ext_fin"][2]
        links += [("Released", "Spent", sp, None), ("Released", "Released, not spent", max(rel - sp, 1e-6), None)]
        colors = {"Government of Uganda": BLUE, "External finance": ORANGE, "Released": AQUA, "Not released": GREY,
                  "Spent": AQUA, "Released, not spent": GREY}
        sv.sankey(ax, [["Government of Uganda", "External finance"], ["Released", "Not released"],
                       ["Spent", "Released, not spent"]], links, colors, unit=" tn",
                  side_labels=["Approved", "Released", "Spent"], scale_total=scale_total, label_size=13)
        lab = yrs[k] if tt < 0.5 else yrs[min(k + 1, len(yrs) - 1)]
        year_lab.set_text(f"FY{lab}")
        appr = v["gou_dev"][0] + v["ext_fin"][0]
        rate_lab.set_text(f"spent {100 * sp / appr:.0f}% of the UGX {appr:.1f} tn approved")

    render(fig, draw, n, "m3_budget_sankey",
           "Morphing Sankey of central-government funds for buildings and structures by funding source, from approval to "
           "release and spending, FY2016/17–FY2021/22 (FY2018/19 omitted). Source: MoFPED budget data portal.")


# ---------- m4: who pays tax (all sectors) ----------
UBOS_ROWS = {"A": "AGRICULTURE,FORESTRY&FISHING", "B": "Mining & quarrying", "C": "Manufacturing", "D": "Electricity",
             "E": "Water", "F": "Construction", "G": "Trade & Repairs", "H": "Transportation & Storage",
             "I": "Accommodation & Food Service", "J": "Information & Communication", "K": "Financial and Insurance",
             "L": "Real Estate Activities", "M": "Professional, Scientific & Technical",
             "N": "Administrative & Support Service", "O": "Public Administration", "P": "Education",
             "Q": "Human Health & Social Work", "R": "Arts, Entertainment & Recreation", "S": "Other Service Activities"}
SHORT = {"A": "Agriculture", "B": "Mining", "C": "Manufacturing", "D": "Electricity", "E": "Water", "F": "Construction",
         "G": "Trade", "H": "Transport", "I": "Hotels and food", "J": "ICT", "K": "Finance", "L": "Real estate",
         "M": "Professional services", "N": "Administrative services", "O": "Public administration", "P": "Education",
         "Q": "Health", "R": "Arts", "S": "Other services"}


def sector_panel():
    from src.report2026.parse_ubos import FY_Q_START, fyq_to_date, latest
    f = latest("*QGDP*Current*")
    d = pd.read_excel(f, sheet_name="Original_VA", header=None)
    fy = d.iloc[2].ffill()
    cols = [c for c in d.columns[1:] if str(d.iloc[3, c]).strip() in FY_Q_START and pd.notna(fy[c])]
    fys = [ext.fy_of(fyq_to_date(fy[c], str(d.iloc[3, c]).strip())) for c in cols]
    lab = d[0].astype(str).str.strip()
    va = {}
    for code, row in UBOS_ROWS.items():
        r = d.index[lab == row][0]
        va[code] = pd.Series(d.loc[r, cols].astype(float).values, index=fys).groupby(level=0).sum()
    gdp = pd.Series(d.loc[d.index[lab == "GDP AT MARKET PRICES"][0], cols].astype(float).values, index=fys) \
        .groupby(level=0).sum()
    nq = pd.Series(1, index=fys).groupby(level=0).sum()
    b, raw = ext.books()
    ura = pd.read_csv(PROCESSED / "sector2026" / "ura_series.csv").pivot(index="fy", columns="series", values="value")
    total = ura.sector_table_total.fillna(ura.sector_rows_sum)
    rows = []
    for code in UBOS_ROWS:
        tax = b.get(f"sector_{code}")
        if tax is None:
            continue
        for y in sorted(set(tax.dropna().index) & set(nq[nq == 4].index)):
            if y < "2017/18":
                continue
            rows.append({"fy": y, "code": code, "sector": SHORT[code], "tax_ugx_bn": tax[y],
                         "tax_share_pct": 100 * tax[y] / total[y], "gdp_share_pct": 100 * va[code][y] / gdp[y]})
    p = pd.DataFrame(rows)
    p["relative_intensity"] = p.tax_share_pct / p.gdp_share_pct
    p.to_csv(sv.S / "sector_tax_vs_gdp.csv", index=False)
    return p


def m4_who_pays():
    p = sector_panel()
    yrs = sorted(p.fy.unique())
    codes = sorted(p.code.unique())
    P = {c: p[p.code == c].set_index("fy").reindex(yrs).interpolate().bfill().ffill() for c in codes}
    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0.08, 0.1, 0.86, 0.72])
    frame_title(fig, "Who pays tax relative to the output they produce?",
                "Each bubble is an economic sector: share of GDP (x) against share of gross URA collections (y); bubble "
                "area proportional to tax paid")
    fig.text(0.08, 0.025, "Sources: URA annual data books; UBOS Quarterly GDP (current prices); author's calculations.",
             fontsize=9.5, color=INK2)
    lo, hi = 0.08, 40
    year_lab = fig.text(0.93, 0.16, "", fontsize=40, fontweight="bold", color="#d9d6cf", ha="right", va="bottom")
    smax = p.tax_ugx_bn.max()
    hold, move = int(1.3 * FPS), int(1.1 * FPS)
    per = hold + move
    n = per * (len(yrs) - 1) + hold + FPS
    trail = []
    big = {"F", "C", "G", "A", "K", "J", "O"}

    def draw(i):
        k, j = divmod(i, per)
        if k >= len(yrs) - 1:
            k, tt = len(yrs) - 1, 0.0
        else:
            tt = ease((j - hold) / move) if j > hold else 0.0
        a, b = yrs[k], yrs[min(k + 1, len(yrs) - 1)]
        ax.clear()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--", zorder=1)
        ax.text(0.16, 0.19, "taxed in proportion to output", fontsize=11, color=INK2, rotation=36.5,
                rotation_mode="anchor", ha="left", va="bottom")
        ax.text(0.1, 25, "pays more than its share", fontsize=11, color=INK2)
        ax.text(12, 0.1, "pays less than its share", fontsize=11, color=INK2, ha="right")
        for c in codes:
            ra, rb = P[c].loc[a], P[c].loc[b]
            gx = np.exp(np.log(ra.gdp_share_pct) + (np.log(rb.gdp_share_pct) - np.log(ra.gdp_share_pct)) * tt)
            gy = np.exp(np.log(ra.tax_share_pct) + (np.log(rb.tax_share_pct) - np.log(ra.tax_share_pct)) * tt)
            sz = ra.tax_ugx_bn + (rb.tax_ugx_bn - ra.tax_ugx_bn) * tt
            col = ORANGE if c == "F" else BLUE
            ax.scatter(gx, gy, s=40 + 2600 * sz / smax, color=col, alpha=0.85 if c == "F" else 0.45,
                       edgecolor=SURF, linewidth=1.2, zorder=4 if c == "F" else 3)
            ax.text(gx * 1.07, gy * 1.07, P[c].sector.iloc[0], fontsize=13 if c == "F" else 10.5 if c in big else 9,
                    color=ORANGE if c == "F" else INK if c in big else INK2, fontweight="bold" if c == "F" else None,
                    zorder=5)
            if c == "F":
                trail.append((gx, gy))
        if len(trail) > 1:
            tx, ty = zip(*trail)
            ax.plot(tx, ty, color=ORANGE, lw=1.2, alpha=0.5, zorder=2)
        ax.set_xlabel("share of GDP (%, log scale)", fontsize=12)
        ax.set_ylabel("share of gross tax collections (%, log scale)", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=GRID, lw=0.6, which="major")
        ax.tick_params(labelsize=11)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        year_lab.set_text(f"FY{a if tt < 0.5 else b}")

    render(fig, draw, n, "m4_who_pays_tax",
           "Animated bubble chart of ISIC sections, FY2017/18–FY2024/25: share of GDP at current prices against share of "
           "gross URA collections (log scales), bubble area proportional to tax paid, construction highlighted with its "
           "path. Sources: URA annual data books; UBOS; author's calculations.")
    return p


# ---------- m5: dispute pulse map ----------
def m5_dispute_pulse():
    g = gpd.read_file(ROOT / "data" / "external" / "procurement-risk-uganda" / "docs" / "data" / "districts.geojson")
    loc = pd.read_parquet(PROCESSED / "case_locations.parquet")
    a = pd.read_parquet(PROCESSED / "analysis_cases.parquet")[["case_id", "year", "stream"]]
    m = loc.merge(a, on="case_id")
    m = m[m.district.notna() & m.district.isin(g.district)]
    cent = g.set_index("district").geometry.representative_point()
    yrs = list(range(2015, 2027))
    cnt = m.groupby(["year", "district"]).size().unstack(fill_value=0).reindex(yrs, fill_value=0)
    cum = cnt.cumsum()
    per_year = m.groupby("year").size().reindex(yrs, fill_value=0)
    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0.02, 0.05, 0.6, 0.8])
    bx = fig.add_axes([0.66, 0.16, 0.3, 0.55])
    frame_title(fig, "Where construction disputes reach the courts, 2015–2026",
                "Construction- and arbitration-related decisions published on ULII, by the district the dispute concerns; "
                "circle area = decisions to date")
    fig.text(0.02, 0.025, "Districts geoparsed from the judgments (rule-based). 2026 to 16 November. Source: ULII; "
             "author's coding.", fontsize=9.5, color=INK2)
    g.plot(ax=ax, color="#efede8", edgecolor="#d9d6cf", linewidth=0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    xy = np.array([[cent[d].x, cent[d].y] for d in cnt.columns])
    scat = ax.scatter(xy[:, 0], xy[:, 1], s=np.zeros(len(xy)), color=ORANGE, alpha=0.75, edgecolor=SURF, lw=0.8, zorder=3)
    rings = ax.scatter(xy[:, 0], xy[:, 1], s=np.zeros(len(xy)), facecolor="none", edgecolor=ORANGE, lw=1.4, zorder=2)
    year_lab = ax.text(0.03, 0.06, "", transform=ax.transAxes, fontsize=30, fontweight="bold", color=INK)
    bars = bx.bar(yrs, np.zeros(len(yrs)), color=BLUE, width=0.7)
    bx.set_ylim(0, per_year.max() * 1.18)
    bx.set_xticks([2015, 2018, 2021, 2024, 2026])
    bx.spines[["top", "right"]].set_visible(False)
    bx.grid(axis="y", color=GRID)
    bx.set_title("Decisions located to a district, by year", fontsize=12, color=INK2, loc="left")
    kampala = bx.text(0.02, -0.22, "", transform=bx.transAxes, fontsize=11, color=INK2)
    smax = cum.values.max()
    per = int(1.3 * FPS)
    n = per * len(yrs) + 2 * FPS

    def draw(i):
        k, j = divmod(i, per)
        k = min(k, len(yrs) - 1)
        tt = ease(j / (per * 0.6)) if i < per * len(yrs) else 1.0
        prev = cum.iloc[k - 1].values if k > 0 else np.zeros(len(xy))
        now = prev + (cum.iloc[k].values - prev) * tt
        scat.set_sizes(10 + 1800 * np.sqrt(now / smax) ** 2)
        new = cnt.iloc[k].values > 0
        ph = (j / per) if i < per * len(yrs) else 1.0
        rings.set_sizes(np.where(new, (10 + 1800 * now / smax) * (1 + 6 * ph), 0))
        rings.set_alpha(max(0.0, 0.7 * (1 - ph)))
        for yy, bar in zip(yrs, bars):
            bar.set_height(per_year[yy] if yy < yrs[k] else per_year[yy] * tt if yy == yrs[k] else 0)
            bar.set_color(ORANGE if yy == yrs[k] else BLUE)
        year_lab.set_text(str(yrs[k]))
        kc = int(round(now[list(cnt.columns).index("Kampala")])) if "Kampala" in cnt.columns else 0
        kampala.set_text(f"Kampala: {kc} of {int(round(now.sum()))} decisions to date")

    render(fig, draw, n, "m5_dispute_pulse",
           "Animated map of construction- and arbitration-related court decisions by the district the dispute concerns, "
           "2015–2026: circles grow with the cumulative count and pulse in years with new decisions; the inset shows the "
           "annual count. Source: ULII; author's coding.")


# ---------- static: v8 CIPI heat stripes, v9 dumbbell ----------
CIPI_SUBS = {"AGG": "Aggregates", "ALU": "Aluminium", "BRICK": "Bricks", "CEM": "Cement", "CLAY": "Clay products",
             "DIESEL": "Diesel", "IRON": "Iron sheets", "IRONSTEEL": "Iron and steel", "LABOUR": "Labour", "LIME": "Lime",
             "MURRAM": "Murram", "NAILS": "Nails", "PAINT": "Paint", "PIPES": "Pipes", "SAND": "Sand", "STEEL": "Steel",
             "TILES": "Tiles", "TIMBER": "Timber", "CIPI_ALL": "All items"}


def v8_heat_stripes():
    m = pd.read_csv(PROCESSED / "sector2026" / "panel_monthly.csv", parse_dates=["date"], index_col="date")
    y = pd.DataFrame({lab: 100 * (m[c] / m[c].shift(12) - 1) for c, lab in CIPI_SUBS.items() if c in m}).dropna(how="all")
    y = y.loc["2018-07-01":"2026-04-01"]
    order = y.drop(columns="All items").mean().sort_values(ascending=False).index.tolist() + ["All items"]
    y = y[order]
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f4f3f0", ORANGE])
    cmap.set_bad("#d9d6cf")                                   # months with no published index
    lim = np.nanpercentile(np.abs(y.values), 97)
    fig, ax = plt.subplots(figsize=(13, 7.2))
    im = ax.imshow(np.ma.masked_invalid(y.T.values), aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest")
    ax.set_yticks(range(len(order)), order, fontsize=10)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.axhline(len(order) - 1.5, color=SURF, lw=3)
    ticks = [i for i, d in enumerate(y.index) if d.month == 7]
    ax.set_xticks(ticks, [f"Jul {d.year}" for d in y.index[ticks]], fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("annual change, %", fontsize=10)
    cb.outline.set_visible(False)
    sv.title(fig, "Construction input inflation, material by material",
             "UBOS Construction Input Price Index sub-indices, annual % change by month, July 2018–April 2026 "
             "(orange = rising, blue = falling, grey = not published); sorted by average inflation")
    fig.text(0.03, 0.02, "Source: UBOS CIPI; author's calculations.", fontsize=9, color=INK2)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.12, right=0.97)
    sv.save(fig, "v8_cipi_heat_stripes.png",
            "Annual percentage change in each UBOS CIPI sub-index and the all-items index, monthly, July 2018–April 2026, "
            "rows sorted by average inflation; colour scale clipped at the 97th percentile. Source: UBOS; author's "
            "calculations.")


def v9_dumbbell(p):
    yr = p.fy.max()
    q = p[p.fy == yr].sort_values("relative_intensity")
    fig, ax = plt.subplots(figsize=(11, 7.6))
    yv = np.arange(len(q))
    for k, r in enumerate(q.itertuples()):
        col = ORANGE if r.code == "F" else INK2
        ax.plot([r.gdp_share_pct, r.tax_share_pct], [k, k], color=GRID if r.code != "F" else ORANGE, lw=3,
                solid_capstyle="round", zorder=1)
        ax.scatter(r.gdp_share_pct, k, s=70, color=GREY if r.code != "F" else ORANGE, zorder=3,
                   edgecolor=SURF, linewidth=1)
        ax.scatter(r.tax_share_pct, k, s=70, color=BLUE if r.code != "F" else ORANGE, marker="D", zorder=3,
                   edgecolor=SURF, linewidth=1)
        ax.text(max(r.gdp_share_pct, r.tax_share_pct) * 1.12, k, f"×{r.relative_intensity:.2f}", va="center",
                fontsize=9, color=col, fontweight="bold" if r.code == "F" else None)
    ax.set_xscale("log")
    ax.set_yticks(yv, q.sector, fontsize=10.5)
    for lbl, c in zip(ax.get_yticklabels(), q.code):
        if c == "F":
            lbl.set_color(ORANGE)
            lbl.set_fontweight("bold")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xlim(0.03, 80)
    ax.scatter([], [], s=70, color=GREY, label="share of GDP")
    ax.scatter([], [], s=70, color=BLUE, marker="D", label="share of tax")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID)
    ax.tick_params(axis="y", length=0)
    sv.title(fig, f"Tax share against output share, by sector, FY{yr}",
             "Grey dot = share of GDP; blue diamond = share of gross URA collections; label = ratio of the two "
             "(1 = taxed in proportion to output)")
    fig.text(0.03, 0.02, "Sources: URA annual data book; UBOS Quarterly GDP (current prices); author's calculations.",
             fontsize=9, color=INK2)
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.2, right=0.97)
    sv.save(fig, "v9_tax_vs_output_dumbbell.png",
            f"Share of GDP at current prices and share of gross URA collections by ISIC section, FY{yr}, sorted by their "
            "ratio. Sources: URA; UBOS; author's calculations.")


def main(only=None):
    jobs = {"m1": m1_flowmap, "m2": m2_cost_shock, "m3": m3_budget_sankey, "m4": m4_who_pays, "m5": m5_dispute_pulse,
            "v8": v8_heat_stripes, "v9": lambda: v9_dumbbell(sector_panel())}
    for k, fn in jobs.items():
        if not only or k in only:
            fn()
    caps = V / "captions_motion.md"
    old = caps.read_text() if caps.exists() and only else ""
    caps.write_text((old or "# Stage 13 motion suite — captions\n\n") +
                    "\n\n".join(f"**{n}** — {c}" for n, c in CAPTIONS + sv.CAPTIONS) + "\n")


if __name__ == "__main__":
    main(only=sys.argv[1].split(",") if len(sys.argv) > 1 else None)
