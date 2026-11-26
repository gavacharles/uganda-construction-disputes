"""Stage 13 visual suite (PhD): Sankeys, animated flow map, animated cost shock, animated budget waterfall, polar quarter
clocks and a tax waffle, built from the stage-13 outputs. Matplotlib only (custom Sankey bands); animations via ffmpeg.

Outputs: outputs/analysis/13_fiscal_triangulation/visuals/ (PNG 300 dpi, GIF, MP4) + captions.md
Usage: python -m src.analysis.s13_visuals [v1,v3,...]   (run src.analysis.s13_fiscal_triangulation first)
"""
import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle, Wedge  # noqa: E402
from matplotlib.path import Path  # noqa: E402

from src.analysis import s13_fiscal_extensions as ext  # noqa: E402
from src.analysis.common import ANALYSIS_DIR, C  # noqa: E402
from src.config import PROCESSED, ROOT  # noqa: E402

S = ANALYSIS_DIR / "13_fiscal_triangulation"
V = S / "visuals"
NE = ROOT / "data" / "external" / "naturalearth" / "ne_110m_admin_0_countries.zip"
INK, INK2, GRID, SURF = "#1c1b19", "#5d5b56", "#e6e4df", "#ffffff"
BLUE, ORANGE, AQUA, VIOLET, GOLD, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#7a5cc4", "#c4a35c", "#a7a6a2"
KAMPALA = (32.58, 0.35)
CAPTIONS = []
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def save(fig, name, caption, dpi=300):
    V.mkdir(parents=True, exist_ok=True)
    fig.savefig(V / name, dpi=dpi)
    plt.close(fig)
    CAPTIONS.append((name, caption))


def title(fig, head, sub, y=0.975):
    fig.text(0.03, y, head, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.03, y - 0.055, sub, fontsize=10.5, color=INK2, va="top")


# ---------- Sankey engine ----------
def band(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=0.45):
    """Filled band from [y0a, y0b] at x0 to [y1a, y1b] at x1 (cubic Bezier edges)."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=1))


def sankey(ax, columns, links, colors, gap=0.025, width=0.018, label_fmt="{:,.1f}", unit="", side_labels=None):
    """columns: list of lists of node names (left to right); links: (src, dst, value, link colour or None)."""
    inflow, outflow = {}, {}
    for s, d, v, _ in links:
        outflow[s] = outflow.get(s, 0) + v
        inflow[d] = inflow.get(d, 0) + v
    size = {n: max(inflow.get(n, 0), outflow.get(n, 0)) for col in columns for n in col}
    total = max(sum(size[n] for n in col) for col in columns)
    pos = {}
    xs = np.linspace(0.06, 0.94, len(columns))
    for x, col in zip(xs, columns):
        h = sum(size[n] for n in col) / total * (1 - gap * (len(columns[0]) + 1))
        y = 0.5 + (h + gap * (len(col) - 1)) / 2
        for n in col:
            hn = size[n] / total * (1 - gap * (len(columns[0]) + 1))
            pos[n] = [x, y - hn, y, y, y]           # x, bottom, top, out-cursor, in-cursor
            y -= hn + gap
    scale = (1 - gap * (len(columns[0]) + 1)) / total
    # stack each node's outgoing links by destination position, incoming links by source position (no crossings
    # at the node ends)
    out_off = {}
    for s_, d, v, col in sorted(links, key=lambda l: (l[0], -pos[l[1]][2])):
        out_off[(s_, d)] = pos[s_][3]
        pos[s_][3] -= v * scale
    in_off = {}
    for s_, d, v, col in sorted(links, key=lambda l: (l[1], -pos[l[0]][2])):
        in_off[(s_, d)] = pos[d][4]
        pos[d][4] -= v * scale
    for s_, d, v, col in links:
        hs = v * scale
        y0a, y1a = out_off[(s_, d)], in_off[(s_, d)]
        band(ax, pos[s_][0] + width / 2, y0a, y0a - hs, pos[d][0] - width / 2, y1a, y1a - hs,
             col or colors.get(s_, GREY))
    last = len(columns) - 1
    for i, col in enumerate(columns):
        for n in col:
            x, b, t = pos[n][:3]
            ax.add_patch(Rectangle((x - width / 2, b), width, t - b, color=colors.get(n, INK2), zorder=3, lw=0))
            ha, tx = ("right", x - width) if i == 0 else ("left", x + width)
            small = (t - b) < 0.045
            txt = f"{n.replace(chr(10), ' ')}  {label_fmt.format(size[n])}{unit}" if small else \
                f"{n}\n{label_fmt.format(size[n])}{unit}"
            box = None if i in (0, last) else dict(boxstyle="round,pad=0.25", fc=SURF, ec="none", alpha=0.85)
            ax.text(tx, (b + t) / 2, txt, ha=ha, va="center", fontsize=9, color=INK, zorder=4, linespacing=1.15,
                    bbox=box)
    if side_labels:
        for x, lab in zip(xs, side_labels):
            ax.text(x, 1.02, lab, ha="center", va="bottom", fontsize=9.5, color=INK2, fontweight="bold")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.02, 1.06)
    ax.axis("off")


# ---------- data ----------
def budget_flows():
    d = ext.mofped()
    c = d[d.constr & d.fy.isin(ext.FULL_YEARS)]
    asset = c.item_desc.astype(str).str.lower()
    c = c.assign(asset=np.select([asset.str.contains("road"), asset.str.contains("non-residential"),
                                  asset.str.contains("residential"), asset.str.contains("other structures")],
                                 ["Roads and bridges", "Non-residential buildings", "Residential buildings",
                                  "Other structures"], "Other"))
    c = c[c.fund.isin(["gou_dev", "ext_fin"])]
    g = c.groupby(["fund", "stage"]).amount.sum().unstack() / 1e12
    sp = c[c.stage == "spent"].groupby("asset").amount.sum() / 1e12
    yearly = c.groupby(["fy", "fund", "stage"]).amount.sum().unstack() / 1e12
    return g, sp, yearly


def trade_flows():
    rows = pd.read_csv(PROCESSED / "fiscal_triangulation" / "ura_trade_2021.csv", dtype={"hs8": str})
    groups = {"Cement and clinker": ["2523"], "Steel": ["72", "7303", "7304", "7305", "7306", "7307", "7308", "7314",
                                                            "7317"],
              "Machinery and trucks": ["8426", "8429", "8430", "8474", "870410"],
              "Other inputs": ["2520", "2522", "2505", "2515", "2516", "2517", "271320", "2714", "2715", "6907", "6908",
                               "6910", "7003", "7004", "7005", "3208", "3209", "3210", "8544"]}
    digits = rows.hs8.str.replace(".", "", regex=False)
    out = []
    for g, pref in groups.items():
        m = digits.str.startswith(tuple(pref)) & ~digits.str.startswith("7204")
        out.append(rows[m].assign(group=g))
    t = pd.concat(out)
    t = t[t.country != "unmatched"]
    fx = pd.read_csv(PROCESSED / "sector2026" / "panel_monthly.csv", parse_dates=["date"], index_col="date") \
        .exchange_rate.loc["2021"].mean()
    t["usd_m"] = t.value_ugx_bn * 1e3 / fx
    return t.groupby(["country", "group"]).usd_m.sum().reset_index()


NAME_FIX = {"United Republic of Tanzania": "Tanzania", "Korea, Republic": "South Korea", "Russian Federation": "Russia",
            "United States": "United States of America", "Viet Nam": "Vietnam", "Iran, Islamic Republic": "Iran",
            "Congo, The Democratic Republic": "Dem. Rep. Congo", "Turkey": "Turkey", "Czech Republic": "Czechia"}


# ---------- V1: public construction shilling ----------
def v_sankey_budget():
    g, sp, _ = budget_flows()
    fund = {"gou_dev": "Government of Uganda", "ext_fin": "External finance"}
    links = []
    for f, name in fund.items():
        links += [(f"{name}\n(approved)", "Released", g.loc[f, "released"], None),
                  (f"{name}\n(approved)", "Not released", g.loc[f, "approved"] - g.loc[f, "released"], None)]
    rel, spent = g.released.sum(), g.spent.sum()
    links += [("Released", "Spent", spent, None), ("Released", "Released, not spent", rel - spent, None)]
    for a, v in sp.sort_values(ascending=False).items():
        links.append(("Spent", a, v, None))
    colors = {"Government of Uganda\n(approved)": BLUE, "External finance\n(approved)": ORANGE, "Released": AQUA,
              "Not released": GREY, "Spent": AQUA, "Released, not spent": GREY}
    colors.update({a: c for a, c in zip(sp.sort_values(ascending=False).index, [VIOLET, BLUE, GOLD, ORANGE, GREY])})
    fig, ax = plt.subplots(figsize=(13, 7.2))
    sankey(ax, [list(colors)[:2], ["Released", "Not released"], ["Spent", "Released, not spent"],
                list(sp.sort_values(ascending=False).index)], links, colors, unit=" tn",
           side_labels=["Approved", "Released", "Spent", "Built"])
    approved = g.approved.sum()
    title(fig, f"Of every UGX 100 budgeted for public buildings and structures, about UGX {100 * spent / g.approved.sum():.0f} was spent",
          f"Central government, items 3121xx, five fiscal years pooled (FY2016/17–17/18, FY2019/20–21/22), UGX trillion. "
          f"Approved {approved:.1f} tn; released {rel:.1f} tn; spent {spent:.1f} tn")
    fig.text(0.03, 0.03, "Externally financed works lose most on release: "
             f"{100 * g.loc['ext_fin', 'spent'] / g.loc['ext_fin', 'approved']:.0f}% of the approved amount was spent, "
             f"against {100 * g.loc['gou_dev', 'spent'] / g.loc['gou_dev', 'approved']:.0f}% for government-funded works. "
             "Source: MoFPED budget data portal; author's calculations.", fontsize=9, color=INK2)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.08, right=0.92)
    save(fig, "v1_sankey_public_construction.png",
         "Flow of central-government funds for buildings and structures (items 3121xx, all votes) from approval to "
         "release, spending and asset type, five fiscal years pooled (FY2016/17, FY2017/18, FY2019/20–FY2021/22; "
         "FY2018/19 omitted for lack of full-year data). Source: MoFPED budget data portal; author's calculations.")
    return approved, rel, spent


# ---------- V2: imports Sankey ----------
def v_sankey_imports(t):
    top = t.groupby("country").usd_m.sum().nlargest(9).index
    t = t.assign(origin=np.where(t.country.isin(top), t.country.replace(
        {"United Arab Emirates": "UAE", "United Republic of Tanzania": "Tanzania", "Russian Federation": "Russia",
         "Korea, Republic": "South Korea"}),
        "Other countries"))
    f = t.groupby(["origin", "group"]).usd_m.sum().reset_index()
    order_o = f.groupby("origin").usd_m.sum().sort_values(ascending=False).index.tolist()
    order_o = [o for o in order_o if o != "Other countries"] + ["Other countries"]
    order_g = ["Steel", "Cement and clinker", "Machinery and trucks", "Other inputs"]
    gcol = {"Steel": BLUE, "Cement and clinker": ORANGE, "Machinery and trucks": VIOLET, "Other inputs": AQUA}
    colors = {o: INK2 for o in order_o} | gcol | {"Uganda": INK}
    links = [(r.origin, r.group, r.usd_m, gcol[r.group]) for r in f.itertuples()]
    links += [(g, "Uganda", f[f.group == g].usd_m.sum(), gcol[g]) for g in order_g]
    fig, ax = plt.subplots(figsize=(13, 7.6))
    sankey(ax, [order_o, order_g, ["Uganda"]], links, colors, label_fmt="{:,.0f}", unit=" m",
           side_labels=["Recorded origin", "Input group", ""])
    title(fig, "Where Uganda's imported construction inputs came from, 2021",
          "Recorded import values, US$ million, by country of origin and input group (countries matched to URA's "
          "country blocks; 92.5% of value assigned)")
    fig.text(0.03, 0.03, "Kenya supplied most of the cement clinker; Japan most of the flat-rolled steel; China most of "
             "the machinery. Source: URA Annual Imports Trade Report 2021; author's calculations.", fontsize=9, color=INK2)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.1, right=0.95)
    save(fig, "v2_sankey_imports_2021.png",
         "Recorded imports of construction inputs in 2021 by country of origin and input group, US$ million. Steel = "
         "HS 72 (excl. scrap) and 7303–7308, 7314, 7317; machinery = HS 8426, 8429, 8430, 8474 and dump trucks; other "
         "inputs = lime, gypsum, aggregates, bitumen, tiles, sanitary ware, flat glass, paints and cable. Source: URA "
         "(2022); author's calculations.")


# ---------- V3: animated flow map ----------
def arc(p0, p1, n=60, bend=0.22):
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    cx, cy = mx - bend * dy, my + bend * dx
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * np.array([x0, y0]) + 2 * (1 - t) * t * np.array([cx, cy]) + t ** 2 * np.array([x1, y1])


def v_flowmap(t):
    w = gpd.read_file(NE)
    pts = {r.NAME: (r.LABEL_X, r.LABEL_Y) for r in w.itertuples()} | {r.ADMIN: (r.LABEL_X, r.LABEL_Y) for r in w.itertuples()}
    t = t.assign(ne_name=t.country.replace(NAME_FIX))
    t = t[t.ne_name.isin(pts) & (t.ne_name != "Uganda")]
    by = t.groupby(["ne_name", "group"]).usd_m.sum().reset_index()
    by = by[by.usd_m >= 1].sort_values("usd_m")
    gcol = {"Steel": BLUE, "Cement and clinker": ORANGE, "Machinery and trucks": VIOLET, "Other inputs": AQUA}
    paths = [(arc(pts[r.ne_name], KAMPALA, bend=0.18 if pts[r.ne_name][0] < KAMPALA[0] else -0.18), r.usd_m, gcol[r.group], r)
             for r in by.itertuples()]
    totals = t.groupby("ne_name").usd_m.sum().nlargest(10)

    def base(ax):
        w.plot(ax=ax, color="#efede8", edgecolor="#d9d6cf", linewidth=0.4)
        w[w.NAME == "Uganda"].plot(ax=ax, color=INK, edgecolor="none")
        ax.set_xlim(-20, 150)
        ax.set_ylim(-38, 62)
        ax.set_aspect("equal")
        ax.axis("off")
        for g, c in gcol.items():
            ax.plot([], [], color=c, lw=3, label=g)
        ax.legend(loc="lower left", frameon=False, fontsize=9)

    def draw(ax, frac=1.0):
        for p, v, c, r in paths:
            k = max(2, int(len(p) * frac))
            ax.plot(p[:k, 0], p[:k, 1], color=c, lw=0.6 + 2.6 * np.sqrt(v / by.usd_m.max()) * 2.2, alpha=0.75,
                    solid_capstyle="round", zorder=3)
            if frac < 1 and k < len(p):
                ax.plot(p[k - 1, 0], p[k - 1, 1], "o", ms=2.5, color=c, zorder=4)
        if frac >= 1:
            for n, v in totals.items():
                x, y = pts[n]
                dx = 7 if n == "Kenya" else 0
                ax.text(x + dx, y + 2.2, f"{n.replace('United Arab Emirates', 'UAE')}  {v:,.0f}", fontsize=7.5,
                        ha="center", color=INK, zorder=5)
        ax.plot(*KAMPALA, "o", ms=6, color=INK, mec=SURF, zorder=6)
        if frac >= 1:
            ax.text(KAMPALA[0] - 3, KAMPALA[1] - 4, f"Kampala\nUS$ {t.usd_m.sum():,.0f} m", ha="right", va="top",
                    fontsize=9, fontweight="bold", color=INK, zorder=6)

    fig, ax = plt.subplots(figsize=(12, 7.4))
    base(ax)
    draw(ax, 1.0)
    title(fig, "Construction inputs flow to Uganda from three regions: East Africa, the Gulf and Asia",
          "Recorded imports by country of origin and input group, 2021, US$ million (line width proportional to value; labels = "
          "ten largest origins, all groups)")
    fig.text(0.03, 0.03, "Source: URA Annual Imports Trade Report 2021; Natural Earth; author's calculations.",
             fontsize=9, color=INK2)
    fig.subplots_adjust(top=0.86, bottom=0.05, left=0.02, right=0.98)
    save(fig, "v3_flowmap_imports_2021.png",
         "Flow map of recorded construction-input imports into Uganda in 2021: one curve per origin country and input "
         "group, width proportional to value (US$ million). Sources: URA (2022); Natural Earth; author's calculations.")

    fig, ax = plt.subplots(figsize=(9.6, 6), dpi=100)
    fig.subplots_adjust(top=0.86, bottom=0.04, left=0.02, right=0.98)
    title(fig, "Construction inputs flowing into Uganda, 2021", "Recorded imports by origin and input group "
          "(line width proportional to US$ value)", y=0.975)
    frames = 70

    def update(i):
        ax.clear()
        base(ax)
        draw(ax, min(1.0, (i + 1) / (frames - 15)))
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=60, blit=False)
    anim.save(V / "v3_flowmap_imports_2021.gif", writer=animation.PillowWriter(fps=16))
    anim.save(V / "v3_flowmap_imports_2021.mp4", writer=animation.FFMpegWriter(fps=24, bitrate=1800))
    plt.close(fig)
    CAPTIONS.append(("v3_flowmap_imports_2021.gif / .mp4", "Animated version of the flow map: import routes draw in "
                     "toward Kampala, then the ten largest origins are labelled."))


# ---------- V4: animated cost shock ----------
def v_cost_shock():
    m = pd.read_csv(PROCESSED / "sector2026" / "panel_monthly.csv", parse_dates=["date"], index_col="date")
    base = "2017-07-01"
    series = {"Brent crude in shillings": ("brent_ugx", ORANGE), "UGX per US$": ("exchange_rate", BLUE),
              "CIPI diesel (UBOS)": ("DIESEL", VIOLET), "Material producer prices": ("ppi_materials", AQUA)}
    d = pd.DataFrame({k: 100 * m[c] / m[c].loc[base] for k, (c, _) in series.items()}).loc[base:]
    d = d.dropna(how="all")
    events = [("2020-04-01", "COVID-19\nlockdown"), ("2022-06-01", "Ukraine war\noil spike"),
              ("2026-04-01", "2026 oil and\nshilling shock")]
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=100)
    fig.subplots_adjust(top=0.82, bottom=0.1, left=0.08, right=0.76)
    title(fig, "The cost of building in Uganda follows oil and the shilling",
          "Monthly indices, July 2017 = 100. Brent in shillings is what a road contractor ultimately pays for fuel and "
          "bitumen", y=0.97)
    n = len(d)
    ymax = np.nanmax(d.values) * 1.08

    def update(i):
        ax.clear()
        k = min(n, int(i * 1.6) + 2)
        cur = d.iloc[:k]
        ends = []
        for name, (_, c) in series.items():
            s = cur[name].dropna()
            ax.plot(s.index, s, color=c, lw=2)
            if len(s):
                ax.plot(s.index[-1], s.iloc[-1], "o", color=c, ms=5)
                ends.append([s.iloc[-1], name, c])
        ends.sort(key=lambda e: e[0])
        for j in range(1, len(ends)):                      # keep end labels at least 6% of the range apart
            ends[j][0] = max(ends[j][0], ends[j - 1][0] + 0.06 * (ymax - 40))
        for yv, name, c in ends:
            v = cur[name].dropna().iloc[-1]
            ax.text(d.index[-1] + pd.Timedelta(days=60), yv, f"{name}  {v:.0f}", color=c, fontsize=9, va="center",
                    clip_on=False)
        for dt, lab in events:
            if pd.Timestamp(dt) <= cur.index[-1]:
                ax.axvline(pd.Timestamp(dt), color=GRID, lw=1, zorder=0)
                ax.text(pd.Timestamp(dt), ymax * 0.97, lab, fontsize=8, color=INK2, ha="center", va="top")
        ax.axhline(100, color=GREY, lw=0.8)
        ax.set_xlim(d.index[0], d.index[-1] + pd.Timedelta(days=30))
        ax.set_ylim(40, ymax)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID)
        ax.text(0.01, 0.04, cur.index[-1].strftime("%B %Y"), transform=ax.transAxes, fontsize=16, color=INK,
                fontweight="bold")
        return []

    frames = int(n / 1.6) + 25
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=70)
    anim.save(V / "v4_cost_shock.gif", writer=animation.PillowWriter(fps=14))
    anim.save(V / "v4_cost_shock.mp4", writer=animation.FFMpegWriter(fps=20, bitrate=1800))
    update(frames - 1)
    fig.text(0.08, 0.02, "Sources: EIA/FRED (Brent), Bank of Uganda (exchange rate), UBOS CIPI (to April 2026) and PPI.",
             fontsize=8.5, color=INK2)
    fig.savefig(V / "v4_cost_shock.png", dpi=300)
    plt.close(fig)
    CAPTIONS.append(("v4_cost_shock.gif / .mp4 / .png", "Animated monthly indices (July 2017 = 100) of Brent crude in "
                     "shillings, the UGX/US$ rate, the UBOS CIPI diesel sub-index (to April 2026) and building-material "
                     "producer prices, with major shocks marked."))


# ---------- V5: animated budget waterfall ----------
def v_budget_waterfall():
    _, _, y = budget_flows()
    yrs = [fy for fy in ext.FULL_YEARS if fy in y.index.get_level_values(0)]
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=100)
    fig.subplots_adjust(top=0.8, bottom=0.12, left=0.1, right=0.96)
    title(fig, "Every year, the construction budget leaks between approval and spending",
          "Central government, buildings and structures (items 3121xx), UGX trillion", y=0.97)
    steps = 20

    def vals(fy):
        r = y.loc[fy]
        out = {}
        for f in ("gou_dev", "ext_fin"):
            out[f] = [r.loc[f, "approved"], r.loc[f, "released"], r.loc[f, "spent"]] if f in r.index else [0, 0, 0]
        return out

    seq = [vals(fy) for fy in yrs]

    def update(i):
        ax.clear()
        k, j = divmod(i, steps)
        k = min(k, len(yrs) - 1)
        a = seq[k]
        b = seq[min(k + 1, len(yrs) - 1)]
        tt = 0 if i >= steps * (len(yrs) - 1) else (j / steps) ** 2 * (3 - 2 * j / steps) if j > steps * 0.5 else 0
        cur = {f: [a[f][s] + (b[f][s] - a[f][s]) * tt for s in range(3)] for f in a}
        x = np.arange(3)
        ax.bar(x, cur["gou_dev"], 0.55, color=BLUE, label="Government of Uganda")
        ax.bar(x, cur["ext_fin"], 0.55, bottom=cur["gou_dev"], color=ORANGE, label="External finance")
        tot = [cur["gou_dev"][s] + cur["ext_fin"][s] for s in range(3)]
        for s in range(3):
            ax.text(s, tot[s] + 0.15, f"{tot[s]:.1f}", ha="center", fontsize=11, color=INK)
        for s in (1, 2):
            ax.annotate("", xy=(s - 0.3, tot[s]), xytext=(s - 0.7, tot[s - 1]),
                        arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
            ax.text(s - 0.5, (tot[s] + tot[s - 1]) / 2 + 0.4, f"−{tot[s - 1] - tot[s]:.1f}", ha="center",
                    fontsize=10, color=ORANGE if s == 1 else INK2)
        ax.set_xticks(x, ["Approved", "Released", "Spent"], fontsize=11)
        ax.set_ylim(0, 9.5)
        ax.set_ylabel("UGX trillion")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID)
        ax.legend(frameon=False, loc="upper right")
        lab = yrs[k] if tt < 0.5 else yrs[min(k + 1, len(yrs) - 1)]
        ax.text(0.02, 0.92, f"FY{lab}", transform=ax.transAxes, fontsize=18, fontweight="bold", color=INK)
        ax.text(0.02, 0.84, f"spent {100 * tot[2] / tot[0]:.0f}% of approved", transform=ax.transAxes, fontsize=11,
                color=INK2)
        return []

    frames = steps * len(yrs) + 10
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=90)
    anim.save(V / "v5_budget_waterfall.gif", writer=animation.PillowWriter(fps=10))
    anim.save(V / "v5_budget_waterfall.mp4", writer=animation.FFMpegWriter(fps=12, bitrate=1500))
    plt.close(fig)
    CAPTIONS.append(("v5_budget_waterfall.gif / .mp4", "Animated approved–released–spent bars for buildings and "
                     "structures by funding source, FY2016/17–FY2021/22 (FY2018/19 omitted). Source: MoFPED budget data "
                     "portal."))


# ---------- V6: polar quarter clocks ----------
def v_quarter_clocks():
    d = ext.mofped()
    c = d[d.constr & d.fy.isin(ext.FULL_YEARS) & (d.stage == "spent")]
    p = c.groupby(["fy", "quarter"]).amount.sum().unstack()
    p = 100 * p.div(p.sum(axis=1), axis=0)
    p.loc["All years"] = 100 * c.groupby("quarter").amount.sum() / c.amount.sum()
    fig, axes = plt.subplots(1, len(p), figsize=(13, 3.6), subplot_kw={"aspect": "equal"})
    qcol = {"Q1": GREY, "Q2": GREY, "Q3": GREY, "Q4": ORANGE}
    for ax, (fy, r) in zip(axes, p.iterrows()):
        for k, q in enumerate(["Q1", "Q2", "Q3", "Q4"]):
            rad = np.sqrt(r[q] / 40)                 # area ∝ share
            ax.add_patch(Wedge((0, 0), rad, 90 - 90 * (k + 1), 90 - 90 * k, facecolor=qcol[q], edgecolor=SURF,
                               lw=2, alpha=0.95))
            ang = np.deg2rad(90 - 90 * k - 45)
            lr = max(rad, np.sqrt(25 / 40)) + 0.32
            ax.text(lr * np.cos(ang), lr * np.sin(ang), f"{q}\n{r[q]:.0f}%", ha="center", va="center",
                    fontsize=8.5, color=ORANGE if q == "Q4" else INK2)
        ax.add_patch(plt.Circle((0, 0), np.sqrt(25 / 40), fill=False, ls="--", color=INK2, lw=0.7))
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.axis("off")
        ax.set_title(fy if fy == "All years" else f"FY{fy}", fontsize=10.5, color=INK, fontweight="bold" if fy == "All years" else None)
    title(fig, "The year-end rush: construction spending bunches in April–June",
          "Share of each year's spending on buildings and structures by quarter (Q1 = Jul–Sep; wedge area proportional to share; "
          "dashed circle = even 25%)", y=0.98)
    fig.subplots_adjust(top=0.76, bottom=0.02, left=0.01, right=0.99, wspace=0.02)
    save(fig, "v6_quarter_clocks.png",
         "Quarterly distribution of central-government spending on buildings and structures (items 3121xx) by fiscal "
         "year and pooled; wedge area proportional to the quarter's share; dashed circle = an even 25% per quarter. "
         "Source: MoFPED budget data portal; author's calculations.")


# ---------- V7: tax waffle ----------
def v_tax_waffle():
    t = pd.read_csv(S / "tax_intensity.csv", index_col=0).loc["2024/25"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    for ax, (val, lab, col) in zip(axes, [(t.tax_per_100_va, "Construction: tax per UGX 100 of value added", BLUE),
                                           (t.economy_tax_per_100_gdp, "Whole economy: tax per UGX 100 of GDP", GREY)]):
        n = int(round(val))
        for k in range(100):
            r, cidx = divmod(k, 10)
            ax.add_patch(FancyBboxPatch((cidx, 9 - r), 0.82, 0.82, boxstyle="round,pad=0,rounding_size=0.18",
                                        facecolor=col if k < n else "#eeece7", edgecolor="none"))
        ax.set_xlim(-0.2, 10)
        ax.set_ylim(-0.4, 10.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0, -0.9, f"UGX {val:.1f}", fontsize=20, fontweight="bold", color=col if col != GREY else INK2)
        ax.text(0, -1.7, lab, fontsize=10.5, color=INK2)
    title(fig, "Construction pays less than half the economy-wide tax rate",
          "FY2024/25: each square is UGX 1 of every UGX 100 of output", y=0.97)
    fig.text(0.03, 0.02, f"Construction's share of GDP {t.gdp_share_pct:.1f}%; share of gross tax "
             f"{t.tax_share_pct:.1f}%. Sources: URA annual data book FY2024/25; UBOS Quarterly GDP; author's "
             "calculations.", fontsize=9, color=INK2)
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.04, right=0.98, wspace=0.12)
    save(fig, "v7_tax_waffle.png",
         "Tax collected per UGX 100 of output in FY2024/25: construction-sector gross collections per UGX 100 of "
         "construction value added (left) and total collections per UGX 100 of GDP (right). Sources: URA; UBOS; "
         "author's calculations.")


def main(only=None):
    V.mkdir(parents=True, exist_ok=True)
    t = trade_flows()
    jobs = {"v1": v_sankey_budget, "v2": lambda: v_sankey_imports(t), "v3": lambda: v_flowmap(t), "v4": v_cost_shock,
            "v5": v_budget_waterfall, "v6": v_quarter_clocks, "v7": v_tax_waffle}
    for k, fn in jobs.items():
        if not only or k in only:
            fn()
    if not only:
        (V / "captions.md").write_text("# Stage 13 visual suite — captions\n\n" +
                                       "\n\n".join(f"**{n}** — {c}" for n, c in CAPTIONS) + "\n")
    print(f"{len(CAPTIONS)} visuals -> {V}")


if __name__ == "__main__":
    import sys
    main(only=sys.argv[1].split(",") if len(sys.argv) > 1 else None)
