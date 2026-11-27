"""Native, smooth rebuilds of the four Atlas animations (square 1080 x 1080, 30 fps, eased), replacing the earlier
browser captures. Data: docs/atlas/data/insights.json (hex layout, district-year counts, race shares) and the
processed case tables (decisions, treatment links, matters). Every frame carries the credit line.

  01_decade_on_the_map       hex map of districts filling year by year (decisions per 100,000 people), new-case pulses
  02_case_law_network_grows  decisions appear in order and the treatment links between them draw in
  03_dispute_pulse           every decision by court and date; arcs join decisions in the same dispute
  04_court_audit_press_race  dispute-family shares in court decisions, Auditor-General reports and The Observer

Outputs: outputs/animations/*.mp4 + *.gif (GIF copies refreshed in outputs/writing/substack)
Usage: python -m src.viz.atlas_motion [a1,a2,a3,a4]
"""
import json
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.collections import LineCollection, PatchCollection  # noqa: E402
from matplotlib.patches import RegularPolygon  # noqa: E402

from src.config import OUT, PROCESSED, ROOT  # noqa: E402
from src.viz.credit import credit  # noqa: E402

A = OUT / "animations"
SUB = OUT / "writing" / "substack"
INS = json.loads((ROOT / "docs" / "atlas" / "data" / "insights.json").read_text())
FPS, DPI, SIZE = 30, 150, (7.2, 7.2)
INK, INK2, GRID, SURF, PAPER = "#1c1b19", "#5d5b56", "#e6e4df", "#ffffff", "#faf9f6"
BLUE, ORANGE, AQUA, VIOLET, GOLD, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#7a5cc4", "#c4a35c", "#a7a6a2", "#c0392b"
STREAM = {"construction": BLUE, "arbitration": ORANGE, "both": AQUA}
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "figure.facecolor": PAPER,
                     "axes.facecolor": PAPER, "savefig.facecolor": PAPER})


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def head(fig, title, sub):
    fig.text(0.05, 0.965, title, fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.05, 0.925, sub, fontsize=10, color=INK2, va="top", wrap=True)
    fig.text(0.05, 0.012, "Uganda Construction & Arbitration Law Atlas · CC BY 4.0", fontsize=8, color=INK2)
    credit(fig)


def render(fig, draw, n, name):
    mp4, gif = A / f"{name}.mp4", A / f"{name}.gif"
    w = animation.FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow"])
    with w.saving(fig, str(mp4), dpi=DPI):
        for i in range(n):
            draw(i)
            w.grab_frame()
    plt.close(fig)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4), "-vf",
                    "fps=15,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=sierra2_4a", str(gif)], check=True)
    if (SUB / gif.name).exists():
        (SUB / gif.name).write_bytes(gif.read_bytes())
    print(f"{name}: {n} frames ({n / FPS:.1f} s)")


def year_progress(i, years, per, hold_end):
    """Return (year index, eased fraction into it) for frame i."""
    k, j = divmod(i, per)
    if k >= len(years):
        return len(years) - 1, 1.0
    return k, ease(j / (per * 0.7))


# ---------- a1: hex map ----------
def a1_hex():
    hx = pd.DataFrame(INS["hex"])
    dy = pd.DataFrame(INS["districtYear"])
    years = list(range(2015, 2027))
    cnt = dy.pivot_table(index="d", columns="y", values="n", aggfunc="sum").reindex(columns=years, fill_value=0) \
        .reindex(hx.district, fill_value=0).fillna(0)
    cum = cnt.cumsum(axis=1)
    rate = cum.div(hx.set_index("district").population, axis=0) * 1e5
    vmax = np.nanpercentile(rate[years[-1]], 97)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#eef3fa", "#8fb6e6", BLUE, "#0b3a75"])
    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.8])
    head(fig, "Construction disputes by district", "Decisions per 100,000 people, cumulative from 2015; number = decisions "
                                                  "in the district; orange ring = new decisions this year")
    ax.set_aspect("equal")
    ax.axis("off")
    r = 1.0
    hexes = [RegularPolygon((x, y), 6, radius=r * 0.98, orientation=0) for x, y in zip(hx.x, hx.y)]
    pc = PatchCollection(hexes, edgecolor=PAPER, linewidth=1.2)
    ax.add_collection(pc)
    rings = PatchCollection([RegularPolygon((x, y), 6, radius=r * 0.98, orientation=0) for x, y in zip(hx.x, hx.y)],
                            facecolor="none", linewidth=2.2)
    ax.add_collection(rings)
    ax.set_xlim(hx.x.min() - 1.5, hx.x.max() + 1.5)
    ax.set_ylim(hx.y.min() - 1.5, hx.y.max() + 1.5)
    labels = [ax.text(x, y, "", ha="center", va="center", fontsize=5.6, color=INK) for x, y in zip(hx.x, hx.y)]
    abbrev = [d[:4] for d in hx.district]
    year_lab = fig.text(0.93, 0.12, "", fontsize=30, fontweight="bold", color=INK, ha="right")
    tot_lab = fig.text(0.93, 0.085, "", fontsize=11, color=INK2, ha="right")
    per = int(1.25 * FPS)
    n = per * len(years) + 2 * FPS

    def draw(i):
        k, f = year_progress(i, years, per, 2 * FPS)
        prev = rate[years[k - 1]].values if k else np.zeros(len(hx))
        now = prev + (rate[years[k]].values - prev) * f
        pc.set_facecolor(cmap(np.clip(now / vmax, 0, 1)))
        new = cnt[years[k]].values > 0
        ph = min(1.0, (i - k * per) / per) if i < per * len(years) else 1.0
        rings.set_edgecolor([(0.92, 0.41, 0.2, (0.9 * (1 - ph)) if nw else 0) for nw in new])
        c_now = (cum[years[k - 1]].values if k else np.zeros(len(hx))) + cnt[years[k]].values * f
        for lab, a, c, v in zip(labels, abbrev, c_now, now):
            lab.set_text(f"{a}\n{int(round(c))}" if c >= 0.5 else a)
            lab.set_color(SURF if v / vmax > 0.55 else INK)
        year_lab.set_text(str(years[k]))
        tot_lab.set_text(f"{int(round(c_now.sum()))} located decisions to date")

    render(fig, draw, n, "01_decade_on_the_map")


# ---------- a2: network ----------
def a2_network():
    a = pd.read_parquet(PROCESSED / "analysis_cases.parquet")[["case_id", "year", "stream"]]
    t = pd.read_parquet(PROCESSED / "treatment.parquet")
    t = t[t.later.isin(a.case_id) & t.earlier.isin(a.case_id)].copy()
    lab = t.label.astype(str).str.lower()
    t["col"] = np.select([t.kind == "appeal", lab.str.contains("follow|appl|approv"), lab.str.contains("disting"),
                          lab.str.contains("doubt|criticis|overrul|not follow")], [RED, AQUA, GREY, GOLD], "#b9b6ae")
    g = nx.Graph()
    g.add_edges_from(zip(t.later, t.earlier))
    pos = nx.spring_layout(g, seed=42, k=0.18, iterations=200)
    info = a.set_index("case_id")
    nodes = list(g.nodes)
    xy = np.array([pos[v] for v in nodes])
    yr = np.array([info.year.get(v, 2015) for v in nodes], dtype=float)
    deg = np.array([g.degree(v) for v in nodes], dtype=float)
    col = [STREAM.get(info.stream.get(v, "construction"), BLUE) for v in nodes]
    t["yr"] = t.later.map(info.year).astype(float)
    e_xy = [(pos[a_], pos[b_]) for a_, b_ in zip(t.later, t.earlier)]
    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.82])
    head(fig, "How Uganda's construction case law grows",
         "Decisions and the later decisions that cite or appeal them, to 2026. Links: green followed, grey distinguished, "
         "amber doubted, red appeal; dots: blue construction, orange arbitration, aqua both")
    ax.axis("off")
    ax.set_xlim(xy[:, 0].min() - 0.08, xy[:, 0].max() + 0.08)
    ax.set_ylim(xy[:, 1].min() - 0.08, xy[:, 1].max() + 0.08)
    lc = LineCollection([], linewidths=1.3, zorder=1)
    ax.add_collection(lc)
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=np.zeros(len(nodes)), c=col, edgecolor=PAPER, linewidth=0.6, zorder=2)
    year_lab = fig.text(0.93, 0.1, "", fontsize=30, fontweight="bold", color=INK, ha="right")
    cnt_lab = fig.text(0.93, 0.07, "", fontsize=11, color=INK2, ha="right")
    years = list(range(2015, 2027))
    per = int(1.25 * FPS)
    n = per * len(years) + 2 * FPS

    def draw(i):
        tc = min(len(years) - 1e-6, i / per)                 # continuous year position
        ycur = years[0] + tc
        grow = np.clip((ycur - yr + 1) / 0.6, 0, 1)            # node grows in over ~0.6 of a year
        sc.set_sizes(ease(grow) * (14 + 26 * np.sqrt(deg)))
        segs, cols = [], []
        for (p0, p1), y_, c in zip(e_xy, t.yr, t.col):
            f = ease((ycur - y_ + 1) / 0.6)
            if f > 0:
                p0, p1 = np.array(p0), np.array(p1)
                segs.append([p0, p0 + (p1 - p0) * f])
                cols.append(matplotlib.colors.to_rgba(c, 0.85))
        lc.set_segments(segs)
        lc.set_color(cols)
        year_lab.set_text(str(int(min(ycur, years[-1]))))
        cnt_lab.set_text(f"{int((grow >= 1).sum())} decisions · {len(segs)} links")

    render(fig, draw, n, "02_case_law_network_grows")


# ---------- a3: dispute pulse timeline ----------
LEVEL = {"UGSC": 3, "UGCA": 2, "UGCC": 2}


def a3_pulse():
    a = pd.read_parquet(PROCESSED / "analysis_cases.parquet")[["case_id", "date_decided", "stream"]]
    m = pd.read_parquet(PROCESSED / "matters.parquet")
    a = a.merge(m[["case_id", "matter_id"]], on="case_id", how="left")
    code = a.case_id.str.extract(r"_(UG[A-Za-z]+)_")[0]
    tribunal = pd.Series(np.where(code.isin(["UGPPDPAAT", "UGCADER", "UGTAT", "UGRSB", "UGIC"]), 0, 1), index=a.index)
    a["level"] = code.map(LEVEL).fillna(tribunal)
    a["date"] = pd.to_datetime(a.date_decided)
    a = a.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    rng = np.random.default_rng(42)
    a["y"] = a.level + rng.uniform(-0.28, 0.28, len(a))
    a["x"] = matplotlib.dates.date2num(a.date)
    arcs = []
    for _, g in a.groupby("matter_id"):
        g = g.sort_values("date")
        for (_, r0), (_, r1) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows()):
            arcs.append((r0.x, r0.y, r1.x, r1.y, r1.level > r0.level))
    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0.2, 0.12, 0.76, 0.7])
    head(fig, "Dispute pulse: every decision by court and date",
         "To November 2026. Arcs join decisions in the same dispute (orange: up to a higher court); dots: blue "
         "construction, orange arbitration, aqua both")
    ax.set_yticks([0, 1, 2, 3], ["Tribunals", "High Court", "Court of Appeal", "Supreme Court"], fontsize=11)
    ax.set_ylim(-0.6, 3.6)
    x0, x1 = a.x.min() - 30, a.x.max() + 30
    ax.set_xlim(x0, x1)
    ax.xaxis_date()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color=GRID)
    sc = ax.scatter(a.x, a.y, s=np.zeros(len(a)), c=[STREAM.get(s, BLUE) for s in a.stream], edgecolor="none", zorder=3)
    halo = ax.scatter(a.x, a.y, s=np.zeros(len(a)), facecolor="none", edgecolor=[STREAM.get(s, BLUE) for s in a.stream],
                      linewidth=1.0, zorder=2)
    lc = LineCollection([], linewidths=0.9, zorder=1)
    ax.add_collection(lc)
    cursor = ax.axvline(x0, color=INK2, lw=0.8, alpha=0.6)
    date_lab = fig.text(0.93, 0.86, "", fontsize=16, fontweight="bold", color=INK, ha="right")
    run, hold = 15 * FPS, 2 * FPS
    n = run + hold
    span = x1 - x0

    def draw(i):
        xc = x0 + span * min(1.0, i / run)
        age = (xc - a.x.values) / span * 30               # seconds of animation since the decision appeared
        vis = age >= 0
        sc.set_sizes(np.where(vis, 16 * ease(age / 0.15), 0))
        halo.set_sizes(np.where(vis & (age < 0.8), 16 + 260 * age, 0))
        halo.set_alpha(0.6)
        segs, cols = [], []
        for xa, ya, xb, yb, up in arcs:
            if xb <= xc:
                xs = np.linspace(xa, xb, 24)
                lift = 0.35 + 0.25 * min(1, (xb - xa) / 900)
                ys = ya + (yb - ya) * np.linspace(0, 1, 24) + lift * np.sin(np.linspace(0, np.pi, 24))
                segs.append(np.column_stack([xs, ys]))
                cols.append(matplotlib.colors.to_rgba(ORANGE if up else GREY, 0.75 if up else 0.45))
        lc.set_segments(segs)
        lc.set_color(cols)
        cursor.set_xdata([xc, xc])
        date_lab.set_text(matplotlib.dates.num2date(xc).strftime("%b %Y"))

    render(fig, draw, n, "03_dispute_pulse")


# ---------- a4: court / audit / press race ----------
def a4_race():
    r = pd.DataFrame(INS["race"])
    years = sorted(r.year.unique())
    fams = list(dict.fromkeys(r.family))
    srcs = list(dict.fromkeys(r.source))
    scol = dict(zip(srcs, [BLUE, VIOLET, GOLD]))
    P = {y: r[r.year == y].pivot(index="family", columns="source", values="share").reindex(index=fams, columns=srcs)
         .fillna(0) for y in years}
    vmax = max(p.values.max() for p in P.values()) * 1.12
    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0.3, 0.1, 0.64, 0.72])
    head(fig, "Courts, auditors and the press see different disputes",
         "Share of each source's items by dispute family. Blue court decisions · purple Auditor-General · gold The Observer")
    hold, move = int(1.2 * FPS), int(1.0 * FPS)
    per = hold + move
    n = per * (len(years) - 1) + hold + FPS
    year_lab = fig.text(0.93, 0.13, "", fontsize=30, fontweight="bold", color=INK, ha="right")
    bh = 0.26

    def draw(i):
        k, j = divmod(i, per)
        if k >= len(years) - 1:
            k, tt = len(years) - 1, 0.0
        else:
            tt = ease((j - hold) / move) if j > hold else 0.0
        a_, b_ = P[years[k]], P[years[min(k + 1, len(years) - 1)]]
        v = a_ + (b_ - a_) * tt
        ax.clear()
        for s_i, s in enumerate(srcs):
            yv = np.arange(len(fams)) + (s_i - 1) * bh
            ax.barh(yv, v[s].values, height=bh * 0.9, color=scol[s])
            for y_, x_ in zip(yv, v[s].values):
                ax.text(x_ + vmax * 0.01, y_, f"{x_:.1f}%", va="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(fams)), fams, fontsize=10.5)
        ax.invert_yaxis()
        ax.set_xlim(0, vmax)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID)
        ax.set_xlabel("% of the source's items", fontsize=10)
        year_lab.set_text(str(years[k] if tt < 0.5 else years[min(k + 1, len(years) - 1)]))

    render(fig, draw, n, "04_court_audit_press_race")


def main(only=None):
    A.mkdir(parents=True, exist_ok=True)
    jobs = {"a1": a1_hex, "a2": a2_network, "a3": a3_pulse, "a4": a4_race}
    for k, fn in jobs.items():
        if not only or k in only:
            fn()


if __name__ == "__main__":
    main(only=sys.argv[1].split(",") if len(sys.argv) > 1 else None)
