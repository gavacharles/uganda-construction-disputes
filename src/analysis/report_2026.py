"""2026 construction sector report — figures and numbers specific to the 2026 review.

Dumbbell chart of cause shares in 2026 vs 2015–2025, and a table of 2026 headline indicators vs the prior decade.
Output: outputs/analysis/report_2026/ (fig_2026_vs_decade.png, t_2026_headline.csv)
"""
import numpy as np
import pandas as pd

from src.analysis.common import C, CAUSE_LABEL, CAUSES, Stage, load, plt

MERITS = ["initiator succeeded", "partly succeeded", "initiator failed", "struck out (procedural)"]


def indicators(d):
    m = d[d.outcome.isin(MERITS)]
    ugx = d.quantum_claimed[d.currency.eq("UGX")]
    return {"decisions": len(d), "initiator success %": round(100 * m.outcome.isin(MERITS[:2]).mean(), 1),
            "state party %": round(100 * d.state_party.eq("Y").mean(), 1),
            "Commercial Division %": round(100 * d.court.str.contains("Commercial", na=False).mean(), 1),
            "PPDA Tribunal %": round(100 * d.court.str.contains("PPDA", na=False).mean(), 1),
            "median claim (UGX m)": round(ugx.median() / 1e6, 1)}


def main():
    s = Stage("report_2026", "2026 construction sector report — supporting figures")
    a = load()
    con = a[a.stream.isin(["construction", "both"])]
    now, before = con[con.year == 2026], con[con.year < 2026]
    t = pd.DataFrame({"2026": indicators(now), "2015–2025": indicators(before)})
    t.loc["decisions per year", :] = [len(now), round(len(before) / 11, 1)]
    s.table(t, "t_2026_headline", "Table 1. Construction-related decisions: 2026 against 2015–2025",
            note="2026 runs to 16 November 2026. Codes are rule-based (v1) with gold labels where available.")

    keep = [c for c in CAUSES if (con[c] == "Y").sum() >= 15]
    sh = pd.DataFrame({"2026": (now[keep] == "Y").mean() * 100, "2015–2025": (before[keep] == "Y").mean() * 100})
    sh.index = [CAUSE_LABEL[c] for c in keep]
    sh = sh.sort_values("2026")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    y = np.arange(len(sh))
    for i, (_, r) in enumerate(sh.iterrows()):
        up = r["2026"] >= r["2015–2025"]
        ax.plot([r["2015–2025"], r["2026"]], [i, i], color=C["up"] if up else C["down"], lw=2, alpha=0.6)
    ax.scatter(sh["2015–2025"], y, color=C["neutral"], s=36, zorder=3, label="2015–2025 average")
    ax.scatter(sh["2026"], y, color=C["ink"], s=36, zorder=3, label="2026")
    ax.set_yticks(y, sh.index, fontsize=8.5)
    ax.set_xlabel("% of construction-related decisions with the cause in issue")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    s.style(ax, grid="x")
    s.fig(fig, "fig_2026_vs_decade.png", f"Dispute causes in construction-related decisions, 2026 (n = {len(now)}) against the "
          f"2015–2025 average (n = {len(before)}). Orange lines mark causes more frequent in 2026; blue lines, less frequent.")
    s.write()


if __name__ == "__main__":
    main()
