"""Stage 2.1 — draw the gold-set sample (150 register judgments to download and hand-code).

Allocation favours the construction core: construction 60, both 50, arbitration-only 40.
Within each stream, systematic sampling over decision date (random start, seed 42) spreads
the sample across years and courts, split between periods in proportion to their size.
Output: outputs/coding/gold_download_list.xlsx
"""
import numpy as np
import pandas as pd

from src.config import INTERIM, OUT, SEED

ALLOC = {"construction": 60, "both": 50, "arbitration": 40}


def systematic(df, n, rng):
    df = df.sort_values(["date_decided", "court"]).reset_index(drop=True)
    if n >= len(df):
        return df
    step = len(df) / n
    idx = (rng.uniform(0, step) + step * np.arange(n)).astype(int)
    return df.iloc[idx]


def main():
    rng = np.random.default_rng(SEED)
    cases = pd.read_parquet(INTERIM / "registers_cases.parquet")
    picks = []
    for stream, n in ALLOC.items():
        s = cases[cases.stream == stream]
        share = s.period.value_counts(normalize=True)
        n_26 = int(round(n * share.get("2026", 0)))
        for period, k in (("2015-2025", n - n_26), ("2026", n_26)):
            picks.append(systematic(s[s.period == period], k, rng))
    g = pd.concat(picks).sort_values(["date_decided"]).reset_index(drop=True)
    g.insert(0, "No.", range(1, len(g) + 1))
    g["save_as"] = g["case_id"] + ".pdf"
    g["downloaded"] = ""
    cols = ["No.", "case_id", "citation", "parties", "court", "judge", "date_decided",
            "case_number", "stream", "period", "key_issues", "ulii_link", "save_as", "downloaded"]
    out = OUT / "coding" / "gold_download_list.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    g["date_decided"] = g["date_decided"].dt.date
    with pd.ExcelWriter(out) as xw:
        g[cols].to_excel(xw, sheet_name="Download list", index=False)
        ws = xw.sheets["Download list"]
        for i, w in enumerate([5, 18, 22, 50, 28, 26, 12, 30, 13, 10, 40, 60, 22, 11]):
            ws.column_dimensions[chr(65 + i)].width = w
        for r in range(2, len(g) + 2):  # clickable links
            c = ws[f"L{r}"]
            c.hyperlink, c.style = c.value, "Hyperlink"
        ws.freeze_panes = "C2"
    print(f"{len(g)} judgments -> {out}")
    print(g.groupby(["stream", "period"]).size().unstack(fill_value=0).to_string())
    print(g.groupby(g.date_decided.map(lambda d: d.year)).size().to_string())
    print(g.court.value_counts().to_string())


if __name__ == "__main__":
    main()
