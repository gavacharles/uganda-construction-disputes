"""Sector report 2026: one-page sector scorecard.

One row per indicator: latest value and period, the change on a year earlier, the percentile of the latest value within
the base window (FY2017/18-FY2024/25), a direction arrow and the source. "Good" says which direction is favourable for
the sector (used to colour the arrow in the report), and is blank where direction is not a judgement.
Output: data/processed/sector2026/scorecard.csv (+ printed table). Run after build_panel and indices.
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED
from src.report2026.indices import BASE, complete_quarters

OUT = PROCESSED / "sector2026"


def _load():
    rd = lambda f, **k: pd.read_csv(OUT / f, parse_dates=[0], index_col=0, **k)
    m = rd("panel_monthly.csv")
    q = complete_quarters(rd("panel_quarterly.csv"))
    return m, q, {k: rd(f"{k}.csv") for k in ("idx_ccpi", "idx_procurement", "idx_payment", "idx_dispute",
                                               "cipi_nowcast")}, pd.read_csv(OUT / "ura_series.csv")


def row(group, name, s, unit, lag, source, good="", change="diff", note=""):
    """s: the indicator series (already transformed); lag: periods in a year (12 monthly, 4 quarterly, 1 annual)."""
    s = s.dropna()
    t, v = s.index[-1], s.iloc[-1]
    prev = s.shift(lag).iloc[-1] if len(s) > lag else np.nan
    ch = v - prev if change == "diff" else 100 * (v / prev - 1)
    base = s.loc[BASE[0]:BASE[1]] if isinstance(s.index, pd.DatetimeIndex) else s
    pctl = 100 * (base < v).mean() if len(base) >= 4 else np.nan
    arrow = "→" if pd.isna(ch) or abs(ch) < 1e-9 else ("↑" if ch > 0 else "↓")
    period = (f"{t.year} Q{t.quarter}" if lag == 4 else t.strftime("%b %Y")) if isinstance(t, pd.Timestamp) else str(t)
    return {"group": group, "indicator": name, "period": period, "value": round(float(v), 2), "unit": unit,
            "change_on_year": round(float(ch), 2) if pd.notna(ch) else None,
            "change_type": "pp/points" if change == "diff" else "%", "pctl_vs_base": round(pctl) if pd.notna(pctl) else None,
            "arrow": arrow, "good": good, "source": source, "note": note, "_series": s}


def build():
    m, q, ix, ura = _load()
    yoy = lambda s, k: 100 * (s / s.shift(k) - 1)
    now = ix["cipi_nowcast"].cipi_nowcast
    cipi_ext = pd.concat([m.CIPI_ALL.dropna(), now])
    ura_s = lambda k: ura[ura.series == k].set_index("fy").value
    rows = [
        row("Economy", "Construction real value added, growth y/y", yoy(q.gdp_construction_va_const, 4), "%", 4, "UBOS QGDP", "up"),
        row("Economy", "Construction share of GDP (current prices)", q.gdp_construction_share_cur.rolling(4).mean(), "% (4-qtr avg)", 4,
            "UBOS QGDP"),
        row("Prices", "CIPI all items, inflation y/y (official)", yoy(m.CIPI_ALL, 12), "%", 12, "UBOS CIPI", "down"),
        row("Prices", "CIPI all items, inflation y/y (H&Q nowcast)", yoy(cipi_ext, 12), "%", 12, "H&Q nowcast", "down",
            note="model estimate after last UBOS release (Apr 2026)"),
        row("Prices", "Construction Cost Pressure Index (CCPI-U)", ix["idx_ccpi"]["index"], "index (100 = base avg)", 12, "H&Q", "down",
            note="provisional" if ix["idx_ccpi"].provisional.iloc[-1] else ""),
        row("Economy", "UGX per US$ (monthly average)", m.exchange_rate, "UGX", 12, "BoU/MoFPED", "down", change="pct"),
        row("Economy", "Brent crude (monthly average)", m.brent_crude_usd, "US$/bbl", 12, "EIA/FRED", "down", change="pct"),
        row("Prices", "PPI construction materials, inflation y/y", yoy(m.ppi_materials, 12), "%", 12, "UBOS PPI", "down"),
        row("Economy", "Headline CPI inflation", m.cpi_headline_inflation, "%", 12, "UBOS CPI", "down"),
        row("Money", "Central Bank Rate (policy rate)", m.central_bank_rate, "%", 12, "BoU MPC", "down",
            note="raised from 9.75% at a special MPC meeting on 14 Oct 2026; held 9 Nov 2026"),
        row("Money", "Commercial lending rate", m.lending_rate, "%", 12, "BoU", "down"),
        row("Materials", "Cement production (IIP 2394), growth y/y", yoy(q.iip_cement, 4), "%", 4, "UBOS IIP", "up"),
        row("Materials", "Iron & steel imports, 3-month total", m.imp_iron_steel.rolling(3).sum() / 1e3, "US$ m", 12, "UBOS trade",
            change="pct"),
        row("Money", "Construction sector tax collected", ura_s("construction_sector_revenue"), "UGX bn (FY)", 1, "URA", change="pct"),
        row("Pipeline", "Public works awards on the GPP (quarter)", q.works_awards, "awards", 4, "PPDA GPP (OCDS)", change="pct",
            note="publication coverage varies"),
        row("Pipeline", "Procurement Health Index", ix["idx_procurement"]["index"], "index", 4, "H&Q (GPP)", "up"),
        row("Money", "Payment Stress Index", ix["idx_payment"]["index"], "index", 4, "H&Q (MoFPED/BoU)", "down"),
        row("Disputes", "Construction court decisions, last 4 quarters", ix["idx_dispute"].court_4q, "decisions", 4, "ULII (H&Q corpus)",
            change="pct", note="rise partly reflects wider ULII publication"),
        row("Disputes", "Payment disputes, share of decisions (4 qtrs)", 100 * ix["idx_dispute"].payment_share_4q, "%", 4,
            "ULII (H&Q corpus)", "down"),
    ]
    return pd.DataFrame(rows)


def main():
    s = build().drop(columns="_series")
    s.to_csv(OUT / "scorecard.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_colwidth", 48):
        print(s[["indicator", "period", "value", "unit", "change_on_year", "change_type", "pctl_vs_base", "arrow"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
