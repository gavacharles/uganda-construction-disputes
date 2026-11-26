"""Sector report 2026: Hove & Quill construction indices.

Formulas, base period and weights are fixed here before any 2026 value is read (blueprint section 4); change them only
with a version bump. Every index is standardised on the BASE window (FY2017/18-FY2024/25, Jul 2017-Jun 2025) and scaled
as 100 + 10 z: 100 = the base-period average, 110 = one standard deviation above it.

  ccpi          Construction Cost Pressure Index, monthly: mean z of 3-month % changes in CIPI, UGX/USD, Brent in UGX and
                PPI construction materials, the 12-month change in the lending rate and the GSCPI level. It is published
                when at least MIN_COMPONENTS are available; fewer than all six = provisional.
  cipi_nowcast  CIPI all-items after the last UBOS release: OLS of the monthly log change on lags 0-2 of Brent-in-UGX and
                UGX/USD and lags 1-2 of PPI materials (fit to the last release), with 80% bands widening with sqrt(h).
                A Hove & Quill estimate, not an official statistic.
  escalation_gap  Unrecovered cost escalation, % of contract value, by base month and duration (12/18/24 months; equal
                monthly valuations). Actual = change in the formula basket since the base date. Recovered = 0 for a
                fixed-price contract, or (1 - FIXED_SHARE) x basket change lagged INDEX_LAG months under a FIDIC 13.7-style
                formula. Basket weights are from the brent-crude-uganda-escalation case study (Uganda road IPC).
  procurement   Procurement Health Index, quarterly (works awards on the GPP): + mean bidders, - single-bid share,
                - non-open method share, - median days from tender to award. Higher = healthier.
  payment       Payment Stress Index, quarterly: + fiscal deficit / expenditure, + lending rate, - private credit growth y/y,
                - public expenditure growth y/y. Higher = more stress. (Arrears data are not yet in the panel.)
  dispute       Dispute Climate Index, quarterly: court/tribunal construction decisions and Observer dispute articles
                (4-quarter sums). Court counts partly reflect ULII publication growth; read with the share-based caveat.
  weather       Weather Working-Days Outlook: expected earthworks days lost per month by region (2006-2025 normal) and the
                change on 1981-2000.
Usage: python -m src.report2026.indices
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED, ROOT

OUT = PROCESSED / "sector2026"
GH = ROOT / "data" / "external" / "sector" / "github"
VERSION = "1.0"
AS_OF = "2026-11-16"    # earliest cut-off across sources (ULII 16 Nov; OCDS mirror 23 Nov)
BASE = ("2017-07-01", "2025-06-01")
MIN_COMPONENTS = 3
FIXED_SHARE = 0.20      # non-adjustable portion (case study formula; FIDIC 13.7 typically 0.10-0.20)
INDEX_LAG = 1           # months between valuation and the index month used
DURATIONS = (12, 18, 24)
MIN_VALUED = 6          # running contracts with fewer valued months are dominated by the index lag; not reported
BASKET = {  # formula element -> CIPI sub-index or proxy (weight renormalised over the adjustable 0.80)
    "fuel": ("DIESEL", 0.10), "bitumen": ("brent_ugx", 0.20), "steel": ("STEEL", 0.05), "cement": ("CEM", 0.10),
    "equipment": ("exchange_rate", 0.10), "labour": ("LABOUR", 0.15), "metal": ("IRONSTEEL", 0.10)}


# ---------- helpers ----------
def zscore(s, base=BASE, sign=1):
    b = s.loc[base[0]:base[1]]
    return sign * (s - b.mean()) / b.std()


def scale(z):
    return 100 + 10 * z


def pct(s, k):
    return 100 * (s / s.shift(k) - 1)


def composite(parts, min_n):
    z = pd.concat(parts, axis=1)
    n = z.notna().sum(axis=1)
    return pd.DataFrame({"index": scale(z.mean(axis=1)).where(n >= min_n), "n_components": n}).join(z.add_prefix("z_"))


# ---------- indices ----------
def ccpi(m):
    parts = {
        "cipi_3m": pct(m.CIPI_ALL, 3), "fx_3m": pct(m.exchange_rate, 3), "brent_ugx_3m": pct(m.brent_ugx, 3),
        "ppi_materials_3m": pct(m.ppi_materials, 3), "lending_12m": m.lending_rate.diff(12),
        "gscpi": m.gscpi_supply_chain_pressure}
    out = composite([zscore(v).rename(k) for k, v in parts.items()], MIN_COMPONENTS)
    out["provisional"] = out.n_components < len(parts)
    return out.loc["2017-10-01":].dropna(subset=["index"])


def cipi_nowcast(m, horizon_end=None):
    y = np.log(m.CIPI_ALL).diff()
    X = pd.DataFrame({f"brent_l{l}": np.log(m.brent_ugx).diff().shift(l) for l in range(3)})
    X = X.join(pd.DataFrame({f"fx_l{l}": np.log(m.exchange_rate).diff().shift(l) for l in range(3)}))
    X = X.join(pd.DataFrame({f"ppi_l{l}": np.log(m.ppi_materials).diff().shift(l) for l in (1, 2)}))
    X.insert(0, "const", 1.0)
    last = m.CIPI_ALL.last_valid_index()
    fit = pd.concat([y.rename("y"), X], axis=1).loc[:last].dropna()
    beta, *_ = np.linalg.lstsq(fit.drop(columns="y").values, fit.y.values, rcond=None)
    resid = fit.y.values - fit.drop(columns="y").values @ beta
    sigma = resid.std(ddof=len(beta))
    fut = X.loc[last:].iloc[1:].dropna()
    if horizon_end:
        fut = fut.loc[:horizon_end]
    path, lvl = [], m.CIPI_ALL.loc[last]
    for h, (d, row) in enumerate(fut.iterrows(), start=1):
        lvl = lvl * np.exp(row.values @ beta)
        band = 1.2816 * sigma * np.sqrt(h)
        path.append({"date": d, "cipi_nowcast": lvl, "lo80": lvl * np.exp(-band), "hi80": lvl * np.exp(band), "h": h})
    coefs = pd.Series(beta, index=X.columns).rename("coef").to_frame()
    coefs.loc["sigma", "coef"] = sigma
    coefs.loc["n_obs", "coef"] = len(fit)
    coefs.loc["r2", "coef"] = 1 - resid.var() / fit.y.var()
    return pd.DataFrame(path).set_index("date") if path else pd.DataFrame(), coefs


def basket_index(m, basket=BASKET):
    w = pd.Series({k: v[1] for k, v in basket.items()})
    w = w / w.sum()
    rel = pd.concat({k: m[col] / m[col].loc[BASE[0]] for k, (col, _) in basket.items()}, axis=1)
    return (rel * w).sum(axis=1, min_count=len(w))


def escalation_gap(I, durations=DURATIONS, fixed=FIXED_SHARE, lag=INDEX_LAG):
    """I: basket index (monthly). One row per base month x duration. Contracts still running at the last index month
    are measured on the months valued so far (complete=False, months_valued < duration): the gap to date."""
    I = I.dropna()
    rows = []
    idx = I.index
    for i, t0 in enumerate(idx):
        if i - lag < 0:
            continue
        for D in durations:
            months = [k for k in range(1, D + 1) if i + k < len(idx)]
            if not months:
                continue
            actual = np.mean([I.iloc[i + k] / I.iloc[i] - 1 for k in months])
            recovered = (1 - fixed) * np.mean([I.iloc[i + k - lag] / I.iloc[i - lag] - 1 for k in months])
            rows.append({"base_month": t0, "duration_m": D, "months_valued": len(months),
                         "complete": len(months) == D, "reportable": len(months) >= MIN_VALUED, "actual_pct": 100 * actual,
                         "gap_formula_pct": 100 * (actual - recovered), "gap_fixed_price_pct": 100 * actual})
    return pd.DataFrame(rows)


def procurement(q):
    parts = [zscore(q.mean_bidders).rename("bidders"), zscore(q.single_bid_share, sign=-1).rename("single_bid"),
             zscore(q.non_open_share, sign=-1).rename("non_open"), zscore(q.median_days_to_award, sign=-1).rename("days")]
    return composite(parts, 3).dropna(subset=["index"])


def payment(q):
    parts = [zscore(-q.fiscal_deficit / q.overall_public_expenditure).rename("deficit_ratio"),
             zscore(q.lending_rate).rename("lending_rate"),
             zscore(pct(q.private_credit, 4), sign=-1).rename("credit_growth"),
             zscore(pct(q.overall_public_expenditure, 4), sign=-1).rename("spend_growth")]
    return composite(parts, 3).dropna(subset=["index"])


def dispute(q):
    c4 = q.court_decisions.fillna(0).rolling(4).sum()
    p4 = q.press_dispute_articles.rolling(4, min_periods=4).sum()
    parts = [zscore(c4).rename("court_4q"), zscore(p4).rename("press_4q")]
    out = composite(parts, 1)
    out["court_4q"], out["press_4q"] = c4, p4
    out["payment_share_4q"] = q.court_payment.fillna(0).rolling(4).sum() / c4
    return out.loc["2016-04-01":].dropna(subset=["index"])


def weather():
    c = pd.read_csv(GH / "uganda-seasons-construction-delay/outputs/monthly_calendar.csv")
    w = c.pivot_table(index=["region", "month"], columns="period", values="lost_days")
    w["change"] = w["2006-2025"] - w["1981-2000"]
    return w.reset_index()


def complete_quarters(q, as_of=AS_OF):
    """Drop the quarter that contains the data cut-off (partial counts)."""
    return q.loc[: (pd.Timestamp(as_of).to_period("Q") - 1).start_time]


def main():
    m = pd.read_csv(OUT / "panel_monthly.csv", parse_dates=["date"], index_col="date")
    q = complete_quarters(pd.read_csv(OUT / "panel_quarterly.csv", parse_dates=["date"], index_col="date"))
    res = {"idx_ccpi": ccpi(m), "idx_procurement": procurement(q), "idx_payment": payment(q),
           "idx_dispute": dispute(q)}
    now, coefs = cipi_nowcast(m)
    res["cipi_nowcast"], res["cipi_nowcast_model"] = now, coefs
    I = basket_index(m)
    res["escalation_basket"] = I.rename("basket").to_frame()
    res["idx_escalation_gap"] = escalation_gap(I).set_index("base_month")
    res["weather_outlook"] = weather().set_index(["region", "month"])
    for k, v in res.items():
        v.to_csv(OUT / f"{k}.csv")
    (OUT / "INDEX_VERSION").write_text(f"{VERSION}\nbase {BASE[0]}..{BASE[1]}\n")
    for k in ("idx_ccpi", "idx_procurement", "idx_payment", "idx_dispute"):
        v = res[k]
        print(f"{k}: {v.index.min():%Y-%m} -> {v.index.max():%Y-%m}; last {v['index'].iloc[-1]:.1f} "
              f"(n={int(v.n_components.iloc[-1])})")
    print("nowcast:\n", now.round(1).to_string(), "\nmodel:\n", coefs.round(4).T.to_string())
    g = res["idx_escalation_gap"]
    print("escalation gap, 18-month contracts, last 6 base months:\n",
          g[g.duration_m == 18].tail(6).round(2).to_string())


if __name__ == "__main__":
    main()
