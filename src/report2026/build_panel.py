"""Sector report 2026: assemble the monthly and quarterly panels that feed the indices.

Monthly (data/processed/sector2026/panel_monthly.csv): CIPI headline/civil/building and sub-indices, macro block
(FX, CBR, lending rate, CPI, private credit, GSCPI) from the CiO panel (UBOS + MoFPED/BoU, to Apr 2026), extended to
Oct 2026 for FX and Brent with the CiO macro extension (same definitions); UBOS PPI (to Sep 2026), CPI inflation
(to Oct 2026), imports by SITC (to Aug 2026); fiscal panel (to Feb 2026).
Quarterly (panel_quarterly.csv, calendar quarters, period start): QGDP, IIP, RPPI; works procurement from the GPP
OCDS mirror; court decisions; Observer press signals.
Usage: python -m src.report2026.build_panel
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import INTERIM, PROCESSED, ROOT

EXT = ROOT / "data" / "external"
CIO = EXT / "sector" / "github" / "cio-construction-costs" / "data" / "shared"
OUT = PROCESSED / "sector2026"


def ubos(freq):
    u = pd.read_csv(OUT / "ubos_series.csv", parse_dates=["date"])
    return u[u.freq == freq].pivot_table(index="date", columns="series", values="value")


def monthly():
    p = pd.read_csv(CIO / "panel_v1.0_extended.csv", parse_dates=["date"]).set_index("date")
    ext = pd.read_csv(CIO / "macro_extension_monthly.csv", parse_dates=["date"]).set_index("date")
    p = p.combine_first(ext[["exchange_rate", "brent_crude_usd"]])
    g = pd.read_csv(CIO / "government_funding_panel.csv", parse_dates=["date"]).set_index("date")
    u = ubos("M")   # latest UBOS releases replace the panel's copies of the same series
    m = p.drop(columns=[c for c in u.columns if c in p.columns]).join(g, how="outer").join(u, how="outer")
    m = m.loc["2015-07-01":]
    cbr = extend_cbr(m.central_bank_rate)
    m = m.reindex(m.index.union(cbr.index))
    m["central_bank_rate"] = cbr
    m["brent_ugx"] = m.brent_crude_usd * m.exchange_rate
    m["ppi_materials"] = m[["ppi_cement", "ppi_iron_steel", "ppi_clay_bricks", "ppi_concrete"]].mean(axis=1, skipna=False)
    m.index.name = "date"
    return m


def extend_cbr(cbr, as_of=None):
    """Extend the policy rate past the panel with verified MPC decisions (src/report2026/cbr_decisions.csv): the rate
    in force at the end of each month. Overlapping months must agree with the panel."""
    d = pd.read_csv(Path(__file__).with_name("cbr_decisions.csv"), parse_dates=["date"]).sort_values("date")
    end = max(pd.Timestamp(as_of or pd.Timestamp.today()), d.date.max()).to_period("M").to_timestamp()
    months = pd.date_range(cbr.dropna().index.max() + pd.offsets.MonthBegin(1), end, freq="MS")
    out = cbr.copy().reindex(cbr.index.union(months))
    last = cbr.dropna().iloc[-1]
    for t in months:
        in_force = d[d.date <= t + pd.offsets.MonthEnd(0)]
        out.loc[t] = in_force.rate.iloc[-1] if len(in_force) else last
    for _, r in d.iterrows():
        t = r.date.to_period("M").to_timestamp()
        if t in cbr.dropna().index and abs(cbr.loc[t] - r.rate) > 1e-9 and r.date.day < 20:
            print(f"warning: panel CBR {cbr.loc[t]} differs from decision {r.rate} in {t:%Y-%m}")
    return out


def works_bids():
    """Bidders per procurement from the OCDS bid records (ocid -> n_bids)."""
    rows = []
    with gzip.open(EXT / "ocds" / "uganda_ppda_full.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            det = (r.get("bids") or {}).get("details") or []
            if det:
                rows.append({"ocid": r.get("ocid"), "n_bids": len(det)})
    return pd.DataFrame(rows).groupby("ocid").n_bids.max()


def procurement_q():
    k = pd.read_parquet(INTERIM / "ocds_contracts.parquet")
    w = k[k.works & k.award_date.notna()].copy()
    w["n_bids"] = w.ocid.map(works_bids())
    w["q"] = w.award_date.dt.to_period("Q").dt.start_time
    ts = pd.to_datetime(w.tender_start, errors="coerce", utc=True).dt.tz_localize(None)
    w["days_to_award"] = (w.award_date - ts).dt.days.where(lambda x: x.between(0, 1500))
    ratio = (w.value_ugx / w.estimate_ugx).replace([np.inf, -np.inf], np.nan)
    w["award_to_estimate"] = ratio.where(ratio.between(0.2, 5))
    w["non_open"] = (w.method != "open").astype(float)
    return w.groupby("q").agg(
        works_awards=("ocid", "nunique"), works_value_ugx_bn=("value_ugx", lambda x: x.sum() / 1e9),
        mean_bidders=("n_bids", "mean"), bid_records=("n_bids", "count"),
        single_bid_share=("n_bids", lambda x: (x.dropna() == 1).mean() if x.notna().any() else np.nan),
        non_open_share=("non_open", "mean"), median_days_to_award=("days_to_award", "median"),
        median_award_to_estimate=("award_to_estimate", "median"))


def disputes_q():
    a = pd.read_parquet(PROCESSED / "analysis_cases.parquet")
    a = a[a.include.fillna(True).astype(bool) & a.stream.isin(["construction", "both"])].copy()
    a["q"] = pd.to_datetime(a.date_decided, errors="coerce").dt.to_period("Q").dt.start_time
    for c in ("c_payment", "c_delay_eot", "c_price_adjust"):   # coded Y/N
        a[c] = a[c].astype(str).str.upper().eq("Y").astype(int)
    c = a.groupby("q").agg(court_decisions=("case_id", "size"), court_payment=("c_payment", "sum"),
                           court_delay=("c_delay_eot", "sum"), court_price_adjust=("c_price_adjust", "sum"))
    o = pd.read_csv(EXT / "the-observer/paper2_online_discourse/outputs_observer_full/monthly_dispute_signals.csv")
    o["q"] = pd.to_datetime(o.month).dt.to_period("Q").dt.start_time
    press = o.groupby("q")["count"].sum().rename("press_dispute_articles")
    return c.join(press, how="outer")


def quarterly(m):
    q = ubos("Q")
    q.index = q.index.to_period("Q").start_time
    mq = m[["CIPI_ALL", "CIPI_CIVIL", "exchange_rate", "brent_crude_usd", "lending_rate", "central_bank_rate",
            "private_credit", "cpi_headline_inflation", "ppi_materials"]].resample("QS").mean()
    fis = m[["overall_public_expenditure", "fiscal_deficit", "total_central_government_revenue"]].resample("QS").sum(min_count=3)
    imp = m[[c for c in m.columns if c.startswith("imp_")]].resample("QS").sum(min_count=3)
    out = q.join([mq, fis, imp, procurement_q(), disputes_q()], how="outer").loc["2015-07-01":]
    out.index.name = "date"
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = monthly()
    m.to_csv(OUT / "panel_monthly.csv")
    q = quarterly(m)
    q.to_csv(OUT / "panel_quarterly.csv")
    cov = pd.concat([pd.DataFrame({"freq": f, "series": df.columns,
                                   "first_obs": [df[c].first_valid_index() for c in df.columns],
                                   "last_obs": [df[c].last_valid_index() for c in df.columns]})
                     for f, df in (("M", m), ("Q", q))])
    cov.to_csv(OUT / "panel_coverage.csv", index=False)
    print(f"monthly {m.shape}, quarterly {q.shape}")
    print(cov.groupby(["freq", "last_obs"]).series.apply(lambda s: ", ".join(s)).to_string())


if __name__ == "__main__":
    main()
