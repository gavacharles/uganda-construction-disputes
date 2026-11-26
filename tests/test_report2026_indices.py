"""Checks on the sector-report index formulas (synthetic data; no downloads needed)."""
import numpy as np
import pandas as pd
import pytest

from src.report2026 import indices as ix
from src.report2026.fetch import upload_rank
from src.report2026.parse_ura import header_years, numbers


def months(n, start="2017-01-01"):
    return pd.date_range(start, periods=n, freq="MS")


def test_scale_and_zscore_centre_on_base():
    s = pd.Series(np.arange(120, dtype=float), index=months(120, "2016-01-01"))
    z = ix.zscore(s)
    base = z.loc[ix.BASE[0]:ix.BASE[1]]
    assert base.mean() == pytest.approx(0, abs=1e-9)
    assert base.std() == pytest.approx(1)
    assert ix.scale(pd.Series([0.0, 1.0])).tolist() == [100, 110]


def test_composite_requires_min_components():
    a = pd.Series([1.0, np.nan], name="a")
    b = pd.Series([1.0, 2.0], name="b")
    out = ix.composite([a, b], min_n=2)
    assert out["index"].iloc[0] == 110
    assert np.isnan(out["index"].iloc[1])


def test_escalation_gap_flat_prices_is_zero():
    g = ix.escalation_gap(pd.Series(100.0, index=months(40)))
    assert np.allclose(g[["actual_pct", "gap_formula_pct", "gap_fixed_price_pct"]], 0)


def test_escalation_gap_no_fixed_share_no_lag_recovers_everything():
    I = pd.Series(100 * 1.01 ** np.arange(40), index=months(40))
    g = ix.escalation_gap(I, durations=(12,), fixed=0.0, lag=0)
    assert np.allclose(g.gap_formula_pct, 0)
    assert (g.gap_fixed_price_pct > 0).all()


def test_escalation_gap_fixed_share_leaves_that_share_unrecovered():
    I = pd.Series(100 * 1.01 ** np.arange(40), index=months(40))
    g = ix.escalation_gap(I, durations=(12,), fixed=0.2, lag=0)
    full = g[g.complete]
    assert np.allclose(full.gap_formula_pct, 0.2 * full.actual_pct)


def test_escalation_gap_marks_running_contracts():
    g = ix.escalation_gap(pd.Series(100.0, index=months(20)), durations=(18,))
    assert g.complete.sum() == 1 and (~g.complete).sum() > 0
    assert g[~g.complete].months_valued.max() < 18


def test_nowcast_recovers_known_relation():
    rng = np.random.default_rng(42)
    n = 110
    idx = months(n)
    brent = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.05, n))), index=idx)
    fx = pd.Series(3600 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    ppi = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    dl = 0.001 + 0.05 * np.log(brent * fx).diff().shift(1).fillna(0)
    cipi = pd.Series(100 * np.exp(np.cumsum(dl)), index=idx)
    cipi.iloc[-6:] = np.nan
    m = pd.DataFrame({"CIPI_ALL": cipi, "brent_ugx": brent * fx, "exchange_rate": fx, "ppi_materials": ppi})
    path, coefs = ix.cipi_nowcast(m)
    assert len(path) == 6
    assert coefs.loc["r2", "coef"] > 0.99
    assert (path.lo80 <= path.cipi_nowcast).all() and (path.cipi_nowcast <= path.hi80).all()


def test_complete_quarters_drops_partial_quarter():
    q = pd.DataFrame({"x": range(4)}, index=pd.to_datetime(["2026-01-01", "2026-04-01", "2026-07-01", "2026-10-01"]))
    assert ix.complete_quarters(q, as_of="2026-11-16").index.max() == pd.Timestamp("2026-07-01")


def test_ura_header_typo_and_numbers():
    assert header_years(["Sector   2017/18   2018/19   2019/21   2021/22"], 0) == ["2018/19", "2019/20", "2020/21", "2021/22"]
    assert numbers("Construction        -     127.03    146.77    1,179.36", 4) == [None, 127.03, 146.77, 1179.36]
    assert numbers("Tax Refunds   (440.05)   (543.00)   (637.26)   (723.44)", 4)[0] == -440.05


def test_ubos_upload_rank_orders_by_month_year():
    names = ["CPI-October-2025-Excel-Tables.xlsx", "09_2026CPI_September_2026.xlsx", "11_2026CPI_October_2026.xlsx"]
    assert max(names, key=upload_rank) == "11_2026CPI_October_2026.xlsx"
