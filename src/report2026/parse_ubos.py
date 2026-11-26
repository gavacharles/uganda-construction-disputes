"""Sector report 2026: construction-relevant series from the latest UBOS releases (data/external/sector/ubos).

Long output data/processed/sector2026/ubos_series.csv: series, date (period start), value, unit, freq, file.
  QGDP   construction value added (constant 2016/17 prices, UGX bn), GDP (constant), construction share of GDP (current %)
  IIP    industrial production index: cement/lime/plaster (2394), concrete articles (2395), clay building materials
         (2392), basic iron and steel (2410), non-metallic minerals (Div 23), total industry
  PPI    producer price index (manufacturing) for the same classes, monthly
  CPI    headline annual inflation, monthly
  Trade  merchandise imports by SITC division (US$ '000), monthly: 66 non-metallic mineral manufactures (incl. cement),
         67 iron and steel, 69 metal manufactures, 72 specialised machinery, 74 general industrial machinery
  RPPI   residential property price index (headline, Greater Kampala), quarterly
  CIPI   latest UBOS release in the folder (to document the publication gap)
"""
import datetime as dt
import re

import pandas as pd

from src.config import PROCESSED, ROOT
from src.report2026.fetch import upload_rank

SRC = ROOT / "data" / "external" / "sector" / "ubos"
OUT = PROCESSED / "sector2026"
FY_Q_START = {"Q1": 7, "Q2": 10, "Q3": 1, "Q4": 4}   # Uganda FY starts 1 July


def latest(pattern):
    hits = sorted(SRC.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return max(hits, key=lambda p: upload_rank(p.name))


def fyq_to_date(fy, q):
    y = int(str(fy)[:4])
    m = FY_Q_START[q]
    return pd.Timestamp(y if m >= 7 else y + 1, m, 1)


def rows(series, dates, values, unit, freq, f):
    return [{"series": series, "date": d, "value": float(v), "unit": unit, "freq": freq, "file": f.name}
            for d, v in zip(dates, values) if pd.notna(d) and pd.notna(v) and str(v).strip() not in ("", "-")]


def qgdp():
    out = []
    for f, sheet, series, unit in [
        (latest("*QGDP*Constant*"), "Original_VA", "gdp_construction_va_const", "UGX bn, 2016/17 prices"),
        (latest("*QGDP*Current*"), "Original_%share", "gdp_construction_share_cur", "% of GDP, current prices"),
    ]:
        d = pd.read_excel(f, sheet_name=sheet, header=None)
        fy = d.iloc[2].ffill()
        cols = [c for c in d.columns[1:] if str(d.iloc[3, c]).strip() in FY_Q_START and pd.notna(fy[c])]
        dates = [fyq_to_date(fy[c], str(d.iloc[3, c]).strip()) for c in cols]
        r = d.index[d[0].astype(str).str.strip().str.lower() == "construction"][0]
        out += rows(series, dates, d.loc[r, cols], unit, "Q", f)
        if series == "gdp_construction_va_const":
            out += rows("gdp_total_const", dates, d.loc[4, cols], unit, "Q", f)
    return out


def iip():
    f = latest("*Index_of_Industrial_Production*Excel_Tables*")
    d = pd.read_excel(f, sheet_name=0, header=None)
    cols = [c for c in d.columns if re.match(r"\d{4}/\d{2}-Q[1-4]", str(d.iloc[2, c]))]
    dates = [fyq_to_date(str(d.iloc[2, c])[:7], str(d.iloc[2, c])[-2:]) for c in cols]
    keys = {"2394": "iip_cement", "2395": "iip_concrete", "2392": "iip_clay_bricks", "2410": "iip_iron_steel",
            "Div 23": "iip_nonmetallic", "TOTAL INDUSTRIAL SECTOR": "iip_total"}
    out = []
    for k, s in keys.items():
        m = d[d.iloc[:, 1:5].astype(str).apply(lambda r: r.str.strip().eq(k).any(), axis=1)]
        if len(m):
            out += rows(s, dates, m.iloc[0][cols], "index, 2016/17=100", "Q", f)
    return out


def ppi():
    f = latest("*PPI*")
    d = pd.read_excel(f, sheet_name="INDICES PER CLASS", header=None)
    cols = [c for c in d.columns[4:] if isinstance(d.iloc[0, c], (dt.datetime, pd.Timestamp))]
    dates = [pd.Timestamp(d.iloc[0, c]).to_period("M").to_timestamp() for c in cols]
    out = []
    for code, s in {"2394": "ppi_cement", "2395": "ppi_concrete", "2392": "ppi_clay_bricks", "2410": "ppi_iron_steel"}.items():
        m = d[d[2].astype(str).str.strip().str.replace(".0", "", regex=False) == code]
        out += rows(s, dates, m.iloc[0][cols], "index", "M", f)
    m = d[d[3].astype(str).str.strip().str.upper() == "MANUFACTURING"]
    out += rows("ppi_manufacturing", dates, m.iloc[0][cols], "index", "M", f)
    return out


def cpi():
    f = latest("*CPI*")
    d = pd.read_excel(f, sheet_name=0, header=None)
    cols = [c for c in d.columns[3:] if isinstance(d.iloc[0, c], (dt.datetime, pd.Timestamp))]
    dates = [pd.Timestamp(d.iloc[0, c]).to_period("M").to_timestamp() for c in cols]
    r = d.index[d[1].astype(str).str.strip() == "Headline"][0]
    return rows("cpi_headline_inflation", dates, d.loc[r, cols], "% y/y", "M", f)


def _trade_date(v):
    if isinstance(v, (dt.datetime, pd.Timestamp)):
        v = pd.Timestamp(v)
        return pd.Timestamp(2000 + v.day, v.month, 1) if v.day != 1 else v   # "Apr-26" auto-read as 26 Apr
    try:  # labels vary: 'Jan-24', 'Jan -21', 'Jan_23'
        return pd.Timestamp(dt.datetime.strptime(re.sub(r"\s*[-_]\s*", "-", str(v).strip()), "%b-%y"))
    except ValueError:
        return pd.NaT


def imports():
    f = latest("*Composition_of_Imports*")
    d = pd.read_excel(f, sheet_name=0, header=None)
    dates = [_trade_date(d.iloc[1, c]) for c in d.columns[2:]]
    out = []
    for code, s in {"66": "imp_nonmetallic_minerals", "67": "imp_iron_steel", "69": "imp_metal_manufactures",
                    "72": "imp_specialised_machinery", "74": "imp_general_machinery"}.items():
        m = d[d[0].astype(str).str.strip().str.zfill(2) == code]
        out += rows(s, dates, m.iloc[0, 2:], "US$ '000", "M", f)
    return out


def rppi():
    f = latest("*RPPI*")
    d = pd.read_excel(f, sheet_name=0, header=None)
    fy = d[0].where(d[0].astype(str).str.match(r"\d{4}/\d{2}")).ffill()
    hcol = [c for c in d.columns if str(d.iloc[2, c]).strip() == "Headline"][0]
    ok = d[1].astype(str).str.strip().isin(FY_Q_START) & fy.notna()
    dates = [fyq_to_date(fy[i], str(d.loc[i, 1]).strip()) for i in d.index[ok]]
    return rows("rppi_headline", dates, d.loc[ok, hcol], "index, 2015/16=100", "Q", f)


def cipi_release():
    f = latest("*CIPI*")
    return {"latest_cipi_file": f.name}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allrows = []
    for fn in (qgdp, iip, ppi, cpi, imports, rppi):
        r = fn()
        allrows += r
        print(f"{fn.__name__}: {len(r)} obs")
    d = pd.DataFrame(allrows).sort_values(["series", "date"])
    d.to_csv(OUT / "ubos_series.csv", index=False)
    print(d.groupby("series").agg(first=("date", "min"), last=("date", "max"), n=("value", "size"),
                                  last_value=("value", "last")).to_string())
    print(cipi_release())


if __name__ == "__main__":
    main()
