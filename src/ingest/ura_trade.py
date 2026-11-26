"""PhD (stage 13 input): URA Annual Imports Trade Report 2021 (HS 8-digit, the only edition on URA's statistics pages).

The report (2,583 pages, exported from Excel) has three parts:
  1.1  summary of countries      country -> value (UGX bn)
  1.2  country by commodity      HS code, description, value, quantity, unit; the country label is lost in the text layer,
                                 so country blocks are recovered where the HS codes restart, and each block is matched to a
                                 country by its total (unique match within MATCH_TOL)
  2.1  summary of commodities    HS code -> value (UGX bn)
  2.2  commodity by country      one value/quantity per HS code (followed by a list of countries without values)
Values in 1.2 add up to 2.1 by HS code; quantities and descriptions in 1.2, and values in 2.2, are unreliable
(see audit()). Only values are used.

Output: data/processed/fiscal_triangulation/ura_trade_2021.csv (one row per country block x HS code),
        ura_trade_2021_summary.csv (2.1), ura_trade_2021_audit.csv
Usage: python -m src.ingest.ura_trade
"""
import re
import subprocess

import pandas as pd

from src.config import PROCESSED, ROOT

SRC = ROOT / "data" / "external" / "sector" / "ura" / "annual-imports-trade-report.pdf"
OUT = PROCESSED / "fiscal_triangulation"
MATCH_TOL = 0.01          # block total within 1% of a country total
ROW = re.compile(r"^\s+(\d{4}\.\d{2}\.\d{2})\s+(.*?)\s+([\d,]+\.\d\d)\s+([\d,.]+)\s+(\w+)\s*$")
SUMMARY_ROW = re.compile(r"^(\d{4}\.\d{2}\.\d{2})\s+(.*?)\s+([\d,]+\.\d\d)\s*$")
COUNTRY_ROW = re.compile(r"^(\d+)\s+(.+?)\s{2,}([\d,]+\.\d\d)\s*$")


def num(s):
    return float(s.replace(",", ""))


def sections(lines):
    s12 = next(i for i, l in enumerate(lines) if l.startswith("1.2 Recorded Imports"))
    s21 = next(i for i, l in enumerate(lines) if i > s12 and l.startswith("2.1 Summary of Commodities"))
    s22 = next(i for i, l in enumerate(lines) if i > s21 and l.startswith("2.2 Recorded Imports"))
    return s12, s21, s22


def countries(lines, end):
    return pd.DataFrame([{"rank": int(m.group(1)), "country": m.group(2).strip(), "total": num(m.group(3))}
                         for l in lines[:end] if (m := COUNTRY_ROW.match(l))])


def blocks(lines):
    """Country-by-commodity rows split into blocks wherever the HS code sequence restarts."""
    out, b, prev = [], 0, None
    for l in lines:
        m = ROW.match(l)
        if not m:
            continue
        code = m.group(1)
        if prev and code < prev:
            b += 1
        out.append({"block": b, "hs8": code, "desc": m.group(2).strip(), "value_ugx_bn": num(m.group(3)),
                    "quantity": num(m.group(4)), "unit": m.group(5)})
        prev = code
    return pd.DataFrame(out)


def match_countries(rows, ctry, tol=MATCH_TOL):
    """Assign each block the country whose total is the unique match within tol (largest blocks first)."""
    tot = rows.groupby("block").value_ugx_bn.sum().sort_values(ascending=False)
    used, assign = set(), {}
    for b, v in tot.items():
        c = ctry[~ctry.country.isin(used) & ((ctry.total - v).abs() <= tol * ctry.total.clip(lower=1))]
        if len(c) == 1:
            assign[b] = c.country.iloc[0]
            used.add(assign[b])
    return rows.assign(country=rows.block.map(assign).fillna("unmatched"))


def summary(lines):
    rows = [{"hs8": m.group(1), "value_ugx_bn": num(m.group(3))} for l in lines if (m := SUMMARY_ROW.match(l))]
    return pd.DataFrame(rows).drop_duplicates("hs8")


def section22(lines):
    rows = []
    for l in lines:
        m = re.match(r"^(\d{4}\.\d{2}\.\d{2})\s+(.*?)\s+([\d,]+\.\d\d)\s+([\d,.]+)\s+(\w+)\s*$", l)
        if m:
            rows.append({"hs8": m.group(1), "value_22": num(m.group(3))})
    return pd.DataFrame(rows).drop_duplicates("hs8")


def audit(rows, summ, s22, ctry):
    by_code = rows.groupby("hs8")
    multi = rows[rows.hs8.isin(by_code.block.nunique().loc[lambda n: n >= 3].index)]
    const_qty = multi.groupby("hs8").quantity.nunique().eq(1)
    desc_var = multi.assign(d=multi.desc.str.lower().str.strip(" -")).groupby("hs8").d.nunique().gt(1)
    chk = summ.merge(by_code.value_ugx_bn.sum().rename("sum_12").reset_index(), on="hs8", how="outer")
    chk = chk.merge(s22, on="hs8", how="left")
    big = chk[chk.value_ugx_bn >= 1]
    matched = rows[rows.country != "unmatched"].value_ugx_bn.sum() / rows.value_ugx_bn.sum()
    return pd.DataFrame([
        {"check": "HS codes (in 3+ country blocks) with the same quantity in every block",
         "n": len(const_qty), "share_pct": 100 * const_qty.mean()},
        {"check": "HS codes (in 3+ country blocks) with differing descriptions across blocks",
         "n": len(desc_var), "share_pct": 100 * desc_var.mean()},
        {"check": "HS codes >= UGX 1 bn where section 1.2 sums to section 2.1 (within 1%)",
         "n": len(big), "share_pct": 100 * ((big.sum_12 - big.value_ugx_bn).abs() <= 0.01 * big.value_ugx_bn).mean()},
        {"check": "HS codes >= UGX 1 bn where the section 2.2 value equals section 2.1 (within 1%)",
         "n": int(big.value_22.notna().sum()),
         "share_pct": 100 * ((big.value_22 - big.value_ugx_bn).abs() <= 0.01 * big.value_ugx_bn)[big.value_22.notna()].mean()},
        {"check": "import value assigned to a country (blocks matched to section 1.1 totals)",
         "n": int(rows.block.nunique()), "share_pct": 100 * matched},
        {"check": "section 1.2 total as % of section 1.1 total", "n": len(ctry),
         "share_pct": 100 * rows.value_ugx_bn.sum() / ctry.total.sum()},
    ])


def main():
    lines = subprocess.run(["pdftotext", "-layout", str(SRC), "-"], capture_output=True, text=True).stdout.splitlines()
    s12, s21, s22 = sections(lines)
    ctry = countries(lines, s12)
    rows = match_countries(blocks(lines[s12:s21]), ctry)
    summ = summary(lines[s21:s22])
    aud = audit(rows, summ, section22(lines[s22:]), ctry)
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT / "ura_trade_2021.csv", index=False)
    summ.to_csv(OUT / "ura_trade_2021_summary.csv", index=False)
    aud.to_csv(OUT / "ura_trade_2021_audit.csv", index=False)
    print(f"{len(rows):,} rows in {rows.block.nunique()} blocks; {len(summ):,} HS codes in summary; "
          f"{(rows.country != 'unmatched').groupby(rows.block).first().sum()} blocks matched to countries")
    print(aud.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
