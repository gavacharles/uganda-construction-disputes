"""Sector report 2026: construction-relevant series from the URA annual data books (text PDFs, via pdftotext -layout).

Each book reports four fiscal years, so successive books overlap. A series is read from every book that has it.
The value from the most recent book is kept, and books that disagree by more than 5% are flagged (URA revises figures,
and some tables in the books are visibly mis-keyed, e.g. cement "production" of 11 billion tonnes in FY2024/25).
The FY2019/20 book is scanned (no text layer) and is skipped; FY2015/16-2018/19 come from the stand-alone sector table.

Series (UGX bn unless stated):
  construction_sector_revenue  gross tax collected from the construction sector (ISIC F)
  cement_excise                domestic excise duty on cement
  cement_vat                   domestic VAT on cement
  construction_vat             VAT from construction (reported from FY2022/23)
  diesel_pump_price            average pump price, UGX/litre (UBOS figures reported by URA)
Output: data/processed/sector2026/ura_series.csv (+ ura_crosscheck.csv)
"""
import re
import subprocess

import pandas as pd

from src.config import PROCESSED, ROOT

SRC = ROOT / "data" / "external" / "sector" / "ura"
OUT = PROCESSED / "sector2026"
FY = re.compile(r"(20\d\d)\s*/\s*(20)?(\d\d)")
NUM = re.compile(r"\(?-?[\d,]+\.?\d*\)?|(?<=\s)-(?=\s|$)")
BOOKS = {  # file -> fiscal year of the book (orders the books: later wins)
    "revenue-collections-ugx-billions-by-economic-sector.pdf": "2018/19",
    "ura-annual-data-book-fy2018-19.pdf": "2018/19",
    "ura-annual-data-book.pdf": "2020/21",
    "ura-annual-data-book-fy-2023-2024.pdf": "2023/24",
    "annual-data-book-fy2024-2025.pdf": "2024/25",
}


def text(pdf):
    return subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout.splitlines()


def fy_label(m):
    return f"{m.group(1)}/{m.group(3)}"


def header_years(lines, i, span=8):
    """Fiscal-year labels of the first line near i that names at least three of them."""
    for j in range(i, min(i + span, len(lines))):
        yrs = [fy_label(m) for m in FY.finditer(lines[j])]
        if len(yrs) >= 3:  # headers are consecutive years; rebuild from the last one (fixes typos like "2019/21")
            last = int(yrs[-1][:4])
            return [f"{y}/{(y + 1) % 100:02d}" for y in range(last - len(yrs) + 1, last + 1)]
    return None


def numbers(line, n):
    """Last n numeric cells of a row ('-' = missing; '(x)' = negative)."""
    body = re.sub(r"^\s*[A-Z]?-?[A-Za-z][A-Za-z ,;&()\[\]/.-]*?(?=\s{2,}|\s+[\d(-])", "", line)
    vals = []
    for tok in NUM.findall(body):
        if tok == "-":
            vals.append(None)
            continue
        neg = tok.startswith("(")
        v = float(tok.strip("()").replace(",", ""))
        vals.append(-v if neg else v)
    return vals[-n:] if len(vals) >= n else None


def is_title(line, pat):
    return re.search(pat, line, re.I) and "...." not in line and "��" not in line


def find_row(lines, start, label, stop=None, within=120):
    for k in range(start, min(start + within, len(lines))):
        if stop and k > start and re.search(stop, lines[k], re.I):
            return None
        if re.match(rf"^\s*{label}\s{{2,}}", lines[k], re.I):
            return k
    return None


def extract(lines, book):
    rows = []

    def add(series, yrs, vals, page_line):
        if yrs and vals:
            for y, v in zip(yrs[-len(vals):], vals):
                rows.append({"series": series, "fy": y, "value": v, "book": book, "line": page_line + 1})

    for i, ln in enumerate(lines):
        # construction sector revenue
        if is_title(ln, r"(Gross )?Revenue Collections.*by economic sector|Revenue collections \(UGX Billions\) by Economic Sector"):
            yrs = header_years(lines, i)
            k = find_row(lines, i, r"(F-)?Construction", within=60)
            if yrs and k:
                add("construction_sector_revenue", yrs, numbers(lines[k], len(yrs)), k)
        # tax heads: cement excise, cement VAT, construction VAT
        if is_title(ln, r"Table 1\.1\.1:? Revenue Collections.*by Tax Head|Table 1\.1\.\d+:? Revenue Collections \(UGX,? ?Bn\) by Tax Head"):
            yrs = header_years(lines, i)
            ex = find_row(lines, i, r"Excise Duty", within=80)
            vat = find_row(lines, i, r"(Value Added Tax|VALUE ADDED TAX)", within=120)
            if yrs and ex:
                k = find_row(lines, ex, "Cement", stop=r"^\s*(Value Added Tax|VALUE ADDED TAX)\s", within=40)
                if k:
                    add("cement_excise", yrs, numbers(lines[k], len(yrs)), k)
            if yrs and vat:
                stop = r"^\s*Taxes on International Trade\s"
                k = find_row(lines, vat, "Cement", stop=stop, within=60)
                if k:
                    add("cement_vat", yrs, numbers(lines[k], len(yrs)), k)
                k = find_row(lines, vat, "Construction", stop=stop, within=60)
                if k:
                    add("construction_vat", yrs, numbers(lines[k], len(yrs)), k)
        # pump prices
        if is_title(ln, r"Average Pump Fuel Prices"):
            yrs = header_years(lines, i)
            k = find_row(lines, i, r"Diesel \[?AGO\]?", within=10)
            if yrs and k:
                add("diesel_pump_price", yrs, numbers(lines[k], len(yrs)), k)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f, book in BOOKS.items():
        rows += extract(text(SRC / f), book)
    d = pd.DataFrame(rows).dropna(subset=["value"])
    d = d.drop_duplicates(["series", "fy", "book"], keep="first")
    d["book_rank"] = d.book.map(lambda b: int(b[:4]))
    d["fy_start"] = d.fy.str[:4].astype(int)
    chk = (d.groupby(["series", "fy"])
             .agg(n_books=("value", "size"), vmin=("value", "min"), vmax=("value", "max")).reset_index())
    chk["max_rel_diff"] = (chk.vmax - chk.vmin) / chk.vmax.abs()
    best = d.sort_values("book_rank").groupby(["series", "fy"]).tail(1)
    out = best.merge(chk, on=["series", "fy"])
    out["flag"] = out.max_rel_diff.gt(0.05).map({True: "books disagree >5%; latest book kept", False: ""})
    out = out.sort_values(["series", "fy"])[["series", "fy", "value", "book", "line", "n_books", "max_rel_diff", "flag"]]
    out.to_csv(OUT / "ura_series.csv", index=False)
    d.sort_values(["series", "fy", "book"]).to_csv(OUT / "ura_crosscheck.csv", index=False)
    print(out.pivot(index="fy", columns="series", values="value").to_string())
    print("\nflags:\n", out[out.flag != ""][["series", "fy", "value", "vmin_vmax"]].to_string()
          if False else out[out.flag != ""][["series", "fy", "value", "book", "max_rel_diff"]].to_string())


if __name__ == "__main__":
    main()
