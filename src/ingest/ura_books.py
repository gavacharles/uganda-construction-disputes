"""PhD (stage 13 input): further tables from the URA annual data books, read from every edition that has them.

  fuel_tax_<product>        Recorded fuel taxes, UGX bn (petrol, diesel, kerosene)
  fuel_volume_<product>     Imported fuel volumes, million litres (annual total of Table 3.1.1)
  pump_price_<product>      Average pump price, UGX per litre (UBOS figures reported by URA)
  vat_returns_<category>    Construction (ISIC F) VAT returns: expected, nil, non_filer, payment, offset (from the
                            FY2020/21 edition on); the FY2018/19 edition gives nil, payment and refund/offset counts
  sector_<ISIC section>     Gross collections by ISIC section (A-T) from the revenue-by-sector table, UGX bn
  subsector_<ISIC4>         Gross collections from construction-related sub-sectors in the top-50 sub-sector table
                            (2394 cement, 4100 buildings, 42xx civil engineering, 43xx specialised, 4663 wholesale of
                            construction materials), UGX bn
Output: data/processed/fiscal_triangulation/ura_books_long.csv (series, fy, value, book, line)
Usage: python -m src.ingest.ura_books
"""
import re

import pandas as pd

from src.report2026.parse_ura import BOOKS, FY, SRC, find_row, header_years, is_title, numbers, text

OUT = SRC.parents[2] / "processed" / "fiscal_triangulation"
PRODUCTS = {"petrol": r"Petrol\s*[\[(]?PMS[\])]?", "diesel": r"Diesel\s*[\[(]?AGO[\])]?",
            "kerosene": r"Kero?se?ne\s*[\[(]?BIK[\])]?"}
SUBSECTORS = ("2394", "4100", "4210", "4220", "4290", "4311", "4312", "4321", "4322", "4329", "4330", "4390", "4663")
VAT_COLS = ["expected", "nil", "non_filer", "payment", "offset"]
SECTIONS = [("A", r"agricultur"), ("B", r"mining"), ("C", r"manufactur"), ("D", r"electricity"), ("E", r"water supply"),
            ("F", r"construction"), ("G", r"wholesale"), ("H", r"transport"), ("I", r"accommodation"),
            ("J", r"information"), ("K", r"financial"), ("L", r"real estate"), ("M", r"professional"),
            ("N", r"administrative"), ("O", r"public admin"), ("P", r"education"), ("Q", r"health"), ("R", r"arts"),
            ("S", r"other service"), ("T", r"households"), ("U", r"extraterritorial")]


def section_of(*texts):
    for t in texts:
        for code, pat in SECTIONS:
            if t and re.search(pat, t, re.I):
                return code
    return None


def sector_rows(lines, i, yrs, add):
    """Every ISIC-section row of the revenue-by-sector table at line i (labels may wrap onto adjacent lines)."""
    end = next((k for k in range(i + 1, min(i + 160, len(lines)))
                if re.search(r"Source: Uganda Revenue|Revenue share \(Net\)", lines[k])), None)
    if end is None:
        return
    numeric = lambda ln: numbers(ln, len(yrs)) is not None and not FY.search(ln)
    for k in range(i + 1, end):
        ln = lines[k]
        if not numeric(ln) or re.match(r"^\s*(total|un\s?sectori)", ln, re.I):
            continue
        here = re.split(r"\s{2,}[\d(-]", ln.strip(), maxsplit=1)[0]
        prev = lines[k - 1].strip() if not numeric(lines[k - 1]) else ""
        nxt = lines[k + 1].strip() if k + 1 < end and not numeric(lines[k + 1]) else ""
        code = section_of(here, prev, nxt)      # a wrapped label starts above its numbers
        if code:
            for y, x in zip(yrs, numbers(ln, len(yrs))):
                add(f"sector_{code}", y, x, k)


FY_ANY = re.compile(r"(20\d\d)\s*[-/]\s*(20)?(\d\d)\b")


def years_any(lines, i, span=12):
    """Fiscal-year header near line i, written either 2021/22 or 2021-22."""
    for j in range(i + 1, min(i + span, len(lines))):
        yrs = list(dict.fromkeys(f"{m.group(1)}/{m.group(3)}" for m in FY_ANY.finditer(lines[j])))
        if len(yrs) >= 3:
            return yrs
    return None


def fy_from(label):
    m = FY.search(label.replace("-", "/"))
    return f"{m.group(1)}/{m.group(3)}" if m else None


def extract(lines, book):
    rows = []

    def add(series, fy, value, k):
        if fy and value is not None:
            rows.append({"series": series, "fy": fy, "value": value, "book": book, "line": k + 1})

    for i, ln in enumerate(lines):
        if is_title(ln, r"(Gross )?Revenue Collections.*by economic sector|Revenue collections \(UGX Billions\) by Economic Sector"):
            yrs = header_years(lines, i)
            if yrs:
                sector_rows(lines, i, yrs, add)
        if is_title(ln, r"Recorded Fuel taxes"):
            yrs = header_years(lines, i)
            for p, pat in PRODUCTS.items():
                k = find_row(lines, i, pat, within=14)
                v = numbers(lines[k], len(yrs)) if yrs and k else None
                for y, x in zip(yrs or [], v or []):
                    add(f"fuel_tax_{p}", y, x, k)
        if is_title(ln, r"Average Pump Fuel Prices"):
            yrs = header_years(lines, i)
            for p, pat in PRODUCTS.items():
                k = find_row(lines, i, pat, within=10)
                v = numbers(lines[k], len(yrs)) if yrs and k else None
                for y, x in zip(yrs or [], v or []):
                    add(f"pump_price_{p}", y, x, k)
        if is_title(ln, r"Monthly Imported Fuel Volumes"):
            hdr = next((lines[j] for j in range(i, i + 8) if len(FY.findall(lines[j])) >= 6), None)
            k = find_row(lines, i, "Total", within=40)
            if hdr and k:
                yrs = [f"{m.group(1)}/{m.group(3)}" for m in FY.finditer(hdr)]
                v = numbers(lines[k], len(yrs))
                if v and len(yrs) % 3 == 0:
                    n = len(yrs) // 3
                    for j, p in enumerate(PRODUCTS):
                        for y, x in zip(yrs[j * n:(j + 1) * n], v[j * n:(j + 1) * n]):
                            add(f"fuel_volume_{p}", y, x, k)
        if is_title(ln, r"Number of VAT Returns Filed"):
            k = find_row(lines, i, r"(F-)?Construction", within=400)
            if k is None:
                continue
            if re.search(r"(Nil|Payment|Refund)", lines[k]):          # FY2018/19 edition: one year, by category
                y = fy_from(ln)
                for j in range(k, k + 3):
                    m = re.search(r"(Nil return|Payment return|Refund/offset return)\s+([\d,]+)", lines[j])
                    if m:
                        cat = {"Nil return": "nil", "Payment return": "payment", "Refund/offset return": "offset"}[m.group(1)]
                        add(f"vat_returns_{cat}", y, float(m.group(2).replace(",", "")), j)
                continue
            for j in range(k, k + 12):                                # later editions: rows by financial year
                y = fy_from(lines[j][:60]) if re.search(r"20\d\d\s*[-/]\s*\d\d", lines[j]) else None
                if j > k and re.match(r"^\s*[A-Z][a-z]", lines[j]) and not y and re.search(r"\d", lines[j]):
                    break
                if y:
                    v = numbers(lines[j], 5)
                    for c, x in zip(VAT_COLS, v or []):
                        add(f"vat_returns_{c}", y, x, j)
        if is_title(ln, r"Top 50 Sub-sectors"):
            yrs = years_any(lines, i)
            for k in range(i, min(i + 260, len(lines))):
                m = re.search(r"\b(\d{4})-[A-Z]", lines[k])
                if m and m.group(1) in SUBSECTORS and yrs:
                    for j in range(k, k + 3):                       # numbers may sit on the next line
                        v = numbers(re.sub(r"^\s*\d{1,2}\s+", " ", lines[j].replace(m.group(0), " ")), len(yrs))
                        if v and all(x is not None and x < 5000 for x in v):
                            for y, x in zip(yrs, v):
                                add(f"subsector_{m.group(1)}", y, x, j)
                            break
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f, book in BOOKS.items():
        rows += extract(text(SRC / f), book)
    d = pd.DataFrame(rows).drop_duplicates(["series", "fy", "book"])
    d.to_csv(OUT / "ura_books_long.csv", index=False)
    piv = d.sort_values("book").groupby(["series", "fy"]).value.last().unstack("series")
    pd.set_option("display.width", 250)
    print(piv.round(1).to_string())


if __name__ == "__main__":
    main()
