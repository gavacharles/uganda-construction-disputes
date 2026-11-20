"""Stage 1.3 — OCR the scanned ULII judgments.

Per PDF:
  1. ocrmypdf (Tesseract, eng) -> searchable PDF in data/interim/ocr/pdf/
  2. pytesseract per page (300 dpi) -> page text + mean word confidence
Outputs
  data/interim/ocr/pages.parquet        file, page, text, conf, n_words
  outputs/logs/ocr_low_conf.xlsx        pages with conf < 80 (manual check)
  outputs/logs/ocr_files.csv            per-file summary
Usage: python -m src.ingest.ocr [--limit N]
"""
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pdfplumber
import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path

from src.config import INTERIM, LOGS

OCR = INTERIM / "ocr"
SRC, PDF_OUT = OCR / "src" / "PDFs", OCR / "pdf"
CONF_MIN, DPI = 80, 300


def text_layer(pdf):
    with pdfplumber.open(pdf) as p:
        return [(pg.extract_text() or "") for pg in p.pages]


def ocr_page(args):
    pdf, page_no = args
    img = convert_from_path(str(pdf), dpi=DPI, first_page=page_no, last_page=page_no)[0]
    d = pytesseract.image_to_data(img, lang="eng", config="--psm 3",
                                  output_type=pytesseract.Output.DATAFRAME)
    d = d[(d.conf >= 0) & d.text.notna() & d.text.astype(str).str.strip().ne("")]
    # rebuild text line by line
    lines = d.groupby(["block_num", "par_num", "line_num"], sort=True)["text"].apply(
        lambda s: " ".join(map(str, s)))
    conf = float(d.conf.mean()) if len(d) else 0.0
    return pdf.name, page_no, "\n".join(lines), conf, len(d)


def searchable_pdf(pdf):
    out = PDF_OUT / pdf.name
    if not out.exists():
        subprocess.run(["ocrmypdf", "--skip-text", "--rotate-pages", "-l", "eng",
                        "--jobs", "2", "-q", str(pdf), str(out)], check=False)
    return out.exists()


def main(limit=None):
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(SRC.glob("*.pdf"))[:limit]
    pages_fp = OCR / "pages.parquet"
    done = pd.read_parquet(pages_fp) if pages_fp.exists() else pd.DataFrame(
        columns=["file", "page", "text", "conf", "n_words", "method"])

    rows, jobs = [], []
    for pdf in pdfs:
        if pdf.name in set(done.file):
            continue
        layer = text_layer(pdf)
        if not layer:  # pdfplumber reads 0 pages on some malformed page trees; trust poppler
            layer = [""] * pdfinfo_from_path(str(pdf))["Pages"]
        if layer and sum(len(t.strip()) for t in layer) > 200 * len(layer) * 0.5:  # born-digital
            rows += [(pdf.name, i + 1, t, 100.0, len(t.split()), "text_layer")
                     for i, t in enumerate(layer)]
        else:
            jobs += [(pdf, i + 1) for i in range(len(layer))]

    print(f"{len(pdfs)} PDFs; {len(jobs)} pages to OCR")
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(ocr_page, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            rows.append((*f.result(), "tesseract"))
            if k % 100 == 0:
                print(f"  {k}/{len(jobs)} pages", flush=True)
        ok = list(ex.map(searchable_pdf, pdfs))
    print(f"searchable PDFs: {sum(ok)}/{len(pdfs)}")

    new = pd.DataFrame(rows, columns=done.columns)
    pages = pd.concat([done, new]).sort_values(["file", "page"]).reset_index(drop=True)
    pages.to_parquet(pages_fp, index=False)

    low = pages[(pages.conf < CONF_MIN) & (pages.n_words > 0)]
    low[["file", "page", "conf", "n_words"]].to_excel(LOGS / "ocr_low_conf.xlsx", index=False)
    files = pages.groupby("file").agg(
        pages=("page", "size"), words=("n_words", "sum"), method=("method", "first"),
        ocr_conf=("conf", lambda c: round(float(c[pages.loc[c.index, "n_words"] > 0].mean()), 1)),
        low_pages=("conf", lambda c: int((c < CONF_MIN).sum())))
    files.to_csv(LOGS / "ocr_files.csv")
    print(files.describe().round(1).to_string())
    print(f"pages below {CONF_MIN}: {len(low)} of {len(pages)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    main(ap.parse_args().limit)
