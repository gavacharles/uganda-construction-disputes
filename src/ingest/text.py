"""Stage 1.4a — extract full text from every downloaded judgment (born-digital via pdfplumber, OCR otherwise).

Output: data/interim/texts.parquet (case_id, text, pages, words, method, ocr_conf)
Usage: python -m src.ingest.text
"""
import re
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pdfplumber
from pdf2image import pdfinfo_from_path

from src.config import INTERIM, JUDGMENTS
from src.ingest.ocr import ocr_page

BOILER = [r"(?im)^\s*Page \d+ of \d+\s*$", r"(?im)^\s*\d+\s*$",
          r"(?i)downloaded from (ulii|www\.ulii)\S*.*$", r"(?i)uganda legal information institute.*$"]


def clean(t):
    for rx in BOILER:
        t = re.sub(rx, "", t)
    t = re.sub(r"-\n(\w)", r"\1", t)          # rejoin hyphenated line breaks
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def extract(row):
    path = JUDGMENTS / row["path"]
    if not row["needs_ocr"]:
        try:
            with pdfplumber.open(path) as pdf:
                pages = [pg.extract_text() or "" for pg in pdf.pages]
            if pages:
                t = clean("\n".join(pages))
                return row["case_id"], t, len(pages), len(t.split()), "text_layer", 100.0
        except Exception:
            pass
    n = pdfinfo_from_path(str(path))["Pages"]
    res = [ocr_page((path, i + 1)) for i in range(n)]
    t = clean("\n".join(r[2] for r in res))
    conf = sum(r[3] * r[4] for r in res) / max(1, sum(r[4] for r in res))
    return row["case_id"], t, n, len(t.split()), "tesseract", round(conf, 1)


def main():
    ix = pd.read_parquet(INTERIM / "judgments_index.parquet")
    out = INTERIM / "texts.parquet"
    done = set(pd.read_parquet(out).case_id) if out.exists() else set()
    todo = [r for r in ix.to_dict("records") if r["case_id"] not in done]
    print(f"{len(todo)} judgments to extract ({sum(r['needs_ocr'] for r in todo)} by OCR)", flush=True)
    rows = []
    with ProcessPoolExecutor() as ex:
        for k, r in enumerate(ex.map(extract, todo, chunksize=4), 1):
            rows.append(r)
            if k % 100 == 0:
                print(f"  {k}/{len(todo)}", flush=True)
    new = pd.DataFrame(rows, columns=["case_id", "text", "pages", "words", "method", "ocr_conf"])
    allt = pd.concat([pd.read_parquet(out), new]) if done else new
    allt.to_parquet(out, index=False)
    print(allt.method.value_counts().to_string())
    print(allt.words.describe().round(0).to_string())


if __name__ == "__main__":
    main()
