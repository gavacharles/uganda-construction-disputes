"""Stage 1.2 — inventory the full-text judgments downloaded by hand (zips in 'Downloaded judgements/').

Each zip is extracted to JUDGMENTS/<zip stem>/ (~/Research/ulii_judgments, outside OneDrive) (originals untouched). Builds one index
case_id -> pdf path, flags duplicates across zips (sha256), checks coverage against the corpus and
the gold set, and tests each PDF for a text layer (born-digital vs needs OCR).
Outputs: data/interim/judgments_index.parquet, outputs/logs/judgments_coverage.csv
"""
import hashlib
import zipfile

import pandas as pd
import pdfplumber
from pdf2image import pdfinfo_from_path

from src.config import DOWNLOADS, INTERIM, JUDGMENTS, LOGS, PROCESSED
from src.ingest.registers import case_aliases

SRC = DOWNLOADS
DEST = JUDGMENTS


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def text_layer(p):
    try:
        with pdfplumber.open(p) as pdf:
            n = len(pdf.pages)
            chars = sum(len((pg.extract_text() or "").strip()) for pg in pdf.pages[:5])
            sample = min(n, 5)
    except Exception:
        n, chars, sample = 0, 0, 0
    if n == 0:
        n = pdfinfo_from_path(str(p))["Pages"]
    return n, (chars / sample if sample else 0)


def main():
    rows, idx = [], []
    for z in sorted(SRC.glob("*.zip")):
        out = DEST / z.stem
        if not out.exists():
            out.mkdir(parents=True)
            with zipfile.ZipFile(z) as zf:
                for m in zf.namelist():  # flat names only; refuse paths that escape the folder
                    if "/" in m or "\\" in m or m.startswith("."):
                        continue
                    (out / m).write_bytes(zf.read(m))
        if (out / "_index.csv").exists():
            ix = pd.read_csv(out / "_index.csv").assign(zip=z.stem)
            idx.append(ix)
        for p in sorted(out.glob("*.pdf")):
            rows.append({"case_id": p.stem, "zip": z.stem, "path": str(p.relative_to(DEST)),
                         "bytes": p.stat().st_size, "sha256": sha(p)})

    f = pd.DataFrame(rows)
    status = pd.concat(idx).drop_duplicates(["file", "zip"]) if idx else pd.DataFrame()
    copies = f.groupby("case_id").agg(n_copies=("zip", "size"), n_versions=("sha256", "nunique"))
    f = f.sort_values("zip").drop_duplicates("case_id", keep="first").merge(copies, on="case_id")

    tl = [text_layer(DEST / p) for p in f.path]
    f["pages"] = [t[0] for t in tl]
    f["chars_per_page"] = [round(t[1]) for t in tl]
    f["needs_ocr"] = f.chars_per_page < 200
    f.to_parquet(INTERIM / "judgments_index.parquet", index=False)

    cases = pd.read_parquet(PROCESSED / "cases.parquet")
    gold = pd.read_excel(SRC / "gold_download_list.xlsx")
    alias = case_aliases(cases)                      # '78 / 85' citations: PDF may be filed under either number
    with_pdf = {alias[i] for i in f.case_id if i in alias}
    cases["has_pdf"] = cases.case_id.isin(with_pdf)
    cov = cases.groupby(["period", "stream", "source"]).agg(cases=("case_id", "size"), with_pdf=("has_pdf", "sum"))
    cov["pct"] = (100 * cov.with_pdf / cov.cases).round(1)
    cov.to_csv(LOGS / "judgments_coverage.csv")

    extra = f[~f.case_id.isin(alias)]
    missing = cases[~cases.has_pdf & cases.source.eq("register")]
    missing[["case_id", "citation", "parties", "court", "stream", "period", "ulii_link"]].to_csv(
        LOGS / "judgments_missing.csv", index=False)
    print(f"PDFs: {len(rows)} files, {len(f)} unique case_ids; "
          f"case_ids with differing copies: {(f.n_versions > 1).sum()}")
    if len(status):
        print("index status:", status["status"].value_counts().to_dict())
    print(f"needs OCR: {f.needs_ocr.sum()} of {len(f)} ({f.loc[f.needs_ocr, 'pages'].sum()} pages)")
    print(f"gold set with PDF: {gold.case_id.isin(f.case_id).sum()} / {len(gold)}")
    print(f"PDFs not matching a corpus case_id: {len(extra)} -> {extra.case_id.tolist()[:15]}")
    print(f"register cases without PDF: {len(missing)}")
    print(cov.to_string())


if __name__ == "__main__":
    main()
