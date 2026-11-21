"""Build one review workbook for the decisions only the researcher can make.

Sheets: to_confirm (include/exclude), duplicates (same/separate), ocr_low_conf (checked), legislation (verified).
Output: outputs/review/review_queue.xlsx  (only undecided items; decisions are read back by src.review.apply_review)
"""
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from src.config import INTERIM, JUDGMENTS, LOGS, OUT, PROCESSED
from src.review import apply_review as review

DEST = OUT / "review"
OCR_PDF = INTERIM / "ocr" / "pdf"
HEAD = PatternFill("solid", fgColor="DCE8F7")
TODO = PatternFill("solid", fgColor="FFF4CC")


def pdf_links():
    ix = pd.read_parquet(INTERIM / "judgments_index.parquet")
    links = {r.case_id: (JUDGMENTS / r.path).as_uri() for r in ix.itertuples()}
    ocr = pd.read_parquet(INTERIM / "ocr_cases.parquet")
    for r in ocr.itertuples():
        p = OCR_PDF / r.file
        if p.exists():
            links.setdefault(r.case_id, p.as_uri())
    return links


def sheet(xw, name, df, decision_col, choices, widths):
    df = df.copy()
    df[decision_col] = ""
    df["note"] = ""
    df.to_excel(xw, sheet_name=name, index=False)
    ws = xw.sheets[name]
    for c in ws[1]:
        c.fill, c.font = HEAD, Font(bold=True)
    for i, w in enumerate(widths + [18, 40]):
        ws.column_dimensions[chr(65 + i)].width = w
    if df.empty:  # nothing left to decide on this sheet
        return
    dcol = chr(65 + df.columns.get_loc(decision_col))
    dv = DataValidation(type="list", formula1='"' + ",".join(choices) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{dcol}2:{dcol}{len(df) + 1}")
    for r in range(2, len(df) + 2):
        ws[f"{dcol}{r}"].fill = TODO
        for c in ws[r]:
            c.alignment = Alignment(vertical="top", wrap_text=False)
        if "pdf" in df.columns:
            cell = ws.cell(row=r, column=df.columns.get_loc("pdf") + 1)
            if cell.value:
                cell.hyperlink, cell.value, cell.style = cell.value, "open PDF", "Hyperlink"
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    links = pdf_links()

    conf = []
    for src, fp in [("scanned (OCR)", "ocr_cases.parquet"), ("download not in registers", "extra_cases.parquet")]:
        d = pd.read_parquet(INTERIM / fp)
        d = d[d.include & d.stream_rule.eq("construction (to confirm)")]
        conf.append(d.assign(source=src))
    conf = pd.concat(conf)
    tc = pd.DataFrame({"case_id": conf.case_id, "citation": conf.citation, "source": conf.source,
                       "court": conf.court, "year": conf.year, "parties": conf.parties,
                       "words": conf.words, "construction_terms": conf.cons_score,
                       "arbitration_terms": conf.arb_score, "pdf": conf.case_id.map(links)})
    tc = tc.sort_values("construction_terms")

    dup = pd.read_csv(LOGS / "merge_report.csv").rename(columns={
        "ocr_case": "added_case", "parties_ocr": "parties_added", "register_case": "register_case"})
    dup["added_case_id"] = dup.added_case.str.replace(r"\[(\d{4})\]\s*(\w+)\s*(\d+)", r"\1_\2_\3", regex=True)
    dup["register_case_id"] = dup.register_case.str.replace(r"\[(\d{4})\]\s*(\w+)\s*(\d+)", r"\1_\2_\3", regex=True)
    dup["pdf_added"] = dup.added_case_id.map(links)
    dup["pdf"] = dup.register_case_id.map(links)
    # evidence for the decision: case number, date and length of each judgment
    from src.ingest.extras import parse_head
    texts = pd.read_parquet(INTERIM / "texts.parquet", columns=["case_id", "text"]).set_index("case_id").text
    cases = pd.read_parquet(PROCESSED / "cases.parquet").set_index("case_id")
    for side, col in [("added", "added_case_id"), ("register", "register_case_id")]:
        dup[f"case_no_{side}"] = [parse_head(texts[i])[0] if i in texts.index else cases.case_number.get(i) for i in dup[col]]
        dup[f"date_{side}"] = [str(cases.date_decided.get(i, ""))[:10] for i in dup[col]]
        dup[f"words_{side}"] = [len(texts[i].split()) if i in texts.index else None for i in dup[col]]
    dup = dup[["added_case", "register_case", "parties_added", "parties_register", "score",
               "case_no_added", "case_no_register", "date_added", "date_register", "words_added", "words_register",
               "pdf_added", "pdf"]]

    low = pd.read_excel(LOGS / "ocr_low_conf.xlsx")
    low["pdf"] = low.file.map(lambda f: (OCR_PDF / f).as_uri() if (OCR_PDF / f).exists() else None)

    leg = pd.read_parquet(INTERIM / "legislation.parquet")
    leg = leg.loc[~leg.verified, ["leg_id", "title", "date", "commencement_status", "source"]]
    leg["date"] = leg["date"].dt.date

    # show only items not yet decided (decisions live in manual/review/*.csv)
    tc = tc[~tc.case_id.isin(review.load("to_confirm").case_id)]
    done = set(zip(review.load("duplicates").added_case, review.load("duplicates").register_case))
    dup = dup[[k not in done for k in zip(dup.added_case, dup.register_case)]]
    low = low[[k not in set(zip(review.load("ocr_low_conf").file, review.load("ocr_low_conf").page)) for k in zip(low.file, low.page)]]
    leg = leg[~leg.leg_id.isin(review.load("legislation").leg_id)]

    out = DEST / "review_queue.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        pd.DataFrame({"Sheet": ["to_confirm", "duplicates", "ocr_low_conf", "legislation"],
                      "What to decide": [
                          "Is the judgment a construction (or arbitration) dispute? Choose Include or Exclude.",
                          "Same parties, different citations: same decision (Duplicate), separate ruling (Separate), or exclude for another reason (Exclude; give the reason in note)? Compare case_no / date / words.",
                          "Open the PDF page and check the OCR text is usable (OK) or needs retyping (Fix).",
                          "Check against the gazetted text; record the commencement date in the note (Verified / Not found)."],
                      "Items": [len(tc), len(dup), len(low), len(leg)]}).to_excel(xw, sheet_name="README", index=False)
        xw.sheets["README"].column_dimensions["A"].width = 16
        xw.sheets["README"].column_dimensions["B"].width = 110
        sheet(xw, "to_confirm", tc, "decision", ["Include", "Exclude"], [18, 20, 22, 30, 6, 60, 8, 10, 10, 11])
        sheet(xw, "duplicates", dup, "decision", ["Duplicate", "Separate", "Exclude"], [20, 20, 50, 50, 6, 28, 28, 11, 11, 8, 8, 11, 11])
        sheet(xw, "ocr_low_conf", low, "decision", ["OK", "Fix"], [70, 6, 8, 8, 11])
        sheet(xw, "legislation", leg, "decision", ["Verified", "Not found"], [8, 60, 11, 45, 50])
    print(f"{out}\n to_confirm {len(tc)} | duplicates {len(dup)} | ocr_low_conf {len(low)} | legislation {len(leg)}")


if __name__ == "__main__":
    main()
