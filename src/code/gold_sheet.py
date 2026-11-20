"""Stage 2.1b — build the blind gold-set coding workbook (coder 1: 150 cases; coder 2: 40-case subset).

Columns and code lists follow codebook.md. Reading aids only (PDF link, issues as framed);
no rule-based codes are shown, so manual coding stays an independent benchmark.
Output: outputs/coding/gold_coding_sheet.xlsx
"""
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from src.config import DOWNLOADS, INTERIM, JUDGMENTS, OUT, SEED

YN, YNNA = ["Y", "N"], ["Y", "N", "NA"]
PARTY = ["state body", "contractor", "subcontractor", "consultant", "supplier", "private employer",
         "bank or insurer", "individual", "other"]
FIELDS = [  # (column, allowed codes or None for free entry)
    ("include", YN), ("exclusion_reason", ["bare interlocutory", "no reasons", "not construction or arbitration", "duplicate", "NA"]),
    ("stream", ["construction", "arbitration", "both"]),
    ("proceeding", ["suit", "appeal", "application", "arbitration cause", "tribunal review", "judicial review", "other"]),
    ("decision_type", ["final judgment", "ruling on application", "appeal decision", "tribunal decision"]),
    ("initiator_type", PARTY), ("respondent_type", PARTY), ("state_party", YN), ("foreign_party", YN),
    ("project_type", ["road", "building", "water and sanitation", "energy and power", "bridge or other civil",
                      "supply-and-install", "not stated"]),
    ("contract_form", ["FIDIC 1999", "FIDIC 2017", "FIDIC other", "PPDA SBD", "JBC or PAM", "bespoke written",
                       "oral or no written contract", "not stated"]),
    ("contract_value_ugx", None),
    *[(c, YN) for c in ["c_payment", "c_final_account", "c_variations", "c_delay_eot", "c_lds", "c_termination",
                        "c_defects", "c_guarantees", "c_procurement", "c_site_land", "c_prof_fees", "c_price_adjust",
                        "c_force_majeure", "c_fraud"]],
    ("notice_timebar_raised", YN), ("notice_timebar_upheld", YNNA), ("tier_skipped_raised", YN),
    ("limitation_raised", YN), ("clauses_cited", None),
    ("outcome", ["initiator succeeded", "partly succeeded", "initiator failed", "referred to arbitration",
                 "struck out (procedural)", "remitted or retrial", "consent"]),
    ("currency", ["UGX", "USD", "EUR", "other", "NA"]), ("quantum_claimed", None), ("quantum_awarded", None),
    ("interest_rate_pct", None), ("interest_basis", ["commercial", "court rate", "contractual", "none", "not stated"]),
    ("costs_to", ["initiator", "respondent", "each party own", "NA"]), ("filing_date", None),
    ("arb_application_type", ["stay or referral", "interim measures", "appointment", "jurisdiction", "set-aside",
                              "enforcement or recognition", "extension of time", "other", "NA"]),
    *[(g, YNNA) for g in ["g_public_policy", "g_jurisdiction", "g_natural_justice", "g_out_of_time",
                          "g_arbitrability", "g_invalid_agreement"]],
    ("award_result", ["upheld", "set aside", "partly set aside", "remitted", "enforced", "enforcement refused", "NA"]),
    ("deference", YNNA), ("seat_foreign", YNNA),
    ("confidence", ["high", "medium", "low"]), ("notes", None),
]
INFO = ["case_id", "citation", "court", "date_decided", "parties", "pages", "pdf", "issues_as_framed"]
HEAD, INFO_FILL, CODE_FILL = PatternFill("solid", fgColor="DCE8F7"), PatternFill("solid", fgColor="F2F2F2"), PatternFill("solid", fgColor="FFF9E6")


def build(ws, n):
    info_n = len(INFO)
    for i, c in enumerate(ws[1], 1):
        c.font, c.fill = Font(bold=True), HEAD
        c.alignment = Alignment(wrap_text=True, vertical="top")
    widths = {"case_id": 17, "citation": 19, "court": 22, "date_decided": 11, "parties": 40, "pages": 6, "pdf": 9,
              "issues_as_framed": 50, "clauses_cited": 28, "notes": 40}
    for i, (col, codes) in enumerate([(c, None) for c in INFO] + FIELDS, 1):
        L = get_column_letter(i)
        ws.column_dimensions[L].width = widths.get(col, 13)
        if codes:
            dv = DataValidation(type="list", formula1='"' + ",".join(codes) + '"', allow_blank=True,
                                showErrorMessage=True, errorTitle="Not in codebook", error="Pick a code from the list (see codebook.md).")
            ws.add_data_validation(dv)
            dv.add(f"{L}2:{L}{n + 1}")
        for r in range(2, n + 2):
            ws[f"{L}{r}"].fill = INFO_FILL if i <= info_n else CODE_FILL
    for r in range(2, n + 2):
        cell = ws.cell(row=r, column=INFO.index("pdf") + 1)
        if cell.value:
            cell.hyperlink, cell.value, cell.style = cell.value, "open PDF", "Hyperlink"
    ws.freeze_panes = ws.cell(row=2, column=3)
    ws.row_dimensions[1].height = 45


def main():
    gold = pd.read_excel(DOWNLOADS / "gold_download_list.xlsx")
    ix = pd.read_parquet(INTERIM / "judgments_index.parquet").set_index("case_id")
    seg = pd.read_parquet(INTERIM / "segments.parquet").set_index("case_id")
    g = pd.DataFrame({
        "case_id": gold.case_id, "citation": gold.citation, "court": gold.court,
        "date_decided": pd.to_datetime(gold.date_decided).dt.date, "parties": gold.parties,
        "pages": gold.case_id.map(ix["pages"]),
        "pdf": gold.case_id.map(lambda c: (JUDGMENTS / ix.loc[c, "path"]).as_uri() if c in ix.index else None),
        "issues_as_framed": gold.case_id.map(lambda c: " | ".join(seg.loc[c, "issue_list"][:5]) if c in seg.index else ""),
    })
    for col, _ in FIELDS:
        g[col] = ""
    # coder 2 subset: 40 cases stratified by stream (seed 42)
    rng = np.random.default_rng(SEED)
    sub = gold.groupby("stream", group_keys=False).apply(
        lambda d: d.sample(n=round(40 * len(d) / len(gold)), random_state=SEED)).case_id
    g2 = g[g.case_id.isin(sub)].copy()

    out = OUT / "coding" / "gold_coding_sheet.xlsx"
    readme = pd.DataFrame({"Step": [
        "1. Read codebook.md before starting; keep it open.",
        "2. Code each row in the 'coder1' sheet: open the PDF, read facts / issues / holding, pick codes from the dropdowns.",
        "3. Light-yellow cells are yours to fill; grey cells are reading aids. 'issues_as_framed' is extracted from the judgment and may be incomplete.",
        "4. Do not look at rule-based outputs while coding (blind coding keeps the gold set an independent benchmark).",
        "5. The second coder fills 'coder2' (40 cases) independently, without seeing coder1.",
        "6. Save the file; then run: python -m src.code.kappa  (agreement per field).",
    ]})
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        xw.sheets["README"].column_dimensions["A"].width = 140
        g.to_excel(xw, sheet_name="coder1", index=False)
        build(xw.sheets["coder1"], len(g))
        g2.to_excel(xw, sheet_name="coder2", index=False)
        build(xw.sheets["coder2"], len(g2))
    print(out, "| coder1", len(g), "| coder2", len(g2), "| with PDF", g.pdf.notna().sum(),
          "| issues pre-extracted", (g.issues_as_framed != "").sum())
    print("coder2 by register stream:", gold[gold.case_id.isin(sub)].stream.value_counts().to_dict())


if __name__ == "__main__":
    main()
