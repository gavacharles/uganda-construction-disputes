"""Stage 1.5 — merge register cases and included OCR'd scanned judgments into one corpus.

Dedupe: exact case_id, then fuzzy parties match (same year, token-set ratio >= 90) flagged for review.
Output: data/processed/cases.parquet, outputs/logs/merge_report.csv, outputs/logs/prisma_counts.csv
"""
import pandas as pd
from rapidfuzz import fuzz

from src.config import INTERIM, LOGS, PROCESSED

OUTCOME_TEXT = {  # OCR rule outcome -> register-style label, so one outcome coder serves both
    "allowed / granted": "Granted", "dismissed": "Dismissed", "struck out": "Struck out",
    "referred to arbitration": "Referred to arbitration", "award set aside": "Award set aside",
    "unclear": "See judgment",
}
STREAM = {"construction": "construction", "construction (to confirm)": "construction",
          "both": "both", "arbitration": "arbitration"}


def main():
    reg = pd.read_parquet(INTERIM / "registers_cases.parquet")
    ocr = pd.read_parquet(INTERIM / "ocr_cases.parquet")
    o = to_rows(ocr[ocr.include].copy(), "ocr")
    extra_fp = INTERIM / "extra_cases.parquet"
    xtr = pd.read_parquet(extra_fp) if extra_fp.exists() else None
    x = to_rows(xtr[xtr.include].copy(), "fulltext") if xtr is not None else o.iloc[0:0]
    o = pd.concat([o, x[~x.case_id.isin(o.case_id)]], ignore_index=True)
    reg = reg.assign(ocr_conf=None, stream_to_confirm=False)
    return finish(reg, o, ocr, xtr)


def to_rows(inc, source):
    o = pd.DataFrame({
        "case_id": inc.case_id, "citation": inc.citation, "period": "2015-2025",
        "year": inc.year.astype("Int64"), "stream": inc.stream_rule.map(STREAM),
        "date_decided": inc.date_decided, "parties": inc.parties, "court": inc.court, "judge": inc.judge,
        "case_number": inc.case_number, "holding_summary": None,
        "outcome_register": inc.outcome_rule.map(OUTCOME_TEXT), "key_issues": None,
        "project_type_register": None, "ulii_link": inc.ulii_link, "statutes_register": None,
        "in_construction": inc.stream_rule.ne("arbitration"), "in_arbitration": inc.stream_rule.isin(["both", "arbitration"]),
        "also_flag": inc.stream_rule.eq("both"), "source": source, "ocr_conf": inc.ocr_conf.round(1),
        "stream_to_confirm": inc.stream_rule.eq("construction (to confirm)"),
    })
    o.loc[o.year >= 2026, "period"] = "2026"
    return o


def finish(reg, o, ocr, xtr):

    # fuzzy duplicate check: same year, near-identical party names
    flags = []
    for _, r in o.iterrows():
        same = reg[reg.year == r.year]
        for _, s in same.iterrows():
            score = fuzz.token_set_ratio(str(r.parties).lower(), str(s.parties).lower())
            if score >= 90:
                flags.append({"ocr_case": r.citation, "register_case": s.citation, "parties_ocr": r.parties,
                              "parties_register": s.parties, "score": score})
    pd.DataFrame(flags).to_csv(LOGS / "merge_report.csv", index=False)

    dup = o.case_id.isin(reg.case_id)
    cases = pd.concat([reg, o[~dup]], ignore_index=True)
    assert cases.case_id.is_unique
    cases.to_parquet(PROCESSED / "cases.parquet", index=False)

    prisma = pd.DataFrame([
        ("register cases (ULII summaries screened)", len(reg)),
        ("scanned candidates OCR'd", len(ocr)),
        ("scanned excluded: minimal reasoning", int((ocr.flag_minimal).sum())),
        ("scanned excluded: not construction/arbitration", int((~ocr.flag_minimal & ocr.flag_irrelevant).sum())),
        ("scanned included", int(ocr.include.sum())),
        ("downloads not in registers screened", 0 if xtr is None else len(xtr)),
        ("downloads excluded: minimal reasoning", 0 if xtr is None else int(xtr.flag_minimal.sum())),
        ("downloads excluded: not construction/arbitration", 0 if xtr is None else int((~xtr.flag_minimal & xtr.flag_irrelevant).sum())),
        ("downloads included", 0 if xtr is None else int(xtr.include.sum())),
        ("duplicates of register cases (case_id)", int(dup.sum())),
        ("possible party-name duplicates to review", len(flags)),
        ("final corpus", len(cases)),
    ], columns=["step", "n"])
    prisma.to_csv(LOGS / "prisma_counts.csv", index=False)
    print(prisma.to_string(index=False))
    print(cases.groupby(["period", "stream", "source"]).size().unstack(fill_value=0).to_string())
    if flags:
        print(pd.DataFrame(flags).to_string(index=False))


if __name__ == "__main__":
    main()
