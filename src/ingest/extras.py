"""Stage 1.4b — screen downloaded judgments that are not in the registers (third source: 'fulltext').

Metadata: citation/date from the download index, court from the citation code, case number, parties
and judge parsed from the first page. Relevance and minimal-reasoning rules reuse src.ingest.ocr_cases.code.
Outputs: data/interim/extra_cases.parquet, outputs/logs/exclusions_extra.csv
"""
import re

import pandas as pd

from src.config import INTERIM, JUDGMENTS, LOGS, PROCESSED
from src.ingest.ocr_cases import code
from src.ingest.registers import case_id, norm_citation

COURT = {"UGSC": "Supreme Court", "UGCA": "Court of Appeal", "UGCC": "Constitutional Court",
         "UGCommC": "High Court (Commercial Division)", "UGHCCD": "High Court (Civil Division)",
         "UGHCLD": "High Court (Land Division)", "UGHCFD": "High Court (Family Division)",
         "UGHCEBD": "High Court (Execution and Bailiffs Division)", "UGHCACD": "High Court (Anti-Corruption Division)",
         "UGHCCRD": "High Court (Criminal Division)", "UGHC": "High Court", "UGIC": "Industrial Court",
         "UGTAT": "Tax Appeals Tribunal", "UGPPDPAAT": "PPDA Appeals Tribunal", "UGCADER": "CADER"}
ROLE_A = r"(?:PLAINTIFF|APPLICANT|APPELLANT|CLAIMANT|PETITIONER)S?"
ROLE_B = r"(?:DEFENDANT|RESPONDENT)S?"
NAME = r"([A-Z][A-Za-z0-9 .,&()'’/\-]{3,120}?)"


def tidy(s):
    s = re.sub(r"^\s*\d+[.)]?\s*", "", s)
    s = re.sub(r"[\s:;.…_\-]+$", "", s).strip(" ,")
    return s.title() if s.isupper() else s


def parse_head(text):
    head = text[:6000]
    num = re.search(r"(?i)((?:civil|misc\w*|miscellaneous|arbitration|company|labour|tax)[^\n]{0,40}?"
                    r"(?:suit|appeal|application|cause|petition|reference|revision)[^\n]{0,15}?no\.?\s*[\w/]+\s*of\s*\d{4})", head)
    a = re.search(rf"\n\s*{NAME}\s*[:.…_\-\s]*(?:\n\s*)?(?:\d+\w*\s*)?{ROLE_A}", head)
    b = re.search(rf"\n\s*{NAME}\s*[:.…_\-\s]*(?:\n\s*)?(?:\d+\w*\s*)?{ROLE_B}", head)
    judge = re.search(r"(?i)before\s*[:\-]?\s*((?:hon\.?|the hon\w*|his lordship|her lordship|lady justice|justice)[^\n]{3,80})", head)
    parties = f"{tidy(a.group(1))} v {tidy(b.group(1))}" if a and b else None
    return (num.group(1).strip() if num else None, parties, judge.group(1).strip() if judge else None)


def main():
    ix = pd.concat([pd.read_csv(p) for p in JUDGMENTS.glob("*/_index.csv")])
    ix = ix[ix.status.eq("downloaded")].drop_duplicates("file")
    ix["case_id"] = ix.file.str.replace(".pdf", "", regex=False)
    cases = pd.read_parquet(PROCESSED / "cases.parquet")
    second = {case_id(c.split(" / ")[1]) for c in cases.citation if " / " in c}
    texts = pd.read_parquet(INTERIM / "texts.parquet")
    ex = texts[~texts.case_id.isin(cases.case_id) & ~texts.case_id.isin(second)].merge(
        ix[["case_id", "citation", "date", "ulii_link"]], on="case_id", how="left")

    meta = ex.text.fillna("").map(parse_head)
    ex["case_number"] = [m[0] for m in meta]
    ex["parties"] = [m[1] for m in meta]
    ex["judge"] = [m[2] for m in meta]
    ex["citation"] = ex.citation.map(norm_citation)
    ex["court"] = ex.case_id.str.split("_").str[1].map(COURT).fillna("Other")
    ex["date_decided"] = pd.to_datetime(ex.date, errors="coerce")
    ex["year"] = ex.case_id.str[:4].astype(int)
    ex["ocr_conf"] = ex["ocr_conf"]
    ex = code(ex)
    ex["source"] = "fulltext"

    ex.drop(columns=["text"]).to_parquet(INTERIM / "extra_cases.parquet", index=False)
    excl = ex.loc[~ex.include, ["case_id", "citation", "words", "cons_score", "arb_score"]].assign(
        reason=lambda x: x.apply(lambda r: "minimal reasoning" if r.words < 800 else "not construction/arbitration", axis=1),
        stage="full-text screen of downloads not in registers")
    excl.to_csv(LOGS / "exclusions_extra.csv", index=False)
    print(f"extras screened: {len(ex)}; included {ex.include.sum()}; excluded {(~ex.include).sum()}")
    print(ex.stream_rule.value_counts().to_string())
    print("parties parsed:", ex.parties.notna().mean().round(2), "| case number parsed:", ex.case_number.notna().mean().round(2),
          "| judge parsed:", ex.judge.notna().mean().round(2))
    print(ex[ex.include].groupby("year").size().to_dict())


if __name__ == "__main__":
    main()
