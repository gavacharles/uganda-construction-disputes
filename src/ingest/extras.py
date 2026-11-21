"""Stage 1.4b — screen downloaded judgments that are not in the registers (third source: 'fulltext').

Metadata: citation/date from the download index, court from the citation code, case number, parties
and judge parsed from the first page. Relevance and minimal-reasoning rules reuse src.ingest.ocr_cases.code.
Outputs: data/interim/extra_cases.parquet, outputs/logs/exclusions_extra.csv
"""
import re

import pandas as pd

from src.config import INTERIM, JUDGMENTS, LOGS
from src.ingest.ocr_cases import code
from src.ingest.registers import case_aliases, norm_citation

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


# party block parser: lines above / below the VERSUS line, so names wrapped over two lines are kept whole
ROLE = r"(?:\w{0,2}(?:laintiff|pplicant|ppellant|laimant|etitioner|efendant|espondent|bjector|ntervener))S?"
VS = re.compile(r"(?i)^\W*(?:versus|vs\.?|v\.?|and)\W*$")
STOP_A = re.compile(r"(?i)\bno\.?\s*\d|\bof\s+(?:19|20)\d{2}|arising|coram|before|republic|\bcourt\b|division|registry|"
                    r"at kampala|^\W*between\W*$|in the matter|judicature|^\W*\(|reference|\bcause\b|application|appeal|suit\b|holden")
STOP_B = re.compile(r"(?i)before|judgment|ruling|coram|\bhon\b|honourable|introduction|background|^\W*\(|arising|"
                    r"\bjustice\b|representation|brief facts|counsel|^\W*\d+\.\s+\w+ (?:is|was|filed)")


def _clean(line):
    s = re.sub(rf"(?i)\b{ROLE}\b", " ", line)
    s = re.sub(r"(?i)\bH\.?C\.?T[\w\-/.]*\d{4}\b|:\s*[A-Za-z]?\s*$", " ", s)   # e-filing case numbers, stray ': I'
    s = re.sub(r"^\s*\d{1,3}\s+(?=[A-Z]|\d\s*[.)])", "", s)          # margin line numbers
    s = re.sub(r"(?:\s*[:;.…_\-=]){2,}|[}{|\[\]=•]|\s\)", " ", s)     # dot/colon leaders, braces, OCR junk
    s = re.sub(r"(?i)\b\d(?:st|nd|rd|th)\b|/\s*judgment (?:debtor|creditor)", " ", s)
    s = re.sub(r"(?:\s+(?:\d{1,3}|[a-z]))+\s*$", "", s)                  # trailing margin numbers / stray letters
    return re.sub(r"\s+", " ", s).strip(" :;.,-")


def _namey(s):
    letters = re.sub(r"[^A-Za-z]", "", s)
    return len(letters) >= 2 and sum(c.isupper() for c in letters) / len(letters) > 0.6


def _side(lines):
    """Join wrapped lines; numbered parties -> first + 'and Another/Others'."""
    parts = [tidy(p) for p in re.split(r"(?:^|\s)\d\s*[.)]\s*", " ".join(lines))]
    parts = [p for p in parts if len(p) > 3]
    if not parts:
        return None
    return parts[0] + ("" if len(parts) == 1 else " and Another" if len(parts) == 2 else " and Others")


def parse_parties(text):
    lines = text[:6000].split("\n")[:80]
    v = next((i for i, l in enumerate(lines) if VS.match(l) and not re.match(r"(?i)^\W*and\W*$", l)), None)
    if v is None:
        v = next((i for i, l in enumerate(lines) if re.match(r"(?i)^\W*and\W*$", l)), None)
    if v is None:
        return None
    a, b = [], []
    for l in reversed(lines[max(0, v - 7):v]):
        c = _clean(l)
        if not c:
            if a:
                break
            continue
        if STOP_A.search(l) or not _namey(c):
            break
        a.insert(0, c)
    for l in lines[v + 1:v + 10]:
        c = _clean(l)
        if STOP_B.search(l) or (c and not _namey(c)):
            break
        if c:
            b.append(c)
        if b and re.search(rf"(?i)\b{ROLE}\b", l):
            break
    A, B = (_side(a), _side(b)) if a and b else (None, None)
    return f"{A} v {B}" if A and B else None


def parse_head(text):
    head = text[:6000]
    num = re.search(r"(?i)((?:civil|misc\w*|miscellaneous|arbitration|company|labour|tax)[^\n]{0,40}?"
                    r"(?:suit|appeal|application|cause|petition|reference|revision)[^\n]{0,15}?no\.?\s*[\w/]+\s*of\s*\d{4})", head)
    a = re.search(rf"\n\s*{NAME}\s*[:.…_\-\s]*(?:\n\s*)?(?:\d+\w*\s*)?{ROLE_A}", head)
    b = re.search(rf"\n\s*{NAME}\s*[:.…_\-\s]*(?:\n\s*)?(?:\d+\w*\s*)?{ROLE_B}", head)
    judge = re.search(r"(?i)before\s*[:\-]?\s*((?:hon\.?|the hon\w*|his lordship|her lordship|lady justice|justice)[^\n]{3,80})", head)
    parties = parse_parties(text) or (f"{tidy(a.group(1))} v {tidy(b.group(1))}" if a and b else None)
    return (num.group(1).strip() if num else None, parties, judge.group(1).strip() if judge else None)


def main():
    ix = pd.concat([pd.read_csv(p) for p in JUDGMENTS.glob("*/_index.csv")])
    ix = ix[ix.status.eq("downloaded")].drop_duplicates("file")
    ix["case_id"] = ix.file.str.replace(".pdf", "", regex=False)
    cases = pd.read_parquet(INTERIM / "registers_cases.parquet")  # registers only: cases.parquet already holds extras
    second = set(case_aliases(cases))
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
