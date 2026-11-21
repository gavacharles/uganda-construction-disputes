"""Stage 1.1 — read the six ULII registers and harmonise them.

Outputs
  data/interim/registers_cases.parquet   one row per unique case (citation)
  data/interim/legislation.parquet       one row per enactment
  data/interim/scanned_candidates.parquet
  outputs/logs/registers_check.csv       register counts vs 'Summary by year'
"""
import re

import pandas as pd

from src.config import HEADER_ROW, INTERIM, LOGS, PERIODS, RAW
from src.review import apply_review as review

CASE_COLS = {
    "Year": "year",
    "Date judgment delivered": "date_decided",
    "Case (parties)": "parties",
    "Citation": "citation",
    "Court": "court",
    "Judge(s)": "judge",
    "Case number": "case_number",
    "What the court held": "holding_summary",
    "Outcome": "outcome_register",
    "Key issues": "key_issues",
    "Contract / project type": "project_type_register",
    "Statutes / rules": "statutes_register",
    "Also in arbitration register": "also_flag",
    "Also in construction register": "also_flag",
    "ULII link": "ulii_link",
}
LEG_COLS = {
    "Date": "date", "Title": "title", "Citation": "citation", "Type": "type",
    "Area": "area", "Commencement / status": "commencement_status",
    "Key provisions": "key_provisions", "Relevance": "relevance", "Source": "source",
}


def _read(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet, header=HEADER_ROW, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(how="all")


def norm_citation(c):
    return re.sub(r"\s+", " ", str(c)).strip() if pd.notna(c) else None


def case_id(cit):
    m = re.match(r"\[(\d{4})\]\s*([A-Za-z]+)\s*(\d+)", cit or "")
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}" if m else None


def load_cases():
    frames = []
    for period in PERIODS:
        for register in ("construction", "arbitration"):
            path = RAW / period / f"{register.title()}_Cases_Register_ULII_{period}.xlsx"
            df = _read(path, f"{register.title()} cases").rename(columns=CASE_COLS)
            df = df[df["citation"].notna()].drop(columns=["No."], errors="ignore")
            df["register"], df["period"] = register, period
            frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["citation"] = raw["citation"].map(norm_citation)
    raw["date_decided"] = pd.to_datetime(raw["date_decided"], errors="coerce")
    raw["year"] = raw["date_decided"].dt.year.astype("Int64")
    raw["also_flag"] = raw["also_flag"].str.strip().str.lower().eq("yes")
    raw_n = raw.groupby(["period", "register"]).size().rename("register_rows")

    # one row per citation; flags from both registers
    flags = raw.groupby("citation").agg(
        in_construction=("register", lambda s: "construction" in set(s)),
        in_arbitration=("register", lambda s: "arbitration" in set(s)),
        also_flag=("also_flag", "any"),
    )
    # prefer construction row for shared fields, fill gaps from arbitration row
    raw = raw.sort_values("register", key=lambda s: s.ne("construction"))
    cases = raw.drop(columns=["also_flag", "register"]).groupby("citation", sort=False).first()
    cases = cases.join(flags).reset_index()
    both = (cases.in_construction & cases.in_arbitration) | cases.also_flag
    cases["stream"] = "arbitration"
    cases.loc[cases.in_construction, "stream"] = "construction"
    cases.loc[both, "stream"] = "both"
    cases["case_id"] = cases["citation"].map(case_id)
    assert cases["case_id"].notna().all(), cases.loc[cases.case_id.isna(), "citation"]
    assert cases["case_id"].is_unique
    cases["source"] = "register"
    lead = ["case_id", "citation", "period", "year", "stream"]
    return cases[lead + [c for c in cases.columns if c not in lead]], raw_n


def load_legislation():
    frames = []
    for period in PERIODS:
        path = RAW / period / f"Construction_Arbitration_Legislation_Uganda_{period}.xlsx"
        df = _read(path, f"Legislation {period}").rename(columns=LEG_COLS)
        df = df[df["title"].notna()].drop(columns=["No."], errors="ignore")
        df["period"] = period
        frames.append(df)
    leg = pd.concat(frames, ignore_index=True)
    leg["date"] = pd.to_datetime(leg["date"], errors="coerce")
    leg = leg.sort_values("date", kind="stable")
    # same enactment listed in both period registers (e.g. Arbitration Amendment Act 2024)
    key = leg["source"].fillna("").str.replace(r"/eng@.*$", "", regex=True)
    leg = leg[~key.duplicated(keep="first")].reset_index(drop=True)
    leg["leg_id"] = [f"LEG{i + 1:03d}" for i in range(len(leg))]
    leg["on_ulii"] = leg["source"].fillna("").str.contains(r"ulii\.org/en/akn/ug/act/")
    unsure = leg["commencement_status"].fillna("").str.contains(r"verify|per .*alert", case=False)
    leg["verified"] = leg["on_ulii"] & ~unsure  # set True manually once checked vs gazette
    leg["amends_repeals"] = None      # filled in stage 13
    return leg[["leg_id"] + [c for c in leg.columns if c != "leg_id"]]


def load_scanned():
    path = RAW / "2015-2025" / "Construction_Cases_Register_ULII_2015-2025.xlsx"
    df = _read(path, "Scanned - not read").rename(columns=CASE_COLS)
    df = df[df["citation"].notna()].drop(columns=["No."], errors="ignore")
    df["citation"] = df["citation"].map(norm_citation)
    df["case_id"] = df["citation"].map(case_id)
    return df


def check_against_summary(cases):
    """Compare per-year counts with the registers' own 'Summary by year' sheet."""
    path = RAW / "2015-2025" / "Construction_Cases_Register_ULII_2015-2025.xlsx"
    s = _read(path, "Summary by year")
    s = s[pd.to_numeric(s["Year"], errors="coerce").notna()].copy()
    s["year"] = s["Year"].astype(int)
    s = s.set_index("year")[["Construction cases", "Arbitration cases", "In both registers"]].astype(int)
    c = cases[cases.period == "2015-2025"].groupby("year").agg(
        construction=("in_construction", "sum"),
        arbitration=("in_arbitration", "sum"),
        both=("stream", lambda x: (x == "both").sum()),
    )
    return s.join(c, how="outer")


def main():
    cases, raw_n = load_cases()
    leg = review.apply_legislation(load_legislation())  # researcher's gazette checks
    scanned = load_scanned()
    cases.to_parquet(INTERIM / "registers_cases.parquet", index=False)
    leg.to_parquet(INTERIM / "legislation.parquet", index=False)
    scanned.to_parquet(INTERIM / "scanned_candidates.parquet", index=False)
    chk = check_against_summary(cases)
    chk.to_csv(LOGS / "registers_check.csv")

    print("Register rows:\n", raw_n.to_string())
    print(f"\nUnique cases: {len(cases)}")
    print(cases.groupby(["period", "stream"]).size().unstack(fill_value=0).to_string())
    print(f"\nLegislation: {len(leg)} ({(~leg.on_ulii).sum()} not on ULII, "
          f"{(~leg.verified).sum()} to verify)")
    print(leg.loc[~leg.verified, ["leg_id", "title"]].to_string(index=False))
    print(f"Scanned candidates: {len(scanned)}")
    print("\nCheck vs 'Summary by year' (2015-2025):\n", chk.to_string())


if __name__ == "__main__":
    main()
