"""Read the researcher's decisions from the filled review workbook and store them as versioned CSVs.

Input:  outputs/review/review_queue-filled.xlsx (falls back to review_queue.xlsx)
Output: manual/review/{to_confirm,duplicates,ocr_low_conf,legislation}.csv  (committed: the only
        non-reproducible inputs; corrections are made in the CSVs). merge.py and registers.py apply them; this script also patches the
        current interim/processed files so a full rebuild is not needed.
"""
import pandas as pd

from src.config import INTERIM, OUT, ROOT

MANUAL = ROOT / "manual" / "review"
SHEETS = {"to_confirm": ["case_id", "citation", "decision", "note"],
          "duplicates": ["added_case", "register_case", "decision", "note"],
          "ocr_low_conf": ["file", "page", "decision", "note"],
          "legislation": ["leg_id", "title", "decision", "note"]}


def workbook():
    d = OUT / "review"
    return next(p for p in [d / "review_queue-filled.xlsx", d / "review_queue.xlsx"] if p.exists())


def load(name):
    p = MANUAL / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=SHEETS[name])


def excluded_cases():
    """case_ids dropped on review: 'Exclude' in to_confirm or 'Duplicate' in duplicates (added side)."""
    tc, dup = load("to_confirm"), load("duplicates")
    dup["case_id"] = dup.added_case.str.replace(r"\[(\d{4})\]\s*(\w+)\s*(\d+)", r"\1_\2_\3", regex=True)
    out = {"excluded on review: not construction/arbitration": set(tc.loc[tc.decision.eq("Exclude"), "case_id"]),
           "excluded on review: duplicate decision": set(dup.loc[dup.decision.eq("Duplicate"), "case_id"])}
    for _, r in dup[dup.decision.eq("Exclude")].iterrows():  # excluded for another reason: text before ' (' / ';' in note
        out.setdefault("excluded on review: " + str(r.note).split(";")[0].split(" (")[0], set()).add(r.case_id)
    return out


def confirmed_cases():
    tc = load("to_confirm")
    return set(tc.loc[tc.decision.eq("Include"), "case_id"])


DATE = r"(\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4})"


def commencement(note):
    """Commencement date from the researcher's gazette note ('Commence(d|ment) ... 1 July 2021')."""
    import re
    m = re.search(r"Commence\w*[^\n]*?" + DATE, str(note), flags=re.I)
    return m.group(1) if m else None


def apply_legislation(leg):
    rv = load("legislation").set_index("leg_id")
    if rv.empty:
        return leg
    leg = leg.copy()
    leg["verification_note"] = leg.leg_id.map(rv.note)
    leg["commencement_date"] = pd.to_datetime(leg.verification_note.map(commencement), errors="coerce")
    leg.loc[leg.leg_id.isin(rv.index[rv.decision.eq("Verified")]), "verified"] = True
    nf = leg.leg_id.isin(rv.index[rv.decision.eq("Not found")])
    leg.loc[nf, "verified"] = False
    leg.loc[nf, "commencement_status"] = "Not commenced (researcher check: " + leg.loc[nf, "verification_note"].fillna("not found") + ")"
    return leg


def main(reimport=False):
    """CSVs are the record once written (later corrections are made there); --reimport overwrites them."""
    MANUAL.mkdir(parents=True, exist_ok=True)
    wb = workbook()
    for name, cols in SHEETS.items():
        if (MANUAL / f"{name}.csv").exists() and not reimport:
            print(f"{name}: kept existing {name}.csv (use --reimport to overwrite from {wb.name})")
            continue
        d = pd.read_excel(wb, sheet_name=name).rename(columns={" ": "leg_id"})
        missing = d.decision.isna().sum()
        d[cols].to_csv(MANUAL / f"{name}.csv", index=False)
        print(f"{name}: {len(d)} rows, {d.decision.value_counts().to_dict()}" + (f", {missing} blank" if missing else ""))

    leg = apply_legislation(pd.read_parquet(INTERIM / "legislation.parquet"))
    leg.to_parquet(INTERIM / "legislation.parquet", index=False)
    print(f"legislation: {int(leg.verified.sum())}/{len(leg)} verified")
    print(f"decisions saved to {MANUAL} (read from {wb.name}); re-run src.ingest.merge to apply case decisions")


if __name__ == "__main__":
    import sys
    main(reimport="--reimport" in sys.argv)
