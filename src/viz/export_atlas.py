"""Export the Uganda Construction & Arbitration Law Atlas data for GitHub Pages (docs/atlas/data/).

Public by design, so it publishes only:
  - case metadata from the judgments (citation, court, date, judge, parties) and the ULII link;
  - the author's coded fields (stream, proceeding, issues, outcome, arbitration type and grounds, state party),
    flagged as machine-coded (v1); statutes and register enactments cited; citation links within the corpus.
It does NOT publish ULII summaries or key issues (ULII's own work), per-case amounts claimed/awarded or geoparsed
districts (unverified), or any judgment text: full judgments are read on ULII.
Usage: python -m src.viz.export_atlas
"""
import json
import re
from datetime import date

import pandas as pd

from src.analysis.common import CAUSE_LABEL, CAUSES
from src.config import INTERIM, OUT, PROCESSED
from src.extract.fulltext import CASENO, caseno_key
from src.ingest.registers import case_aliases

DEST = OUT.parent / "docs" / "atlas" / "data"
GROUNDS = {"g_public_policy": "Public policy", "g_jurisdiction": "Excess of jurisdiction", "g_natural_justice": "Natural justice",
           "g_out_of_time": "Out of time", "g_arbitrability": "Arbitrability", "g_invalid_agreement": "Invalid agreement"}


def court_group(c):
    c = str(c)
    for k, rx in [("Supreme Court", "Supreme"), ("Court of Appeal", r"Court of Appeal|Constitutional"),
                  ("High Court (Commercial)", "Commercial"), ("High Court (other divisions)", "High Court"),
                  ("PPDA Appeals Tribunal", "PPDA"), ("CADER", "CADER")]:
        if re.search(rx, c):
            return k
    return "Other tribunals"


def nn(x):
    return None if pd.isna(x) else x


def citation_links(a):
    alias = case_aliases(a)
    own = {}
    for cid, num in zip(a.case_id, a.case_number.fillna("")):
        for m in CASENO.findall(num):
            own.setdefault(caseno_key(*m), set()).add(cid)
    own = {k: next(iter(v)) for k, v in own.items() if len(v) == 1}
    e = pd.read_parquet(PROCESSED / "citation_edges.parquet")
    e["citing"] = e.citing.map(alias)
    e["cited"] = e.cited.map(lambda c: alias.get(c, own.get(c, c)))
    e = e.dropna(subset=["citing"])
    e = e[e.cited.isin(set(a.case_id)) & (e.citing != e.cited)].drop_duplicates(["citing", "cited"])
    return e.groupby("citing").cited.apply(sorted).to_dict(), e.groupby("cited").citing.apply(sorted).to_dict()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    a = pd.read_parquet(PROCESSED / "analysis_cases.parquet")
    c = pd.read_parquet(PROCESSED / "cases.parquet").set_index("case_id")
    cites, cited_by = citation_links(a)
    leg = pd.read_parquet(INTERIM / "legislation.parquet")
    rows = []
    for r in a.sort_values("date_decided", ascending=False).itertuples():
        rows.append({
            "id": r.case_id, "cit": r.citation, "court": r.court, "cg": court_group(r.court),
            "date": str(r.date_decided)[:10] if pd.notna(r.date_decided) else None, "year": int(r.year),
            "judge": nn(r.judge), "parties": nn(r.parties), "stream": r.stream, "proc": nn(r.proceeding),
            "issues": [CAUSE_LABEL[k] for k in CAUSES if getattr(r, k) == "Y"],
            "outcome": nn(r.outcome), "arb": None if r.arb_application_type in (None, "NA") or pd.isna(r.arb_application_type) else r.arb_application_type,
            "grounds": [v for k, v in GROUNDS.items() if getattr(r, k) == "Y"],
            "state": r.state_party == "Y",
            "statutes": list(r.statutes_cited) if r.statutes_cited is not None else [],
            "leg": list(r.leg_ids_cited) if r.leg_ids_cited is not None else [],
            "cites": cites.get(r.case_id, []), "citedBy": cited_by.get(r.case_id, []),
            "url": nn(c.ulii_link.get(r.case_id)), "src": r.source,
        })
    legs = []
    n_cite = pd.Series([l for r in rows for l in r["leg"]]).value_counts()
    for r in leg.sort_values("date").itertuples():
        eff = r.commencement_date if pd.notna(r.commencement_date) else r.date
        legs.append({"id": r.leg_id, "title": re.sub(r"\s*\((?:per|package)[^)]*\)", "", r.title), "citation": nn(r.citation),
                     "type": nn(r.type), "area": nn(r.area), "effective": str(eff.date()) if pd.notna(eff) else None,
                     "status": nn(r.commencement_status), "provisions": nn(r.key_provisions),
                     "url": r.source if str(r.source or "").startswith("http") else None,
                     "onUlii": bool(r.on_ulii), "verified": bool(r.verified), "citedIn": int(n_cite.get(r.leg_id, 0))})
    meta = {"name": "Uganda Construction & Arbitration Law Atlas", "generated": str(date.today()),
            "coverage": f"{min(r['year'] for r in rows)}–{max(r['year'] for r in rows)}",
            "lastDecision": max(r["date"] for r in rows if r["date"]), "cases": len(rows),
            "coding": "v1 (rule-based, validated against a provisional AI-coded reference set)",
            "streams": pd.Series([r["stream"] for r in rows]).value_counts().to_dict()}
    (DEST / "cases.json").write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
    (DEST / "legislation.json").write_text(json.dumps(legs, ensure_ascii=False, separators=(",", ":")))
    (DEST / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    kb = sum(p.stat().st_size for p in DEST.glob("*.json")) / 1024
    print(f"atlas: {len(rows)} cases, {len(legs)} enactments, {sum(len(r['cites']) for r in rows)} in-corpus citations; {kb:.0f} KB -> {DEST}")


if __name__ == "__main__":
    main()
