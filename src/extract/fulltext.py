"""Stage 3 (rule-based) — extract citations, statutes, money, interest, contract forms and clauses from full text.

No LLM: transparent regex rules only (ULII content signals: ai-input=no).
Outputs
  data/processed/fulltext_features.parquet   one row per judgment
  data/processed/citation_edges.parquet      citing case_id -> cited citation (neutral + law-report)
  data/processed/statute_mentions.parquet    case_id -> statute (+ sections)
"""
import re

import pandas as pd

from src.config import INTERIM, PROCESSED
from src.ingest.registers import case_id as cid

NEUTRAL = re.compile(r"\[((?:19|20)\d{2})\]\s+(UG[A-Za-z]+)\s+(\d{1,5})")
REPORT = re.compile(r"[\[(]?((?:19|20)\d{2})[\])]?\s*(?:(\d{1,2})\s+)?(HCB|KALR|ULR|ULSR|EA|E\.\s?A\.?|UGLR)\s+(\d{1,4})")
CASENO = re.compile(r"(?i)\b(S\.?C\.?C\.?A\.?|supreme court civil appeal|court of appeal civil appeal|civil appeal|civil application|constitutional (?:petition|appeal)|misc(?:ellaneous|\.)?\s*(?:civil\s+)?(?:application|cause)|civil suit|arbitration cause|company cause|h\.?c\.?c\.?s\.?)\s*(?:no\.?|number)?\s*(\d{1,5})\s*of\s*((?:19|20)\d{2})")
STATUTES = {
    "Arbitration and Conciliation Act": r"arbitration and conciliation act",
    "PPDA Act": r"public procurement and disposal of public assets act|\bPPDA act",
    "PPDA Regulations": r"public procurement and disposal of public assets \([^)]*\) regulations|PPDA regulations",
    "Contracts Act": r"\bcontracts act",
    "Civil Procedure Act": r"civil procedure act",
    "Civil Procedure Rules": r"civil procedure rules|\bO\.\s?\d+\s?r\.\s?\d+",
    "Judicature Act": r"judicature act",
    "Limitation Act": r"limitation act",
    "Government Proceedings Act": r"government proceedings act",
    "Evidence Act": r"evidence act",
    "Building Control Act": r"building control act",
    "Roads Act": r"\broads act",
    "UNRA Act": r"uganda national roads authority act",
    "Local Governments Act": r"local governments act",
    "Land Act": r"\bland act",
    "Registration of Titles Act": r"registration of titles act",
    "Engineers Registration Act": r"engineers registration act",
    "Architects Registration Act": r"architects registration act",
    "Surveyors Registration Act": r"surveyors registration act",
    "Public Private Partnerships Act": r"public private partnerships? act",
    "Companies Act": r"companies act",
    "Insolvency Act": r"insolvency act",
    "Sale of Goods Act": r"sale of goods (and supply of services )?act",
    "Constitution": r"constitution of (the republic of )?uganda",
    "National Environment Act": r"national environment act",
    "Judicature (Court Annexed Mediation) Rules": r"court annexed mediation\)? rules|mediation rules",
}
FORMS = {
    "FIDIC 2017": r"FIDIC[^\n]{0,60}2017|2017[^\n]{0,20}FIDIC",
    "FIDIC 1999": r"FIDIC[^\n]{0,60}1999|1999[^\n]{0,20}FIDIC",
    "FIDIC (unspecified)": r"\bFIDIC\b",
    "PPDA SBD": r"standard bidding document|\bSBD\b|standard bidding documents?",
    "JBC/PAM": r"\bJBC\b|joint building council|\bPAM\b",
    "World Bank/MDB": r"world bank|african development bank|\bMDB\b harmoni[sz]ed",
}
CUR = r"(?:UGX|Ug\.?\s?Shs?\.?|U\.?\s?Shs?\.?|Shs\.?|USh)"
MONEY = re.compile(rf"({CUR}|US\$|USD|\$|EUR|€)\s?([\d][\d,]{{3,}}(?:\.\d+)?)(?:\s?(?:/=|=/))?", re.I)
CLAIM_W = re.compile(r"(?i)claim|claimed|seeks?|sought|prays?|prayed|special damages|plaint|demand")
AWARD_W = re.compile(r"(?i)award(?:ed)?|grant(?:ed)?|judgment is entered|ordered to pay|shall pay|decree|entitled to")
INTEREST = re.compile(r"(?i)interest[^.\n]{0,120}?(\d{1,2}(?:\.\d+)?)\s?%\s*(?:p\.?a\.?|per annum|per year)?")
CLAUSE = re.compile(r"(?i)\b(?:sub-?)?clauses?\s+(\d{1,2}(?:\.\d{1,2}){0,3})")
SECTION = re.compile(r"(?i)\bs(?:ection|\.)\s?(\d{1,3}[A-Z]?)(?:\s?\(\d+\))?[^.\n]{0,40}?(arbitration and conciliation act|ACA\b)")


def money(text):
    """Return list of (currency, amount, role) with role from words within 120 chars before the figure."""
    out = []
    for m in MONEY.finditer(text):
        cur = m.group(1).upper()
        cur = "USD" if cur in ("US$", "USD", "$") else "EUR" if cur in ("EUR", "€") else "UGX"
        try:
            amt = float(m.group(2).replace(",", ""))
        except ValueError:
            continue
        if amt < 1000:
            continue
        ctx = text[max(0, m.start() - 120): m.start()]
        role = "awarded" if AWARD_W.search(ctx[-60:]) else "claimed" if CLAIM_W.search(ctx) else "other"
        out.append((cur, amt, role))
    return out


def caseno_key(kind, n, y):
    k = re.sub(r"[^a-z ]", "", kind.lower()).strip()
    k = "scca" if k.replace(" ", "") in ("scca",) or k.startswith("supreme court") else k
    k = "hccs" if k.replace(" ", "") == "hccs" else k
    k = re.sub(r"^misc(ellaneous)?\s*(civil\s+)?", "misc ", k)
    return f"{k} {int(n)}/{y}"


def features(case, text, holding, own_no=""):
    t = text or ""
    own = {caseno_key(*m) for m in CASENO.findall(own_no or "")}
    head = t[:1500]
    own |= {caseno_key(*m) for m in CASENO.findall(head)}  # numbers in the title block are this case's own
    casenos = sorted({caseno_key(*m) for m in CASENO.findall(t[1500:])} - own)
    neutral = [cid(f"[{y}] {c} {n}") for y, c, n in NEUTRAL.findall(t)]
    neutral = [c for c in neutral if c and c != case]
    reports = [f"[{y}] {v + ' ' if v else ''}{re.sub(r'[ .]', '', r)} {p}" for y, v, r, p in REPORT.findall(t)]
    stats = {k: len(re.findall(rx, t, re.I)) for k, rx in STATUTES.items()}
    forms = [k for k, rx in FORMS.items() if re.search(rx, t, re.I)]
    if any(f.startswith("FIDIC 1") or f.startswith("FIDIC 2") for f in forms):
        forms = [f for f in forms if f != "FIDIC (unspecified)"]
    mon = money(t)
    hol = money(holding or "")
    ugx = [a for c, a, r in mon if c == "UGX"]
    claimed = [a for c, a, r in mon if c == "UGX" and r == "claimed"]
    awarded = [a for c, a, r in hol if c == "UGX" and r == "awarded"]
    rates = [float(r) for r in INTEREST.findall(holding or t[-int(len(t) * .3):]) if 0 < float(r) <= 40]
    return {
        "case_id": case,
        "n_neutral_cited": len(set(neutral)), "n_report_cited": len(set(reports)), "n_caseno_cited": len(casenos),
        "cited_casenos": casenos,
        "cited_neutral": sorted(set(neutral)), "cited_reports": sorted(set(reports)),
        "statutes": sorted(k for k, v in stats.items() if v), **{f"stat_n:{k}": v for k, v in stats.items()},
        "aca_sections": sorted({s for s, _ in SECTION.findall(t)}),
        "contract_forms": forms, "clauses_cited": sorted(set(CLAUSE.findall(t)))[:40],
        "ugx_max_mentioned": max(ugx) if ugx else None,
        "ugx_claimed_max": max(claimed) if claimed else None,
        "ugx_awarded_in_holding": sum(sorted(set(awarded))) if awarded else None,
        "usd_mentioned": any(c == "USD" for c, _, _ in mon),
        "interest_rates_pct": sorted(set(rates)),
    }


def main():
    texts = pd.read_parquet(INTERIM / "texts.parquet")
    seg = pd.read_parquet(INTERIM / "segments.parquet").set_index("case_id")["holding"]
    cases = pd.read_parquet(PROCESSED / "cases.parquet").set_index("case_id")["case_number"]
    rows = [features(r.case_id, r.text, seg.get(r.case_id, ""), str(cases.get(r.case_id, "") or "")) for r in texts.itertuples()]
    f = pd.DataFrame(rows)
    f.to_parquet(PROCESSED / "fulltext_features.parquet", index=False)

    edges = [(r["case_id"], c, "neutral") for r in rows for c in r["cited_neutral"]] + \
            [(r["case_id"], c, "report") for r in rows for c in r["cited_reports"]] + \
            [(r["case_id"], c, "case_no") for r in rows for c in r["cited_casenos"]]
    e = pd.DataFrame(edges, columns=["citing", "cited", "kind"])
    e.to_parquet(PROCESSED / "citation_edges.parquet", index=False)
    st = f[["case_id", "statutes"]].explode("statutes").dropna()
    st.to_parquet(PROCESSED / "statute_mentions.parquet", index=False)

    corpus = set(texts.case_id)
    print(f"judgments: {len(f)} | citation edges: {len(e)} (neutral {int((e.kind == 'neutral').sum())}, "
          f"to cases inside corpus {int(e.cited.isin(corpus).sum())})")
    print("edges by kind:", e.kind.value_counts().to_dict(), "| judgments citing anything:", e.citing.nunique())
    for k in ["neutral", "report", "case_no"]:
        print(f"most cited ({k}):", e[e.kind == k].cited.value_counts().head(6).to_dict())
    print("statutes (share of judgments):", (f[[c for c in f if c.startswith('stat_n:')]] > 0).mean()
          .sort_values(ascending=False).head(10).round(2).rename(lambda c: c[7:]).to_dict())
    print("contract forms:", f.contract_forms.explode().value_counts().to_dict())
    print("with UGX claimed:", f.ugx_claimed_max.notna().sum(), "| UGX awarded in holding:", f.ugx_awarded_in_holding.notna().sum(),
          "| interest rate found:", (f.interest_rates_pct.str.len() > 0).sum())


if __name__ == "__main__":
    main()
