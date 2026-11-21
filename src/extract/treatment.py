"""Subsequent treatment of decisions within the corpus (rule-based citator; no LLM).

1. Matters and appeals: decisions with the same parties on both sides (either order; token-set ratio >= 90) form one
   matter (union-find). Within a matter, an appeal is linked to the most recent earlier decision of a lower court, and
   labelled by the appeal's outcome ("Later appeal allowed / dismissed / ..."). Matters are stored for timelines.
2. Citation treatment: for each in-corpus citation (citing -> cited), every mention of the cited decision in the citing
   text (neutral citation, case number or lead party name) is located and the surrounding words classified:
   doubted / not followed > distinguished > followed / applied > cited.
Only labels are stored (no ULII text). Outputs: data/processed/treatment.parquet (later, earlier, kind, label),
         data/processed/matters.parquet (case_id, matter_id, matter_size)
"""
import re

import pandas as pd

from src.code.rule_coder import load_texts
from src.config import PROCESSED
from src.extract.fulltext import CASENO, caseno_key
from src.ingest.registers import case_aliases

NEG = re.compile(r"overrul|wrongly decided|per incuriam|declin\w* to follow|not (?:be )?followed|disagree|doubt(?:ed|ful)|depart(?:ed)? from", re.I)
DIST = re.compile(r"distinguish", re.I)
POS = re.compile(r"\bfollowed\b|\bfollowing (?:the|that|this) (?:decision|reasoning|holding|authority|case)|\bapplied\b(?! for| to)|"
                 r"\badopt(?:ed|s)?\b|\brel(?:ied|y|ying) (?:up)?on\b|\bpersuaded by\b|\bguided by\b|\bbound by\b|\bendorsed?\b|"
                 r"\bwith approval\b|\breaffirmed\b|\bconsistent with\b", re.I)
STOP = {"uganda", "limited", "ltd", "the", "attorney", "general", "company", "and", "another", "others", "national", "of", "co",
        "construction", "engineering", "services", "international", "authority", "bank", "investments", "enterprises", "group"}
APPEAL_OUT = {"initiator succeeded": "Later appeal allowed", "partly succeeded": "Later appeal partly allowed",
              "initiator failed": "Later appeal dismissed", "remitted or retrial": "Remitted on later appeal",
              "struck out (procedural)": "Later appeal struck out"}


def rank(court):
    c = str(court)
    return 3 if "Supreme" in c else 2 if "Court of Appeal" in c or "Constitutional" in c else \
        0 if any(k in c for k in ["Tribunal", "CADER", "Registrar", "Bureau"]) else 1


def norm_sides(p):
    sides = re.split(r"\s+v\.?s?\.?\s+", str(p or "").lower(), maxsplit=1) + [""]
    f = lambda x: re.sub(r"\s+", " ", re.sub(r"\b(ltd|limited|company|co|the|u|uganda|and|another|others|anor|ors|m/s)\b|[^a-z ]", " ", x)).strip()
    return f(sides[0]), f(sides[1])


def matters(a):
    """Union-find over decisions whose two party sides match (either order)."""
    from rapidfuzz import fuzz
    ids, sides = list(a.case_id), [norm_sides(p) for p in a.parties]
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(ids)):
        a1, b1 = sides[i]
        if len(a1) < 4 or len(b1) < 4:
            continue
        for j in range(i + 1, len(ids)):
            a2, b2 = sides[j]
            if len(a2) < 4 or len(b2) < 4:
                continue
            s = max(min(fuzz.token_set_ratio(a1, a2), fuzz.token_set_ratio(b1, b2)),
                    min(fuzz.token_set_ratio(a1, b2), fuzz.token_set_ratio(b1, a2)))
            if s >= 90:
                parent[find(ids[i])] = find(ids[j])
    m = pd.DataFrame({"case_id": ids, "matter_id": [find(i) for i in ids]})
    m["matter_size"] = m.groupby("matter_id").case_id.transform("size")
    return m


def own_keys(a):
    own = {}
    for cid, num in zip(a.case_id, a.case_number.fillna("")):
        for m in CASENO.findall(num):
            own.setdefault(caseno_key(*m), set()).add(cid)
    return {k: next(iter(v)) for k, v in own.items() if len(v) == 1}


def mention_patterns(r):
    pats = []
    m = re.match(r"\[(\d{4})\]\s*(\w+)\s*(\d+)", str(r.citation))
    if m:
        pats.append(rf"\[\s*{m.group(1)}\s*\]\s*{m.group(2)}\s*{m.group(3)}\b")
    for kind, n, y in CASENO.findall(str(r.case_number or "")):
        pats.append(rf"No\.?\s*0*{int(n)}\s*of\s*{y}")
    first = re.split(r"\s+v\.?s?\.?\s+", str(r.parties or ""), maxsplit=1, flags=re.I)[0]
    words = [w for w in re.findall(r"[A-Za-z]{4,}", first) if w.lower() not in STOP]
    if words:
        pats.append(rf"\b{re.escape(words[0])}\b")
    return re.compile("|".join(pats), re.I) if pats else None


def classify(text, rx):
    labels = set()
    for m in rx.finditer(text):
        w = text[max(0, m.start() - 260): m.end() + 260]
        labels.add("Doubted / not followed" if NEG.search(w) else "Distinguished" if DIST.search(w)
                   else "Followed / applied" if POS.search(w) else "Cited")
    for lab in ["Doubted / not followed", "Distinguished", "Followed / applied", "Cited"]:
        if lab in labels:
            return lab
    return "Cited"


def main():
    a = pd.read_parquet(PROCESSED / "analysis_cases.parquet")
    a["date_decided"] = pd.to_datetime(a.date_decided, errors="coerce")
    texts = load_texts(a)
    own = own_keys(a)
    alias = case_aliases(a)
    by = a.set_index("case_id")
    rows = []

    # 1. matters and appeals
    m = matters(a)
    m.to_parquet(PROCESSED / "matters.parquet", index=False)
    a = a.merge(m, on="case_id")
    a["rank"] = a.court.map(rank)
    for r in a[a.proceeding.eq("appeal") & a.matter_size.gt(1)].itertuples():
        low = a[(a.matter_id == r.matter_id) & (a["rank"] < r.rank) & (a.date_decided < r.date_decided)]
        if len(low):
            rows.append({"later": r.case_id, "earlier": low.sort_values("date_decided").case_id.iloc[-1], "kind": "appeal",
                         "label": APPEAL_OUT.get(r.outcome, "Later appeal")})

    # 2. citation treatment
    e = pd.read_parquet(PROCESSED / "citation_edges.parquet")
    e["citing"] = e.citing.map(alias)
    e["cited"] = e.cited.map(lambda c: alias.get(c, own.get(c, c)))
    e = e.dropna(subset=["citing"])
    e = e[e.cited.isin(by.index) & (e.citing != e.cited)].drop_duplicates(["citing", "cited"])
    appeal_pairs = {(x["later"], x["earlier"]) for x in rows}
    for r in e.itertuples():
        if (r.citing, r.cited) in appeal_pairs or r.citing not in texts.index:
            continue
        rx = mention_patterns(by.loc[r.cited].rename(None).to_frame().T.itertuples().__next__())
        rows.append({"later": r.citing, "earlier": r.cited, "kind": "citation",
                     "label": classify(re.sub(r"\s+", " ", texts[r.citing]), rx) if rx else "Cited"})
    t = pd.DataFrame(rows).drop_duplicates(["later", "earlier", "kind"])
    t.to_parquet(PROCESSED / "treatment.parquet", index=False)
    print(f"matters with 2+ decisions: {(m.groupby('matter_id').size() > 1).sum()} covering {(m.matter_size > 1).sum()} decisions")
    print(f"treatment links: {len(t)}")
    print(t.groupby(["kind", "label"]).size().to_string())


if __name__ == "__main__":
    main()
