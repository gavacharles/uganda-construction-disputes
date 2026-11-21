"""Export the Uganda Construction & Arbitration Law Atlas data for GitHub Pages (docs/atlas/data/).

Publishes: case metadata from the judgments, the ULII link, the author's case-register summary and key issues
(paraphrased; credited to ULII where derived), the author's coded fields (machine-coded, v1), legislation cited,
subsequent treatment and matter histories (src.extract.treatment), and aggregated analysis (insights.json) drawn from
the analysis stages. Does not publish judgment text, per-case amounts or geoparsed districts.
Licence for the author's data and analysis: CC BY 4.0 (docs/atlas/LICENSE.md).
Run after: src.code.freeze, src.extract.treatment, and the analysis stages (outputs/analysis/*).
Usage: python -m src.viz.export_atlas
"""
import json
import re
from datetime import date

import numpy as np
import pandas as pd

from src.analysis.common import CAUSE_LABEL, CAUSES
from src.analysis.s9_reform import DISPUTE_TIER, LEVER
from src.config import INTERIM, OUT, PROCESSED, ROOT

DEST = OUT.parent / "docs" / "atlas" / "data"
AN = OUT / "analysis"
GROUNDS = {"g_public_policy": "Public policy", "g_jurisdiction": "Excess of jurisdiction", "g_natural_justice": "Natural justice",
           "g_out_of_time": "Out of time", "g_arbitrability": "Arbitrability", "g_invalid_agreement": "Invalid agreement"}
MERITS = ["initiator succeeded", "partly succeeded", "initiator failed", "struck out (procedural)"]


def court_group(c):
    c = str(c)
    for k, rx in [("Supreme Court", "Supreme"), ("PPDA Appeals Tribunal", "PPDA"), ("Court of Appeal", r"Court of Appeal|Constitutional"),
                  ("High Court (Commercial)", "Commercial"), ("High Court (other divisions)", "High Court"), ("CADER", "CADER")]:
        if re.search(rx, c):
            return k
    return "Other tribunals"


def nn(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return x.item() if hasattr(x, "item") else x


def rec(df):
    return json.loads(df.to_json(orient="records", force_ascii=False))


def csv(stage, name, **kw):
    p = AN / stage / f"{name}.csv"
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


def insights(a, cases):
    con = a[a.stream.isin(["construction", "both"])]
    arb = a[a.stream.isin(["arbitration", "both"])]
    from src.temporal.register_temporal import ulii_totals
    tot = ulii_totals()
    yr = pd.crosstab(a.year, a.stream).reindex(columns=["construction", "both", "arbitration"], fill_value=0)
    years = [{"year": int(y), **{k: int(v) for k, v in r.items()}, "ulii": int(tot.get(y, 0))} for y, r in yr.iterrows()]
    now, before = con[con.year == 2026], con[con.year < 2026]
    causes = [{"cause": CAUSE_LABEL[c], "decade": round(100 * (con[c] == "Y").mean(), 1),
               "before": round(100 * (before[c] == "Y").mean(), 1), "y2026": round(100 * (now[c] == "Y").mean(), 1),
               "n": int((con[c] == "Y").sum())} for c in CAUSES if (con[c] == "Y").sum() >= 10]
    oc = []
    for st in ["construction", "both", "arbitration"]:
        d = a[a.stream == st]
        oc.append({"stream": st, "n": int(d.outcome.notna().sum()),
                   **{o: round(100 * (d.outcome == o).sum() / max(1, d.outcome.notna().sum()), 1)
                      for o in MERITS + ["referred to arbitration", "remitted or retrial", "consent"]}})
    m = con[con.outcome.isin(MERITS)]
    bench_issue = {CAUSE_LABEL[c]: {"n": int((m[c] == "Y").sum()),
                                    "success": round(100 * m[m[c] == "Y"].outcome.isin(MERITS[:2]).mean(), 1)}
                   for c in CAUSES if (m[c] == "Y").sum() >= 8}
    dur = csv("04_outcomes_econ", "t1_duration_by_proceeding")
    arbt = csv("05_arbitration", "t1_application_types")
    dfr = arb[arb.deference.isin(["Y", "N"])].groupby("year").deference.agg(lambda x: (x == "Y").mean())
    dfn = arb[arb.deference.isin(["Y", "N"])].groupby("year").size()
    dist = csv("10_spatial", "district_table")
    rep = csv("06_networks", "t4_repeat_players")
    cited = csv("06_networks", "t1_most_cited")
    central = csv("06_networks", "t2_central_decisions")
    cit2id = dict(zip(a.citation, a.case_id))
    reform = csv("09_reform", "t1_cause_clause_cost")
    lever_key = {CAUSE_LABEL[k]: k for k in CAUSES}
    principal = csv("03_temporal_legislation", "t8_principal_act_share")
    state = csv("02_causes_themes", "t1_state_vs_private")
    filt = csv("11_litigation_filter", "t1_family_shares")
    grounds = csv("05_arbitration", "t2_setaside_grounds")
    rec_ = csv("04_outcomes_econ", "t3_recovery_by_stream")
    interest = csv("04_outcomes_econ", "t6_interest")
    succ_all = m.outcome.isin(MERITS[:2]).mean()
    d_suit = dur.set_index("proceeding").loc["suit", "median"] if len(dur) else None
    rr = rec_.set_index("stream")["mean"].get("construction") if len(rec_) else None
    return {
        "kpi": {"decisions": len(a), "construction": int(a.stream.eq("construction").sum()), "both": int(a.stream.eq("both").sum()),
                "arbitration": int(a.stream.eq("arbitration").sum()), "successConstruction": round(100 * succ_all, 0),
                "medianSuitYears": nn(d_suit), "recoveryConstruction": None if rr is None else round(100 * rr, 0),
                "deference": round(100 * arb.deference.eq("Y").sum() / max(1, arb.deference.isin(["Y", "N"]).sum()), 0),
                "enactments": len(json.load(open(DEST / "legislation.json"))),
                "enactmentsCited": int(sum(1 for l in json.load(open(DEST / "legislation.json")) if l["citedIn"] > 0)),
                "matters": int(cases["matterSize"].gt(1).sum()) if "matterSize" in cases else None},
        "years": years, "causes": causes, "outcomes": oc,
        "duration": rec(dur[dur.n >= 10]) if len(dur) else [],
        "recovery": rec(rec_) if len(rec_) else [], "interest": rec(interest) if len(interest) else [],
        "arbTypes": rec(arbt[arbt.n >= 5]) if len(arbt) else [],
        "grounds": rec(grounds.dropna(subset=["raised %"])) if len(grounds) else [],
        "deference": [{"year": int(y), "pct": round(100 * v, 1), "n": int(dfn[y])} for y, v in dfr.items()],
        "filter": rec(filt.rename(columns={"filter ratio vs OAG": "rOAG", "filter ratio vs Observer": "rObs"})) if len(filt) else [],
        "districts": rec(dist[["district", "region", "population", "cases", "road_cases", "rate_100k", "hotspot"]]) if len(dist) else [],
        "repeat": rec(rep.head(20)) if len(rep) else [],
        "mostCited": rec(cited.head(15)) if len(cited) else [],
        "central": [{"id": cit2id.get(r.citation), "citation": r.citation, "citedBy": int(r._3)}
                    for r in central.head(10).itertuples()] if len(central) else [],
        "principal": rec(principal) if len(principal) else [],
        "state": rec(state) if len(state) else [],
        "reform": [{"rank": int(r["rank"]), "cause": r.cause, "decisions": int(r.decisions), "prevalence": r["prevalence %"],
                    "unrecovered": nn(r["unrecovered share"]), "failure": r["initiator failure %"], "score": r["score (equal)"],
                    "lever": LEVER.get(lever_key.get(r.cause), ("", ""))[0], "reform": LEVER.get(lever_key.get(r.cause), ("", ""))[1]}
                   for _, r in reform.iterrows()] if len(reform) else [],
        "tier": {"lever": DISPUTE_TIER[0], "reform": DISPUTE_TIER[1]},
        "bench": {"issue": bench_issue, "issueBase": round(100 * succ_all, 1),
                  "duration": {r.proceeding: r.median for r in dur.itertuples()} if len(dur) else {},
                  "arb": {r.arb_application_type: {"n": int(r.n), "success": r.initiator_success} for r in arbt.itertuples()} if len(arbt) else {}},
    }


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    a = pd.read_parquet(PROCESSED / "analysis_cases.parquet")
    a["date_decided"] = pd.to_datetime(a.date_decided, errors="coerce")
    c = pd.read_parquet(PROCESSED / "cases.parquet").set_index("case_id")
    t = pd.read_parquet(PROCESSED / "treatment.parquet")
    mt = pd.read_parquet(PROCESSED / "matters.parquet").set_index("case_id")
    later = t.groupby("earlier").apply(lambda g: [{"id": r.later, "label": r.label, "kind": r.kind} for r in g.itertuples()]).to_dict()
    earlier = t.groupby("later").apply(lambda g: [{"id": r.earlier, "label": r.label, "kind": r.kind} for r in g.itertuples()]).to_dict()
    leg = pd.read_parquet(INTERIM / "legislation.parquet")

    rows = []
    for r in a.sort_values("date_decided", ascending=False).itertuples():
        msize = int(mt.matter_size.get(r.case_id, 1))
        rows.append({
            "id": r.case_id, "cit": r.citation, "court": r.court, "cg": court_group(r.court),
            "date": str(r.date_decided)[:10] if pd.notna(r.date_decided) else None, "year": int(r.year),
            "judge": nn(r.judge), "parties": nn(r.parties), "stream": r.stream, "proc": nn(r.proceeding),
            "issues": [CAUSE_LABEL[k] for k in CAUSES if getattr(r, k) == "Y"],
            "outcome": nn(r.outcome), "project": None if r.project_type in (None, "not stated") else nn(r.project_type),
            "arb": None if r.arb_application_type in (None, "NA") or pd.isna(r.arb_application_type) else r.arb_application_type,
            "grounds": [v for k, v in GROUNDS.items() if getattr(r, k) == "Y"], "state": r.state_party == "Y",
            "statutes": list(r.statutes_cited) if r.statutes_cited is not None else [],
            "leg": list(r.leg_ids_cited) if r.leg_ids_cited is not None else [],
            "summary": nn(c.holding_summary.get(r.case_id)), "keyIssues": nn(c.key_issues.get(r.case_id)),
            "regOutcome": nn(c.outcome_register.get(r.case_id)) if pd.notna(c.holding_summary.get(r.case_id)) else None,
            "later": later.get(r.case_id, []), "earlier": earlier.get(r.case_id, []),
            "matter": mt.matter_id.get(r.case_id) if msize > 1 else None, "matterSize": msize,
            "url": nn(c.ulii_link.get(r.case_id)), "src": r.source,
        })
    n_cite = pd.Series([l for r in rows for l in r["leg"]]).value_counts()
    legs = []
    for r in leg.sort_values("date").itertuples():
        eff = r.commencement_date if pd.notna(r.commencement_date) else r.date
        legs.append({"id": r.leg_id, "title": re.sub(r"\s*\((?:per|package)[^)]*\)", "", r.title), "citation": nn(r.citation),
                     "type": nn(r.type), "area": nn(r.area), "effective": str(eff.date()) if pd.notna(eff) else None,
                     "status": nn(r.commencement_status), "provisions": nn(r.key_provisions),
                     "url": r.source if str(r.source or "").startswith("http") else None,
                     "verified": bool(r.verified), "citedIn": int(n_cite.get(r.leg_id, 0))})
    (DEST / "legislation.json").write_text(json.dumps(legs, ensure_ascii=False, separators=(",", ":")))
    (DEST / "cases.json").write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
    ins = insights(a, pd.DataFrame(rows))
    (DEST / "insights.json").write_text(json.dumps(ins, ensure_ascii=False, separators=(",", ":"), default=nn))

    # district boundaries, coordinates rounded to ~100 m
    g = json.load(open(ROOT / "data/external/procurement-risk-uganda/docs/data/districts.geojson"))
    rnd = lambda x: [rnd(v) for v in x] if isinstance(x, list) else round(x, 3)
    for f in g["features"]:
        f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
        f["properties"] = {k: f["properties"][k] for k in ("district", "region")}
    (DEST / "districts.geojson").write_text(json.dumps(g, separators=(",", ":")))

    meta = {"name": "Uganda Construction & Arbitration Law Atlas", "generated": str(date.today()),
            "coverage": f"{min(r['year'] for r in rows)}–{max(r['year'] for r in rows)}",
            "lastDecision": max(r["date"] for r in rows if r["date"]), "cases": len(rows),
            "coding": "v1 (rule-based, validated against a provisional AI-coded reference set)",
            "licence": "CC BY 4.0", "withSummary": sum(1 for r in rows if r["summary"])}
    (DEST / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    kb = {p.name: round(p.stat().st_size / 1024) for p in DEST.glob("*")}
    print(f"atlas: {len(rows)} cases ({meta['withSummary']} with summaries), {len(legs)} enactments, "
          f"{len(t)} treatment links, {sum(1 for r in rows if r['matter'])} decisions in multi-decision matters; files KB {kb}")


if __name__ == "__main__":
    main()
