"""Export aggregated tables (no case text, no party names) for the online explorer dashboard.

Each output is JSON {rows, meta}; meta carries this script's code so every number can be traced.
Usage: python -m src.viz.export_dashboard   -> outputs/dashboard/*.json
"""
import json
from pathlib import Path

import numpy as np
import pymannkendall as mk
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from src.config import INTERIM, OUT
from src.temporal.register_temporal import BINS, annual, quarterly, ulii_totals

DEST = OUT.parent / "docs" / "data"  # served by GitHub Pages
CODE = Path(__file__).read_text()
STREAM = {"construction": "Construction", "both": "Construction + arbitration", "arbitration": "Arbitration"}
PROC = {"suit": "Suit", "appeal": "Appeal", "application/cause": "Application / cause",
        "arbitration cause": "Arbitration cause", "other": "Other proceeding"}
OUTC = {"successful": "Initiator succeeded", "partial": "Partly succeeded",
        "referred to arbitration": "Referred to arbitration", "unsuccessful": "Initiator failed",
        "unclear": "Outcome unclear"}
ARB = {"stay/referral": "Stay / referral", "set-aside": "Set-aside", "enforcement/recognition": "Enforcement",
       "interim measures": "Interim measures", "appointment": "Appointment", "other": "Other arbitration"}
PARTY = {"company": "Company", "individual": "Individual", "state": "State body", "bank/insurer": "Bank / insurer",
         "unknown": "Unknown"}


def dump(name, rows, inputs):
    rows = json.loads(pd.DataFrame(rows).replace({np.nan: None}).to_json(orient="records"))
    meta = [{"label": "script", "value": "src/viz/export_dashboard.py"},
            {"label": "Input", "value": inputs}]
    (DEST_SCOPE / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": meta}, ensure_ascii=False))
    return len(rows)


SCOPES = {  # scope -> (periods kept, period bins)
    "2015-2025": (["2015-2025"], {"2015–2018": (2015, 2018), "2019–2021": (2019, 2021),
                                  "2022–2024": (2022, 2024), "2025": (2025, 2025)}),
    "2015-2026": (["2015-2025", "2026"], BINS),
}


def flows(c, bins):
    out = []
    for period, d in [("All years", c)] + [(b, c[c.bin == b]) for b in bins]:
        specs = {
            "pathway": [("stream", STREAM, "proceeding", PROC), ("proceeding", PROC, "outcome_class", OUTC)],
            "parties": [("party_type_1", PARTY, "party_type_2", PARTY), ("party_type_2", PARTY, "outcome_class", OUTC)],
            "arbitration": [("arb_application_type", ARB, "outcome_class", OUTC)],
        }
        for view, steps in specs.items():
            dv = d[d.arb_application_type.notna()] if view == "arbitration" else d
            for step, (a, amap, b, bmap) in enumerate(steps):
                g = dv.groupby([a, b]).size().reset_index(name="value")
                for _, r in g.iterrows():
                    out.append({"view": view, "period": period, "step": step,
                                "source": amap[r[a]], "target": bmap[r[b]], "value": int(r["value"])})
    return out


def export(scope):
    periods, bins = SCOPES[scope]
    dest = DEST / scope
    dest.mkdir(parents=True, exist_ok=True)
    global DEST_SCOPE
    DEST_SCOPE = dest
    c = pd.read_parquet(INTERIM / "registers_coded.parquet")
    c = c[c.period.isin(periods)].copy()
    c["bin"] = c["year"].map(lambda y: next(k for k, (a, b) in bins.items() if a <= y <= b))
    src = f"data/interim/registers_coded.parquet ({len(c)} ULII cases {scope}: registers + OCR'd scanned judgments, rule-coded)"
    tot = ulii_totals()
    tot = tot[tot.index.isin(c.year.unique())]
    n = {}
    n["flows"] = dump("flows", flows(c, bins), src)

    a = annual(c, tot).reset_index().rename(columns={"index": "year"})
    a.columns = [x.replace(" ", "_").replace("-", "_") for x in a.columns]
    n["annual"] = dump("annual", a, src + "; ULII totals from register 'Summary by year' (2026 ≈ 3,560)")

    q = quarterly(c, ulii_totals())
    q = q[[p.year in set(c.year) for p in q.index]]
    q = pd.DataFrame({"quarter": [str(p) for p in q.index], "start": [p.start_time.strftime("%Y-%m-%d") for p in q.index],
                      "construction_related": q["construction-related"].astype(int),
                      "arbitration_related": q["arbitration-related"].astype(int)})
    n["quarterly"] = dump("quarterly", q, src)

    leg = pd.read_parquet(INTERIM / "legislation.parquet")
    leg = leg[leg.date.dt.year <= c.year.max()]
    area = leg["area"].fillna("")
    group = np.select([area.str.contains("procurement", case=False), area.str.contains("road", case=False),
                       area.str.contains("Arbitration", case=False),
                       area.str.contains("building|standards|professions|materials|environment", case=False)],
                      ["Procurement & PPP", "Roads", "Arbitration & ADR", "Building & standards"], "Other construction")
    leg = pd.DataFrame({"leg_id": leg.leg_id, "date": leg.date.dt.strftime("%Y-%m-%d"), "title": leg.title,
                        "type": leg.type, "group": group, "status": leg.commencement_status,
                        "verified": leg.verified})
    n["legislation"] = dump("legislation", leg, "data/interim/legislation.parquet (legislation registers)")

    rows = []
    for sname, m in [("Construction-related", ["construction", "both"]), ("Arbitration-related", ["arbitration", "both"])]:
        s = c[c.stream.isin(m) & (c.source == "register")]  # issue coding from register summaries only
        for period, d in [("All years", s)] + [(b, s[s.bin == b]) for b in bins]:
            for col in [x for x in c if x.startswith("issue:")]:
                rows.append({"stream": sname, "period": period, "issue": col[6:],
                             "cases": int(d[col].sum()), "total": len(d),
                             "share_pct": round(100 * d[col].mean(), 1) if len(d) else None})
    n["issues"] = dump("issues", rows, src)

    rows = []
    d = c[c.outcome_class.isin(["successful", "unsuccessful", "partial"])]
    for sname, m in [("Construction-related", ["construction", "both"]), ("Arbitration-related", ["arbitration", "both"])]:
        for b in bins:
            s = d[d.stream.isin(m) & (d.bin == b)]
            k = int(s.outcome_class.ne("unsuccessful").sum())
            lo, hi = proportion_confint(k, len(s), method="wilson")
            rows.append({"stream": sname, "period": b, "succeeded": k, "decided": len(s),
                         "success_pct": round(100 * k / len(s), 1), "ci_low": round(100 * lo, 1), "ci_high": round(100 * hi, 1)})
    n["success"] = dump("success", rows, src + "; Wilson 95% intervals; unclear and referred outcomes excluded")

    rows = []
    dd = c[c.years_to_decision.notna()]
    for b in bins:
        x = dd.loc[dd.bin == b, "years_to_decision"]
        q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        rows.append({"period": b, "cases": len(x), "q1": q1, "median": med, "q3": q3,
                     "whisker_low": x[x >= q1 - 1.5 * iqr].min(), "whisker_high": x[x <= q3 + 1.5 * iqr].max(),
                     "max": x.max()})
    n["duration"] = dump("duration", rows, src + "; suits and appeals only, filing year from case number")

    rows = []
    for col in ["all", "construction_related", "arbitration_related"]:
        ser = a.set_index("year")[col]
        for name, v in [("count", ser), ("rate per 1,000 ULII judgments", 1000 * ser / a.set_index("year")["ulii_judgments"])]:
            r = mk.original_test(v.values)
            rows.append({"series": col.replace("_", "-"), "measure": name, "trend": r.trend, "tau": round(r.Tau, 2),
                         "p": round(r.p, 4), "sen_slope_per_year": round(r.slope, 3)})
    n["trends"] = dump("trends", rows, src + "; Mann-Kendall test and Sen's slope on annual series (2026 unadjusted partial year)")
    print(scope, n)


def main():
    for scope in SCOPES:
        export(scope)


if __name__ == "__main__":
    main()
