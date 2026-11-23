"""Entity-year dispute signals and bid-level red flags for the early-warning model (v2).

Signals (per OCDS buyer x year):
  court      decisions of the courts naming the entity as a party (construction or arbitration; tax-only excluded)
  tribunal   PPDA Appeals Tribunal decisions naming the entity
  oag        Auditor-General infrastructure sentences (dispute-relevant drivers) mentioning the entity (report year)
  press      The Observer articles (dispute drivers) mentioning the entity
Bid red flags (all procurements with bid records, per buyer x award year): procurements with bids, mean bidders,
single-bid share, coefficient of variation of bid prices, share where the winning bid was not the lowest valid bid.
Entity mentions are matched by full name or by an explicit abbreviation list; matches are rule-based and should be
spot-checked (outputs/review/entity_matches.xlsx covers the court links).
Outputs: data/interim/entity_signals.parquet, data/interim/entity_bids.parquet, outputs/review/entity_matches.xlsx
"""
import gzip
import json
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.analysis.common import CAUSES
from src.config import INTERIM, OUT, PROCESSED, ROOT
from src.model.link_contracts import core, sides
from src.model.entity_risk import SUCCESSOR

EXT = ROOT / "data" / "external"
ABBR = {"Uganda National Roads Authority": ["UNRA"], "Kampala Capital City Authority": ["KCCA"],
        "National Water & Sewerage Corporation": ["NWSC"], "Ministry of Works and Transport": ["MoWT"],
        "National Social Security Fund": ["NSSF"], "Uganda Electricity Transmission Company": ["UETCL"],
        "Uganda Electricity Generation Company Limited": ["UEGCL"], "Uganda Revenue Authority": ["URA"],
        "Public Procurement and Disposal of Public Assets Authority": ["PPDA"], "Uganda National Bureau of Standards": ["UNBS"],
        "National Housing and Construction Company Limited": ["NHCC"], "Uganda Civil Aviation Authority": ["UCAA", "CAA"],
        "Rural Electrification Agency": ["REA"], "Uganda Railways Corporation": ["URC"], "National Environment Management Authority": ["NEMA"],
        "Ministry of Water and Environment": ["MWE"], "Ministry of Health": ["MoH"], "Ministry of Education and Sports": ["MoES"]}
# regulators and oversight bodies appear in audits and the press as watchdogs, not as parties to a works dispute
REGULATORS = re.compile(r"Public Procurement and Disposal|Inspectorate of Government|Regulatory Authority|Bank of Uganda|"
                        r"Uganda Revenue Authority|Auditor General|Capital Markets|Insurance Regulatory|Financial Intelligence", re.I)
OAG_DISPUTE = {"delayed_payments", "budget_and_funding_issues", "delays_and_time_overruns", "procurement_irregularities",
               "infrastructure_and_equipment", "land_and_right_of_way", "contract_management", "claims_and_liabilities",
               "governance_and_controls", "compliance_and_regulatory"}


def name_patterns(buyers):
    pats = {}
    for b in buyers:
        parts = [re.escape(x.strip()).replace(r"\ ", r"\s+") for x in re.sub(r"\s+", " ", b.strip()).split("&")]
        full = r"\s*(?:&|and)\s*".join(parts)
        alts = [full] + [rf"\b{re.escape(x)}\b" for x in ABBR.get(b, [])]
        if len(b.split()) >= 2:
            pats[b] = re.compile("|".join(alts))
    return pats


def bid_flags():
    rows = []
    with gzip.open(EXT / "ocds" / "uganda_ppda_full.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            det = (r.get("bids") or {}).get("details") or []
            if not det:
                continue
            vals = [float(d["value"]["amount"]) for d in det if (d.get("value") or {}).get("amount") and d.get("status") != "disqualified"]
            aw = (r.get("awards") or [{}])[0]
            win = ((aw.get("value") or {}).get("amount"))
            yr = pd.to_datetime(aw.get("date") or r.get("date"), errors="coerce")
            rows.append({"buyer": (r.get("buyer") or {}).get("name"), "year": yr.year if pd.notna(yr) else None, "n_bids": len(det),
                         "cv": float(np.std(vals) / np.mean(vals)) if len(vals) >= 2 and np.mean(vals) > 0 else np.nan,
                         "not_lowest": float(win > min(vals) * 1.001) if win and len(vals) >= 2 else np.nan})
    b = pd.DataFrame(rows).dropna(subset=["buyer", "year"])
    b["year"] = b.year.astype(int)
    g = b.groupby(["buyer", "year"]).agg(bid_procs=("n_bids", "size"), mean_bidders=("n_bids", "mean"),
                                          single_bid_share=("n_bids", lambda x: (x == 1).mean()), bid_cv=("cv", "mean"),
                                          not_lowest_share=("not_lowest", "mean")).reset_index()
    return g


def main():
    k = pd.read_parquet(INTERIM / "ocds_contracts.parquet")
    buyers = sorted(k.buyer.dropna().unique())
    bc = {b: core(b) for b in buyers}
    # court / tribunal links (as in entity_risk), with a review sheet
    a = pd.read_parquet(PROCESSED / "analysis_cases.parquet")
    a = a[~a.court.str.contains("Tax", na=False)]
    links = []
    for r in a.itertuples():
        if re.search(r"revenue authority", str(r.parties), re.I) and not any(getattr(r, c) == "Y" for c in CAUSES):
            continue
        for sd in sides(r.parties):
            c = core(sd)
            if len(c) < 6 or "attorney general" in c:
                continue
            best = max(buyers, key=lambda b: fuzz.token_set_ratio(c, bc[b]))
            sc = fuzz.token_set_ratio(c, bc[best])
            if sc >= 92 and abs(len(c) - len(bc[best])) < 40:
                links.append({"case_id": r.case_id, "citation": r.citation, "year": int(r.year), "court": r.court, "parties": r.parties,
                              "party_side": sd, "entity": best, "score": round(sc, 1),
                              "kind": "tribunal" if "PPDA" in str(r.court) else "court"})
    L = pd.DataFrame(links).drop_duplicates(["case_id", "entity"])
    (OUT / "review").mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT / "review" / "entity_matches.xlsx") as xw:
        L.assign(decision="", note="").to_excel(xw, sheet_name="matches", index=False)
    for old, (new, y0) in SUCCESSOR.items():
        L.loc[(L.entity == old) & (L.year >= y0), "entity"] = new

    pats = name_patterns(buyers)
    oag = pd.read_csv(EXT / "oag-reports/government_Auditor_General/oag_infrastructure_sentence_corpus_2017_2025_expanded.csv")
    oag = oag[oag.driver_label_expanded.fillna(oag.driver_label).isin(OAG_DISPUTE)]
    obs = pd.read_csv(EXT / "the-observer/paper2_online_discourse/outputs_observer_full_corpus/extended_analysis/articles_extended.csv",
                      usecols=["year", "title", "text", "main_driver_nice"]).dropna(subset=["main_driver_nice"])
    ment = []
    pats = {b: rx for b, rx in pats.items() if not REGULATORS.search(b)}
    for src, df, col in [("oag", oag, "sentence"), ("press", obs, "text")]:
        for r in df.itertuples():
            body, title = str(getattr(r, col))[:20000], str(getattr(r, "title", "") or "")
            for b, rx in pats.items():
                n = len(rx.findall(body))
                # press: headline mention or at least two mentions; audits: any mention in a dispute-relevant finding
                if (src == "oag" and n) or (src == "press" and (rx.search(title) or n >= 2)):
                    ment.append({"source": src, "entity": b, "year": int(r.year)})
    M = pd.DataFrame(ment)
    for old, (new, y0) in SUCCESSOR.items():
        M.loc[(M.entity == old) & (M.year >= y0), "entity"] = new
    S = pd.concat([L.rename(columns={"kind": "source"})[["source", "entity", "year"]], M]).groupby(["entity", "year", "source"]).size().unstack(fill_value=0)
    S = S.reindex(columns=["court", "tribunal", "oag", "press"], fill_value=0).reset_index()
    S.to_parquet(INTERIM / "entity_signals.parquet", index=False)
    B = bid_flags()
    B.to_parquet(INTERIM / "entity_bids.parquet", index=False)
    print(f"court/tribunal links: {len(L)} ({L.entity.nunique()} entities) -> review sheet outputs/review/entity_matches.xlsx")
    print("signal totals:", S[["court", "tribunal", "oag", "press"]].sum().to_dict(), "| entities with any signal:", S.entity.nunique())
    print("entity-years with each signal:", (S[["court", "tribunal", "oag", "press"]] > 0).sum().to_dict())
    print(f"bid flags: {len(B)} entity-years; mean bidders {B.mean_bidders.mean():.2f}; single-bid share {B.single_bid_share.mean():.2f}")


if __name__ == "__main__":
    main()
