"""Feasibility step for a contract-level dispute model: link public works contracts (OCDS) to court decisions.

A decision is linked to a contract when its parties include the contract's supplier (fuzzy name match) AND its buyer
(fuzzy match, aliases such as UNRA/KCCA, or the Attorney General standing for central government), and the
contract was awarded before the decision (within 12 years). 'Supplier only' matches are counted but not used as labels.
Output: data/interim/contract_links.parquet, outputs/model/link_report.md
"""
import re

import pandas as pd
from rapidfuzz import fuzz, process

from src.config import INTERIM, OUT, PROCESSED

STOP = r"\b(ltd|limited|company|co|the|u|uganda|and|another|others|anor|ors|m/s|t/a|smc|plc|jv|joint venture|group|holdings|enterprises|services|international|construction|engineering|contractors?)\b"
ALIAS = {"unra": "uganda national roads authority", "kcca": "kampala capital city authority", "nwsc": "national water and sewerage corporation",
         "ppda": "public procurement and disposal of public assets authority", "uec": "uganda electricity"}
CENTRAL = re.compile(r"ministry|office of the prime minister|state house|parliament|judiciary|courts of judicature|"
                     r"office of the president|directorate|police|prisons|defence", re.I)
LOCAL = re.compile(r"district|municipal|city council|town council|local government", re.I)
EXTRA_NOT_WORKS = re.compile(r"advertis|publish|notice of|decoration|license|licence|services\b", re.I)


def core(name):
    s = str(name or "").lower()
    for k, v in ALIAS.items():
        s = re.sub(rf"\b{k}\b", v, s)
    s = re.sub(r"\(.*?\)", " ", s)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()


def short(name):
    """Distinctive part of a company name for supplier matching."""
    return re.sub(r"\s+", " ", re.sub(STOP, " ", core(name))).strip()


def sides(parties):
    s = re.split(r"\s+(?:v|vs|versus)\.?\s+", str(parties or ""), maxsplit=1, flags=re.I)
    return [x for x in s if x.strip()]


def buyer_matches(side, buyer):
    a, b = core(side), core(buyer)
    if "attorney general" in a:
        return bool(CENTRAL.search(buyer)) and not LOCAL.search(buyer)
    return len(a) > 5 and fuzz.token_set_ratio(a, b) >= 88


def main():
    (OUT / "model").mkdir(parents=True, exist_ok=True)
    k = pd.read_parquet(INTERIM / "ocds_contracts.parquet")
    k = k[k.works & ~k.title.str.contains(EXTRA_NOT_WORKS) & k.award_date.notna() & (k.year >= 2010)].copy()
    k = k.explode("suppliers").dropna(subset=["suppliers"])
    k["sup_short"] = k.suppliers.map(short)
    k = k[k.sup_short.str.len() >= 4]
    sup_index = k.groupby("sup_short").contract_id.apply(list).to_dict()
    names = list(sup_index)
    byid = k.drop_duplicates("contract_id").set_index("contract_id")

    a = pd.read_parquet(PROCESSED / "analysis_cases.parquet")
    a["date_decided"] = pd.to_datetime(a.date_decided, errors="coerce")
    rows, sup_only = [], 0
    for r in a.itertuples():
        sd = sides(r.parties)
        if len(sd) < 2 or pd.isna(r.date_decided):
            continue
        for i in (0, 1):
            ss = short(sd[i])
            if len(ss) < 4:
                continue
            for name, score, _ in process.extract(ss, names, scorer=fuzz.token_set_ratio, limit=3, score_cutoff=92):
                if min(len(name), len(ss)) < 5 and score < 100:
                    continue
                cands = [byid.loc[c] for c in sup_index[name]]
                cands = [c for c in cands if c.award_date < r.date_decided and (r.date_decided - c.award_date).days < 12 * 365]
                if not cands:
                    continue
                strong = [c for c in cands if buyer_matches(sd[1 - i], c.buyer)]
                if strong:
                    for c in strong:
                        rows.append({"case_id": r.case_id, "stream": r.stream, "contract_id": c.name, "supplier": c.suppliers, "buyer": c.buyer,
                                     "title": c.title, "value_ugx": c.value_ugx, "award_date": c.award_date, "decided": r.date_decided,
                                     "match_score": score, "role": "initiator" if i == 0 else "respondent"})
                else:
                    sup_only += 1
    L = pd.DataFrame(rows).drop_duplicates(["case_id", "contract_id"])
    L.to_parquet(INTERIM / "contract_links.parquet", index=False)
    n_contracts = byid.shape[0]
    rep = [f"# Contract–dispute linkage (feasibility)\n",
           f"- Works contracts with dates, 2010+: **{n_contracts}** ({k.sup_short.nunique()} suppliers)",
           f"- Court decisions linked to at least one contract (supplier + buyer match): **{L.case_id.nunique() if len(L) else 0}**",
           f"- Contracts linked to a decision: **{L.contract_id.nunique() if len(L) else 0}** (positive rate {100 * (L.contract_id.nunique() if len(L) else 0) / n_contracts:.2f}%)",
           f"- Supplier-only matches (not used as labels): {sup_only}"]
    if len(L):
        rep += ["\n## Linked decisions by stream\n", L.drop_duplicates("case_id").stream.value_counts().to_frame().to_markdown(),
                "\n## Sample links\n", L.sort_values("decided").head(25)[["case_id", "supplier", "buyer", "title", "award_date", "decided"]].to_markdown(index=False)]
    (OUT / "model" / "link_report.md").write_text("\n".join(rep))
    print("\n".join(rep[:5]))


if __name__ == "__main__":
    main()
