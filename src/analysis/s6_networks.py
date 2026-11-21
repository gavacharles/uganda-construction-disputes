"""Analysis stage 6 (RQ5) — precedent (citation) network and repeat-player (party) network.

A. Citations from corpus decisions to any authority (neutral citations, law reports, case numbers). Most-cited
   authorities; directed in-corpus network (PageRank, betweenness, Louvain communities, seed 42).
B. Parties: normalised names, repeat players (>= 3 decisions), their roles and initiator success; co-litigation
   network. Judges are not analysed individually (aggregate reporting only).
Output: outputs/analysis/06_networks/
"""
import re
import warnings

import networkx as nx
import numpy as np
import pandas as pd

from src.analysis.common import C, SEED, Stage, load, plt
from src.config import PROCESSED
from src.extract.fulltext import CASENO, caseno_key
from src.ingest.registers import case_aliases

warnings.filterwarnings("ignore")


def norm_party(p):
    p = re.sub(r"(?i)\band (?:another|others|anor|ors)\b|\bt/a\b.*|\(.*?\)", " ", str(p))
    p = re.sub(r"(?i)\b(?:limited|ltd|company|co|plc|the|m/s|uganda|u)\b", " ", p)
    p = re.sub(r"[^a-z ]", " ", p.lower())
    p = re.sub(r"\s+", " ", p).strip()
    return {"attorney general": "Attorney General", "national roads authority": "UNRA", "unra": "UNRA",
            "kampala capital city authority": "KCCA", "kcca": "KCCA",
            "national water and sewerage corporation": "NWSC", "nwsc": "NWSC"}.get(p, p.title())


def main():
    s = Stage("06_networks", "Stage 6 — Precedent and repeat-player networks (RQ5)")
    a = load()
    alias = case_aliases(a)
    e = pd.read_parquet(PROCESSED / "citation_edges.parquet")
    e["citing"] = e.citing.map(alias)
    e = e.dropna(subset=["citing"])
    # cited case numbers -> corpus decisions with that case number (unique keys only; court not checked)
    own = {}
    for cid, num in zip(a.case_id, a.case_number.fillna("")):
        for m in CASENO.findall(num):
            own.setdefault(caseno_key(*m), set()).add(cid)
    own = {k: next(iter(v)) for k, v in own.items() if len(v) == 1}
    e["cited_id"] = e.cited.map(lambda c: alias.get(c, own.get(c, c)))
    e = e[e.citing != e.cited_id].drop_duplicates(["citing", "cited_id"])
    cit = a.set_index("case_id").citation
    e["in_corpus"] = e.cited_id.isin(a.case_id)
    s.text(f"Citation links from **{e.citing.nunique()}** corpus decisions (full-text decisions only; the 65 OCR'd scanned "
           f"judgments were not parsed for citations): {len(e)} links, {e.cited_id.nunique()} distinct authorities, "
           f"{int(e.in_corpus.sum())} links to other decisions in the corpus.")

    top = e.groupby(["cited_id", "kind"]).citing.nunique().sort_values(ascending=False).head(25).rename("citing decisions").reset_index()
    top["authority"] = top.cited_id.map(lambda c: cit.get(c, c))
    s.table(top[["authority", "kind", "citing decisions"]].set_index("authority"), "t1_most_cited",
            "Table 1. Most-cited authorities in the corpus (all kinds)",
            note="kind: neutral = [year] court number; report = law report (HCB, KALR, EA...); case_no = cited by case number only "
                 "(matched to a corpus decision where its own case number is unique). "
                 "The same decision cited in different forms is counted separately (forms are not yet reconciled).")

    G = nx.DiGraph()
    G.add_nodes_from(a.case_id)
    G.add_edges_from(e[e.in_corpus][["citing", "cited_id"]].itertuples(index=False))
    H = G.subgraph([n for n in G if G.degree(n) > 0]).copy()
    pr = nx.pagerank(H, alpha=0.85)
    bt = nx.betweenness_centrality(H.to_undirected(), seed=SEED, k=min(200, len(H)))
    comms = nx.community.louvain_communities(H.to_undirected(), seed=SEED)
    comm_of = {n: i for i, c in enumerate(sorted(comms, key=len, reverse=True)) for n in c}
    nodes = pd.DataFrame({"citation": [cit[n] for n in H], "stream": [a.set_index("case_id").stream[n] for n in H],
                          "cited by (corpus)": [H.in_degree(n) for n in H], "PageRank": [pr[n] for n in H],
                          "betweenness": [bt[n] for n in H], "community": [comm_of[n] for n in H]}, index=list(H))
    s.text(f"In-corpus network: {H.number_of_nodes()} decisions linked by {H.number_of_edges()} citations; "
           f"{len(comms)} Louvain communities (largest {len(max(comms, key=len))}); "
           f"{(nodes['cited by (corpus)'] > 0).sum()} decisions cited at least once by a later corpus decision.")
    s.table(nodes.sort_values("PageRank", ascending=False).head(15).round(4).set_index("citation"), "t2_central_decisions",
            "Table 2. Most central decisions in the in-corpus precedent network (PageRank)")
    comm = nodes.groupby("community").agg(size=("citation", "size"), arbitration_share=("stream", lambda x: (x != "construction").mean()),
                                          lead=("PageRank", lambda x: nodes.loc[x.idxmax(), "citation"]))
    s.table(comm[comm["size"] >= 5].round(2), "t3_communities", "Table 3. Precedent communities (size ≥ 5), share arbitration-related and lead decision")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    pos = nx.spring_layout(H.to_undirected(), seed=SEED, k=0.25)
    sizes = [20 + 3000 * pr[n] for n in H]
    cols = [C[nodes.stream[n]] for n in H]
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.15, arrows=False, width=0.5)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=sizes, node_color=cols, alpha=0.85, linewidths=0)
    for n in nodes.sort_values("PageRank", ascending=False).head(6).index:
        ax.annotate(cit[n], pos[n], fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    for st in ["construction", "both", "arbitration"]:
        ax.scatter([], [], color=C[st], label=st, s=25)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.axis("off")
    s.fig(fig, "fig1_precedent_network.png", "In-corpus precedent network: decisions (nodes, sized by PageRank, coloured by stream) "
          "linked by citations of earlier corpus decisions. The six most central decisions are labelled.")

    # ---- B. parties
    rows = []
    for r in a.itertuples():
        sides = re.split(r"\s+(?:v|vs|versus)\.?\s+", str(r.parties or ""), maxsplit=1, flags=re.I)
        for role, side, ptype in zip(["initiator", "respondent"], sides, [r.initiator_type, r.respondent_type]):
            name = norm_party(side)
            if len(name) > 2:
                rows.append({"case_id": r.case_id, "party": name, "role": role, "type": ptype, "stream": r.stream,
                             "outcome": r.outcome, "year": r.year})
    p = pd.DataFrame(rows)
    p["won"] = np.where(p.role == "initiator", p.outcome.isin(["initiator succeeded", "partly succeeded"]),
                        p.outcome.isin(["initiator failed", "struck out (procedural)"]))
    rep = p.groupby("party").agg(decisions=("case_id", "nunique"), as_initiator=("role", lambda x: (x == "initiator").sum()),
                                 type=("type", lambda x: x.mode().iat[0] if x.notna().any() else None),
                                 win_rate=("won", "mean"), first=("year", "min"), last=("year", "max"))
    rep = rep[rep.decisions >= 3].sort_values("decisions", ascending=False)
    rep["win_rate"] = (100 * rep.win_rate).round(0)
    share = p[p.party.isin(rep.index)].case_id.nunique() / a.case_id.nunique()
    s.table(rep.head(25), "t4_repeat_players", f"Table 4. Repeat players (in ≥ 3 decisions): {len(rep)} parties appear in "
            f"{share:.0%} of decisions", note="win_rate = share of the party's decisions in which it prevailed (as initiator: succeeded or "
            "partly; as respondent: initiator failed or struck out). Party names normalised; groups of companies not merged.")
    one = p.groupby("party").case_id.nunique()
    p["repeat"] = p.party.map(one) >= 3
    wr = p.groupby("repeat").won.mean().mul(100).round(1)
    s.text(f"Win rate of repeat players {wr.get(True, np.nan)}% vs one-shotters {wr.get(False, np.nan)}% (Galanter's repeat-player "
           "thesis); state actors among repeat players: " + ", ".join(rep[rep.type == "state body"].index[:8]) + ".")

    B = nx.Graph()
    for cid, g in p.groupby("case_id"):
        names = g.party.tolist()
        if len(names) == 2 and names[0] != names[1]:
            u, v = names
            B.add_edge(u, v, weight=B[u][v]["weight"] + 1 if B.has_edge(u, v) else 1)
    core = B.subgraph([n for n in B if n in rep.index or any(m in rep.index for m in B[n])]).copy()
    deg = pd.Series(dict(core.degree())).sort_values(ascending=False)
    s.table(deg.head(12).rename("distinct opponents").to_frame(), "t5_party_hubs",
            "Table 5. Parties with the most distinct opponents (co-litigation network around repeat players)")
    s.write()


if __name__ == "__main__":
    main()
