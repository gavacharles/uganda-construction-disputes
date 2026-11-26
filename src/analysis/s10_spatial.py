"""Analysis stage 10 (RQ7) — where construction disputes arise, and what they overlap with.

Geoparsing (rule-based, no LLM): district names (135, from the district boundary layer) counted in the facts, issues
and early reasoning; strong cues ('X District', 'X Municipality', 'X Town Council', ...) weigh 3; the court's own seat
('High Court at Kampala') is ignored. A decision is located in its top-scoring district (score >= 2).
Spatial statistics: rate per 100,000 people; global Moran's I and local Getis-Ord Gi* (queen contiguity, 999
permutations, seed 42). Overlays (Spearman): public works contract value per capita and procurement risk (OCDS),
rainy-season access penalty; road-dispute rate in districts crossed by tracked road corridors (Mann-Whitney).
Rainfall is available only by climate zone in the source repos, so the rainy-season access penalty is the proxy.
Output: outputs/analysis/10_spatial/ (+ data/processed/case_locations.parquet)
"""
import json
import re
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from esda.getisord import G_Local
from esda.moran import Moran
from libpysal.weights import Queen
from scipy.stats import mannwhitneyu, spearmanr
from shapely.geometry import LineString

from src.analysis.common import C, PROCESSED, SEED, SEQ, THEME, Stage, load, plt
from src.code.rule_coder import load_texts
from src.code.segment import segment
from src.config import ROOT

warnings.filterwarnings("ignore")
EXT = ROOT / "data" / "external"
CUE = r"(?:district|municipal(?:ity| council)?|town council|town|city|local government|\bdlg\b|sub-county|county|hospital|market|road)"
SEAT = r"(?:at|holden at|sitting at|registry|registrar|division,?)\s+$"


def name_rx(d):
    body = re.escape(d).replace("\\-", "[- ]?")   # 'Madi-Okollo' also matches 'Madi Okollo'
    return re.compile(r"\b" + body + r"\b")


def locate(text, names):
    seg, issues, _ = segment(text)
    zone = " ".join([seg["facts"], seg["issues"], " ".join(issues), seg["analysis"][:8000]]) or text[1500:20000]
    zone = re.sub(r"\s+", " ", zone)
    scores = {}
    for d, rx in names.items():
        sc = 0
        for m in rx.finditer(zone):
            if re.search(SEAT, zone[max(0, m.start() - 25): m.start()], re.I):
                continue
            sc += 3 if re.match(rf"\W{{0,3}}{CUE}", zone[m.end(): m.end() + 25], re.I) else 1
        if sc:
            scores[d] = sc
    if re.search(r"kampala capital city|\bKCCA\b|\b(?:nakawa|makindye|kawempe|rubaga|lubaga) division", zone, re.I):
        scores["Kampala"] = scores.get("Kampala", 0) + 3
    if not scores:
        return None, 0
    d, s = max(scores.items(), key=lambda kv: kv[1])
    return (d, s) if s >= 2 else (None, s)


def main():
    s = Stage("10_spatial", "Stage 10 — Spatial concentration of construction disputes (RQ7)")
    a = load()
    gdf = gpd.read_file(EXT / "procurement-risk-uganda/docs/data/districts.geojson")
    gdf["district"] = gdf.district.str.strip()
    names = {d: name_rx(d) for d in gdf.district}
    texts = load_texts(a)
    loc = [locate(texts[c], names) for c in a.case_id]
    a["district"], a["loc_score"] = [x[0] for x in loc], [x[1] for x in loc]
    if THEME == "default":   # the themed (report) run must not rewrite processed data
        a[["case_id", "district", "loc_score"]].to_parquet(PROCESSED / "case_locations.parquet", index=False)
    con = a[a.stream.isin(["construction", "both"])]
    s.text(f"Located decisions: **{a.district.notna().sum()} of {len(a)}** ({a.district.notna().mean():.0%}); construction-related "
           f"{con.district.notna().sum()} of {len(con)}. Unlocated decisions name no district in the facts or issues (common in "
           "procedural rulings). Geoparsing is rule-based and unvalidated: spot-check before mapping individual cases.")

    cnt = con.groupby("district").size().rename("cases")
    road = con[con.project_type.eq("road")].groupby("district").size().rename("road_cases")
    g = gdf.merge(cnt, left_on="district", right_index=True, how="left").merge(road, left_on="district", right_index=True, how="left")
    g[["cases", "road_cases"]] = g[["cases", "road_cases"]].fillna(0)
    g["rate_100k"] = 1e5 * g.cases / g.population
    top = g.sort_values("cases", ascending=False).head(12)[["district", "region", "cases", "road_cases", "rate_100k"]].round(2)
    s.table(top.set_index("district"), "t1_top_districts", "Table 1. Districts with the most construction-related decisions")
    reg = g.groupby("region").agg(cases=("cases", "sum"), population=("population", "sum"))
    reg["rate_100k"] = (1e5 * reg.cases / reg.population).round(2)
    reg["share %"] = (100 * reg.cases / reg.cases.sum()).round(1)
    s.table(reg, "t2_regions", "Table 2. Construction-related decisions by region")

    w = Queen.from_dataframe(g, use_index=False, silence_warnings=True)
    w.transform = "r"
    np.random.seed(SEED)  # Moran draws permutations from the global RNG
    mi = Moran(g.rate_100k.values, w, permutations=999)
    gi = G_Local(g.cases.values, w, star=True, permutations=999, seed=SEED)
    g["gi_z"], g["gi_p"] = gi.Zs, gi.p_sim
    g["hotspot"] = np.select([(g.gi_z > 0) & (g.gi_p < 0.05), (g.gi_z < 0) & (g.gi_p < 0.05)], ["hot spot", "cold spot"], "not significant")
    s.text(f"**Global Moran's I** (rate per 100,000, queen contiguity): I = {mi.I:.3f}, permutation p = {mi.p_sim:.3f}. "
           f"**Getis-Ord Gi\\*** hot spots (p < 0.05): {', '.join(g[g.hotspot == 'hot spot'].district) or 'none'}; "
           f"cold spots: {', '.join(g[g.hotspot == 'cold spot'].district) or 'none'}.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.6))
    g.plot(column="rate_100k", cmap=SEQ, linewidth=0.2, edgecolor="#999", ax=ax1, legend=True,
           legend_kwds={"shrink": 0.6, "label": "decisions per 100,000"}, missing_kwds={"color": "#eee"})
    ax1.set_title("Rate per 100,000 people", fontsize=9)
    colors = {"hot spot": C["up"], "cold spot": C["down"], "not significant": "#e9e9e6"}
    g.plot(color=g.hotspot.map(colors), linewidth=0.2, edgecolor="#999", ax=ax2)
    for k, v in colors.items():
        ax2.scatter([], [], color=v, label=k, s=30)
    ax2.legend(frameon=False, fontsize=7, loc="lower left")
    ax2.set_title("Getis-Ord Gi* (counts, p < 0.05)", fontsize=9)
    for ax in (ax1, ax2):
        ax.axis("off")
    s.fig(fig, "fig1_maps.png", "Construction-related decisions by district of the dispute (rule-based geoparsing of the facts "
          "and issues): rate per 100,000 people (left) and Getis-Ord Gi* hot and cold spots of counts (right; queen contiguity, "
          "999 permutations).")

    # overlays
    met = json.load(open(EXT / "procurement-risk-uganda/docs/data/district_metrics.json"))["data"]
    ov = pd.DataFrame({d: {"works value per capita": v.get("all|all", {}).get("value_per_capita"),
                           "road works value per capita": v.get("all|roads", {}).get("value_per_capita"),
                           "procurement risk score": v.get("all|all", {}).get("mean_risk"),
                           "single-bid share": v.get("all|all", {}).get("single_bid_share")} for d, v in met.items()}).T
    acc = pd.read_csv(EXT / "uganda-rainy-season-access-study/outputs/access_districts.csv").groupby("district").seasonal_penalty_h.mean()
    ov["rainy-season access penalty (h)"] = acc
    g = g.merge(ov, left_on="district", right_index=True, how="left")
    rows = []
    for col in ov.columns:
        x = g[["rate_100k", col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(x) > 20:
            r, p = spearmanr(x.rate_100k, x[col])
            rows.append({"district indicator": col, "districts": len(x), "Spearman rho with dispute rate": round(r, 2), "p": round(p, 3)})
    s.table(pd.DataFrame(rows).set_index("district indicator"), "t3_overlays",
            "Table 3. Dispute rate and district overlays (public works spending, procurement risk, rainy-season access)",
            note="Spending and risk from OCDS public contracts 2018/19–2026/27 (procurement-risk-uganda); access penalty = extra "
                 "travel hours in the worst rainy month (uganda-rainy-season-access-study). Associations only.")

    roads = json.load(open(EXT / "procurement-risk-uganda/docs/data/roads.json"))["roads"]
    lines = gpd.GeoDataFrame({"road": [r["name"] for r in roads]},
                             geometry=[LineString([(lon, lat) for lat, lon in r["route"]]) for r in roads], crs=g.crs)
    hit = gpd.sjoin(g[["district", "geometry"]], lines, predicate="intersects").district.unique()
    g["corridor"] = g.district.isin(hit)
    g["road_rate"] = 1e5 * g.road_cases / g.population
    u, p = mannwhitneyu(g[g.corridor].road_rate, g[~g.corridor].road_rate)
    s.text(f"**Road corridors.** {g.corridor.sum()} districts are crossed by the {len(lines)} tracked national road projects. Road-dispute "
           f"rate per 100,000: corridor districts mean {g[g.corridor].road_rate.mean():.2f} vs others {g[~g.corridor].road_rate.mean():.2f} "
           f"(Mann–Whitney p = {p:.3f}).")
    g.drop(columns="geometry").to_csv(s.dir / "district_table.csv", index=False)
    s.write()


if __name__ == "__main__":
    main()
