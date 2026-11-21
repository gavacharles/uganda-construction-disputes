"""Analysis stage 12 (RQ7) — spatial spillover: do disputes in neighbouring districts predict disputes next year?

District-year panel of construction-related decisions located by the Stage 10 geoparser (2015–2026). Negative binomial
model: cases_t ~ own cases_{t-1} + neighbours' cases_{t-1} (queen contiguity, row-standardised spatial lag) + year
effects, offset log(population). Global Moran's I of counts by year (999 permutations, seed 42). Locations exist for
about half of decisions and are rule-based, so results are exploratory.
Output: outputs/analysis/12_spillover/ (summary.md, t1_model.csv, t2_moran_by_year.csv)
"""
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from esda.moran import Moran
from libpysal.weights import Queen

from src.analysis.common import C as COL, PROCESSED, SEED, Stage, load, plt
from src.config import ROOT

warnings.filterwarnings("ignore")


def main():
    s = Stage("12_spillover", "Stage 12 — Spatial spillover of construction disputes (RQ7)")
    a = load().merge(pd.read_parquet(PROCESSED / "case_locations.parquet"), on="case_id")
    con = a[a.stream.isin(["construction", "both"]) & a.district.notna()]
    g = gpd.read_file(ROOT / "data/external/procurement-risk-uganda/docs/data/districts.geojson")
    g["district"] = g.district.str.strip()
    w = Queen.from_dataframe(g, use_index=False, silence_warnings=True)
    w.transform = "r"
    years = list(range(2015, 2027))
    Y = pd.crosstab(con.district, con.year).reindex(index=g.district, columns=years, fill_value=0)
    lag = pd.DataFrame(np.column_stack([w.sparse @ Y[y].values for y in years]), index=Y.index, columns=years)
    rows = []
    for t in years[1:]:
        for d in Y.index:
            rows.append({"district": d, "year": t, "cases": Y.at[d, t], "own_lag": Y.at[d, t - 1], "nbr_lag": lag.at[d, t - 1]})
    p = pd.DataFrame(rows).merge(g[["district", "population"]], on="district")
    p["log_pop"] = np.log(p.population)
    p = p[p.district != "Kampala"]  # Kampala hosts the courts and most head offices: an outlier in both level and lag
    m = smf.glm("cases ~ own_lag + nbr_lag + C(year)", data=p, family=sm.families.NegativeBinomial(alpha=1.0),
                offset=p.log_pop).fit()
    ci = np.exp(m.conf_int())
    t1 = pd.DataFrame({"IRR": np.exp(m.params), "low": ci[0], "high": ci[1], "p": m.pvalues}).loc[["own_lag", "nbr_lag"]].round(3)
    t1.index = ["own district, previous year (per decision)", "neighbouring districts, previous year (per decision, average)"]
    s.text(f"District-years: {len(p)} (134 districts × 11 years, Kampala excluded). Located construction decisions: "
           f"{len(con)} of {a.stream.isin(['construction', 'both']).sum()}.")
    s.table(t1, "t1_model", "Table 1. Negative binomial model of construction decisions per district-year",
            note="IRR > 1: more decisions next year. Year effects and a population offset included. Exploratory: locations cover about half of decisions.")
    mor = []
    for y in years:
        np.random.seed(SEED)
        mi = Moran(Y[y].values.astype(float), w, permutations=999)
        mor.append({"year": y, "located decisions": int(Y[y].sum()), "Moran's I": round(mi.I, 3), "p": round(mi.p_sim, 3)})
    mt = pd.DataFrame(mor).set_index("year")
    s.table(mt, "t2_moran_by_year", "Table 2. Spatial clustering of construction decisions by year (Moran's I, queen contiguity)")
    fig, ax = plt.subplots(figsize=(6.4, 3))
    ax.plot(mt.index, mt["Moran's I"], marker="o", color=COL["construction"])
    ax.axhline(0, color=COL["ink2"], lw=0.8)
    ax.scatter(mt.index[mt.p < 0.05], mt["Moran's I"][mt.p < 0.05], s=70, facecolors="none", edgecolors=COL["arbitration"], label="p < 0.05")
    ax.set_ylabel("Moran's I")
    ax.legend(frameon=False, fontsize=8)
    s.style(ax)
    s.fig(fig, "fig1_moran_by_year.png", "Spatial clustering of located construction decisions by year (Moran's I). Circled years are "
          "significant at 5%; values near zero mean no clustering beyond chance.")
    s.write()
    return t1, mt


if __name__ == "__main__":
    main()
