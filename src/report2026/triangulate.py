"""Sector report 2026: UBOS x URA triangulation (data already in hand; MoFPED vote-level data not yet collected).

  tax_intensity  URA construction-sector collections against UBOS construction value added (current prices, fiscal-year
                 sums): share of tax vs share of GDP, tax per UGX 100 of value added, and the same on sectorised revenue
                 (total less unsectorised). FY2015/16-2016/17 are flagged: they come from a stand-alone URA table whose
                 rows sum to net (not gross) collections and whose unsectorised row is 2.5x the next year's.
  cement         three views of cement: UBOS output (IIP 2394), the volume implied by URA domestic excise (a specific duty
                 of UGX 500 per 50 kg, unchanged FY2015/16-FY2024/25), and URA domestic VAT on cement deflated by the UBOS
                 cement PPI. VAT is net of input credit, so it indexes value added, not sales; it is used as an index only.
  imports_lead   do UBOS monthly imports (US$ value) of iron and steel (SITC 67) and machinery (72 + 74) lead construction
                 real value added? Correlation of year-on-year growth at leads 0-4 quarters, and an out-of-sample test of
                 a nowcast (expanding-window OLS; AR(1) baseline), and the latest import reading for the quarter after
                 the last GDP release.
  trade_2021     URA imports trade report 2021 (HS 8-digit values; quantities unusable): construction inputs by group and
                 recorded origin, and a check against UBOS SITC 66 and 67 for the same calendar year.
  data_audit     revisions across URA data books, totals that do not add, URA- vs UBOS-reported GDP shares, and the URA
                 diesel pump price against the CIPI diesel sub-index.
Outputs: data/processed/sector2026/tri_*.csv
Usage: python -m src.report2026.triangulate
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED

D = PROCESSED / "sector2026"
EXCISE_UGX_PER_TONNE = 500 * 20        # UGX 500 per 50 kg bag (Excise Duty Act 2014, Sch. 2); 750 from FY2026/27
EARLY = ("2015/16", "2016/17")         # URA stand-alone table: different basis (see docstring)
IMPORT_GROUPS = {"iron_steel": ["imp_iron_steel"], "machinery": ["imp_general_machinery", "imp_specialised_machinery"],
                 "nonmetallic": ["imp_nonmetallic_minerals"]}
# construction inputs by HS prefix (HS 2017; 8-digit codes matched on their leading digits)
HS_GROUPS = {
    "cement clinker": ["252310"], "cement": ["252321", "252329", "252330", "252390"],
    "lime, gypsum and plaster": ["2520", "2522"], "sand, stone and aggregates": ["2505", "2515", "2516", "2517"],
    "bitumen and asphalt": ["271320", "2714", "2715"],
    "steel: flat-rolled incl. roofing sheet": ["7208", "7209", "7210", "7211", "7212"],
    "steel: bars, rods and sections": ["7213", "7214", "7215", "7216", "7217"],
    "steel: billets and semi-finished": ["7206", "7207"],
    "steel tubes, structures and fittings": ["7303", "7304", "7305", "7306", "7307", "7308", "7314", "7317"],
    "ceramic tiles and sanitary ware": ["6907", "6908", "6910"], "flat glass": ["7003", "7004", "7005"],
    "paints and varnishes": ["3208", "3209", "3210"], "electrical cable": ["8544"],
    "earthmoving and construction machinery": ["8426", "8429", "8430", "8474"], "dump trucks": ["870410"]}
# approximate HS -> SITC Rev.4 concordance for the cross-check with UBOS (UBOS series are US$ '000)
SITC_CHECK = {
    "imp_iron_steel": {"label": "SITC 67 iron and steel", "include": ["72", "7302", "7303", "7304", "7305", "7306"],
                       "exclude": ["7204"]},
    "imp_nonmetallic_minerals": {"label": "SITC 66 non-metallic mineral manufactures",
                                 "include": ["2522", "2523", "68", "69", "70", "7101", "7102", "7103", "7104"],
                                 "exclude": ["6910", "7019"]}}
OOS_START = "2021-01-01"                # first quarter forecast out of sample
COVID = ("2020-04-01", "2021-06-01")    # quarters whose y/y growth is dominated by the lockdowns


def fy_of(t):
    y = t.year if t.month >= 7 else t.year - 1
    return f"{y}/{(y + 1) % 100:02d}"


def load():
    u = pd.read_csv(D / "ubos_series.csv", parse_dates=["date"])
    ubos = u.pivot_table(index="date", columns="series", values="value")
    ura = pd.read_csv(D / "ura_series.csv").pivot(index="fy", columns="series", values="value")
    xc = pd.read_csv(D / "ura_crosscheck.csv")
    m = pd.read_csv(D / "panel_monthly.csv", parse_dates=["date"], index_col="date")
    return ubos, ura, xc, m


def fy_sum(s, n_needed):
    g = s.dropna().groupby(s.dropna().index.map(fy_of))
    return g.sum().where(g.size() == n_needed)


def tax_intensity(ubos, ura):
    va, gdp = fy_sum(ubos.gdp_construction_va_cur, 4), fy_sum(ubos.gdp_total_cur, 4)
    t = pd.DataFrame({"construction_tax": ura.construction_sector_revenue,
                      "total_tax": ura.sector_table_total.fillna(ura.sector_rows_sum),
                      "unsectorised": ura.unsectorised, "construction_vat": ura.construction_vat,
                      "va_cur": va, "gdp_cur": gdp, "ura_rev_share_pct": ura.ura_rev_share_pct,
                      "ura_gdp_share_pct": ura.ura_gdp_share_pct})
    t["tax_share_pct"] = 100 * t.construction_tax / t.total_tax
    t["tax_share_sectorised_pct"] = 100 * t.construction_tax / (t.total_tax - t.unsectorised)
    t["gdp_share_pct"] = 100 * t.va_cur / t.gdp_cur
    t["tax_per_100_va"] = 100 * t.construction_tax / t.va_cur
    t["economy_tax_per_100_gdp"] = 100 * t.total_tax / t.gdp_cur
    t["relative_intensity"] = t.tax_per_100_va / t.economy_tax_per_100_gdp   # 1 = taxed like the average sector
    t["vat_per_100_va"] = 100 * t.construction_vat / t.va_cur
    t["comparable"] = ~t.index.isin(EARLY)
    return t.dropna(subset=["construction_tax"])


def cement(ubos, ura):
    iip, ppi = ubos.iip_cement.dropna(), ubos.ppi_cement.dropna()
    c = pd.DataFrame({"cement_excise": ura.cement_excise, "cement_vat": ura.cement_vat,
                      "iip_cement": iip.groupby(iip.index.map(fy_of)).mean().where(
                          iip.groupby(iip.index.map(fy_of)).size() == 4),
                      "ppi_cement": ppi.groupby(ppi.index.map(fy_of)).mean().where(
                          ppi.groupby(ppi.index.map(fy_of)).size() == 12)})
    c["implied_taxed_mt"] = c.cement_excise * 1e9 / EXCISE_UGX_PER_TONNE / 1e6
    c["real_cement_vat"] = c.cement_vat / c.ppi_cement
    base = "2017/18"
    for k in ("iip_cement", "implied_taxed_mt", "real_cement_vat"):
        c[f"{k}_idx"] = 100 * c[k] / c.loc[base, k]
    c["excise_vs_iip_gap_pct"] = c.implied_taxed_mt_idx / c.iip_cement_idx * 100 - 100
    return c.dropna(subset=["cement_excise"])


def _yoy(s):
    return 100 * (s / s.shift(4) - 1)


def imports_quarterly(ubos):
    m = ubos[[c for cols in IMPORT_GROUPS.values() for c in cols]].dropna(how="all")
    m = m.loc[m.index.notna()]
    grp = pd.DataFrame({g: m[cols].sum(axis=1, min_count=len(cols)) for g, cols in IMPORT_GROUPS.items()})
    grp["iron_steel_machinery"] = grp.iron_steel + grp.machinery
    qs = grp.resample("QS-JAN").agg(["sum", "count"])
    out = pd.DataFrame({g: qs[(g, "sum")].where(qs[(g, "count")] == 3) for g in grp.columns})
    partial = grp.resample("QS-JAN").count().iloc[:, 0]
    return out, grp, partial


def imports_lead(ubos):
    imq, imm, _ = imports_quarterly(ubos)
    va = ubos.gdp_construction_va_const.dropna()
    y = _yoy(va).rename("va_yoy")
    rows = []
    for g in imq.columns:
        x = _yoy(imq[g])
        for k in range(5):
            d = pd.concat([y, x.shift(k)], axis=1).dropna()
            dx = d.drop(index=d.loc[COVID[0]:COVID[1]].index)
            rows.append({"group": g, "lead_q": k, "r_all": d.corr().iloc[0, 1], "n_all": len(d),
                         "r_ex_covid": dx.corr().iloc[0, 1], "n_ex_covid": len(dx)})
    corr = pd.DataFrame(rows)

    # out-of-sample test: VA y/y(t) on VA y/y(t-1) [AR] vs + imports y/y(t) [IMP]; expanding window, COVID excluded.
    # Pre-specified group: iron_steel_machinery; the other groups are reported as diagnostics (chosen after the fact).
    covid_q = lambda idx: idx.to_series().between(*map(pd.Timestamp, COVID))
    preds, rmse = [], []
    for g in imq.columns:
        d = pd.concat([y, y.shift(1).rename("va_lag"), _yoy(imq[g]).rename("imp_yoy")], axis=1)
        cov = covid_q(d.index)
        for t in d.loc[OOS_START:].dropna().index:
            tr = d.loc[:t].iloc[:-1].dropna()
            tr = tr[~cov.reindex(tr.index)]
            row = {"group": g, "date": t, "actual": d.loc[t, "va_yoy"]}
            for name, cols in {"ar": ["va_lag"], "imp": ["va_lag", "imp_yoy"]}.items():
                X = np.column_stack([np.ones(len(tr))] + [tr[c] for c in cols])
                b, *_ = np.linalg.lstsq(X, tr.va_yoy.values, rcond=None)
                row[name] = b[0] + sum(bi * d.loc[t, c] for bi, c in zip(b[1:], cols))
            preds.append(row)
    oos = pd.DataFrame(preds)
    for g, o in oos.groupby("group"):
        rmse.append({"group": g, "prespecified": g == "iron_steel_machinery", "n_oos": len(o),
                     "rmse_ar": float(np.sqrt(((o.ar - o.actual) ** 2).mean())),
                     "rmse_imp": float(np.sqrt(((o.imp - o.actual) ** 2).mean()))})
    rmse = pd.DataFrame(rmse)
    rmse["imp_beats_ar"] = rmse.rmse_imp < rmse.rmse_ar

    # latest reading: iron and steel imports in the months after the last GDP quarter, against a year earlier
    nxt = y.index.max() + pd.DateOffset(months=3)
    months = imm.loc[nxt:nxt + pd.DateOffset(months=2)].dropna(how="all")
    prev = imm.loc[[t - pd.DateOffset(years=1) for t in months.index]]
    latest = pd.DataFrame([{"quarter": nxt, "months": len(months), "last_month": months.index.max(),
                            **{f"{g}_yoy_to_date": 100 * (months[g].sum() / prev[g].sum() - 1) for g in imm.columns},
                            "va_yoy_last_quarter": y.iloc[-1], "last_gdp_quarter": y.index.max()}]).set_index("quarter")
    return corr, oos.set_index("date"), rmse, latest


def _hs_mask(codes, include, exclude=()):
    digits = codes.str.replace(".", "", regex=False)
    inc = digits.str.startswith(tuple(include))
    return inc & ~digits.str.startswith(tuple(exclude)) if exclude else inc


def trade_2021(ubos, m):
    rows = pd.read_csv(D / "ura_trade_2021.csv", dtype={"hs8": str})
    summ = pd.read_csv(D / "ura_trade_2021_summary.csv", dtype={"hs8": str})
    fx = m.exchange_rate.loc["2021-01-01":"2021-12-01"].mean()
    out = []
    for g, pref in HS_GROUPS.items():
        r = rows[_hs_mask(rows.hs8, pref)]
        total = summ[_hs_mask(summ.hs8, pref)].value_ugx_bn.sum()
        by = r[r.country != "unmatched"].groupby("country").value_ugx_bn.sum().sort_values(ascending=False)
        shares = by / r.value_ugx_bn.sum() if r.value_ugx_bn.sum() else by
        top = "; ".join(f"{c} {100 * v:.0f}%" for c, v in shares.head(3).items())
        out.append({"group": g, "value_ugx_bn": total, "value_usd_m": total * 1e3 / fx, "top_origins": top,
                    "top1_share_pct": 100 * shares.iloc[0] if len(shares) else np.nan,
                    "unmatched_share_pct": 100 * r[r.country == "unmatched"].value_ugx_bn.sum() / r.value_ugx_bn.sum()
                    if r.value_ugx_bn.sum() else np.nan})
    groups = pd.DataFrame(out).sort_values("value_ugx_bn", ascending=False)
    chk = []
    for series, spec in SITC_CHECK.items():
        ura = summ[_hs_mask(summ.hs8, spec["include"], spec["exclude"])].value_ugx_bn.sum() * 1e3 / fx
        ub = ubos[series].loc["2021-01-01":"2021-12-01"]
        chk.append({"series": spec["label"], "ura_hs_usd_m": ura, "ubos_sitc_usd_m": ub.sum() / 1e3,
                    "ubos_months": int(ub.notna().sum()), "diff_pct": 100 * (ura / (ub.sum() / 1e3) - 1)})
    return groups, pd.DataFrame(chk), fx


def data_audit(ura, xc, ubos, m, tax):
    rows = []
    multi = xc.groupby(["series", "fy"]).value.agg(["size", "min", "max"])
    multi = multi[multi["size"] > 1]
    for (s, fy), r in multi.iterrows():
        if s == "sector_rows_sum":      # reported under 'rows do not sum to total'
            continue
        rel = (r["max"] - r["min"]) / abs(r["max"]) if r["max"] else 0
        if rel > 0.005:
            rows.append({"check": "revised between data books", "series": s, "fy": fy, "value_a": r["min"],
                         "value_b": r["max"], "rel_diff_pct": 100 * rel})
    for fy in ura.index:
        a, b = ura.loc[fy, "sector_rows_sum"], ura.loc[fy, "sector_table_total"]
        if pd.notna(a) and pd.notna(b) and abs(a - b) / b > 0.005:
            rows.append({"check": "sector rows do not sum to total", "series": "sector_table_total", "fy": fy,
                         "value_a": a, "value_b": b, "rel_diff_pct": 100 * (a - b) / b})
        a, b = ura.loc[fy, "department_table_total"], ura.loc[fy, "sector_table_total"]
        if pd.notna(a) and pd.notna(b) and abs(a - b) / b > 0.005:
            rows.append({"check": "department total differs from sector total", "series": "total_revenue", "fy": fy,
                         "value_a": a, "value_b": b, "rel_diff_pct": 100 * (a - b) / b})
    # within-book check: rows vs total in each book
    for book, g in xc[xc.series.isin(["sector_rows_sum", "sector_table_total"])].groupby("book"):
        p = g.pivot_table(index="fy", columns="series", values="value")
        if {"sector_rows_sum", "sector_table_total"} <= set(p.columns):
            for fy, r in p.dropna().iterrows():
                if abs(r.sector_rows_sum - r.sector_table_total) / r.sector_table_total > 0.005:
                    rows.append({"check": f"rows do not sum to total in the {book} book", "series": "sector_table_total",
                                 "fy": fy, "value_a": r.sector_rows_sum, "value_b": r.sector_table_total,
                                 "rel_diff_pct": 100 * (r.sector_rows_sum - r.sector_table_total) / r.sector_table_total})
    # URA-printed GDP share vs UBOS
    for book, g in xc[xc.series == "ura_gdp_share_pct"].groupby("book"):
        for _, r in g.iterrows():
            ub = tax.gdp_share_pct.get(r.fy, np.nan)
            if pd.notna(ub) and abs(r.value - ub) > 0.3:
                rows.append({"check": f"URA-printed GDP share differs from UBOS ({book} book)", "series": "gdp_share",
                             "fy": r.fy, "value_a": r.value, "value_b": ub, "rel_diff_pct": 100 * (r.value - ub) / ub})
    audit = pd.DataFrame(rows).drop_duplicates()
    # diesel: URA (UBOS-reported) pump price vs CIPI diesel sub-index, fiscal-year growth
    dz = m.DIESEL.dropna()
    cipi = dz.groupby(dz.index.map(fy_of)).mean().where(dz.groupby(dz.index.map(fy_of)).size() == 12)
    diesel = pd.DataFrame({"ura_pump_price": ura.diesel_pump_price, "cipi_diesel": cipi}).dropna()
    diesel["ura_growth_pct"] = 100 * diesel.ura_pump_price.pct_change()
    diesel["cipi_growth_pct"] = 100 * diesel.cipi_diesel.pct_change()
    diesel["gap_pp"] = diesel.ura_growth_pct - diesel.cipi_growth_pct
    return audit, diesel


def main():
    ubos, ura, xc, m = load()
    tax = tax_intensity(ubos, ura)
    cem = cement(ubos, ura)
    corr, oos, rmse, latest = imports_lead(ubos)
    audit, diesel = data_audit(ura, xc, ubos, m, tax)
    trade_groups, trade_chk, fx21 = trade_2021(ubos, m)
    for k, v in {"tri_tax_intensity": tax, "tri_cement": cem, "tri_imports_corr": corr, "tri_imports_oos": oos,
                 "tri_imports_rmse": rmse, "tri_imports_latest": latest, "tri_data_audit": audit, "tri_diesel": diesel,
                 "tri_trade_2021_groups": trade_groups, "tri_trade_2021_sitc_check": trade_chk}.items():
        v.to_csv(D / f"{k}.csv", index=k not in ("tri_imports_corr", "tri_data_audit", "tri_imports_rmse", "tri_trade_2021_groups",
                                          "tri_trade_2021_sitc_check"))
    pd.set_option("display.width", 200)
    print("tax intensity:\n", tax[["construction_tax", "tax_share_pct", "tax_share_sectorised_pct", "gdp_share_pct",
                                   "tax_per_100_va", "relative_intensity", "vat_per_100_va", "comparable"]].round(2))
    print("\ncement:\n", cem.round(2))
    print("\nimports lead (r by lead):\n", corr.pivot(index="lead_q", columns="group", values="r_ex_covid").round(2))
    print("\nout-of-sample RMSE (pp):\n", rmse.round(2))
    print("\nlatest:\n", latest.round(1).T)
    print("\naudit:\n", audit.round(1).to_string())
    print("\ndiesel:\n", diesel.round(1))
    print(f"\nURA imports 2021 (UGX/US$ {fx21:,.0f}):\n", trade_groups.round(1).to_string(index=False))
    print("\nHS vs SITC 2021:\n", trade_chk.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
