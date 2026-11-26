"""Stage 13, part B (PhD): extensions to the UBOS x URA triangulation, adding MoFPED budget data.

  fuel            URA fuel taxes, import volumes and pump prices: implied duty per litre, duty share of the pump price,
                  and pass-through from Brent in shillings to pump prices and the CIPI diesel sub-index.
  price_views     CIPI material sub-indices against UBOS PPI for the same materials (monthly y/y), and PPI cement around
                  the 2026 excise increase (UGX 500 -> 750 per 50 kg; effective date uncertain, see working paper).
  formality       construction VAT returns (expected, nil, non-filer, payment, offset) and construction-related
                  sub-sectors in URA's top-50 table.
  budget          MoFPED approved / released / spent for buildings and structures (items 3121xx, all votes): execution
                  rates, quarterly profile, public construction spending vs construction value added, arrears releases,
                  construction-sector tax per shilling released, externally financed releases vs machinery imports, and
                  state-party construction cases by filing year.
Inputs: data/processed/fiscal_triangulation/{ura_books_long.csv, mofped_budget_long.csv.gz} (src.ingest.ura_books,
src.ingest.mofped) and the shared series in data/processed/sector2026/.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.common import C
from src.config import PROCESSED

T = PROCESSED / "fiscal_triangulation"
CONSTR = "3121"                                   # buildings and structures (both charts of accounts)
FUND_LABEL = {"gou_dev": "GoU development", "ext_fin": "external finance", "nonwage": "non-wage recurrent",
              "wage": "wage", "arrears": "arrears", "aia": "AIA/NTR"}
WORKS_VOTES = r"Works and Transport|National Roads|Road Fund|Kampala Capital City|Energy and Mineral"
FULL_YEARS = ["2016/17", "2017/18", "2019/20", "2020/21", "2021/22"]   # FY2018/19: Q1 releases/spending only


def fy_of(t):
    y = t.year if t.month >= 7 else t.year - 1
    return f"{y}/{(y + 1) % 100:02d}"


def books():
    d = pd.read_csv(T / "ura_books_long.csv")
    d["rank"] = d.book.str[:4].astype(int)
    return d.sort_values("rank").groupby(["series", "fy"]).value.last().unstack("series"), d


# ---------- fuel ----------
def fuel(b, m):
    out = []
    for p in ("diesel", "petrol"):
        f = pd.DataFrame({"tax_ugx_bn": b[f"fuel_tax_{p}"], "volume_ml": b[f"fuel_volume_{p}"],
                          "pump_price": b[f"pump_price_{p}"]})
        f["duty_per_litre"] = 1e3 * f.tax_ugx_bn / f.volume_ml
        f["copy_error"] = (f.tax_ugx_bn - f.volume_ml).abs() < 0.01          # tax column = volume column
        f.loc[f.copy_error, "duty_per_litre"] = np.nan
        f["duty_share_pct"] = 100 * f.duty_per_litre / f.pump_price
        f.insert(0, "product", p)
        out.append(f)
    f = pd.concat(out)
    br = m.brent_ugx.dropna()
    brent = br.groupby(br.index.map(fy_of)).mean().where(br.groupby(br.index.map(fy_of)).size() == 12)
    dz = m.DIESEL.dropna()
    cipi = dz.groupby(dz.index.map(fy_of)).mean().where(dz.groupby(dz.index.map(fy_of)).size() == 12)
    d = f[f["product"] == "diesel"][["pump_price"]].join(brent.rename("brent_ugx")).join(cipi.rename("cipi_diesel"))
    g = 100 * d.pct_change()
    g.columns = [f"{c}_growth_pct" for c in d.columns]
    pass_through = d.join(g).dropna(subset=["brent_ugx_growth_pct"])
    slope = np.polyfit(g.brent_ugx_growth_pct.dropna(), g.pump_price_growth_pct.loc[g.brent_ugx_growth_pct.dropna().index], 1)[0] \
        if g.brent_ugx_growth_pct.notna().sum() >= 4 else np.nan
    return f.reset_index(names="fy"), pass_through.reset_index(names="fy"), slope


# ---------- prices ----------
PAIRS = {"cement": ("CEM", "ppi_cement"), "iron and steel": ("IRONSTEEL", "ppi_iron_steel"),
         "clay bricks": ("BRICK", "ppi_clay_bricks")}


def price_views(m, ubos):
    rows = []
    for name, (cipi, ppi) in PAIRS.items():
        a = 100 * (m[cipi] / m[cipi].shift(12) - 1)
        b = 100 * (ubos[ppi] / ubos[ppi].shift(12) - 1)
        d = pd.concat([a.rename("cipi"), b.rename("ppi")], axis=1).dropna()
        rows.append({"material": name, "cipi_series": cipi, "ppi_series": ppi, "months": len(d),
                     "from": d.index.min().strftime("%Y-%m"), "to": d.index.max().strftime("%Y-%m"),
                     "corr_yoy": d.corr().iloc[0, 1], "mean_cipi_yoy": d.cipi.mean(), "mean_ppi_yoy": d.ppi.mean(),
                     "mean_gap_pp": (d.cipi - d.ppi).mean()})
    comp = pd.DataFrame(rows)
    p = ubos.ppi_cement.dropna()
    mom = 100 * p.pct_change()
    hist = mom.loc[:"2026-05-01"]
    window = pd.DataFrame({"ppi_cement": p.loc["2026-03-01":], "mom_pct": mom.loc["2026-03-01":]})
    window["z_vs_history"] = (window.mom_pct - hist.mean()) / hist.std()
    return comp, window.reset_index(names="month")


# ---------- formality ----------
def formality(b, raw, tax):
    v = raw[raw.series.str.startswith("vat_returns")].pivot_table(index=["book", "fy"], columns="series", values="value")
    v.columns = [c.replace("vat_returns_", "") for c in v.columns]
    if {"expected", "nil", "non_filer"} <= set(v.columns):
        v["non_filer_pct"] = 100 * v.non_filer / v.expected
        v["nil_pct_of_filed"] = 100 * v.nil / (v.expected - v.non_filer)
    sub = b[[c for c in b.columns if c.startswith("subsector_")]].copy()
    sub.columns = [c.replace("subsector_", "isic_") for c in sub.columns]
    sub = sub.join(tax[["construction_tax"]])
    for c in [c for c in sub.columns if c.startswith("isic_4")]:
        sub[f"{c}_share_pct"] = 100 * sub[c] / sub.construction_tax
    return v.reset_index(), sub.dropna(how="all", subset=[c for c in sub.columns if c.startswith("isic_")])


# ---------- budget ----------
def mofped():
    d = pd.read_csv(T / "mofped_budget_long.csv.gz", dtype={"item_code": str, "vote_code": str})
    d["item_code"] = d.item_code.fillna("")
    d["constr"] = d.item_code.str.startswith(CONSTR)
    d["roads"] = d.item_desc.astype(str).str.contains("Roads and Bridges", case=False)
    return d


def budget(d, ubos, ura_tax, imq, cases):
    c = d[d.constr]
    a = c.groupby(["fy", "stage"]).amount.sum().unstack() / 1e12
    a["release_rate_pct"] = 100 * a.released / a.approved
    a["absorption_pct"] = 100 * a.spent / a.released
    a["execution_pct"] = 100 * a.spent / a.approved
    a.loc[~a.index.isin(FULL_YEARS), ["released", "spent", "release_rate_pct", "absorption_pct", "execution_pct"]] = np.nan
    byfund = (c[c.fy.isin(FULL_YEARS)].groupby(["fund", "stage"]).amount.sum().unstack() / 1e12)
    byfund["release_rate_pct"] = 100 * byfund.released / byfund.approved
    byfund["execution_pct"] = 100 * byfund.spent / byfund.approved
    # quarterly profile of construction spending
    q = c[c.fy.isin(FULL_YEARS) & c.stage.isin(["released", "spent"])]
    prof = q.groupby(["stage", "quarter"]).amount.sum().unstack("quarter")
    prof = 100 * prof.div(prof.sum(axis=1), axis=0)
    # public construction spending vs construction value added (current prices)
    va = ubos.gdp_construction_va_cur.dropna()
    va_fy = va.groupby(va.index.map(fy_of)).sum().where(va.groupby(va.index.map(fy_of)).size() == 4) / 1e3   # UGX tn
    share = a[["approved", "spent"]].join(va_fy.rename("construction_va_tn"))
    share["spent_pct_of_va"] = 100 * share.spent / share.construction_va_tn
    share = share.join(ura_tax.rename("construction_tax_bn"))
    share["tax_per_100_released"] = 100 * (share.construction_tax_bn / 1e3) / a.released
    # arrears releases (all items): total, paid centrally (Treasury Operations), and to works-related votes
    arr = d[(d.fund == "arrears") & (d.stage == "released") & d.fy.isin(FULL_YEARS)]
    works = arr.vote_name.astype(str).str.contains(WORKS_VOTES, case=False)
    treas = arr.vote_name.astype(str).str.contains("Treasury Operations", case=False)
    arrears = pd.DataFrame({"arrears_released_ugx_bn": arr.groupby("fy").amount.sum() / 1e9,
                            "treasury_operations_ugx_bn": arr[treas].groupby("fy").amount.sum() / 1e9,
                            "works_votes_ugx_bn": arr[works].groupby("fy").amount.sum() / 1e9}).fillna(0)
    arrears["treasury_share_pct"] = 100 * arrears.treasury_operations_ugx_bn / arrears.arrears_released_ugx_bn
    arrears["works_share_pct"] = 100 * arrears.works_votes_ugx_bn / arrears.arrears_released_ugx_bn
    arrears["vat_in_works_arrears_ugx_bn_if_vatable"] = arrears.works_votes_ugx_bn * 18 / 118
    # quarterly series: construction spending, external-finance releases, construction VA, machinery imports
    qs = c[c.stage.isin(["released", "spent"])].copy()
    y0 = qs.fy.str[:4].astype(int)
    qn = qs.quarter.str[1].astype(int)
    qs["date"] = pd.to_datetime(pd.DataFrame({"year": np.where(qn <= 2, y0, y0 + 1),
                                              "month": np.select([qn == 1, qn == 2, qn == 3], [7, 10, 1], 4), "day": 1},
                                             index=qs.index))
    spent_q = qs[qs.stage == "spent"].groupby("date").amount.sum() / 1e9
    ext_q = qs[(qs.stage == "released") & (qs.fund == "ext_fin")].groupby("date").amount.sum() / 1e9
    qdf = pd.DataFrame({"constr_spent_bn": spent_q, "ext_fin_released_bn": ext_q}).join(
        (va / 1).rename("construction_va_cur_bn")).join(imq.machinery.rename("machinery_imports_usd_k"))
    qdf["fy"] = qdf.index.map(fy_of)
    dev = qdf.groupby("fy").transform(lambda s: s - s.mean())          # within-year (seasonal) co-movement
    links = []
    for x, y_, lag in [("constr_spent_bn", "construction_va_cur_bn", 0), ("constr_spent_bn", "construction_va_cur_bn", 1),
                       ("ext_fin_released_bn", "machinery_imports_usd_k", 0), ("ext_fin_released_bn", "machinery_imports_usd_k", 1)]:
        dd = pd.concat([qdf[x].shift(lag), qdf[y_]], axis=1).dropna()
        ddv = pd.concat([dev[x].shift(lag), dev[y_]], axis=1).dropna()
        links.append({"x": x, "y": y_, "x_leads_by_q": lag, "n": len(dd), "r_levels": dd.corr().iloc[0, 1],
                      "r_within_year": ddv.corr().iloc[0, 1]})
    # disputes: state-party construction cases by filing year
    cases = cases[cases.stream.isin(["construction", "both"])].copy()
    cases["filed"] = pd.to_numeric(cases.case_number.astype(str).str.extract(r"(?:of|/)\s*((?:19|20)\d\d)\b",
                                                                              expand=False), errors="coerce")
    sp = cases[cases.state_party.astype(str) == "Y"]
    disp = pd.DataFrame({"state_party_cases": sp.groupby("filed").size(),
                         "state_party_payment_cases": sp[sp.c_payment.astype(str).isin(["Y", "1", "True", "1.0"])].groupby("filed").size()}).fillna(0)
    return a.reset_index(), byfund.reset_index(), prof.reset_index(), share.reset_index(), arrears.reset_index(), \
        qdf.reset_index(names="quarter"), pd.DataFrame(links), disp.loc[2014:2026].reset_index(names="filing_year")


# ---------- figures ----------
def figures(st, fuel_t, bud, prof, share, sub, vat):
    f = fuel_t
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
    yrs = sorted(f.dropna(subset=["duty_share_pct"]).fy.unique())
    for p, col in (("diesel", C["construction"]), ("petrol", C["arbitration"])):
        g = f[f["product"] == p].set_index("fy").dropna(subset=["duty_share_pct"])
        full = g.duty_share_pct.reindex(yrs)                 # gaps stay gaps (petrol FY2021/22-22/23 errors)
        a.plot(range(len(yrs)), full.values, marker="o", color=col, label=p)
        for y, v in g.duty_share_pct.items():
            a.annotate(f"{v:.0f}%", (yrs.index(y), v), xytext=(0, 7), textcoords="offset points", ha="center",
                       fontsize=7.5, color=col)
    a.set_xticks(range(len(yrs)), [y[2:] for y in yrs], fontsize=8)
    a.set_ylabel("% of pump price")
    a.set_ylim(0, 35)
    a.legend(frameon=False, fontsize=8)
    a.set_title("(a) Fuel duty as a share of the pump price", loc="left", fontsize=10)
    g = f[f["product"] == "diesel"].set_index("fy").dropna(subset=["pump_price"]).loc["2017/18":]
    x = np.arange(len(g))
    b.bar(x, g.pump_price, color=C["neutral"], label="diesel pump price")
    b.bar(x, g.duty_per_litre.fillna(0), color=C["construction"], label="duty per litre (where volume reported)")
    b.set_xticks(x, [y[2:] for y in g.index], fontsize=8)
    b.set_ylabel("UGX per litre")
    b.legend(frameon=False, fontsize=8, loc="upper left")
    b.set_title("(b) Diesel: pump price and duty", loc="left", fontsize=10)
    for ax in (a, b):
        st.style(ax)
    st.fig(fig, "fig5_fuel.png",
           "Implied fuel duty per litre (recorded fuel taxes ÷ imported volumes) as a share of the average pump price "
           "(a), and the diesel pump price with the implied duty, FY2017/18–FY2024/25 (b). Petrol FY2021/22–FY2022/23 "
           "omitted: the published tax figures equal the volume figures. Sources: URA annual data books; author's "
           "calculations.")

    b_ = bud.set_index("fy")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(b_))
    w = 0.27
    a.bar(x - w, b_.approved, w, color=C["neutral"], label="approved")
    a.bar(x, b_.released, w, color=C["construction"], label="released")
    a.bar(x + w, b_.spent, w, color=C["arbitration"], label="spent")
    a.set_xticks(x, [y[2:] for y in b_.index], fontsize=7.5, rotation=45)
    a.set_ylabel("UGX trillion")
    a.legend(frameon=False, fontsize=8, loc="upper left")
    a.set_title("(a) Buildings and structures (items 3121xx)", loc="left", fontsize=10)
    pr = prof.set_index("stage")[["Q1", "Q2", "Q3", "Q4"]]
    for k, (stg, col) in enumerate((("released", C["construction"]), ("spent", C["arbitration"]))):
        b.bar(np.arange(4) + (k - 0.5) * 0.38, pr.loc[stg], 0.36, color=col, label=stg)
    b.axhline(25, color=C["ink2"], ls="--", lw=0.8)
    b.set_xticks(range(4), ["Q1\nJul–Sep", "Q2\nOct–Dec", "Q3\nJan–Mar", "Q4\nApr–Jun"], fontsize=8)
    b.set_ylabel("% of the year's total")
    b.legend(frameon=False, fontsize=8)
    b.set_title("(b) Quarterly profile", loc="left", fontsize=10)
    for ax in (a, b):
        st.style(ax)
    st.fig(fig, "fig6_budget_execution.png",
           "Central-government approved budget, releases and expenditure on buildings and structures (items 3121xx, "
           "all votes), FY2015/16–FY2026/27 (a; releases and expenditure available for FY2016/17–FY2017/18 and "
           "FY2019/20–FY2021/22), and the quarterly distribution of releases and expenditure in those years (b; dashed "
           "line = even spread). Source: MoFPED budget data portal; author's calculations.")

    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
    s = sub.loc["2017/18":]
    for col, name, colr in (("isic_4100", "4100 buildings", C["construction"]), ("isic_4210", "4210 roads and railways", C["arbitration"]),
                            ("isic_4390", "4390 specialised", C["both"]), ("isic_2394", "2394 cement manufacture", C["neutral"])):
        if col in s:
            z = s[col].dropna()
            a.plot([s.index.get_loc(i) for i in z.index], z, marker="o", color=colr, label=name)
    a.set_xticks(range(len(s)), [y[2:] for y in s.index], fontsize=8)
    a.set_ylabel("UGX billion")
    a.legend(frameon=False, fontsize=7.5)
    a.set_title("(a) Tax from construction-related sub-sectors", loc="left", fontsize=10)
    vv = vat.dropna(subset=["non_filer_pct"])
    for bk, mk in zip(sorted(vv.book.unique()), ("o", "s")):
        g = vv[vv.book == bk]
        b.plot(g.fy.str[2:], g.non_filer_pct, marker=mk, color=C["construction"], label=f"non-filers ({bk} edition)")
        b.plot(g.fy.str[2:], g.nil_pct_of_filed, marker=mk, color=C["arbitration"], ls="--", label=f"nil returns ({bk} edition)")
    b.set_ylabel("%")
    b.legend(frameon=False, fontsize=7)
    b.set_title("(b) Construction VAT returns", loc="left", fontsize=10)
    for ax in (a, b):
        st.style(ax)
    st.fig(fig, "fig7_formality.png",
           "Gross collections from construction-related ISIC sub-sectors appearing in URA's top-50 table (a; a series "
           "ends when the sub-sector leaves the table), and construction VAT returns: non-filers as a share of expected "
           "returns and nil returns as a share of returns filed, by data-book edition (b; the editions define the "
           "categories differently). Source: URA annual data books; author's calculations.")


def run(st, ubos, m, tax, imq, cases):
    b, raw = books()
    fuel_t, pt, slope = fuel(b, m)
    comp, window = price_views(m, ubos)
    vat, sub = formality(b, raw, tax)
    d = mofped()
    bud, byfund, prof, share, arrears, qdf, links, disp = budget(d, ubos, tax.construction_tax, imq, cases)
    figures(st, fuel_t, bud, prof, share, sub, vat)
    st.table(fuel_t.round(2), "fuel", "Fuel: taxes, volumes, implied duty per litre and share of pump price", index=False)
    st.table(pt.round(2), "fuel_pass_through", f"Diesel pass-through (slope of pump-price growth on Brent-UGX growth: "
                                               f"{slope:.2f})", index=False)
    st.table(comp.round(2), "price_views", "CIPI vs PPI by material (monthly y/y)", index=False)
    st.table(window.round(2), "ppi_cement_2026", "PPI cement around the 2026 excise increase", index=False)
    st.table(vat.round(1), "vat_returns", "Construction VAT returns by data-book edition", index=False)
    st.table(sub.round(1), "subsectors", "Construction-related sub-sectors in URA's top-50 table (UGX bn)")
    st.table(bud.round(2), "budget_execution", "Buildings and structures (3121xx): approved, released, spent (UGX tn)",
             index=False)
    st.table(byfund.round(2), "budget_by_fund", "Buildings and structures by fund, FY2016/17–FY2021/22 (excl. 2018/19)",
             index=False)
    st.table(prof.round(1), "budget_quarter_profile", "Quarterly profile of construction releases and spending (%)",
             index=False)
    st.table(share.round(2), "public_share", "Public construction spending vs construction value added; tax per 100 "
                                             "released", index=False)
    st.table(arrears.round(1), "arrears", "Arrears releases (all items) and to the eight largest construction votes",
             index=False)
    st.table(links.round(2), "budget_links", "Quarterly links: spending vs output; external finance vs machinery imports",
             index=False)
    qdf.round(2).to_csv(st.dir / "budget_quarterly.csv", index=False)
    st.table(disp, "state_party_cases", "State-party construction decisions by filing year (published decisions only)",
             index=False)
    return {"fuel": fuel_t, "pass_through": pt, "slope": slope, "price_views": comp, "ppi_2026": window, "vat": vat,
            "subsectors": sub, "budget": bud, "byfund": byfund, "profile": prof, "share": share, "arrears": arrears,
            "links": links, "disputes": disp}
