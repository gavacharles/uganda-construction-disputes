"""Stage 1.4 — assemble OCR'd judgments into cases and code them with transparent rules.

Rule-based (dictionary/regex) coding only — no LLM. Every rule is in this file.
Outputs
  data/interim/ocr_cases.parquet
  outputs/scanned/*.csv, figures (300 dpi) and summary.md
Usage: python -m src.ingest.ocr_cases
"""
import re

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import INTERIM, LOGS, OUT  # noqa: E402
from src.ingest.registers import case_id, norm_citation  # noqa: E402
from src.viz.credit import credit

OCR = INTERIM / "ocr"
DEST = OUT / "scanned"
MIN_WORDS = 800  # below this = minimal reasoning flag

THEMES = {  # dispute-cause / issue dictionary (case-insensitive regex)
    "payment & certification": r"interim (payment )?certificate|certified|unpaid|outstanding (sum|balance|amount)|final account|payment certificate",
    "delay, EOT & LDs": r"extension of time|liquidated damages|\bdelay(ed|s)?\b|completion date|time is of the essence",
    "termination": r"terminat(e|ed|ion)",
    "variations": r"variation|additional works|extra works",
    "defects & quality": r"defect|workmanship|substandard|collapse",
    "guarantees & bonds": r"performance (guarantee|bond|security)|advance payment (guarantee|bond)|bank guarantee|retention",
    "procurement & award": r"procurement|bid(der|ding)?\b|tender|best evaluated|PPDA",
    "land & trespass": r"trespass|encroach|land title|certificate of title",
    "professional fees": r"architect|quantity surveyor|consultan(t|cy) fees|engineer'?s fees",
    "arbitration referral/stay": r"refer(red)? (the matter |the dispute )?to arbitration|section 5|stay of proceedings",
    "award set-aside/enforcement": r"set(ting)? aside (the|an|of) (arbitral )?award|section 34|enforce(ment)? of (the )?(arbitral )?award|section 35|section 36",
}
STATUTES = {
    "Arbitration and Conciliation Act": r"arbitration and conciliation act",
    "PPDA Act": r"public procurement and disposal of public assets act|PPDA act",
    "Contracts Act": r"contracts act",
    "Civil Procedure Act/Rules": r"civil procedure (act|rules)",
    "Judicature Act": r"judicature act",
    "Limitation Act": r"limitation act",
    "Government Proceedings Act": r"government proceedings act",
    "Evidence Act": r"evidence act",
    "Building Control Act": r"building control act",
    "Roads Act": r"roads act",
    "Land Act / Registration of Titles Act": r"\bland act|registration of titles act",
    "Local Governments Act": r"local governments act",
    "Companies Act": r"companies act",
    "Physical Planning Act": r"physical planning act",
    "Engineers Registration Act": r"engineers registration act",
}
FORMS = {
    "FIDIC": r"\bFIDIC\b",
    "PPDA SBD": r"standard bidding document|\bSBD\b",
    "JBC / PAM": r"\bJBC\b|joint building council",
}
OUTCOME = [  # first match in the last 20% of the text wins
    ("referred to arbitration", r"referred to arbitration|stay(ed)? pending arbitration"),
    ("award set aside", r"award is (hereby )?set aside"),
    ("struck out", r"struck out"),
    ("dismissed", r"(suit|application|appeal|claim|petition) (is|stands) (hereby )?dismissed|hereby dismissed"),
    ("allowed / granted", r"(appeal|application) (is|stands) (hereby )?(allowed|granted)|judgment is (hereby )?entered|in favour of the plaintiff"),
]
NEUTRAL_CIT = r"\[(?:19|20)\d{2}\]\s+UG[A-Za-z]+\s+\d+"
REPORT_CIT = r"\b(?:19|20)\d{2}\s*\)?\s*(?:\d+\s+)?(?:HCB|KALR|ULR|EA|E\.A\.?)\b"
MONEY = r"(?:UGX|Ug\.?\s?Shs?\.?|Shs\.?|USh)\s*([\d,]{5,})"


def load_cases():
    pages = pd.read_parquet(OCR / "pages.parquet")
    docs = pages.groupby("file").agg(
        text=("text", "\n\n".join), pages=("page", "size"), words=("n_words", "sum"),
        method=("method", "first"),
        ocr_conf=("conf", lambda c: float(c[pages.loc[c.index, "n_words"] > 0].mean())))
    meta = pd.read_csv(next((OCR / "src").glob("*.csv")), encoding="utf-8-sig")
    meta["file"] = meta["PDF file in this ZIP"].str.replace("PDFs/", "", regex=False)
    meta = meta.rename(columns={"Year": "year", "Date judgment delivered": "date_decided",
                                "Case (parties)": "parties", "Citation": "citation",
                                "Court": "court", "Judge(s)": "judge", "Case number": "case_number",
                                "Likely register": "likely_register", "ULII link": "ulii_link"})
    df = meta.merge(docs, left_on="file", right_index=True, how="left")
    df["citation"] = df["citation"].map(norm_citation)
    df["case_id"] = df["citation"].map(case_id)
    df["date_decided"] = pd.to_datetime(df["date_decided"], errors="coerce")
    return df


def code(df):
    t = df["text"].fillna("")
    for k, rx in THEMES.items():
        df[f"theme:{k}"] = t.str.count(rf"(?i){rx}")
    for k, rx in STATUTES.items():
        df[f"stat:{k}"] = t.str.contains(rf"(?i){rx}")
    for k, rx in FORMS.items():
        df[f"form:{k}"] = t.str.contains(rx)
    df["contract_form"] = df[[f"form:{k}" for k in FORMS]].idxmax(axis=1).str[5:].where(
        df[[f"form:{k}" for k in FORMS]].any(axis=1), "not stated / bespoke")
    tail = t.map(lambda s: s[int(len(s) * 0.8):])
    df["outcome_rule"] = "unclear"
    for label, rx in reversed(OUTCOME):  # earlier entries take precedence
        df.loc[tail.str.contains(rf"(?i){rx}"), "outcome_rule"] = label
    df["precedents_cited"] = t.map(lambda s: sorted({" ".join(x.split()) for x in re.findall(NEUTRAL_CIT, s)}))
    df["n_precedents"] = df["precedents_cited"].str.len() + t.str.count(REPORT_CIT)
    df["clauses_cited"] = t.str.findall(r"(?i)\b(?:sub-?)?clause\s+(\d+(?:\.\d+)*)").map(
        lambda x: sorted(set(x)))
    amounts = t.str.findall(MONEY).map(lambda xs: [int(x.replace(",", "")) for x in xs if x.replace(",", "").isdigit()])
    df["max_amount_ugx"] = amounts.map(lambda xs: max(xs) if xs else None)
    df["year_filed"] = pd.to_numeric(df["case_number"].str.extract(r"of\s+(\d{4})")[0], errors="coerce")
    df["years_to_decision"] = df["year"] - df["year_filed"]
    df["doc_type"] = t.str[:3000].str.extract(r"(?i)\b(JUDGMENT|RULING)\b")[0].str.title().fillna("unknown")
    # relevance + PRISMA flags
    cons = t.str.count(
        r"(?i)contractor|construction|\bworks\b|building|\broads?\b|bill of quantities|\bBOQ\b|engineer|procur|"
        r"tender|\bbid|subcontract|\bsite\b|architect|surveyor|certificate of (completion|payment)|consultan|"
        r"materials|cement|excavat|\bplant\b")
    arb = t.str.count(r"(?i)arbitra|adjudicat")
    df["cons_score"], df["arb_score"] = cons, arb
    df["stream_rule"] = "neither"
    df.loc[cons >= 10, "stream_rule"] = "construction"
    df.loc[arb >= 5, "stream_rule"] = "arbitration"
    df.loc[(cons >= 10) & (arb >= 5), "stream_rule"] = "both"
    df["flag_minimal"] = df["words"].fillna(0) < MIN_WORDS
    df["flag_irrelevant"] = (cons < 4) & (arb < 2)          # almost no domain vocabulary
    df["flag_review"] = df["stream_rule"].eq("neither") & ~df["flag_irrelevant"]  # borderline: read it
    df.loc[df.flag_review, "stream_rule"] = "construction (to confirm)"
    df["include"] = ~(df.flag_minimal | df.flag_irrelevant)
    df["source"] = "ocr"
    return df


def fig_years(df, path):
    c = df[df.include].groupby(["year", "stream_rule"]).size().unstack(fill_value=0)
    order = [s for s in ["construction", "construction (to confirm)", "both", "arbitration"] if s in c]
    colors = {"construction": "#2a78d6", "construction (to confirm)": "#9fc3ee", "both": "#1baf7a", "arbitration": "#eb6834"}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bottom = 0
    for s in order:
        ax.bar(c.index, c[s], bottom=bottom, color=colors[s], label=s, width=0.7,
               edgecolor="white", linewidth=2)
        bottom = bottom + c[s]
    ax.set_xticks(c.index)
    ax.set_ylabel("Judgments")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    credit(fig)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_themes(share, path):
    s = share.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(s.index, s.values, color="#2a78d6", height=0.6)
    for y, v in enumerate(s.values):
        ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=8, color="#52514e")
    ax.set_xlabel("Share of included judgments (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    credit(fig)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    df = code(load_cases())
    df.drop(columns=["text"]).to_parquet(INTERIM / "ocr_cases.parquet", index=False)
    df[["text", "case_id"]].to_parquet(INTERIM / "ocr_texts.parquet", index=False)

    reg = pd.read_parquet(INTERIM / "registers_cases.parquet")
    df["in_registers"] = df["case_id"].isin(reg["case_id"])

    excl = df.loc[~df.include, ["case_id", "citation", "words", "stream_rule"]].assign(
        reason=lambda x: x.apply(lambda r: "minimal reasoning (<%d words)" % MIN_WORDS
                                 if r.words < MIN_WORDS else "not construction/arbitration", axis=1),
        stage="ocr screen")
    excl.to_csv(LOGS / "exclusions_ocr.csv", index=False)

    inc = df[df.include]
    tcols = [c for c in df if c.startswith("theme:")]
    theme_share = (inc[tcols] > 0).mean().mul(100).rename(lambda c: c[6:]).round(1)
    stat_share = inc[[c for c in df if c.startswith("stat:")]].mean().mul(100).rename(lambda c: c[5:]).round(1)
    tables = {
        "theme_share": theme_share.sort_values(ascending=False),
        "statute_share": stat_share.sort_values(ascending=False),
        "by_year_stream": inc.groupby(["year", "stream_rule"]).size().unstack(fill_value=0),
        "by_court": inc["court"].value_counts(),
        "outcomes": inc.groupby(["stream_rule", "outcome_rule"]).size().unstack(fill_value=0),
        "contract_form": inc["contract_form"].value_counts(),
        "duration": inc.groupby("stream_rule")["years_to_decision"].describe().round(1),
    }
    for k, v in tables.items():
        v.to_csv(DEST / f"{k}.csv")
    fig_years(df, DEST / "fig_scanned_by_year.png")
    fig_themes(theme_share, DEST / "fig_scanned_themes.png")

    top_prec = inc["precedents_cited"].explode().value_counts().head(10)
    md = [
        "# Scanned judgments — OCR and rule-based analysis\n",
        f"- PDFs: {len(df)}; pages: {int(df.pages.sum())}; mean OCR confidence: {df.ocr_conf.mean():.1f}",
        f"- Already in Excel registers: {int(df.in_registers.sum())}",
        f"- Excluded: {int((~df.include).sum())} (minimal reasoning {int(df.flag_minimal.sum())}, "
        f"not construction/arbitration {int((df.flag_irrelevant & ~df.flag_minimal).sum())}) → see outputs/logs/exclusions_ocr.csv",
        f"- Included: {len(inc)}\n",
    ]
    for k, v in tables.items():
        md += [f"## {k.replace('_', ' ').title()}\n", v.to_markdown(), ""]
    md += ["## Most-cited neutral citations\n", top_prec.to_markdown(), "",
           "**Figure 1.** Included scanned judgments by year of decision and stream (rule-coded). "
           "Source: ULII scanned judgments, OCR (Tesseract).\n",
           "**Figure 2.** Share of included scanned judgments mentioning each issue family "
           "(dictionary coding; a judgment can fall in several).\n",
           "_Caveats: coding is keyword-based and must be validated against the manual gold set; "
           "outcome and amounts are rule-extracted and indicative only._"]
    (DEST / "summary.md").write_text("\n".join(md))
    print("\n".join(md[:5]))


if __name__ == "__main__":
    main()
