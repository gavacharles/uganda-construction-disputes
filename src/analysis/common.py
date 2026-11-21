"""Shared helpers for the analysis stages: data loading, house style, 300-dpi figures with captions, caveats."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import OUT, PROCESSED, SEED  # noqa: E402,F401

C = {"construction": "#2a78d6", "both": "#1baf7a", "arbitration": "#eb6834",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e5e5e2", "accent": "#7a5cc4"}
STREAMS = ["construction", "both", "arbitration"]
CAUSES = ["c_payment", "c_final_account", "c_variations", "c_delay_eot", "c_lds", "c_termination", "c_defects",
          "c_guarantees", "c_procurement", "c_site_land", "c_prof_fees", "c_price_adjust", "c_force_majeure", "c_fraud"]
CAUSE_LABEL = {"c_payment": "Payment & certification", "c_final_account": "Final account", "c_variations": "Variations",
               "c_delay_eot": "Delay & EOT", "c_lds": "Liquidated damages", "c_termination": "Termination & abandonment",
               "c_defects": "Defects & workmanship", "c_guarantees": "Guarantees & bonds", "c_procurement": "Procurement",
               "c_site_land": "Site, land & nuisance", "c_prof_fees": "Professional fees", "c_price_adjust": "Price adjustment",
               "c_force_majeure": "Force majeure", "c_fraud": "Fraud & corruption"}
CAVEATS = """**Caveats.** Litigated cases are a filtered sample of disputes (selection bias); the corpus is small, so
interpretable models are preferred; judge effects are reported only in aggregate. Codes come from rule-based extraction
validated against a provisional AI-coded gold set (see outputs/coding/coding_v1_accuracy.csv); fields with low agreement
are flagged in the text. Register holdings and provisions are paraphrased from ULII summaries and must be verified
against the full judgment before citation. 2026 is a partial year (to 16 Nov 2026)."""
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.titlesize": 11})


def load():
    a = pd.read_parquet(PROCESSED / "analysis_cases.parquet")
    a["year"] = a.year.astype(int)
    a["date_decided"] = pd.to_datetime(a.date_decided, errors="coerce")
    return a


class Stage:
    """Collects figures (with captions) and tables for one analysis stage and writes summary.md."""

    def __init__(self, name, title):
        self.dir = OUT / "analysis" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.title, self.parts, self.n_fig = title, [], 0

    def style(self, ax, grid="y"):
        ax.spines[["top", "right"]].set_visible(False)
        if grid:
            ax.grid(axis=grid, color=C["grid"], linewidth=0.8)
        ax.set_axisbelow(True)

    def fig(self, fig, name, caption):
        self.n_fig += 1
        fig.tight_layout()
        fig.savefig(self.dir / name, dpi=300)
        plt.close(fig)
        self.parts.append(f"![Figure {self.n_fig}]({name})\n\n*Figure {self.n_fig}. {caption}*\n")

    def table(self, df, name, heading, note=None, index=True):
        df.to_csv(self.dir / f"{name}.csv", index=index)
        self.parts.append(f"### {heading}\n\n{df.to_markdown(index=index)}\n" + (f"\n{note}\n" if note else ""))

    def text(self, md):
        self.parts.append(md + "\n")

    def write(self):
        body = f"# {self.title}\n\n" + "\n".join(self.parts) + f"\n{CAVEATS}\n"
        (self.dir / "summary.md").write_text(body)
        print(f"wrote {self.dir / 'summary.md'} ({self.n_fig} figures)")
