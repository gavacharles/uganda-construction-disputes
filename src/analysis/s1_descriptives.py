"""Analysis stage 1 — descriptive profile of the corpus by stream (who, what, which forum, how it ended).

Output: outputs/analysis/01_descriptives/ (tables, 300-dpi figures, summary.md)
"""
import numpy as np
import pandas as pd

from src.analysis.common import C, CAUSE_LABEL, CAUSES, STREAMS, Stage, load, plt


def share_table(a, col, order=None):
    t = pd.crosstab(a[col].fillna("not coded"), a.stream).reindex(columns=[x for x in STREAMS if x in set(a.stream)], fill_value=0)
    t["all"] = t.sum(axis=1)
    if order:
        t = t.reindex([o for o in order if o in t.index] + [i for i in t.index if i not in order])
    pct = (100 * t / t.sum()).round(1)
    return t.astype(str) + " (" + pct.astype(str) + "%)"


def court_group(c):
    for k, rx in [("Supreme Court", "Supreme"), ("PPDA Appeals Tribunal", "PPDA"),  # before "Appeal": 'Appeals Tribunal'
                  ("Court of Appeal", "Court of Appeal|Constitutional"), ("High Court (Commercial)", "Commercial"),
                  ("High Court (other divisions)", "High Court"), ("CADER", "CADER")]:
        if pd.notna(c) and pd.Series([c]).str.contains(rx).iloc[0]:
            return k
    return "Other tribunals"


def main():
    s = Stage("01_descriptives", "Stage 1 — Descriptive profile of the corpus")
    a = load()
    a["court_group"] = a.court.map(court_group)
    n = a.stream.value_counts().reindex(STREAMS)
    s.text(f"Corpus: **{len(a)} decisions** (2015 – 16 Nov 2026): construction {n['construction']}, "
           f"construction-via-arbitration ('both') {n['both']}, arbitration {n['arbitration']}. "
           f"Sources: {a.source.value_counts().to_dict()}. Gold-labelled: {int(a.in_gold.sum())}.")

    s.table(share_table(a, "court_group"), "t1_forum", "Table 1. Forum by stream")
    s.table(share_table(a, "proceeding"), "t2_proceeding", "Table 2. Type of proceeding by stream")
    a["parties_pair"] = a.initiator_type.fillna("?") + " → " + a.respondent_type.fillna("?")
    s.table(share_table(a, "initiator_type"), "t3_initiator", "Table 3. Initiating party by stream")
    s.table(share_table(a, "respondent_type"), "t4_respondent", "Table 4. Responding party by stream")
    st = a.groupby("stream").state_party.apply(lambda x: f"{(x == 'Y').mean():.0%}").reindex(STREAMS)
    s.text(f"State party involved: {st.to_dict()} (all: {(a.state_party == 'Y').mean():.0%}). "
           f"Foreign party: {(a.foreign_party == 'Y').mean():.0%} (low-agreement field; indicative only).")
    con = a[a.stream.isin(["construction", "both"])]
    s.table(share_table(con, "project_type"), "t5_project", "Table 5. Project type (construction-related decisions)")
    s.table(share_table(con, "contract_form"), "t6_contract_form", "Table 6. Contract form (construction-related decisions)",
            note="Contract form has low agreement with the gold labels (κ≈0.3); standard forms are rarely named in judgments.")
    s.table(share_table(a, "outcome"), "t7_outcome", "Table 7. Outcome for the initiating party by stream")

    # Figure 1: decisions per year by stream
    yr = pd.crosstab(a.year, a.stream).reindex(columns=STREAMS, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bottom = np.zeros(len(yr))
    for st_ in STREAMS:
        ax.bar(yr.index, yr[st_], bottom=bottom, color=C[st_], label=st_, width=0.75)
        bottom += yr[st_].values
    ax.set_ylabel("Decisions")
    ax.set_xticks(yr.index)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    s.style(ax)
    s.fig(fig, "fig1_decisions_by_year.png",
          "Decisions per year by stream, 2015–2026 (2026 to 16 November). Raw counts track ULII's growing coverage; "
          "rates per 1,000 ULII judgments are reported in Stage 3.")

    # Figure 2: cause prevalence by stream (construction-related)
    prev = pd.DataFrame({st_: (con[con.stream == st_][CAUSES] == "Y").mean() for st_ in ["construction", "both"]})
    prev = prev.rename(index=CAUSE_LABEL).sort_values("construction")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    y = np.arange(len(prev))
    ax.barh(y - 0.2, 100 * prev["construction"], 0.4, color=C["construction"], label="construction")
    ax.barh(y + 0.2, 100 * prev["both"], 0.4, color=C["both"], label="both (via arbitration)")
    ax.set_yticks(y, prev.index)
    ax.set_xlabel("% of decisions in which the cause is in issue")
    ax.legend(frameon=False, loc="lower right")
    s.style(ax, grid="x")
    s.fig(fig, "fig2_causes_by_stream.png",
          "Share of construction-related decisions in which each dispute cause is in issue (several causes may apply). "
          "Causes are coded from the framed issues and the reasoning.")
    s.table((100 * prev).round(1), "t8_cause_prevalence", "Table 8. Cause prevalence (%) in construction-related decisions")

    # Figure 3: outcome mix by stream
    order = ["initiator succeeded", "partly succeeded", "initiator failed", "struck out (procedural)",
             "referred to arbitration", "remitted or retrial", "consent"]
    oc = pd.crosstab(a.stream, a.outcome, normalize="index").reindex(index=STREAMS, columns=order, fill_value=0)
    cols = ["#1baf7a", "#8fd3b6", "#eb6834", "#f2a97f", "#2a78d6", "#a7a6a2", "#d9d8d4"]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    left = np.zeros(len(oc))
    for k, col in zip(order, cols):
        ax.barh(oc.index, 100 * oc[k], left=left, color=col, label=k)
        left += 100 * oc[k].values
    ax.set_xlabel("% of decisions with a coded outcome")
    ax.legend(frameon=False, ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    s.style(ax, grid=None)
    s.fig(fig, "fig3_outcomes_by_stream.png",
          "Outcome for the initiating party (plaintiff, applicant or appellant) by stream. Outcome agreement with the gold "
          "labels is moderate (κ≈0.57); decisions without a coded outcome are excluded.")

    # arbitration application mix
    arb = a[a.stream.isin(["arbitration", "both"])]
    s.table(share_table(arb, "arb_application_type"), "t9_arb_type", "Table 9. Arbitration application type")
    s.write()


if __name__ == "__main__":
    main()
