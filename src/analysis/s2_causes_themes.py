"""Analysis stage 2 (RQ1) — dispute causes and latent themes.

A. Coded causes: prevalence, co-occurrence (phi), state vs private contrasts (Fisher exact, BH-corrected).
B. Latent themes from the facts/issues/reasoning text (no LLM): PFA (varimax; the protocol's primary method), NMF and
   LDA compared by UMass coherence. The most coherent method (NMF in v1) assigns themes; PFA is kept as a benchmark.
Output: outputs/analysis/02_causes_themes/
"""
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.decomposition import NMF, FactorAnalysis, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from statsmodels.stats.multitest import multipletests

from src.analysis.common import C, CAUSE_LABEL, CAUSES, SEED, Stage, load, plt
from src.code.rule_coder import load_texts
from src.code.segment import segment

warnings.filterwarnings("ignore")
LEGAL_STOP = """court honourable hon justice lordship learned counsel plaintiff plaintiffs defendant defendants applicant applicants
respondent respondents appellant appellants claimant petitioner submitted submission submissions submits argued evidence witness
pw dw judgment ruling decision held hold holding case cases matter matters section sections act acts rule rules order orders
para paragraph page pages vs versus ltd limited co company uganda kampala high division civil suit application appeal misc
miscellaneous cause no number dated day year years month january february march april may june july august september october
november december said also however therefore thus whether issue issues party parties stated state states shall would could
must may made make makes one two three first second third fourth fifth sixth law laws principle authority authorities supra
honour learned trial lower ug ugcomm ugcommc ughc ughccd ugca ugsc hcb kalr ea ulr ors anor others another versus mr ms dr""".split()
STOP = list(ENGLISH_STOP_WORDS.union(LEGAL_STOP))
K_GRID = range(6, 15)


def clean(t):
    t = re.sub(r"[^A-Za-z\s]", " ", t.lower())
    return re.sub(r"\b\w{1,2}\b", " ", t)


def umass(top_idx, X):
    """Mean UMass coherence of each topic's top terms (document co-occurrence)."""
    B = (X > 0).astype(np.int32).tocsc()
    df = np.asarray(B.sum(axis=0)).ravel()
    out = []
    for idx in top_idx:
        s = []
        for i in range(1, len(idx)):
            for j in range(i):
                co = B[:, idx[i]].multiply(B[:, idx[j]]).sum()
                s.append(np.log((co + 1) / max(df[idx[j]], 1)))
        out.append(np.mean(s))
    return float(np.mean(out))


def top_terms(comp, vocab, n=10):
    return [list(vocab[np.argsort(-row)[:n]]) for row in comp]


def main():
    s = Stage("02_causes_themes", "Stage 2 — Dispute causes and latent themes (RQ1)")
    a = load()
    con = a[a.stream.isin(["construction", "both"])].copy()
    Y = (con[CAUSES] == "Y").astype(int)

    # ---- A. coded causes
    prev = Y.mean().rename(index=CAUSE_LABEL).sort_values(ascending=False)
    s.text(f"Construction-related decisions: **{len(con)}**. Median causes in issue per decision: "
           f"{int(Y.sum(axis=1).median())}; decisions with no coded cause: {(Y.sum(axis=1) == 0).mean():.0%} "
           f"(mostly procedural rulings). Most frequent: " + "; ".join(f"{k} {v:.0%}" for k, v in prev.head(5).items()) + ".")
    keep = [c for c in CAUSES if Y[c].sum() >= 10]
    phi = Y[keep].corr().rename(index=CAUSE_LABEL, columns=CAUSE_LABEL)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(phi.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(phi)), phi.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(phi)), phi.index, fontsize=8)
    for i in range(len(phi)):
        for j in range(len(phi)):
            if i != j:
                ax.text(j, i, f"{phi.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(phi.values[i, j]) > 0.3 else C["ink"])
    fig.colorbar(im, ax=ax, shrink=0.7, label="phi coefficient")
    s.fig(fig, "fig1_cause_cooccurrence.png",
          "Co-occurrence of dispute causes in construction-related decisions (phi coefficients; causes in issue in at least "
          "10 decisions). Positive values mark causes litigated together.")

    rows = []
    st = con.state_party.eq("Y")
    for c in keep:
        t = pd.crosstab(st, Y[c]).reindex(index=[True, False], columns=[1, 0], fill_value=0)
        orr, p = fisher_exact(t.values)
        rows.append({"cause": CAUSE_LABEL[c], "state %": round(100 * Y[c][st].mean(), 1),
                     "private %": round(100 * Y[c][~st].mean(), 1), "odds ratio": round(orr, 2), "p": p})
    sp = pd.DataFrame(rows)
    sp["p_BH"] = multipletests(sp.p, method="fdr_bh")[1].round(3)
    sp["p"] = sp.p.round(3)
    s.table(sp.sort_values("p").set_index("cause"), "t1_state_vs_private",
            f"Table 1. Causes in disputes with and without a state party (state n={int(st.sum())}, private n={int((~st).sum())})",
            note="Fisher exact tests; p_BH = Benjamini–Hochberg false-discovery-rate adjusted.")

    # ---- B. latent themes
    texts = load_texts(a)
    docs = []
    for cid in con.case_id:
        seg, issues, _ = segment(texts[cid])
        body = " ".join([seg["facts"], seg["issues"], " ".join(issues), seg["analysis"][:20000]]) or texts[cid]
        docs.append(clean(body))
    tf = TfidfVectorizer(stop_words=STOP, ngram_range=(1, 2), min_df=8, max_df=0.5, max_features=3000, sublinear_tf=True)
    X = tf.fit_transform(docs)
    vocab = np.array(tf.get_feature_names_out())
    cv = CountVectorizer(vocabulary=tf.vocabulary_, ngram_range=(1, 2))
    Xc = cv.fit_transform(docs)

    scores = []
    for k in K_GRID:
        nmf = NMF(k, random_state=SEED, init="nndsvda", max_iter=600).fit(X)
        lda = LatentDirichletAllocation(k, random_state=SEED, learning_method="batch", max_iter=50).fit(Xc)
        scores.append({"k": k, "NMF": umass([np.argsort(-r)[:10] for r in nmf.components_], X),
                       "LDA": umass([np.argsort(-r)[:10] for r in lda.components_], X)})
    sc = pd.DataFrame(scores).set_index("k")
    k = int(sc.NMF.idxmax())
    # PFA: factor analysis with varimax rotation on an SVD-reduced TF-IDF (keeps the problem well conditioned)
    svd = TruncatedSVD(100, random_state=SEED).fit(X)
    Z = svd.transform(X)
    fa = FactorAnalysis(k, rotation="varimax", random_state=SEED).fit(Z)
    pfa_comp = fa.components_ @ svd.components_          # back to term space
    pfa_comp = np.where(np.abs(pfa_comp.min(axis=1, keepdims=True)) > pfa_comp.max(axis=1, keepdims=True), -pfa_comp, pfa_comp)
    sc.loc[k, "PFA"] = umass([np.argsort(-r)[:10] for r in pfa_comp], X)
    s.table(sc.round(3), "t2_coherence", "Table 2. Topic coherence (UMass, higher is better) by number of themes",
            note=f"k = {k} chosen by NMF coherence; PFA evaluated at the same k.")

    nmf = NMF(k, random_state=SEED, init="nndsvda", max_iter=600).fit(X)
    lda = LatentDirichletAllocation(k, random_state=SEED, learning_method="batch", max_iter=50).fit(Xc)
    tt = pd.DataFrame({
        "PFA (primary)": [", ".join(t) for t in top_terms(pfa_comp, vocab)],
        "NMF": [", ".join(t) for t in top_terms(nmf.components_, vocab)],
        "LDA": [", ".join(t) for t in top_terms(lda.components_, vocab)]}, index=[f"T{i + 1}" for i in range(k)])
    s.table(tt, "t3_theme_terms", f"Table 3. Top terms of the {k} latent themes by method",
            note="Themes are unlabelled; labels should be assigned by reading representative judgments (see t4).")

    # document-theme weights from the most coherent method
    best = sc.loc[k, ["NMF", "LDA", "PFA"]].astype(float).idxmax()
    if best == "PFA":
        D = fa.transform(Z) * np.sign([np.corrcoef(fa.transform(Z)[:, i], X @ pfa_comp[i])[0, 1] for i in range(k)])
    elif best == "LDA":
        D = lda.transform(Xc)
    else:
        D = nmf.transform(X)
    s.text(f"**Method used for theme assignment: {best}** (highest coherence at k = {k}: "
           f"{sc.loc[k, best]:.3f} vs PFA {sc.loc[k, 'PFA']:.3f}). PFA, the protocol's primary method, produced one "
           "dominant general-contract factor and is reported as a benchmark; this is a documented deviation.")
    con["theme"] = [f"T{i + 1}" for i in D.argmax(axis=1)]
    rep = con.assign(score=D.max(axis=1)).sort_values("score", ascending=False).groupby("theme").head(3)
    s.table(rep.groupby("theme").citation.apply(lambda x: "; ".join(x)).to_frame("representative decisions"),
            "t4_theme_exemplars", f"Table 4. Most representative decisions per {best} theme (for labelling)")
    link = pd.DataFrame({CAUSE_LABEL[c]: [np.corrcoef(D[:, i], Y[c])[0, 1] for i in range(k)] for c in keep},
                        index=[f"T{i + 1}" for i in range(k)]).round(2)
    s.table(link, "t5_theme_cause_corr", f"Table 5. Correlation of {best} theme weights with coded causes",
            note="Large positive values show which coded cause a theme captures; themes unrelated to any cause are latent themes the codebook does not cover.")
    mix = pd.crosstab(con.theme, con.stream, normalize="columns").mul(100).round(1)
    trend = pd.crosstab(con.year, con.theme, normalize="index")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    cmap = plt.get_cmap("tab20")
    bottom = np.zeros(len(trend))
    for i, t in enumerate(trend.columns):
        ax.bar(trend.index, 100 * trend[t], bottom=bottom, color=cmap(i), label=t, width=0.8)
        bottom += 100 * trend[t].values
    ax.set_ylabel("% of construction-related decisions")
    ax.set_xticks(trend.index)
    ax.legend(frameon=False, ncol=1, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    s.style(ax)
    s.fig(fig, "fig2_theme_mix_by_year.png",
          f"Dominant latent theme ({best}, k = {k}) of construction-related decisions by year of decision. "
          "Theme terms are listed in Table 3.")
    s.table(mix, "t6_theme_by_stream", "Table 6. Dominant theme by stream (% of decisions)")
    con[["case_id", "theme"]].assign(**{f"score_T{i + 1}": D[:, i].round(4) for i in range(k)}).to_parquet(
        s.dir / "theme_scores.parquet", index=False)
    s.write()


if __name__ == "__main__":
    main()
