"""Stage 1.4c — segment each judgment into facts / issues / analysis / holding by its headings.

Rule-based: finds heading lines (BACKGROUND, BRIEF FACTS, ISSUES, RESOLUTION, ANALYSIS, DECISION, ORDERS...).
Falls back to position (first 25% facts, last 15% holding) when headings are absent.
Output: data/interim/segments.parquet (case_id, facts, issues, analysis, holding, issue_list, method)
"""
import re

import pandas as pd

from src.config import INTERIM

H = r"(?im)^\s*(?:[A-Z]\.|\d+\.|[ivx]+\.)?\s*"
HEADS = {
    "facts": H + r"(?:brief\s+)?(?:background|facts|the facts|brief facts|facts of the case|introduction|the claim|the plaintiff'?s case|summary of (?:the )?(?:facts|evidence))\b[^\n]{0,40}$",
    "issues": H + r"(?:the\s+)?(?:issues?|agreed issues|issues for (?:determination|resolution)|grounds of (?:appeal|the application))\b[^\n]{0,40}$",
    "analysis": H + r"(?:resolution|analysis|determination|consideration|discussion|decision of (?:the )?court|court'?s (?:analysis|decision|findings)|issue (?:no\.?\s*)?(?:1|one))\b[^\n]{0,50}$",
    "holding": H + r"(?:conclusion|orders?|final orders?|decision|disposition|result|in the result|remed(?:y|ies))\b[^\n]{0,30}$",
}


def segment(text):
    t = text or ""
    marks = []
    for part, rx in HEADS.items():
        for m in re.finditer(rx, t):
            marks.append((m.start(), part))
    marks.sort()
    # keep the first occurrence of each part, in document order
    seen, cuts = set(), []
    for pos, part in marks:
        if part not in seen:
            seen.add(part)
            cuts.append((pos, part))
    out = {k: "" for k in HEADS}
    if len(cuts) >= 2:
        cuts.append((len(t), None))
        for (a, part), (b, _) in zip(cuts, cuts[1:]):
            out[part] = t[a:b].strip()
        method = "headings"
    else:
        n = len(t)
        out["facts"], out["analysis"], out["holding"] = t[: n // 4], t[n // 4: int(n * .85)], t[int(n * .85):]
        method = "position"
    if not out["holding"]:
        out["holding"] = t[int(len(t) * .85):]
    items = re.findall(r"(?m)^\s*(?:\d+[.)]|[ivx]+[.)]|\([a-z0-9]+\))\s+((?:whether|what|is|was|did|does|if)\b[^\n]{10,250})",
                       out["issues"] or t[: len(t) // 2], flags=re.I)
    return out, [i.strip() for i in items[:12]], method


def main():
    texts = pd.read_parquet(INTERIM / "texts.parquet")
    rows = []
    for r in texts.itertuples():
        seg, items, method = segment(r.text)
        rows.append({"case_id": r.case_id, **seg, "issue_list": items, "n_issues": len(items), "method": method})
    s = pd.DataFrame(rows)
    s.to_parquet(INTERIM / "segments.parquet", index=False)
    print(s.method.value_counts().to_string())
    print("issues found:", (s.n_issues > 0).mean().round(2), "| holding non-empty:", (s.holding.str.len() > 0).mean().round(2))
    for k in HEADS:
        print(k, "found by heading:", ((s.method == "headings") & (s[k].str.len() > 0)).mean().round(2))


if __name__ == "__main__":
    main()
