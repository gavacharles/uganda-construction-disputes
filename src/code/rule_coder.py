"""Stage 2.3 — rule-based coding of every codebook field from the full judgment text (no LLM).

Transparent regex rules over the segmented text (facts / issues / analysis / holding). Codes use the
codebook / gold-sheet vocabulary so src.code.validate can score them against the manual gold set.
Inputs : data/processed/cases.parquet, data/interim/{texts,ocr_texts,legislation}.parquet
Outputs: data/processed/rule_codes.parquet   one row per case, codebook fields (+ evidence columns)
         data/processed/cases.parquet        schema columns filled (contract_form, dispute_causes, ...)
"""
import json
import re

import pandas as pd

from src.code.segment import segment
from src.config import INTERIM, OUT, PROCESSED
from src.extract.fulltext import CLAUSE, FORMS, INTEREST, features, money
from src.ingest.registers import case_aliases

F = re.I

# ---------- C. parties ----------
STATE = (r"attorney general|\bUNRA\b|national roads authority|ministry|local government|district (?:local )?(?:council|government)|"
         r"municipal|city council|\bKCCA\b|kampala capital city|town council|\bauthority\b|commission|national water|\bNWSC\b|"
         r"government|uganda electricity|\bUEGCL\b|\bUETCL\b|\bUEDCL\b|rural electrification|\bPPDA\b|uganda revenue|"
         r"\bURA\b|university|parliament|police|prisons|national housing|\bNHCC\b|uganda railways|civil aviation|"
         r"\bUPDF\b|national (?:social security|medical stores|forestry|agricultural)|\bNSSF\b|uganda national|"
         r"secretary to the treasury|permanent secretary|registered trustees of .* (?:hospital|school)")
BANK = r"\bbank\b|insurance|assurance|\bfinance\b|microfinance|\bsacco\b"
CONSULT = r"consult|architect|surveyor|associates|engineers? (?:&|and) (?:planners|consultants)|planners|design(?:ers)?\b"
SUPPLY = r"suppl(?:y|ies|iers?)|trading|traders|hardware|cement|steel|motors|equipment|machinery|stores|distributors|logistics"
CONTRACTOR = r"construction|contractors?|builders?|building|engineering|civil works|roads? (?:works|construction)|works\b|infrastructure"
COMPANY = r"\bltd\b|limited|company|\bco\.?\b|corporation|\bplc\b|\binc\b|enterprises|group|holdings|\bllc\b|\(pty\)|\bs\.?a\.?\b|joint venture|\bjv\b|trustees|\bgmbh\b"
FOREIGN = (r"china|chinese|kenya|tanzania|rwanda|burundi|south sudan|congo|ethiopia|egypt|india|turk|italia|japan|korea|"
           r"\(pty\)|\bgmbh\b|\bs\.?p\.?a\b|\bs\.?a\.?r\.?l\b|\bllc\b|\binc\b|overseas|\bbv\b|\bag\b|\bspa\b|dubai|emirates|\buae\b|"
           r"united kingdom|\bUK\b|mauritius|south africa|israel|portugal|spain|france|germany|nigeria|ghana")


def party_type(side):
    s = str(side or "").strip()
    if len(s) < 3 or s.lower() in ("none", "nan"):
        return None
    if re.search(STATE, s, F):
        return "state body"
    if re.search(BANK, s, F):
        return "bank or insurer"
    if not re.search(COMPANY + "|" + CONTRACTOR + "|" + CONSULT, s, F):
        return "individual"
    if re.search(CONSULT, s, F):
        return "consultant"
    if re.search(CONTRACTOR, s, F):
        return "contractor"
    if re.search(SUPPLY, s, F):
        return "supplier"
    return "private employer"


ROLE_I = r"(?:plaintiff|applicant|appellant|claimant|petitioner)s?"
ROLE_R = r"(?:defendant|respondent)s?"


def role_from_text(text, role):
    """'the plaintiff was the contractor' / 'engaged the defendant to construct' -> contractor / subcontractor / employer."""
    t = text[:20000]
    gap = r"\W+(?:\w+\W+){0,8}?"
    cues = {
        "subcontractor": rf"\bthe {role}{gap}(?:was|is|as|being) (?:the |a |an )?sub-?contractor",
        "contractor": rf"\bthe {role}{gap}(?:was|is|as|being) (?:the |a |an )?(?:main |principal )?contractor\b|"
                      rf"(?:engaged|contracted|hired|awarded|appointed|retained) the {role}\W+(?:\w+\W+){{0,4}}?to (?:construct|build|carry out|execute|undertake|rehabilitate|renovate|erect|complete)",
        "consultant": rf"\bthe {role}{gap}(?:was|is|as|being) (?:the |a |an )?(?:project |supervising |design )?(?:consultant|architect|engineer|quantity surveyor)\b",
        "private employer": rf"\bthe {role}{gap}(?:was|is|as|being) (?:the )?(?:employer|client|developer|owner)\b",
    }
    n = {k: len(re.findall(rx, t, F)) for k, rx in cues.items()}
    k, v = max(n.items(), key=lambda kv: kv[1])
    return k if v else None


def sides(parties):
    s = re.split(r"\s+(?:v|vs|versus)\.?\s+", "" if pd.isna(parties) else str(parties), maxsplit=1, flags=F)
    return (s + [""])[:2]


# ---------- B. proceeding ----------
def proceeding(court, head, case_number):
    own = "\n".join(l for l in head[:1500].split("\n") if not re.search(r"aris(?:ing|es)|from", l, F))
    h = f"{case_number or ''} {own}"
    if re.search(r"PPDA|Appeals Tribunal|CADER", str(court)):
        return "tribunal review", "tribunal decision"
    if re.search(r"judicial review|certiorari|prerogative", h, F):
        p = "judicial review"
    elif re.search(r"arbitration cause|arb\.? cause|\bCAD/ARB\b", h, F):
        p = "arbitration cause"
    elif re.search(r"(?:civil|misc\w*\.?) appeal|appeal no|\bSCCA\b|\bCACA\b", h[:600], F) and not re.search(r"civil application|misc\w*\.? application", h[:300], F):
        p = "appeal"
    elif re.search(r"misc\w*\.?\s*(?:civil\s+)?(?:application|cause)|civil application|\bcause no|-M[AC]-|company cause|originating summons", h, F):
        p = "application"
    elif re.search(r"civil suit|\bH\.?C\.?C\.?S\b|-CS-|suit no", h, F):
        p = "suit"
    else:
        p = "other"
    if p == "appeal":
        d = "appeal decision"
    elif p == "suit" or re.search(r"^\W*judg(?:e)?ment\W*$", head, re.M | F) and p != "application":
        d = "final judgment"
    else:
        d = "ruling on application"
    return p, d


# ---------- D. project and contract ----------
PROJECT = {
    "road": r"\broads?\b|highway|carriageway|tarmac|bitumen|asphalt|gravel(?:ling)?|murram|pavement|bypass|expressway",
    "bridge or other civil": r"\bbridges?\b|culverts?|\bdams?\b|embankment|railway|airport|runway|\bport\b|landing site|irrigation",
    "water and sanitation": r"water supply|water works|piped water|sewer|sewage|sanitation|borehole|drainage|\bNWSC\b|reservoir|water treatment",
    "energy and power": r"power (?:line|station|plant|project)|electricity|electrification|transmission line|substation|solar|hydro\s?power|grid|\bMW\b|\bkV\b",
    "building": r"\bbuildings?\b|\bhouse\b|residential|apartments?|\bschool\b|classrooms?|hospital|health cent(?:re|er)|office block|hostel|"
                r"renovation|\bmarket\b|warehouse|storeyed|church|mosque|hotel|\bblock\b|dormitor|staff houses?|plaza|arcade|mall",
    "supply-and-install": r"suppl(?:y|ied),? (?:delivery,? )?(?:and )?install|supply and installation|installation of (?:equipment|machinery|lifts|cctv|generators?)",
}
FORM_MAP = {"FIDIC 2017": "FIDIC 2017", "FIDIC 1999": "FIDIC 1999", "FIDIC (unspecified)": "FIDIC other",
            "PPDA SBD": "PPDA SBD", "JBC/PAM": "JBC or PAM"}
WRITTEN = r"(?:entered into|executed|signed|concluded)\s+(?:a|an|the)\s+(?:written\s+)?(?:\w+\s+){0,3}(?:contract|agreement)|written (?:contract|agreement)|contract (?:agreement )?dated|agreement dated"
SBD_CUES = r"general conditions of contract|special conditions of contract|\bGCC\b|\bSCC\b|standard bidding|bid(?:ding)? document"
ORAL = r"oral(?:ly)? (?:agreement|contract)|verbal (?:agreement|contract)|no written (?:contract|agreement)|orally agreed"


def project_type(text):
    scores = {k: len(re.findall(rx, text, F)) for k, rx in PROJECT.items()}
    k, v = max(scores.items(), key=lambda kv: kv[1])
    return k if v >= 2 else "not stated"


def contract_form(text, forms, state):
    for f in ["FIDIC 2017", "FIDIC 1999", "FIDIC (unspecified)", "PPDA SBD", "JBC/PAM"]:
        if f in forms:
            return FORM_MAP[f]
    if state and re.search(SBD_CUES, text, F):
        return "PPDA SBD"
    if re.search(ORAL, text, F):
        return "oral or no written contract"
    if re.search(WRITTEN, text, F):
        return "bespoke written"
    return "not stated"


def contract_value(text):
    for m in re.finditer(r"contract (?:sum|price|value|amount)|(?:total|lump) sum (?:contract )?price", text, F):
        vals = [a for c, a, _ in money(text[m.start(): m.start() + 200]) if c == "UGX"]
        if vals:
            return vals[0]
    return None


# ---------- E. causes (issues framed > analysis mentions) ----------
CAUSES = {
    "c_payment": r"non.?payment|unpaid|interim (?:payment )?certificates?|certificates? of payment|payment certificate|certif(?:y|ied|ication) (?:of|for) (?:payment|works)|outstanding (?:sum|balance|amount)|sum of .{0,40} due|failed to pay|recovery of (?:ugx|shs|usd|\$|the sum)",
    "c_final_account": r"final account|final (?:payment )?certificate|re-?measure|measurement|valuation of (?:the )?works|bills? of quantities|\bBOQ\b",
    "c_variations": r"variation|additional works?|extra works?|change (?:in|of) scope|scope of works? (?:was|were) (?:changed|varied|increased)|variation orders?",
    "c_delay_eot": r"\bdelay(?:ed|s)?\b|extension of time|\bEOT\b|completion (?:date|period)|programme of works|time for completion",
    "c_lds": r"liquidated damages|\bLDs?\b|penalt(?:y|ies) for delay|damages for delay",
    "c_termination": r"\bterminat(?:e|ed|ing|ion)|\babandon(?:ed|ment)|repudiat|\bfrustrat|rescind|rescission",  # \b: not de-termination
    "c_defects": r"defects?\b|defective|workmanship|substandard|sub-standard|collaps|cracks?\b|poor quality|negligen\w* (?:in|construction|design)|defects liability",
    "c_guarantees": r"(?:performance|advance payment|bid|bank) (?:guarantee|security|bond)|\bguarantee\b|\bbonds?\b|retention (?:money|sum)?|\bURDG\b|call on the (?:guarantee|bond)",
    "c_procurement": r"procurement|bid(?:der|ding)? evaluation|best evaluated bidder|\btender(?:s|ing|er|ers)?\b(?! (?:in|as) evidence)|administrative review|accounting officer|\bPPDA\b|bid security|award of (?:the )?contract",
    "c_site_land": r"trespass|encroach|nuisance|site possession|possession of (?:the )?site|access to (?:the )?site|hand(?:ed|ing)? over (?:of )?(?:the )?site|compensation for (?:land|crops|property)|right of way",
    "c_prof_fees": r"professional fees|consultan(?:cy|t'?s?) fees|architect'?s? fees|engineer'?s? fees|quantity surveyor'?s? fees|supervision (?:fees|consultant)|design fees",
    "c_price_adjust": r"price adjustment|price fluctuation|fluctuation|exchange rate|escalation|currency (?:loss|depreciation)",
    "c_force_majeure": r"force majeure|covid|corona|pandemic|lockdown|unforeseen (?:physical )?conditions|unforeseeable|act of god",
    "c_fraud": r"\bfraud|corrupt|forg(?:ed|ery)|bribe|collusion|misrepresent",
}

_TH = OUT / "coding" / "cause_thresholds.json"
CAUSE_MIN = json.loads(_TH.read_text()) if _TH.exists() else {}   # {cause: {"issues": bool, "body": int}}

# ---------- F. procedure ----------
NOTICE = (r"notice of (?:claim|dispute|intention to claim)|(?:sub-)?clause 20\.1\b|contractual (?:notice|time.?bar)|"
          r"fail(?:ed|ure) to (?:give|issue|serve) (?:the )?(?:requisite |required |contractual )?notice (?:of (?:the )?claim|under (?:the )?(?:contract|clause))")
NOTICE_UPHELD = (r"(?:claim|claims) (?:is|are|was|were|stands?) (?:time.?barred|barred)|"
                 r"fail(?:ed|ure) to (?:give|issue|serve) (?:the )?(?:requisite |required )?notice[^.]{0,80}(?:fatal|bar|defeat|cannot)")
TIER = (r"amicable settlement|dispute adjudication board|\bDAB\b|\bDAAB\b|adjudicator|engineer'?s decision|"
        r"refer(?:red)? (?:the (?:dispute|matter) )?to the engineer|condition precedent to (?:arbitration|litigation|filing)|"
        r"premature(?:ly)?|(?:sub-)?clause 67\b|(?:sub-)?clause 20\.[2-8]\b|multi.?tier|escalation clause|mediation (?:first|before)")
LIMITATION = r"limitation act|statute.?barred|barred by limitation|limitation period|section 3 of the limitation|out of time under the limitation"

# ---------- G. outcome ----------
SUBJ = r"(?:suit|application|appeal|claim|petition|cause|reference|plaintiff'?s case|motion|complaint|applicant'?s case)"  # not objection: usually the respondent's
DISP = [
    ("consent", r"consent (?:judgment|order|decree)|by consent of (?:the )?parties|parties (?:have )?(?:settled|reached a settlement)|withdrawn by consent"),
    ("referred to arbitration", r"referred to arbitration|(?:proceedings|suit) (?:is|are|be) (?:hereby )?stayed|stay(?:ed)? (?:of )?(?:the )?(?:suit|proceedings) pending arbitration|parties (?:are|be) referred to arbitration"),
    ("remitted or retrial", r"remitted|retrial|re-?trial|sent back to (?:the )?(?:trial|lower)|fresh hearing"),
    ("partly succeeded", rf"{SUBJ}[^.]{{0,60}}(?:partly|partially|in part)|(?:partly|partially) (?:allowed|granted|succeeds?)|allowed in part|granted in part|succeeds? in part"),
    ("struck out (procedural)", rf"{SUBJ}[^.]{{0,60}}(?:struck out|is incompetent|strike[sd]? out)|(?:hereby|is|be) struck out"),
    ("initiator failed", rf"{SUBJ}[^.]{{0,80}}\b(?:dismissed|refused|rejected|fails|failed|lacks merit|devoid of merit|declined|not proved|disallowed)\b|"
                         r"(?:hereby|is|be) dismissed|judgment (?:is )?(?:hereby )?entered for the (?:defendant|respondent)"),
    ("initiator succeeded", rf"{SUBJ}[^.]{{0,80}}\b(?:allowed|granted|succeeds|upheld|proved|merited)\b|"
                            r"judgment (?:is )?(?:hereby )?entered (?:for|in favour of) the (?:plaintiff|claimant|applicant)|"
                            r"award (?:is|be) (?:hereby )?(?:set aside|recogni[sz]ed|enforced|registered)|(?:hereby|is) set aside|"
                            r"(?:defendant|respondent)s? (?:is|are|shall)[^.]{0,30}(?:ordered|liable) to pay"),
]


def outcome(holding):
    h = re.sub(r"\s+", " ", holding)
    # 'objection upheld' is the respondent's success: the initiator's case is struck out or dismissed
    if re.search(r"(?:preliminary )?objection[^.]{0,40}(?:is|are) (?:hereby )?(?:upheld|sustained)", h, F):
        return "struck out (procedural)"
    hits = {k: len(re.findall(rx, h, F)) for k, rx in DISP}
    for k in ["consent", "referred to arbitration", "remitted or retrial", "partly succeeded"]:
        if hits[k]:
            return k
    win, lose, so = hits["initiator succeeded"], hits["initiator failed"], hits["struck out (procedural)"]
    if win and (lose or so):
        # counterclaim dismissed with suit allowed is still a win; compare the last disposition stated
        last = max(((m.start(), k) for k, rx in DISP[3:] for m in re.finditer(rx, h, F)), default=(0, None))
        return last[1] if last[1] else "partly succeeded"
    if so:
        return "struck out (procedural)"
    if lose:
        return "initiator failed"
    if win:
        return "initiator succeeded"
    return None


def costs_to(holding, out):
    h = re.sub(r"\s+", " ", holding)
    if re.search(r"each party (?:shall |to |will )?(?:bear|meet) (?:its|their|his|her) own costs|no order as to costs|own costs", h, F):
        return "each party own"
    if re.search(r"costs (?:to|of|in favour of|are awarded to|shall be paid to) the (?:defendant|respondent)", h, F):
        return "respondent"
    if re.search(r"costs (?:to|of|in favour of|are awarded to|shall be paid to) the (?:plaintiff|applicant|appellant|claimant|petitioner)", h, F):
        return "initiator"
    if re.search(r"with costs|costs (?:of|to) (?:the )?(?:suit|application|appeal)|is awarded costs|pay (?:the )?costs", h, F):
        return {"initiator succeeded": "initiator", "initiator failed": "respondent",
                "struck out (procedural)": "respondent"}.get(out, "NA")
    return "NA"


def interest(holding):
    h = re.sub(r"\s+", " ", holding)
    rates = [float(r) for r in INTEREST.findall(h) if 0 < float(r) <= 40]
    if re.search(r"commercial rate", h, F):
        basis = "commercial"
    elif re.search(r"court rate", h, F):
        basis = "court rate"
    elif re.search(r"contractual rate|agreed rate|rate (?:agreed|provided|stipulated) (?:in|under) the (?:contract|agreement)", h, F):
        basis = "contractual"
    elif re.search(r"no (?:order (?:as to|for) )?interest|interest (?:is )?(?:declined|refused|denied)", h, F) or not re.search(r"interest", h, F):
        basis = "none"
    else:
        basis = "not stated"
    return (rates[0] if rates else None), basis


def quantum(text, facts, holding, out):
    claim_zone = facts or text[: len(text) // 3]
    claimed = [(c, a) for c, a, r in money(claim_zone) if r == "claimed"]
    awarded = [(c, a) for c, a, r in money(holding) if r == "awarded"]
    cur = None
    for lst in (awarded, claimed):
        if lst:
            cur = max(set(c for c, _ in lst), key=[c for c, _ in lst].count)
            break
    q_claim = max((a for c, a in claimed if c == cur), default=None)
    q_award = max((a for c, a in awarded if c == cur), default=None)
    if out in ("initiator failed", "struck out (procedural)") and q_claim is not None:
        q_award = 0.0
    return cur or "NA", q_claim, q_award


FILED = re.compile(r"(?:filed|instituted|lodged|commenced)[^.]{0,60}?on\s+(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(?:day of\s+)?"
                   r"(January|February|March|April|May|June|July|August|September|October|November|December),?\s+((?:19|20)\d{2})", F)


def filing_date(facts):
    m = FILED.search(facts)
    if not m:
        return None
    d = pd.to_datetime(f"{m.group(1)} {m.group(2)} {m.group(3)}", errors="coerce", dayfirst=True)
    return None if pd.isna(d) else d.strftime("%d/%m/%Y")


# ---------- H. arbitration ----------
ARB = [  # (code, regex), scored on title block + issues; first-listed wins ties
    ("set-aside", r"set(?:ting)? aside (?:the |an |of the )?(?:arbitral )?award|section 34\b|s\.\s?34\b"),
    ("enforcement or recognition", r"enforce(?:ment)? (?:of )?(?:the |an )?(?:arbitral |foreign )?award|recogni[sz]|registration of (?:the )?award|section 3[56]\b|new york convention"),
    ("stay or referral", r"stay (?:of )?(?:the )?(?:suit|proceedings)|refer(?:ral|red)? (?:the (?:matter|dispute|parties) )?to arbitration|section 5\b|s\.\s?5\b"),
    ("interim measures", r"interim (?:measures?|protection|relief|order)|section 6\b|s\.\s?6\b|injunction"),
    ("appointment", r"appoint(?:ment of)? (?:an? |the )?(?:sole )?arbitrators?|section 11\b|s\.\s?11\b"),
    ("jurisdiction", r"jurisdiction of the (?:arbitral )?tribunal|section 16\b|kompetenz"),
    ("extension of time", r"extension of time|enlarge(?:ment of)? time|leave to (?:file|appeal) out of time"),
]
GROUNDS = {
    "g_public_policy": r"public policy",
    "g_jurisdiction": r"beyond the scope|exceed(?:ed|s)? (?:its|his|her|their) (?:jurisdiction|mandate|powers)|ultra petita|outside the (?:scope|terms) of (?:the )?(?:submission|reference)|34\s?\(2\)\s?\(a\)\s?\((?:iv|v)\)",
    "g_natural_justice": r"natural justice|fair hearing|right to be heard|unable to present|bias(?:ed)?\b|34\s?\(2\)\s?\(a\)\s?\(iii\)",
    "g_out_of_time": r"out of time|outside the (?:statutory )?(?:time|period)|within one month|time.?barred|34\s?\(3\)",
    "g_arbitrability": r"arbitrab|not capable of settlement by arbitration|34\s?\(2\)\s?\(b\)\s?\(i\)",
    "g_invalid_agreement": r"(?:no|invalid|void) (?:valid )?arbitration (?:agreement|clause)|arbitration (?:agreement|clause) (?:is|was) (?:invalid|void|null|inoperative)|34\s?\(2\)\s?\(a\)\s?\(ii\)",
}
FOREIGN_SEAT = r"foreign (?:arbitral )?award|new york convention|\bICC\b|\bLCIA\b|london court of international|nairobi (?:centre|center)|\bNCIA\b|seat(?:ed)? (?:of (?:the )?arbitration )?(?:is |was |shall be |in )(?!kampala|uganda)[A-Z]\w+"


def arbitration(head_issues, analysis, holding, out):
    scores = {k: len(re.findall(rx, head_issues, F)) for k, rx in ARB}
    k, v = max(scores.items(), key=lambda kv: kv[1])
    atype = k if v else "other"
    challenge = atype in ("set-aside", "enforcement or recognition")
    zone = head_issues + (" " + analysis if challenge else "")   # grounds argued at length only matter in challenges
    g = {f: "Y" if re.search(rx, zone, F) else "N" for f, rx in GROUNDS.items()}
    h = re.sub(r"\s+", " ", holding)
    if atype == "set-aside":
        res = ("partly set aside" if re.search(r"set aside in part|partly set aside|partially set aside", h, F) else
               "remitted" if re.search(r"remit", h, F) else
               "set aside" if out == "initiator succeeded" else "upheld" if out in ("initiator failed", "struck out (procedural)") else "NA")
    elif atype == "enforcement or recognition":
        res = "enforced" if out == "initiator succeeded" else "enforcement refused" if out == "initiator failed" else "NA"
    else:
        res = "NA"
    defer = ("Y" if res in ("upheld", "enforced") or out == "referred to arbitration"
             or (atype == "stay or referral" and out == "initiator succeeded") else
             "N" if res in ("set aside", "partly set aside", "enforcement refused")
             or (atype == "stay or referral" and out == "initiator failed") else
             "Y" if atype in ("appointment", "interim measures", "extension of time") and out == "initiator succeeded" else
             "N" if atype in ("appointment", "interim measures", "extension of time") and out == "initiator failed" else "NA")
    return atype, g, res, defer


# ---------- legislation links ----------
def leg_patterns(leg):
    pats = {}
    for r in leg.itertuples():
        t = re.sub(r"\s*\(per [^)]*\)|\s*\(package\)", "", str(r.title))
        m = re.match(r"(.+?),?\s*((?:19|20)\d{2})$", t.strip())
        if not m:
            continue
        name, year = m.groups()
        name = re.escape(name).replace(r"\ ", r"\s+")
        pats[r.leg_id] = re.compile(rf"{name},?\s*(?:No\.?\s*\d+\s*of\s*)?{year}", F)
    return pats


# ---------- driver ----------
def load_texts(cases):
    t = pd.read_parquet(INTERIM / "texts.parquet", columns=["case_id", "text"])
    alias = case_aliases(cases)
    t["case_id"] = t.case_id.map(alias)
    t = t.dropna(subset=["case_id"]).drop_duplicates("case_id")
    o = pd.read_parquet(INTERIM / "ocr_texts.parquet")[["case_id", "text"]]
    t = pd.concat([t, o[~o.case_id.isin(t.case_id)]], ignore_index=True)
    return t.set_index("case_id").text


def code_case(c, text, legpats):
    seg, issue_list, _ = segment(text)
    head = text[:3000]
    facts, issues, analysis, holding = seg["facts"], seg["issues"], seg["analysis"], seg["holding"]
    if len(holding) < 300:
        holding = text[int(len(text) * .8):]
    issues_txt = issues + " " + " ".join(issue_list)
    fx = features(c.case_id, text, holding, str(c.case_number or ""))
    a, b = sides(c.parties)
    ta, tb = party_type(a), party_type(b)
    # name gives state / bank / individual reliably; for companies the text says who built and who employed
    if ta in ("private employer", "contractor", "supplier", "consultant"):
        ta = role_from_text(text, ROLE_I) or ta
    if tb in ("private employer", "contractor", "supplier", "consultant"):
        tb = role_from_text(text, ROLE_R) or tb
    proc, dec = proceeding(c.court, head, c.case_number)
    tail = text[int(len(text) * .75):]
    out = outcome(holding) or outcome(tail) or REG_OUTCOME.get(c.outcome_class)   # last resort: register summary
    cur, q_c, q_a = quantum(text, facts, holding, out)
    if cur == "NA":   # no claimed/awarded figure: dominant currency among all sums mentioned
        allc = [c_ for c_, _, _ in money(text)]
        cur = max(set(allc), key=allc.count) if allc else "NA"
    rate, basis = interest(holding)
    body = text[1500:]
    r = {"case_id": c.case_id, "include": "Y", "exclusion_reason": "NA", "stream": c.stream,
         "proceeding": proc, "decision_type": dec, "initiator_type": ta, "respondent_type": tb,
         "state_party": "Y" if "state body" in (ta, tb) else "N",
         "foreign_party": "Y" if re.search(FOREIGN, str(c.parties), F) and not re.search(r"\(U\)|uganda", a if re.search(FOREIGN, a, F) else b, F) else "N",
         "project_type": project_type(facts + " " + issues_txt if len(facts) > 500 else text[: len(text) // 2]),
         "contract_form": contract_form(text, fx["contract_forms"], "state body" in (ta, tb)), "contract_value_ugx": contract_value(text)}
    for k, rx in CAUSES.items():  # raised in the issues, or argued at length in the body
        r[f"n_issues:{k}"], r[f"n_body:{k}"] = len(re.findall(rx, issues_txt, F)), len(re.findall(rx, body, F))
        r[k] = "Y" if (r[f"n_issues:{k}"] and CAUSE_MIN.get(k, {}).get("issues", True)) or r[f"n_body:{k}"] >= CAUSE_MIN.get(k, {}).get("body", 3) else "N"
    r["notice_timebar_raised"] = "Y" if re.search(NOTICE, body, F) else "N"
    r["notice_timebar_upheld"] = ("Y" if re.search(NOTICE_UPHELD, analysis + holding, F) else "N") if r["notice_timebar_raised"] == "Y" else "NA"
    r["tier_skipped_raised"] = "Y" if re.search(TIER, body, F) else "N"
    r["limitation_raised"] = "Y" if re.search(LIMITATION, body, F) else "N"
    forms = [FORM_MAP[f] for f in fx["contract_forms"] if f in FORM_MAP]
    cl = sorted(set(CLAUSE.findall(body)), key=lambda x: [int(p) for p in x.split(".")])[:25]
    r["clauses_cited"] = (f"{forms[0] + ' ' if forms else ''}cl. " + "; ".join(cl)) if cl else None
    r.update({"outcome": out, "currency": cur, "quantum_claimed": q_c, "quantum_awarded": q_a,
              "interest_rate_pct": rate, "interest_basis": basis, "costs_to": (lambda x: x if x != "NA" else costs_to(tail, out))(costs_to(holding, out)),
              "filing_date": filing_date(facts or text[:6000])})
    if c.stream in ("arbitration", "both"):
        atype, g, res, defer = arbitration(head + " " + issues_txt, analysis, holding, out)
        r.update({"arb_application_type": atype, **g, "award_result": res, "deference": defer,
                  "seat_foreign": "Y" if re.search(FOREIGN_SEAT, text, F) else "N"})
    else:
        r.update({"arb_application_type": "NA", **{g: "NA" for g in GROUNDS}, "award_result": "NA",
                  "deference": "NA", "seat_foreign": "NA"})
    # evidence / schema extras
    r.update({"statutes_cited": fx["statutes"], "aca_sections": fx["aca_sections"],
              "precedents_cited": fx["cited_neutral"] + fx["cited_reports"],
              "leg_ids_cited": sorted(k for k, p in legpats.items() if p.search(text)),
              "segment_method": "headings" if seg["issues"] or seg["holding"] else "position",
              "holding_chars": len(holding)})
    return r


REG_OUTCOME = {"successful": "initiator succeeded", "unsuccessful": "initiator failed", "partial": "partly succeeded",
               "referred to arbitration": "referred to arbitration"}

SECTOR = {"road": "transport", "bridge or other civil": "transport/civil", "water and sanitation": "water",
          "energy and power": "energy", "building": "buildings", "supply-and-install": "supply", "not stated": None}


def main():
    cases = pd.read_parquet(PROCESSED / "cases.parquet")
    reg = pd.read_parquet(INTERIM / "registers_coded.parquet")[["case_id", "outcome_class"]]
    cases = cases.drop(columns=["outcome_class"], errors="ignore").merge(reg, on="case_id", how="left")
    texts = load_texts(cases)
    legpats = leg_patterns(pd.read_parquet(INTERIM / "legislation.parquet"))
    rows = [code_case(c, texts[c.case_id], legpats) for c in cases.itertuples() if c.case_id in texts.index]
    rc = pd.DataFrame(rows)
    missing = sorted(set(cases.case_id) - set(rc.case_id))
    rc.to_parquet(PROCESSED / "rule_codes.parquet", index=False)

    # fill the schema columns of cases.parquet (idempotent: drop then re-add)
    causes = [k for k in CAUSES]
    s = rc.set_index("case_id")
    schema = pd.DataFrame({
        "party_types": s.initiator_type + " v " + s.respondent_type,
        "project_type": s.project_type, "sector": s.project_type.map(SECTOR),
        "contract_form": s.contract_form, "clauses_cited": s.clauses_cited,
        "dispute_causes": s[causes].eq("Y").apply(lambda r: [c[2:] for c in causes if r[c]], axis=1),
        "outcome": s.outcome, "quantum_claimed": s.quantum_claimed, "quantum_awarded": s.quantum_awarded,
        "currency": s.currency, "precedents_cited": s.precedents_cited, "statutes_cited": s.statutes_cited,
        "leg_ids_cited": s.leg_ids_cited, "date_filed": pd.to_datetime(s.filing_date, format="%d/%m/%Y", errors="coerce"),
        "arb_application_type": s.arb_application_type.where(s.arb_application_type.ne("NA")),
        "arb_grounds": s[list(GROUNDS)].eq("Y").apply(lambda r: [g[2:] for g in GROUNDS if r[g]], axis=1),
    })
    cases = cases.drop(columns=[c for c in schema.columns if c in cases.columns] + ["outcome_class"]).merge(
        schema, left_on="case_id", right_index=True, how="left")
    cases.to_parquet(PROCESSED / "cases.parquet", index=False)

    print(f"coded {len(rc)} of {len(cases)} cases" + (f"; no text: {missing}" if missing else ""))
    for col in ["proceeding", "decision_type", "initiator_type", "project_type", "contract_form", "outcome",
                "currency", "costs_to", "interest_basis", "arb_application_type", "award_result", "deference"]:
        print(f"{col}: {rc[col].value_counts(dropna=False).to_dict()}")
    print("cause shares:", (rc[causes] == "Y").mean().round(2).to_dict())
    print("procedure shares:", {k: round((rc[k] == "Y").mean(), 2) for k in
                                ["notice_timebar_raised", "tier_skipped_raised", "limitation_raised", "state_party", "foreign_party"]})
    print(f"quantum claimed found: {rc.quantum_claimed.notna().sum()} | awarded: {rc.quantum_awarded.notna().sum()} | "
          f"interest rate: {rc.interest_rate_pct.notna().sum()} | filing date: {rc.filing_date.notna().sum()} | "
          f"clauses: {rc.clauses_cited.notna().sum()} | leg_ids linked: {(rc.leg_ids_cited.str.len() > 0).sum()}")


if __name__ == "__main__":
    main()
