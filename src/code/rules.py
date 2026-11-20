"""Transparent rule-based coding of register fields (no LLM). Validated later against the gold set."""
import re

import pandas as pd

ISSUES = {  # issue families: regex over key_issues + holding_summary + project type
    "payment & final accounts": r"payment|unpaid|final account|certificat|debt|invoice|sum due|quantum meruit",
    "delay, EOT & LDs": r"delay|extension of time|liquidated damages|\bEOT\b|completion",
    "termination & abandonment": r"terminat|abandon|repudiat|frustrat",
    "variations & scope": r"variation|additional work|extra work|scope",
    "defects & quality": r"defect|workmanship|collaps|negligen|substandard",
    "guarantees & bonds": r"guarantee|bond|URDG|retention|security for",
    "procurement & tender": r"procure|tender|bid|PPDA|evaluat",
    "land, site & nuisance": r"trespass|encroach|nuisance|site possession|give site|land",
    "arbitration agreement & referral": r"arbitration clause|refer|stay|kompetenz|s\.?\s?5\b|s\.?\s?6\b|arbitrab|jurisdiction",
    "award challenge & enforcement": r"set aside|s\.?\s?34|ss\.\s?.*34|enforce|registration of award|recogni|public policy|s\.?\s?3[56]",
    "tribunal appointment & conduct": r"appoint|arbitrator|adjudicator|s\.?\s?11\b|bias|natural justice|fair hearing",
    "execution & veil-lifting": r"execution|attach|garnish|veil",
    "limitation & time bars": r"time.?bar|limitation|out of time|time limit|notice",
}

GOV = (r"attorney general|\bUNRA\b|national roads|ministry|local government|district|municipal|"
       r"city council|\bKCCA\b|kampala capital|town council|authority|commission|national water|"
       r"\bNWSC\b|government|uganda electricity|\bUEGCL\b|\bUETCL\b|\bREA\b|rural electrification|"
       r"public procurement|\bPPDA\b|uganda revenue|\bURA\b|university|parliament|police|prisons|"
       r"national housing|\bNHCC\b|uganda railways|civil aviation")
BANK = r"\bbank\b|insurance|assurance|finance|microfinance"
COMPANY = r"\bltd\b|limited|company|\bco\.|corporation|\bplc\b|\binc\b|enterprises|contractors|construction|engineering|group|holdings|\bllc\b|\bS\.?A\.?\b"


def party_types(parties):
    p = str(parties)
    sides = re.split(r"\s+v\.?s?\.?\s+", p, maxsplit=1, flags=re.I)
    out = []
    for s in sides:
        if re.search(GOV, s, re.I):
            out.append("state")
        elif re.search(BANK, s, re.I):
            out.append("bank/insurer")
        elif re.search(COMPANY, s, re.I):
            out.append("company")
        else:
            out.append("individual")
    return out + ["unknown"] * (2 - len(out))


def proceeding(case_number):
    s = str(case_number).lower()
    if "appeal" in s or re.search(r"\bca\b", s):
        return "appeal"
    if "arbitration cause" in s or "arb" in s:
        return "arbitration cause"
    if "misc" in s or "cause" in s or "application" in s or re.search(r"-mc-|-ma-", s):
        return "application/cause"
    if "suit" in s or "cs" in s:
        return "suit"
    return "other"


SUCCESS = (r"granted|allowed|judgment for|damages|enforced|set aside|leave granted|"
           r"stay granted|quashed|registered|appointed|recogni|awarded|entered|upheld claim|"
           r"unpaid|outstanding sums|specific performance|refund|us\$|ugx|ordered|reinstated|"
           r"discharged|vacated|decided for|extended")
FAIL = r"dismiss|refused|struck out|overruled|upheld|rejected|declined|time-barred|failed|no case"
REFER = r"referred|stay(ed)? (pending|for) arbitration|arbitration route confirmed"
PARTIAL = r"partly|in part|remitted|retrial|varied|mixed|conditional"


def outcome_class(o):
    """Proxy: did the initiating party (claimant/applicant/appellant) obtain what it sought?"""
    s = str(o).lower()
    if re.search(REFER, s):
        return "referred to arbitration"
    if re.search(PARTIAL, s):
        return "partial"
    if re.search(r"^(award )?upheld|stay dismissed|objection overruled", s):
        return "unsuccessful"
    if re.search(SUCCESS, s):
        return "successful"
    if re.search(FAIL, s):
        return "unsuccessful"
    return "unclear"


ARB_TYPE = {  # arbitration application type (first match wins)
    "set-aside": r"set aside|s\.?\s?34|ss\.\s?[\d, ]*34|public policy|ultra petita",
    "enforcement/recognition": r"enforce|recogni|registration of award|s\.?\s?3[56]|new york",
    "stay/referral": r"refer|stay|s\.?\s?5\b|kompetenz|arbitration clause|jurisdictional defence",
    "interim measures": r"interim|s\.?\s?6\b|injunction",
    "appointment": r"appoint|s\.?\s?11\b|arbitrator",
}


def first_match(text, table, default="other"):
    for k, rx in table.items():
        if re.search(rx, text, re.I):
            return k
    return default


def code_registers(c):
    c = c.copy()
    txt = (c["key_issues"].fillna("") + " | " + c["holding_summary"].fillna("") + " | "
           + c["project_type_register"].fillna(""))
    for k, rx in ISSUES.items():
        c[f"issue:{k}"] = txt.str.contains(rx, case=False, regex=True)
    pt = c["parties"].map(party_types)
    c["party_type_1"], c["party_type_2"] = pt.str[0], pt.str[1]
    c["state_party"] = (c.party_type_1 == "state") | (c.party_type_2 == "state")
    c["proceeding"] = c["case_number"].map(proceeding)
    c["outcome_class"] = c["outcome_register"].map(outcome_class)
    is_arb = c.stream.isin(["arbitration", "both"])
    arb_txt = c["key_issues"].fillna("") + " | " + c["outcome_register"].fillna("") + " | " + c["holding_summary"].fillna("")
    c["arb_application_type"] = arb_txt.map(lambda t: first_match(t, ARB_TYPE)).where(is_arb)
    c["year_filed"] = pd.to_numeric(c["case_number"].str.extract(r"of\s+(\d{4})")[0], errors="coerce")
    # filing-to-decision only meaningful where the number is the originating suit/appeal
    dur = (c["year"] - c["year_filed"]).where(lambda x: x.between(0, 30))
    c["years_to_decision"] = dur.where(c["proceeding"].isin(["suit", "appeal"]))
    c["subject_unconfirmed"] = c["project_type_register"].fillna("").str.contains("subject matter not stated")
    return c
