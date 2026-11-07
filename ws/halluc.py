"""Automatic hallucination checks for one note against its unit: anything in the note the unit doesn't support.

A cheap proxy for the judges' "invented / flipped / wrong speaker" errors, used to compare edits on dev data.
It catches substituted or invented words, added negations, condition markers with no source, foreign speaker
tags and numbers not in the unit. It does not catch merges or omissions that change meaning; the judges do.

    from ws.halluc import check
    check(unit, note)  ->  [(kind, token), ...]   kinds: word, negation, marker, speaker, number
"""

import re

MARKERS = {"not", "never", "only", "if", "unless", "because", "vs", "but", "no", "or", "and"}
SHORT = {"w", "w/o", "wo", "approx", "info", "asks", "maybe", "thinks", "claims", "rumor", "hopes", "per", "e.g",
         "eg", "etc", "via", "re", "incl", "min", "mins", "hr", "hrs", "sec", "secs", "pct", "aka", "ok", "q",
         "est", "avg", "max", "yr", "yrs", "mo", "wk", "wks", "nr", "num", "no.", "ea", "x", "pros", "cons"}
FUNC = set("""a an the and or of to in on at for with by from as is are was were be been being it its this that these
those he she they we you i me my our your their his her them us there here then than so do does did has have had will
would can could should may might must shall into onto over under about after before up down out off""".split())
NEG = re.compile(r"\b(not|no|never|none|nothing|nobody|without|cannot|neither|nor|stop|avoid|don'?t|doesn'?t|"
                 r"didn'?t|isn'?t|aren'?t|wasn'?t|weren'?t|won'?t|can'?t|shouldn'?t|couldn'?t|wouldn'?t)\b|n['’]t\b",
                 re.I)
NUM = re.compile(r"\d+(?:[.,:]\d+)*")


def words(s):
    return re.findall(r"[a-z0-9]+(?:['’.\-/][a-z0-9]+)*", s.lower())


def _prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _grounded(w, src_words, src):
    if w in src_words or (len(w) >= 3 and w in src):                 # same word, or part of a compound / abbreviation
        return True
    for s in src_words:                                                 # inflection: glides/gliding, opening/opens
        if _prefix(w, s) >= max(4, min(len(w), len(s)) - 3):
            return True
    return False


def check(unit, note):
    note = (note or "").strip()
    if note in ("", "="):
        return []
    src = unit.lower()
    flat = src.replace(",", "")
    src_words = set(words(unit))
    probs = []
    for n in NUM.findall(note):
        if n not in unit and n.replace(",", "") not in flat:
            probs.append(("number", n))
    if re.search(r"\b(NOT|NEVER|NO|not|never)\b", note) and not NEG.search(unit):
        probs.append(("negation", "NOT"))
    if re.search(r"\bunless\b", note, re.I) and "unless" not in src:
        probs.append(("marker", "UNLESS"))
    if re.search(r"\bONLY\b", note) and not re.search(r"\b(only|just|solely|exclusively)\b", src):
        probs.append(("marker", "ONLY"))
    m = re.match(r"\s*(?:\[[^\]]*\]\s*)?(?:↳\s*)?([A-Za-z][\w.'-]{0,30}(?: [A-Za-z0-9][\w.'-]{0,30})?):\s", note)
    tag = m.group(1) if m else ""
    looks_like_name = tag and (tag[0].isupper() or re.search(r"[_\d]", tag)) and tag.split()[0] not in (
        "IF", "NOT", "ONLY", "UNLESS", "BECAUSE", "VS", "BUT", "NEVER", "Q", "A")
    if looks_like_name and tag.lower() not in src:
        probs.append(("speaker", tag))
    for w in (x for t in words(note) for x in re.split(r"[/\-]", t) if x):
        if w in FUNC or w in MARKERS or w in SHORT or len(w) <= 2 or NUM.fullmatch(w):
            continue
        if not _grounded(w, src_words, src):
            probs.append(("word", w))
    return probs


def summarize(records):
    """records: [(unit, note)] -> dict of counts per kind and per 100 notes (notes = non-'=' notes)."""
    notes = [(u, n) for u, n in records if n and n.strip() != "="]
    kinds = {}
    for u, n in notes:
        for k, _ in check(u, n):
            kinds[k] = kinds.get(k, 0) + 1
    tot = sum(v for k, v in kinds.items() if k != "number")
    return {"notes": len(notes), "flags_per_100_notes": round(100 * tot / max(1, len(notes)), 1),
            "numbers_per_100_notes": round(100 * kinds.get("number", 0) / max(1, len(notes)), 1), **kinds}
