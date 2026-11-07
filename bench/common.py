"""Shared paths for bench scripts. No paid APIs: test data and grading are done in-session."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
GOLD = ROOT / "data" / "gold"
RESULTS = ROOT / "results"

import re  # noqa: E402

STOP = set("""the and for are was were with that this from have has had will would could should into onto than then
them they their there here what when where which while about after before over under also just very much more most
some such only into your yours ours mine been being does doing done each every both other same says said""".split())
NEG = {"not", "never", "no", "nothing", "nobody", "none", "only", "unless", "without", "cannot", "can't", "won't", "don't"}


def _tokens(s: str) -> list[str]:
    s = s.lower().replace("£", "").replace("$", "").replace("€", "")
    return re.findall(r"\d+(?:[.:,]\d+)*|[a-z][a-z'-]*", s)


def fact_kept(must: list[str], output: str) -> bool:
    """A fact is kept if every number and negation word of its must-strings appears in the output,
    and at least 2/3 of its other key words (4+ letters) do. Word order and phrasing are free."""
    out = set(t.replace(",", "") for t in _tokens(output))
    out |= {"not" for t in out if t.endswith("n't")}
    need_hard, need_soft = set(), set()
    for m in must:
        for t in _tokens(m):
            t = t.replace(",", "")
            if t[0].isdigit() or t in NEG:
                need_hard.add("not" if t.endswith("n't") else t)
            elif len(t) >= 4 and t not in STOP:
                need_soft.add(t)
    if not need_hard <= out:
        return False
    return not need_soft or len(need_soft & out) / len(need_soft) >= 2 / 3
