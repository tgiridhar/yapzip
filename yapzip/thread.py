"""Thread adapter: a comment tree -> units (depth-first) -> chunks with a carried header.

Thread JSON shape: {"title": str, "op": {"author": str, "text": str},
                    "comments": [{"id", "parent", "author", "text", "depth", "score"?}, ...]}  (depth-first order)
"""

import re
from dataclasses import dataclass, field

from .split import ntok

MAX_UNIT_TOKENS = 200
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class TUnit:
    id: int
    meta: str
    text: str
    src: str            # comment id this unit came from
    parent_src: str     # parent comment id ("" for top level)

    def render(self) -> str:
        return f"u{self.id} [{self.meta}] {self.text}"


@dataclass
class Chunk:
    header: str
    units: list[TUnit] = field(default_factory=list)

    def body(self) -> str:
        return "\n".join(u.render() for u in self.units)


def _split(text: str) -> list[str]:
    if ntok(text) <= MAX_UNIT_TOKENS:
        return [text]
    parts, cur = [], ""
    for s in SENT.split(text):
        if cur and ntok(cur) + ntok(s) > MAX_UNIT_TOKENS:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    return parts + ([cur] if cur else [])


def snippet(text: str, words: int) -> str:
    w = text.split()
    return " ".join(w[:words]) + (" ..." if len(w) > words else "")


def units(thread: dict) -> list[TUnit]:
    out, n = [], 0
    for c in thread["comments"]:
        meta = f"d{c['depth']} {c['author']}"
        if c.get("score") not in (None, 0):
            meta = f"+{c['score']} " + meta
        for part in _split(re.sub(r"\s+", " ", c["text"]).strip()):
            n += 1
            out.append(TUnit(n, meta, part, str(c["id"]), str(c.get("parent") or "")))
    return out


def chunks(thread: dict, budget: int = 1500) -> list[Chunk]:
    """Pack whole units into chunks; each chunk carries title + OP snippet + parent snippets
    for units whose parent comment lives in an earlier chunk."""
    by_src = {str(c["id"]): c for c in thread["comments"]}
    base = f"title: {thread['title']}"
    if thread["op"].get("text"):
        base += f"\nOP ({thread['op']['author']}): {snippet(thread['op']['text'], 60)}"
    result, cur, size = [], [], 0
    for u in units(thread):
        t = ntok(u.render())
        if cur and size + t > budget:
            result.append(cur)
            cur, size = [], 0
        cur.append(u)
        size += t
    if cur:
        result.append(cur)

    out = []
    for us in result:
        inside = {u.src for u in us}
        parents = []
        for u in us:
            p = u.parent_src
            if p and p not in inside and p in by_src and p not in [x[0] for x in parents]:
                parents.append((p, f"re {by_src[p]['author']}: {snippet(by_src[p]['text'], 25)}"))
        header = base + ("\n" + "\n".join(x[1] for x in parents[:4]) if parents else "")
        out.append(Chunk(header, us))
    return out
