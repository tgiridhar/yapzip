"""Keep the model honest: a note line with a number that is not in the block is replaced by
the source line it most overlaps with. Worst case is no compression, never an invented number."""

import re

NUM = re.compile(r"\d+(?:[.,:]\d+)*")
WORD = re.compile(r"[a-z0-9$%]+")


def nums(s: str) -> set[str]:
    return {n.replace(",", "") for n in NUM.findall(s)}


def closest(line: str, sources: list[str]) -> str:
    w = set(WORD.findall(line.lower()))
    return max(sources, key=lambda s: len(w & set(WORD.findall(s.lower()))))


def process(model_text: str, sources: list[str]) -> tuple[list[str], dict]:
    block_nums = nums(" ".join(sources))
    out, seen = [], set()
    stats = {"lines": 0, "fallback": 0, "repeats": 0}
    for raw in model_text.splitlines():
        line = raw.strip().lstrip("-•* ").strip()
        if not line:
            continue
        if line in seen:                      # decoding loop
            stats["repeats"] += 1
            continue
        seen.add(line)
        stats["lines"] += 1
        if nums(line) - block_nums:
            stats["fallback"] += 1
            src = closest(line, sources)
            if src not in seen:
                seen.add(src)
                out.append(src)
        else:
            out.append(line)
    return out, stats
