"""Build results/compare.md: original text, Claude's compression, Qwen's compression, per test file."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.common import GOLD, RAW, RESULTS, ROOT, fact_kept  # noqa: E402
from yapzip.split import ntok  # noqa: E402

QWEN = RESULTS / "qwen_two_stage"
CLAUDE = ROOT / "data" / "claude"


def norm(s):
    return re.sub(r"[\s,$£€]+", " ", s.lower()).strip()


def recall(text, facts):
    return sum(fact_kept(f["must"], text) for f in facts) / len(facts)


def main():
    names = sorted(p.stem for p in GOLD.glob("*.json"))
    rows, sections = [], []
    for n in names:
        raw, cl, qw = (RAW / f"{n}.txt").read_text(), (CLAUDE / f"{n}.txt").read_text(), (QWEN / f"{n}.txt").read_text()
        facts = json.loads((GOLD / f"{n}.json").read_text())["facts"]
        r = dict(name=n, raw=ntok(raw), cl=ntok(cl), qw=ntok(qw), rc=recall(cl, facts), rq=recall(qw, facts), nf=len(facts))
        rows.append(r)
        sections.append(
            f"## {n}\n\n"
            f"| | Size (tokens, est.) | % of original | Key facts kept ({r['nf']}) |\n|---|---|---|---|\n"
            f"| Original | {r['raw']} | 100% | — |\n"
            f"| Claude yapzip | {r['cl']} | {r['cl'] / r['raw']:.0%} | {r['rc']:.0%} |\n"
            f"| Qwen 0.8B yapzip | {r['qw']} | {r['qw'] / r['raw']:.0%} | {r['rq']:.0%} |\n\n"
            f"### Original\n\n```text\n{raw.strip()}\n```\n\n"
            f"### Claude yapzip\n\n```text\n{cl.strip()}\n```\n\n"
            f"### Qwen 0.8B yapzip\n\n```text\n{qw.strip()}\n```\n")
    tot = {k: sum(r[k] for r in rows) for k in ("raw", "cl", "qw")}
    avg = lambda k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
    summary = "\n".join(f"| {r['name']} | {r['raw']} | {r['cl'] / r['raw']:.0%} | {r['rc']:.0%} | {r['qw'] / r['raw']:.0%} | {r['rq']:.0%} |" for r in rows)
    md = f"""# yapzip: original vs Claude vs Qwen 0.8B

Six generated test texts (team chats, meeting transcript, email threads, forum thread).
**Claude yapzip** = compression written by Claude (the target). **Qwen 0.8B yapzip** = untuned Qwen3.5-0.8B (Q8_0, llama.cpp),
two-stage: per-line yes/no keep score, then one rewrite call per ~400-token block over the kept lines, numbers checked against source.
"Key facts kept" = share of the test file's key facts present in the output: every number and negation (NOT, only, never)
of the fact must appear, plus at least 2/3 of its key words, in any order. Rewording is fine; dropping a number or a NOT is not.

| File | Original tokens | Claude size | Claude facts kept | Qwen size | Qwen facts kept |
|---|---|---|---|---|---|
{summary}
| **Total / avg** | {tot['raw']} | {tot['cl'] / tot['raw']:.0%} | {avg('rc'):.0%} | {tot['qw'] / tot['raw']:.0%} | {avg('rq'):.0%} |

Notable Qwen errors: copies lines instead of rewriting, keeps chatter ("+1", "morning ☕"), cuts lines mid-sentence,
repeats lines, scrambles order, and in `forum_union` invents a fact ("Tobias R is 58 years old" — from "58 points").
That one passed the number check because 58 appears in the source.

""" + "\n".join(sections)
    (RESULTS / "compare.md").write_text(md)
    print(md[:md.find("Notable")])


if __name__ == "__main__":
    main()
