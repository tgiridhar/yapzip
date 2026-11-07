"""Un-blind the LLM-judge verdicts and put a judge table at the top of results/compare.md.

Reads results/judge/<name>.verdict.json (A/B as the judge saw them) and results/judge_key.json (which was which).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.common import RESULTS  # noqa: E402

J = RESULTS / "judge"


def main():
    key = json.loads((RESULTS / "judge_key.json").read_text())
    rows, errs = [], []
    for name in sorted(key):
        f = J / f"{name}.verdict.json"
        if not f.exists():
            print("missing verdict:", name)
            continue
        v = json.loads(f.read_text())
        for slot, system in key[name].items():
            g = v[slot]
            labels = list(g["facts"].values())
            n = len(labels)
            rows.append({
                "file": name, "system": system, "facts": n,
                "kept": sum(l == "kept" for l in labels), "partial": sum(l == "partial" for l in labels),
                "wrong": sum(l == "wrong" for l in labels), "missing": sum(l == "missing" for l in labels),
                "errors": len(g.get("errors", [])),
                "qa": sum(bool(q.get("correct")) for q in g["qa"].values()), "qa_n": len(g["qa"]),
            })
            errs += [(name, system, e["quote"], e["why"]) for e in g.get("errors", [])]

    def total(system):
        rs = [r for r in rows if r["system"] == system]
        t = {k: sum(r[k] for r in rs) for k in ("facts", "kept", "partial", "wrong", "missing", "errors", "qa", "qa_n")}
        return t

    lines = ["## LLM judge (blind A/B, Claude subagents)", "",
             "Each file was graded by a judge that saw the original and two unlabeled outputs in random order. "
             "Facts: kept / partial / wrong / missing. Errors: statements not supported by the original. "
             "Quiz: the file's 8 questions answered from the output alone.", "",
             "| File | System | Facts kept | Partial | Wrong | Missing | Errors | Quiz correct |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['file']} | {r['system']} | {r['kept']}/{r['facts']} | {r['partial']} | {r['wrong']} | "
                     f"{r['missing']} | {r['errors']} | {r['qa']}/{r['qa_n']} |")
    for s in ("claude", "qwen"):
        t = total(s)
        if t["facts"]:
            lines.append(f"| **all** | **{s}** | **{t['kept'] / t['facts']:.0%}** | {t['partial']} | {t['wrong']} | "
                         f"{t['missing']} | **{t['errors']}** | **{t['qa']}/{t['qa_n']}** |")
    lines += ["", "### Errors the judges found", ""]
    lines += [f"- `{n}` · {s}: \"{q}\" — {w}" for n, s, q, w in errs]
    block = "\n".join(lines) + "\n\n"

    md = (RESULTS / "compare.md").read_text()
    if "## LLM judge" in md:
        md = md[:md.index("## LLM judge")] + md[md.index("## chat_incident"):]
    md = md.replace("## chat_incident", block + "## chat_incident", 1)
    (RESULTS / "compare.md").write_text(md)
    print(block)


if __name__ == "__main__":
    main()
