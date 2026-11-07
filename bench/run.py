"""Pipe each test file through `yapzip` and score it against its gold file.

Scores: fact recall (must-keep strings present in output), fact units kept (line containing the fact
was cited at all), junk kept (lines that should be dropped but were cited), compression, fallbacks, speed.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.common import GOLD, RAW, RESULTS, fact_kept  # noqa: E402
from yapzip.split import ntok  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"[\s,$£€]+", " ", s.lower()).strip()   # lenient on case, commas, currency signs


def score(name: str, block: int) -> dict:
    raw = (RAW / f"{name}.txt").read_text()
    gold = json.loads((GOLD / f"{name}.json").read_text())
    out_dir, st_dir = RESULTS / "out", RESULTS / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    st_dir.mkdir(parents=True, exist_ok=True)
    st_path = st_dir / f"{name}.jsonl"
    st_path.unlink(missing_ok=True)

    t0 = time.time()
    extra = __import__("os").environ.get("YAPZIP_ARGS", "").split()
    out = subprocess.run(["uv", "run", "-q", "yapzip", "--block", str(block), "--stats", str(st_path), *extra],
                         input=raw, capture_output=True, text=True, check=True).stdout
    wall = time.time() - t0
    (out_dir / f"{name}.txt").write_text(out)
    stats = [json.loads(l) for l in st_path.read_text().splitlines()]

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    o = norm(out)
    facts = gold["facts"]
    recall = sum(fact_kept(f["must"], out) for f in facts) / len(facts)
    # junk leak: a drop span counts as leaked if most of its longer words show up in the output
    owords = set(re.findall(r"[a-z]{4,}", o))
    def leaked(span):
        w = set(re.findall(r"[a-z]{4,}", span.lower()))
        return bool(w) and len(w & owords) / len(w) >= 0.6
    junk_kept = sum(map(leaked, gold["drop_spans"])) / max(1, len(gold["drop_spans"]))

    return {
        "file": name, "units": len(lines), "blocks": len(stats),
        "in_tok": ntok(raw), "out_tok": ntok(out), "ratio": round(ntok(out) / ntok(raw), 2),
        "fact_recall": round(recall, 2), "junk_kept": round(junk_kept, 2),
        "model_lines": sum(s.get("lines", 0) for s in stats),
        "fallback_lines": sum(s.get("fallback", 0) for s in stats),
        "repeats": sum(s.get("repeats", 0) for s in stats),
        "first_block_s": round(stats[0]["since_start_s"], 2) if stats else None,
        "avg_block_s": round(sum(s["wall_s"] for s in stats) / len(stats), 2),
        "prompt_tps": round(sum(s["prompt_tps"] or 0 for s in stats) / len(stats)),
        "gen_tps": round(sum(s["gen_tps"] or 0 for s in stats) / len(stats)),
        "total_s": round(wall, 1),
    }


def main():
    block = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    names = sys.argv[2:] or sorted(p.stem for p in GOLD.glob("*.json"))
    rows = [score(n, block) for n in names]
    with open(RESULTS / f"scores_b{block}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cols = ["file", "units", "blocks", "ratio", "fact_recall", "junk_kept",
            "model_lines", "fallback_lines", "repeats", "avg_block_s", "gen_tps"]
    print("  ".join(f"{c:>12}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):>12}" for c in cols))


if __name__ == "__main__":
    main()
