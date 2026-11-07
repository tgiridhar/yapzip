"""Evaluate the native binary end to end: line units (--no-split) vs sentence units (default).

1. Test documents with answer keys (data/raw + data/gold): key facts kept, output size.
2. Out-of-sample cases (colab/data/oos.json): output size next to Opus.
3. Code safety: every code/data line of a mixed input must come out byte for byte.
Usage: uv run python bench/binary_eval.py [path/to/yapzip] [extra input files...]
"""

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.common import GOLD, RAW, ROOT, fact_kept  # noqa: E402

BIN = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "build" / "yapzip")
EXTRA = sys.argv[2:]
MODES = {"line units (old)": ["--no-split"], "sentence units (new)": []}

CODE_INPUT = """Here is how to run it. It is quick, honestly.
```bash
pip install fastcache==3.6.5
fastcache serve --port 6895. # trailing dot on purpose
```
{"msg": "Hello. World", "retries": 3, "url": "https://x.io/a. b"}
$ curl -X POST https://api.example.com/v1/jobs. -d '{"a": 1}'
docker run -p 8080:80 -e MODE=prod nginx:1.27
print("done. ok"); x = load(path)
| region | p99 ms. | errors |
lol nice
"""
CODE_LINES = [l for l in CODE_INPUT.splitlines()[1:-1]]  # everything except the first prose line and "lol nice"


def run(text, flags):
    t = time.time()
    out = subprocess.run([BIN, *flags], input=text, capture_output=True, text=True).stdout
    return out, time.time() - t


def words(s):
    return len(s.split())


def main():
    print(f"binary: {BIN}\n")
    print("1. Test documents with answer keys (key facts kept / output size vs input)")
    print(f"   {'file':16s}" + "".join(f"{m:>30s}" for m in MODES))
    tot = {m: [0, 0, 0, 0] for m in MODES}  # facts kept, facts, out words, in words
    for g in sorted(GOLD.glob("*.json")):
        raw = (RAW / f"{g.stem}.txt").read_text()
        facts = json.loads(g.read_text())["facts"]
        row = f"   {g.stem:16s}"
        for m, flags in MODES.items():
            out, _ = run(raw, flags)
            k = sum(fact_kept(f["must"], out) for f in facts)
            tot[m][0] += k; tot[m][1] += len(facts); tot[m][2] += words(out); tot[m][3] += words(raw)
            row += f"{f'{k}/{len(facts)} facts, {words(out) / words(raw):.0%} size':>30s}"
        print(row)
    print(f"   {'ALL':16s}" + "".join(f"{f'{t[0] / t[1]:.0%} facts, {t[2] / t[3]:.0%} size':>30s}" for t in tot.values()))

    print("\n2. Out-of-sample cases (output words; Opus for reference)")
    cases = json.loads((ROOT / "colab/data/oos.json").read_text())
    for c in cases:
        src = "\n".join(c["units"]) + "\n"
        row = f"   {c['name'][:34]:34s} in {words(src):4d}  opus {words(c['opus']):4d}"
        for m, flags in MODES.items():
            out, _ = run(src, flags)
            row += f"  {m.split()[0]} {words(out):4d}"
        print(row)

    for path in EXTRA:
        src = Path(path).read_text()
        print(f"\n   extra input {Path(path).name}: {words(src)} words")
        for m, flags in MODES.items():
            out, dt = run(src, flags)
            print(f"     {m:22s} {words(out):5d} words ({words(out) / words(src):.0%}) in {dt:.0f}s")

    print("\n3. Code safety (code/data lines must come out byte for byte)")
    for m, flags in MODES.items():
        out, _ = run(CODE_INPUT, flags)
        outs = out.splitlines()
        fence = "\n".join(CODE_LINES[0:4])
        ok = [fence in out] + [l in outs for l in CODE_LINES[4:]]
        print(f"   {m:22s} {sum(ok)}/{len(ok)} code units intact")


if __name__ == "__main__":
    main()
