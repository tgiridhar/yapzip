"""Run the native binary on the dev articles (data/slop_dev) and score every unit against its label.

- keep/drop vs the labels: accuracy, false keeps by kind (rhetoric, repeat, heading...), facts dropped
- notes checked against their own unit (ws/halluc.py): invented words, added negations, unsupported markers,
  foreign speaker tags, numbers not in the unit
- output size vs input, code units copied exactly

    .venv/bin/python ws/dev_eval.py TAG [--model models/ws/x.gguf] [--names a,b] [--threshold 0.5]

Per-unit records go to results/ws/dev/TAG/<name>.json; the summary is printed and saved as summary.json.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ws.dev_data import load, names, split_units, text  # noqa: E402
from ws.halluc import check  # noqa: E402

BIN = ROOT / "build" / "yapzip"
NUMS = re.compile(r"\d+(?:[.,:]\d+)*")


def markers(note):
    """Meaning-carrying markers in a note: a dropped NOT / ONLY IF / UNLESS flips the fact."""
    up = note.upper()
    return {m for m in ("NOT", "NEVER", "ONLY", "UNLESS", "IF", "BECAUSE", "VS", "BUT")
            if re.search(rf"\b{m}\b", up if m != "IF" else note)} | ({"NOT"} if re.search(r"n't\b", note) else set())


def _flat(binary):
    """--flat if this binary has it (newer builds lay notes out as points by default)."""
    h = subprocess.run([str(binary), "-h"], capture_output=True, text=True)
    return ["--flat"] if "--flat" in h.stderr + h.stdout else []


def run(txt, model=None, threshold=None, block=None):
    cmd = [str(BIN), "--scores"] + _flat(BIN) + (["-m", model] if model else []) + (["--threshold", str(threshold)] if threshold else [])
    cmd += ["--block", str(block)] if block else []
    r = subprocess.run(cmd, input=txt, capture_output=True, text=True, check=True)
    p = [float(m.group(1)) for m in re.finditer(r"^\[score\] ([0-9.]+)\t", r.stderr, re.M)]
    return r.stdout, p


def align(units, p, stdout, threshold=0.5):
    """Map output lines back to units: a kept unit yields either its note (one line) or itself (note '=')."""
    lines = stdout.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    i, notes = 0, [None] * len(units)
    for k, u in enumerate(units):
        if p[k] < threshold:
            continue
        ul = u.split("\n")
        if lines[i:i + len(ul)] == ul:
            notes[k], i = "=", i + len(ul)
        else:
            notes[k], i = (lines[i] if i < len(lines) else ""), i + 1
    if i != len(lines):
        raise RuntimeError(f"alignment used {i} of {len(lines)} output lines")
    return notes


def score(name, model=None, threshold=0.5, block=None):
    d = load(name)
    units = [u["text"] for u in d["units"]]
    txt = text(d)
    assert split_units(txt) == units
    out, p = run(txt, model, threshold if threshold != 0.5 else None, block)
    assert len(p) == len(units), (len(p), len(units))
    notes = align(units, p, out, threshold)
    recs = []
    for u, pk, note in zip(d["units"], p, notes):
        recs.append(dict(id=u["id"], kind=u["kind"], label=u["keep"], p=pk, kept=note is not None, note=note,
                         gold=u["note"], flags=check(u["text"], note) if note else []))
    return recs, out, txt


def summarize(all_recs, outs, txts):
    s = dict(units=0, label_kept=0, kept=0, agree=0, false_keep={}, fact_dropped=0, code_units=0, code_exact=0,
             notes=0, flags={}, gold_markers=0, markers_dropped=0, gold_numbers=0, numbers_dropped=0)
    for recs in all_recs:
        for r in recs:
            s["units"] += 1
            s["label_kept"] += r["label"]
            s["kept"] += r["kept"]
            s["agree"] += r["label"] == r["kept"]
            if r["kept"] and not r["label"]:
                s["false_keep"][r["kind"]] = s["false_keep"].get(r["kind"], 0) + 1
            if r["label"] and not r["kept"] and r["kind"] == "fact":
                s["fact_dropped"] += 1
            if r["kind"] == "code" and r["label"]:
                s["code_units"] += 1
                s["code_exact"] += r["note"] == "="
            if r["kept"] and r["note"] not in (None, "="):
                s["notes"] += 1
                for k, _ in r["flags"]:
                    s["flags"][k] = s["flags"].get(k, 0) + 1
            if r["kept"] and r["label"] and r["gold"] not in (None, "=") and r["note"] not in (None, "="):
                gm, nm = markers(r["gold"]), markers(r["note"])
                s["gold_markers"] += len(gm)
                s["markers_dropped"] += len(gm - nm)
                gn, nn = set(NUMS.findall(r["gold"])), set(NUMS.findall(r["note"]))
                s["gold_numbers"] += len(gn)
                s["numbers_dropped"] += len(gn - nn)
    s["out_words"] = sum(len(o.split()) for o in outs)
    s["in_words"] = sum(len(t.split()) for t in txts)
    s["size"] = round(s["out_words"] / s["in_words"], 4)
    s["keep_acc"] = round(s["agree"] / s["units"], 4)
    s["halluc_per_100_notes"] = round(100 * sum(v for k, v in s["flags"].items() if k != "number") / max(1, s["notes"]), 1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--model")
    ap.add_argument("--names")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--block", type=int)
    a = ap.parse_args()
    out_dir = ROOT / "results" / "ws" / "dev" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = a.names.split(",") if a.names else names()
    all_recs, outs, txts = [], [], []
    for n in sel:
        recs, out, txt = score(n, a.model, a.threshold, a.block)
        (out_dir / f"{n}.json").write_text(json.dumps(recs, indent=0, ensure_ascii=False))
        (out_dir / f"{n}.txt").write_text(out)
        all_recs.append(recs); outs.append(out); txts.append(txt)
        s1 = summarize([recs], [out], [txt])
        print(f"{n:22s} size {s1['size']:.0%}  keep_acc {s1['keep_acc']:.0%}  false_keep {s1['false_keep']}  "
              f"facts_dropped {s1['fact_dropped']}  halluc/100 {s1['halluc_per_100_notes']}  flags {s1['flags']}", flush=True)
    s = summarize(all_recs, outs, txts)
    s["model"], s["names"], s["block"] = a.model or "embedded v2", sel, a.block or 600
    (out_dir / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: s[k] for k in ("size", "keep_acc", "kept", "label_kept", "false_keep", "fact_dropped",
                                        "code_exact", "code_units", "notes", "halluc_per_100_notes", "flags",
                                        "markers_dropped", "gold_markers", "numbers_dropped", "gold_numbers")}))


if __name__ == "__main__":
    main()
