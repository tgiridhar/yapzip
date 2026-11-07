"""Training package v4 (relabeled data) for Colab: colab/yapzip_data_v4.zip.

    .venv/bin/python ws/make_train_v4.py

Contents:
  data/claude_train.jsonl   the 510 Claude-written chunks with relabel v4b notes (data/relabel_v4b/part_*.jsonl:
                            faithful AND compressed, data/STYLE.md "Priority order"); same chunks and order as before,
                            so the notebook's 30-chunk held-out split is the same units
  data/toy_train.jsonl      template chunks, unchanged (right by construction)
  data/oos.json             as before
  data/extra_train.jsonl    training-only: the 20 dev articles (data/relabel_v4/slop_dev) and the 19 Medium articles
                            (data/pasted/v4, text kept out of git) cut into <=300-token blocks like the binary, plus
                            repeat blocks rebuilt from the new notes (a bulleted note followed by the line it came from,
                            labeled drop) with matching controls
  start/v2ws3_start.npz     the mainline weights as a delta on base Qwen (ws/make_start_delta.py)
  native/                   the binary source
"""

import json
import random
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ws.model import Yap, chunks  # noqa: E402
from ws.make_train_v3 import doc_chunks, unit  # noqa: E402

OUT = ROOT / "colab" / "yapzip_data_v4.zip"


def main():
    parts = sorted((ROOT / "data" / "relabel_v4b").glob("part_*.jsonl"))
    assert len(parts) == 18, f"expected 18 final parts, found {len(parts)}"
    claude = "".join(p.read_text() for p in parts)
    v4dev = {p.name: p for p in (ROOT / "data" / "relabel_v4" / "slop_dev").glob("*.json")}
    dev = [v4dev.get(p.name, p) for p in sorted((ROOT / "data" / "slop_dev").glob("*.json"))]   # old label if not redone
    print(f"dev articles: {sum(p.parent.name == 'slop_dev' and 'relabel_v4' in str(p) for p in dev)} relabeled, "
          f"{sum('relabel_v4' not in str(p) for p in dev)} old labels")
    docs = dev + sorted((ROOT / "data" / "pasted" / "v4").glob("*.json"))
    y = Yap(ROOT / "models" / "ws" / "v4c.gguf")      # tokenizer for the block cut only
    rng = random.Random(0)
    extra, loaded = [], []
    for p in docs:
        d = json.loads(p.read_text())
        loaded.append(d)
        extra += [{"genre": d.get("genre", "article"), "units": us} for us in doc_chunks(y, d["units"])]
    n_doc = len(extra)
    for d in loaded:
        facts = [u for u in d["units"] if u.get("kind") == "fact" and u["keep"] and u["note"] and u["note"] != "="]
        ctx = [u for u in d["units"] if u.get("kind") in ("fact", "rhetoric")][:4]
        for f in rng.sample(facts, min(4, len(facts))):
            others = [g for g in facts if g is not f]
            for is_rep in (True, False):
                if not is_rep and not others:
                    continue
                prev = f if is_rep else rng.choice(others)
                block = [u for u in ctx if u is not f][:2]
                units = [unit(k + 1, u["text"], u["keep"], u["note"]) for k, u in enumerate(block)]
                units.append(unit(len(units) + 1, "- " + prev["note"], True, prev["note"]))
                units.append(unit(len(units) + 1, f["text"], not is_rep, f["note"]))
                extra.append({"genre": "repeat_block" if is_rep else "repeat_control", "units": units})
    old = zipfile.ZipFile(ROOT / "colab" / "yapzip_data.zip")
    with zipfile.ZipFile(OUT, "w") as z:
        z.writestr("data/claude_train.jsonl", claude, compress_type=zipfile.ZIP_DEFLATED)
        for name in ("data/toy_train.jsonl", "data/oos.json"):
            z.writestr(name, old.read(name), compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("data/extra_train.jsonl", "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in extra),
                   compress_type=zipfile.ZIP_DEFLATED)
        for f in ("yapzip.cpp", "CMakeLists.txt", "embed.sh"):
            z.write(ROOT / "native" / f, f"native/{f}", compress_type=zipfile.ZIP_DEFLATED)
        z.write(ROOT / "colab" / "v2ws3_start.npz", "start/v2ws3_start.npz", compress_type=zipfile.ZIP_STORED)
    reps = sum(c["genre"] == "repeat_block" for c in extra)
    print(f"claude_train: {claude.count(chr(10))} chunks; extra: {n_doc} article blocks from {len(docs)} docs, "
          f"{reps} repeat blocks, {len(extra) - n_doc - reps} controls -> {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
