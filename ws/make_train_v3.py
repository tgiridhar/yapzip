"""Training package for continuing from the mainline weights (v2ws3) on Colab: colab/yapzip_data_v3.zip.

    .venv/bin/python ws/make_train_v3.py

Contents:
  data/claude_train.jsonl, data/toy_train.jsonl, data/oos.json   as before (same 30-chunk held-out split)
  data/extra_train.jsonl  new chunks, training only:
     - the 20 labeled dev documents (data/slop_dev: long AI-written articles, rhetoric-heavy posts, pasted web
       pages with their chrome), cut into blocks of <=300 tokens exactly like the binary
     - repeat blocks: a bulleted note followed by the line it came from (that line labeled drop, as a repeat),
       plus matching controls where the bullet is a different fact (line labeled keep)
     - articles the user copied from Medium (data/pasted/*.json, labeled; the texts stay out of git)
  start/v2ws3_start.npz   the starting weights (v2ws3) as a delta on base Qwen: low-rank factors per linear matrix
                          plus the exact rows/columns of the weight-space edits (ws/make_start_delta.py, ~90 MB)
  native/                 the binary source
The test articles (data/raw/slop_*), the 14 eval inputs and the trap sets are not included.
"""

import json
import random
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ws.dev_data import load, names  # noqa: E402
from ws.model import Yap, chunks  # noqa: E402

OUT = ROOT / "colab" / "yapzip_data_v3.zip"
START = ROOT / "colab" / "v2ws3_start.npz"


def unit(i, text, keep, note):
    return {"id": i, "text": text, "keep": bool(keep), "note": note if keep else None}


def doc_chunks(y, units):
    """Cut a labeled document into <=300-token blocks like the binary; labels follow position, not text."""
    i = 0
    for ch in chunks(y, [u["text"] for u in units], block=300):
        part, i = units[i:i + len(ch)], i + len(ch)
        assert [u["text"] for u in part] == ch
        yield [unit(k + 1, u["text"], u["keep"], u["note"]) for k, u in enumerate(part)]


def main():
    y = Yap(ROOT / "models" / "ws" / "v4c.gguf")      # tokenizer for the block cut only
    rng = random.Random(0)
    extra = []
    for n in names():
        d = load(n)
        extra += [{"genre": d["genre"], "units": us} for us in doc_chunks(y, d["units"])]
    pasted = sorted((ROOT / "data" / "pasted").glob("*.json"))
    for p in pasted:
        d = json.loads(p.read_text())
        extra += [{"genre": d.get("genre", "pasted"), "units": us} for us in doc_chunks(y, d["units"])]
    n_doc = len(extra)
    for n in names():
        d = load(n)
        facts = [u for u in d["units"] if u["kind"] == "fact" and u["note"] and u["note"] != "="]
        ctx = [u for u in d["units"] if u["kind"] in ("fact", "rhetoric")][:4]
        for f in rng.sample(facts, min(5, len(facts))):
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
    (ROOT / "colab").mkdir(exist_ok=True)
    old = zipfile.ZipFile(ROOT / "colab" / "yapzip_data.zip")
    with zipfile.ZipFile(OUT, "w") as z:
        for name in ("data/claude_train.jsonl", "data/toy_train.jsonl", "data/oos.json"):
            z.writestr(name, old.read(name), compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("data/extra_train.jsonl", "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in extra),
                   compress_type=zipfile.ZIP_DEFLATED)
        for f in ("yapzip.cpp", "CMakeLists.txt", "embed.sh"):
            z.write(ROOT / "native" / f, f"native/{f}", compress_type=zipfile.ZIP_DEFLATED)
        z.write(START, "start/v2ws3_start.npz", compress_type=zipfile.ZIP_STORED)
    reps = sum(c["genre"] == "repeat_block" for c in extra)
    print(f"{len(extra)} extra chunks: {n_doc} from {len(names())} dev documents + {len(pasted)} pasted articles, {reps} repeat blocks, "
          f"{len(extra) - n_doc - reps} controls -> {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
