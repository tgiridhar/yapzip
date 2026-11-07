"""Regression check for weight edits on the notebook's 30 held-out chunks (same split as
colab/yapzip_train_a100.ipynb): keep/drop accuracy + AUC and teacher-forced notes loss, one forward pass per chunk.

    python ws/regress.py [--edit results/ws/edit.npz]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ws.model import Yap  # noqa: E402


def heldout_chunks(n=30, seed=0):
    claude = []
    for f in sorted((ROOT / "data/claude_train").glob("part_*.jsonl")):
        claude += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    base = claude[:300]
    random.Random(seed).shuffle(base)
    return base[:n]


def build(y, chunk):
    units = chunk["units"]
    ids = y.enc("units:\n")
    for u in units:
        ids += y.enc(f"u{u['id']} {u['text']}\n")
    ids += y.enc("keep?\n")
    pos = []
    for u in units:
        ids += y.enc(f" u{u['id']}") + [y.q]
        pos.append(len(ids) - 1)
    ids += y.enc("\nnotes:\n")
    labels, tgt = [bool(u["keep"]) for u in units], []
    for u in units:
        if u["keep"]:
            ids += y.enc(f"u{u['id']} {u['text']} =>")
            note = y.enc(f" {u['note']}\n")
            tgt += range(len(ids), len(ids) + len(note))
            ids += note
    return ids, pos, labels, tgt


def auc(scores, labels):
    s, l = np.asarray(scores), np.asarray(labels, dtype=bool)
    pos, neg = s[l], s[~l]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def evaluate(y, chunks=None):
    chunks = chunks or heldout_chunks()
    d_all, lab_all, nll, ntok, eq_hit, eq_n = [], [], 0.0, 0, 0, 0
    eq = y.single(" =")
    for c in chunks:
        ids, pos, labels, tgt = build(y, c)
        hn = y.forward(ids[:-1], y.new_cache())
        d = hn[mx.array(pos)] @ (y.E[y.yes] - y.E[y.no])
        d_all += np.array(d).tolist()
        lab_all += labels
        if tgt:
            t = mx.array(tgt)
            lg = y.logits(hn[t - 1])
            lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            gold = mx.array([ids[i] for i in tgt])
            tok_lp = mx.take_along_axis(lp, gold[:, None], axis=-1)[:, 0]
            nll -= float(tok_lp.sum())
            ntok += len(tgt)
            # first note token: is the model's argmax " =" exactly when the gold note is "="?
            firsts = [k for k, i in enumerate(tgt) if k == 0 or tgt[k - 1] != i - 1]
            am = np.array(mx.argmax(lg, axis=-1))
            for k in firsts:
                eq_n += 1
                eq_hit += int((am[k] == eq) == (ids[tgt[k]] == eq))
    d_all, lab = np.array(d_all), np.array(lab_all)
    return dict(keep_acc=float(((d_all >= 0) == lab).mean()), keep_auc=auc(d_all, lab),
                notes_nll=nll / max(1, ntok), first_eq_agree=eq_hit / max(1, eq_n), n_units=len(lab), n_note_tokens=ntok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=str(ROOT / "models/ws/v4c.gguf"))
    ap.add_argument("--edit", default=None)
    a = ap.parse_args()
    y = Yap(a.gguf)
    base = evaluate(y)
    print("original:", json.dumps(base))
    if a.edit:
        from ws.edits import Edited, load
        with Edited(y, load(a.edit)):
            print("edited:  ", json.dumps(evaluate(y)))


if __name__ == "__main__":
    main()
