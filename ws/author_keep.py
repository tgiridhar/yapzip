"""Edit 3: keep author lines in threads, so a kept post never lands under the previous poster's name.

Problem (v2ws2, forum thread in the test set): 5 of its 6 remaining meaning errors are posts whose author line
("tobias_r · 58 points") was dropped at the keep stage, so the post reads as the previous poster's.

    .venv/bin/python ws/author_keep.py collect results/ws/author_keep_feats.npz     # practice threads, both splits

Uses the same step construction as Edit 2 (ws/keep_edit.build_edit) with the push reversed (toward ' yes').
"""

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ws.edits import apply_npz  # noqa: E402
from ws.model import Yap  # noqa: E402
from ws.practice import practice_chunks  # noqa: E402

LAYER = 23
NEURONS = (1078, 2423)        # the next two quietest layer-23 neurons (max |act| ~0.28 on dev), after Edit 2's pair


def collect(y, split):
    """Layer-23 MLP input and keep logit at every '?' marker of the practice threads (author on its own line).
    role: 1 author line, 2 the post right after an author line, 0 anything else."""
    X, logit, role, thread, style = [], [], [], [], []
    for name, st, ch, hidx in practice_chunks(y, split):
        seq, marks = y.score_seq(ch)
        tap = {}
        hn = y.forward(seq, y.new_cache(), tap=tap)
        mk = mx.array(marks)
        X.append(np.array(tap[("mlp_in", LAYER)][mk].astype(mx.float32)))
        logit += list(np.array((hn[mk] @ (y.E[y.yes] - y.E[y.no])).astype(mx.float32)))
        role += [1 if i in hidx else 2 if (i - 1) in hidx else 0 for i in range(len(ch))]
        thread += [name] * len(ch)
        style += [st] * len(ch)
    return dict(X=np.concatenate(X), logit=np.array(logit), role=np.array(role), thread=np.array(thread),
                style=np.array(style))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("out")
    ap.add_argument("--gguf", default=str(ROOT / "models/ws/v4c.gguf"))
    ap.add_argument("--base-edit", default=str(ROOT / "results/ws/edit_combined.npz"))
    a = ap.parse_args()
    y = Yap(a.gguf)
    apply_npz(y, a.base_edit)
    if a.cmd == "collect":
        out = {}
        for split in ("analysis", "heldout"):
            d = collect(y, split)
            out.update({f"{split}_{k}": v for k, v in d.items()})
            print(split, len(d["role"]), "units", int((d["role"] == 1).sum()), "author lines", flush=True)
        np.savez(a.out, **out)
        print("saved", a.out)


if __name__ == "__main__":
    main()
