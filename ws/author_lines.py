"""Weight edit #1: author lines ("ravi_s · 36 points · reply to karen_b_dairy", "From: Priya <...>") get "=" notes.

Problem (v2): training units always had the author inline ("handle (+3): text"). When an author line stands on its
own line, v2 sometimes writes it a note that pulls text from a neighbouring post ("ravi_s ... reply to karen_b_dairy:
part-timers under 20 hours NEVER pay dues?" is karen's question), which misattributes facts.

Mechanism found with forward passes: two MLP neurons write the " =" (copy verbatim) token, L20 n58 and L22 n274.
Clean author-line notes open their gate wide (+3.4/+3.8); leaky ones open it less (+2.0/+2.5); prose keeps it shut.
The MLP input at the note position also carries a near-perfect "this unit is an author/metadata line" direction
(difference of means, held-out AUC 0.997).

Edit: add beta * that direction to the two neurons' gate rows, so author lines open the copy-verbatim neurons as
wide as the clean cases do. 2 rows x 1024 weights; no gradients, no optimizer.

    python ws/author_lines.py detect results/ws/author_detector.npz     # forward passes on 20 practice threads (~20 min)
    python ws/author_lines.py edit results/ws/author_detector.npz results/ws/edit_author_lines.npz --beta 0.25
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ws.edits import save  # noqa: E402
from ws.model import Yap  # noqa: E402
from ws.practice import practice_chunks  # noqa: E402

NEURONS = ((20, 58), (22, 274))   # the " =" writers (read off ffn_down columns through the output embedding)


def collect(y, split="analysis", layers=(20, 22)):
    """MLP inputs at the first-note-token position for author lines and for the post that follows them.
    Every author line is kept (forced) so that group is large; posts keep the model's own decision."""
    import mlx.core as mx
    feats = {l: {"hdr": [], "body": []} for l in layers}
    for name, style, ch, hidx in practice_chunks(y, split):
        p, seq, marks, cache = y.keep_scores(ch)
        keep = [bool(p[i] >= .5) or i in hidx for i in range(len(ch))]
        first = True
        for i, u in enumerate(ch):
            if not keep[i]:
                continue
            prompt = (y.enc("notes:\n") if first else []) + y.enc(f"u{i + 1} {u} =>")
            first = False
            tap = {}
            hn = y.forward(prompt, cache, tap=tap)
            kind = "hdr" if i in hidx else ("body" if (i - 1) in hidx else None)
            if kind:
                for l in layers:
                    feats[l][kind].append(np.array(tap[("mlp_in", l)][-1], dtype=np.float32))
            # greedy note, fed back exactly as native/yapzip.cpp does, so the context matches the binary
            t = int(mx.argmax(y.logits(hn[-1])).item())
            ended = False
            for _ in range(96):
                if t in y.eog:
                    break
                if b"\n" in y.piece(t):
                    y.forward([t], cache)
                    ended = True
                    break
                hn = y.forward([t], cache)
                t = int(mx.argmax(y.logits(hn[-1])).item())
            if not ended:
                y.forward(y.nl, cache)
        print(name, {l: (len(feats[l]["hdr"]), len(feats[l]["body"])) for l in layers}, flush=True)
    return {l: (np.stack(v["hdr"]), np.stack(v["body"])) for l, v in feats.items()}


def detector(hdr, body):
    d = hdr.mean(0) - body.mean(0)
    return d / np.linalg.norm(d)


def build_edit(y, dets, beta):
    edits = []
    for layer, j in NEURONS:
        row = np.array(y.lm.model.layers[layer].mlp.gate_proj.weight[j], dtype=np.float32)
        edits.append(dict(layer=layer, mat="gate", op="set_rows", rows=[j], values=(row + beta * dets[layer])[None, :]))
    return edits


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("out")
    e = sub.add_parser("edit")
    e.add_argument("detector")
    e.add_argument("out")
    e.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--gguf", default=str(ROOT / "models/ws/v4c.gguf"))
    a = ap.parse_args()
    y = Yap(a.gguf)
    if a.cmd == "detect":
        f = collect(y)
        np.savez(a.out, **{f"L{l}": detector(h, b) for l, (h, b) in f.items()})
        print("saved", a.out)
    else:
        z = np.load(a.detector)
        dets = {l: z[f"L{l}"] for l, _ in NEURONS}
        save(build_edit(y, dets, a.beta), a.out, meta=dict(beta=a.beta, neurons=NEURONS,
             note="author-line detector added to the gate rows of the ' =' writer neurons"))
        print("saved", a.out)


if __name__ == "__main__":
    main()
