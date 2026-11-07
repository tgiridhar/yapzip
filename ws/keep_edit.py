"""Edit 2: drop generic rhetoric and recap lines in the keep/drop stage.

Problem (v2, long AI-written articles): the keep stage scores generic claims ("It's a gap in trust.", "Control or
trust.") and Key Takeaways / FAQ restatements at P(keep) ~ 1.0, the same as facts, so a third of the output is slop.

    .venv/bin/python ws/keep_edit.py collect results/ws/keep_feats.npz [--names a,b]
        forward passes over the dev articles (data/slop_dev): MLP inputs at every '?' marker, P(keep), labels

The detector and edit steps are added once the mechanism is pinned down (see ws/README.md).
"""

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ws.dev_data import load, names  # noqa: E402
from ws.edits import Edited, apply_npz, save  # noqa: E402
from ws.model import Yap, chunks  # noqa: E402

KINDS = ["fact", "code", "rhetoric", "repeat", "heading", "boilerplate", "story", "cta", "question"]
LAYERS = tuple(range(12, 24))
SLOP_KINDS = [KINDS.index(k) for k in ("rhetoric", "heading", "boilerplate", "story", "cta", "question")]
LAYER, NEURONS = 23, (2664, 2416)   # the two quietest layer-23 neurons (max |act| 0.25 vs typical 1.76 on dev)


def collect_train(y, n=160, seed=1, layer=LAYER):
    """Marker activations on training chunks (all genres) that are NOT in ws/regress.py's 30 held-out chunks,
    with their keep labels: the 'content' side of the detector must cover chats, fiction, legal text and so on."""
    import json
    import random
    from ws.regress import build, heldout_chunks
    held = {json.dumps(c["units"])[:500] for c in heldout_chunks()}
    pool = []
    for f in sorted((ROOT / "data/claude_train").glob("part_*.jsonl")):
        pool += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    pool = [c for c in pool if json.dumps(c["units"])[:500] not in held]
    random.Random(seed).shuffle(pool)
    X, keep, genre = [], [], []
    for ch in pool[:n]:
        ids, pos = build(y, ch)[:2]
        tap = {}
        y.forward(ids, y.new_cache(), tap=tap)
        X.append(np.array(tap[("mlp_in", layer)][mx.array(pos)].astype(mx.float32)))
        keep += [bool(u["keep"]) for u in ch["units"]]
        genre += [ch["genre"]] * len(ch["units"])
    return dict(X=np.concatenate(X), keep=np.array(keep), genre=np.array(genre))


def lda_direction(slop, content, shrink):
    """Closed-form linear discriminant (no optimizer): pooled within-class covariance, shrunk toward a scaled
    identity, applied to the difference of class means. Whitening steers the detector away from directions along
    which real content varies (narrative prose, chat questions, legal clauses), which a plain mean difference picks up."""
    mu_s, mu_c = slop.mean(0), content.mean(0)
    S = np.cov(np.concatenate([slop - mu_s, content - mu_c]).T)
    S = (1 - shrink) * S + shrink * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    w = np.linalg.solve(S, mu_s - mu_c)
    return w / np.linalg.norm(w)


def directions(z, train=None, layer=LAYER, content=None, shrink=None, quantile=99.5):
    """Directions in the layer's MLP-input space:
       d  slop detector at the '?' markers: slop minus content (plain mean difference, or whitened with `shrink`
          and rescaled so content spans 10 units from its median to the threshold)
       q  marker direction: markers minus everything else (other stage-1 positions and the notes phase)
       c  mean marker direction (projection varies ~5% across markers), used as a bias carrier
    plus qm, cm = median projections at the markers."""
    X, kind = z[f"L{layer}"], z["kind"]
    train = np.ones(len(X), bool) if train is None else train
    slop, fact = np.isin(kind, SLOP_KINDS) & train, (kind == 0) & train
    pos = X[fact] if content is None else np.concatenate([X[fact], content])   # content: kept units from all genres
    if shrink is None:
        d = X[slop].mean(0) - pos.mean(0)
        d /= np.linalg.norm(d)
    else:
        d = lda_direction(X[slop], pos, shrink)
    if shrink is not None:                               # whitened scores are compressed: rescale so content's
        sc = pos @ d                                     # median-to-threshold span is 10 units, like the plain detector
        d = d * 10.0 / (np.percentile(sc, quantile) - np.median(sc))
    theta = float(np.percentile(pos @ d, quantile))      # threshold: above all but the top 0.5% of content
    rest = np.concatenate([z[f"other_L{layer}"], z[f"notes_L{layer}"]])
    q = X[train].mean(0) - rest.mean(0)
    q /= np.linalg.norm(q)
    c = X[train].mean(0)                       # mean marker activation: projection varies ~5% across markers
    c /= np.linalg.norm(c)
    return dict(d=d, q=q, c=c, qm=float(np.median(X[train] @ q)), cm=float(np.median(X[train] @ c)), theta=theta)


def no_direction(y):
    """Residual direction that raises logit(' no') - logit(' yes') through the final norm (unit vector)."""
    E, g = y.lm.model.embed_tokens.weight, y.lm.model.norm.weight
    w = np.array(((E[y.no] - E[y.yes]) * g).astype(mx.float32))
    return w / np.linalg.norm(w)


def build_edit(y, dirs, theta, delta, beta, kappa, lam, layer=LAYER, neurons=NEURONS):
    """Two quiet neurons rewired into a step: at a '?' marker, when the slop detector s = d.x passes theta, write a
    fixed push toward ' no' (keep logit down by ~K); below theta, nothing.
        gate_A = beta * (s - theta*q/qm - lam*(c/cm - q/qm))          (c/cm - q/qm ~ 0 at markers, > 1 in notes)
        gate_B = same with theta + delta
        up = q/qm (marker direction);  down_A = +kappa * w_no, down_B = -kappa * w_no
    A - B is a step of height ~beta*delta in the detector; the lam term keeps the gates shut while notes are written."""
    d, q, c, qm, cm = dirs["d"], dirs["q"], dirs["c"], dirs["qm"], dirs["cm"]
    w = no_direction(y)
    a, b = neurons
    row = lambda th: beta * (d - th / qm * q - lam * (c / cm - q / qm))
    gate = np.stack([row(theta), row(theta + delta)])
    up = np.stack([q / qm, q / qm])
    down = np.stack([kappa * w, -kappa * w], 1)
    return [dict(layer=layer, mat="gate", op="set_rows", rows=[a, b], values=gate),
            dict(layer=layer, mat="up", op="set_rows", rows=[a, b], values=up),
            dict(layer=layer, mat="down", op="set_cols", cols=[a, b], values=down)]


def _raw(y, ch):
    seq, marks = y.score_seq(ch)
    hn = y.forward(seq, y.new_cache())
    dl = hn[mx.array(marks)] @ (y.E[y.yes] - y.E[y.no])
    return list(np.array(dl.astype(mx.float32)))


def keep_logits(y, units, edits=None):
    """Stage-1 logit(' yes') - logit(' no') at every unit's marker (unsaturated), chunked like the binary,
    optionally with edits applied. P(keep) = sigmoid of this."""
    out = []
    for ch in chunks(y, units):
        if edits:
            with Edited(y, edits):
                out += _raw(y, ch)
        else:
            out += _raw(y, ch)
    return np.array(out)


def collect(y, sel, layers=LAYERS, other_layers=(), seed=0):
    """MLP inputs at the '?' marker of every unit (the position where P(keep) is read), chunked like the binary.
    other_layers: also sample non-marker positions of stage 1, and every position of teacher-forced gold notes
    (the notes phase), to check that an edit stays silent away from the markers."""
    rng = np.random.default_rng(seed)
    X = {l: [] for l in layers}
    O = {l: [] for l in other_layers}
    N = {l: [] for l in other_layers}
    meta = dict(p=[], label=[], kind=[], art=[], idx=[], ntok=[], chunk=[])
    nchunk = 0
    for a, n in enumerate(sel):
        d = load(n)
        units = [u["text"] for u in d["units"]]
        k = 0
        for ch in chunks(y, units):
            tap = {}
            p, seq, marks, cache = y.keep_scores(ch, tap=tap)
            mk = mx.array(marks)
            for l in layers:
                X[l].append(np.array(tap[("mlp_in", l)][mk].astype(mx.float32)))
            if other_layers:
                rest = np.setdiff1d(np.arange(len(seq)), marks)
                pick = mx.array(np.sort(rng.choice(rest, size=min(40, len(rest)), replace=False)))
                for l in other_layers:
                    O[l].append(np.array(tap[("mlp_in", l)][pick].astype(mx.float32)))
                labs = d["units"][k:k + len(ch)]
                toks = y.enc("notes:\n")
                for i, (u, lab) in enumerate(zip(ch, labs)):
                    if lab["keep"]:
                        toks += y.enc(f"u{i + 1} {u} =>") + y.enc(" " + lab["note"] + "\n")
                ntap = {}
                y.forward(toks, cache, tap=ntap)
                for l in other_layers:
                    N[l].append(np.array(ntap[("mlp_in", l)].astype(mx.float32)))
                del ntap
            for i, u in enumerate(ch):
                lab = d["units"][k]
                meta["p"].append(float(p[i]))
                meta["label"].append(bool(lab["keep"]))
                meta["kind"].append(KINDS.index(lab["kind"]))
                meta["art"].append(a)
                meta["idx"].append(k)
                meta["ntok"].append(len(y.enc(u)))
                meta["chunk"].append(nchunk)
                k += 1
            nchunk += 1
            del tap, cache
        print(f"{n}: {k} units", flush=True)
    out = {f"L{l}": np.concatenate(v) for l, v in X.items()}
    out.update({f"other_L{l}": np.concatenate(v) for l, v in O.items()})
    out.update({f"notes_L{l}": np.concatenate(v) for l, v in N.items()})
    out.update({k: np.array(v) for k, v in meta.items()})
    out["names"] = np.array(sel)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("out")
    c.add_argument("--names")
    c.add_argument("--other", default="", help="layers to also sample away from the markers, e.g. 23")
    e = sub.add_parser("edit", help="build the step edit from collected features and save it")
    e.add_argument("feats")
    e.add_argument("out")
    e.add_argument("--theta", type=float, default=-9.0)
    e.add_argument("--delta", type=float, default=2.0)
    e.add_argument("--beta", type=float, default=4.0)
    e.add_argument("--K", type=float, default=12.0, help="keep-logit drop for flagged units (step height)")
    e.add_argument("--lam", type=float, default=10.0)
    e.add_argument("--content", help="training-chunk features (collect_train) whose kept units join the content side")
    e.add_argument("--shrink", type=float, help="whiten the detector (closed-form LDA) with this shrinkage")
    ap.add_argument("--gguf", default=str(ROOT / "models/ws/v4c.gguf"))
    ap.add_argument("--base-edit", help="edit file applied first (e.g. Edit 1), so directions match the final model")
    a = ap.parse_args()
    y = Yap(a.gguf)
    if a.base_edit:
        apply_npz(y, a.base_edit)
    if a.cmd == "collect":
        sel = a.names.split(",") if a.names else names()
        other = tuple(int(x) for x in a.other.split(",") if x)
        np.savez(a.out, **collect(y, sel, other_layers=other))
        print("saved", a.out)
    elif a.cmd == "edit":
        z = np.load(a.feats)
        content = None
        if a.content:
            t = np.load(a.content)
            content = t["X"][t["keep"]]
        dirs = directions(z, content=content, shrink=a.shrink)
        theta = dirs["theta"] if a.shrink is not None else a.theta
        kappa = 0.5 * a.K / 26.6          # measured: kappa 0.5 with beta*delta = 8 drops the keep logit by ~26.6
        P = dict(theta=theta, delta=a.delta, beta=a.beta, kappa=kappa, lam=a.lam)
        save(build_edit(y, dirs, **P), a.out,
             meta=dict(P, K=a.K, shrink=a.shrink, layer=LAYER, neurons=NEURONS, feats=a.feats, content=a.content,
                       names=[str(n) for n in z["names"]],
                       note="slop step: two quiet layer-23 neurons push the keep score down by ~K when the slop "
                            "detector passes theta at a '?' marker"))
        print("saved", a.out)


if __name__ == "__main__":
    main()
