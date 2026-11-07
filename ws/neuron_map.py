"""Neuron map at the keep/drop markers: what separates facts from slop, chrome and repeats, layer by layer.

    .venv/bin/python ws/neuron_map.py MODEL.gguf OUT.json [--docs data/relabel_v4/slop_dev,data/pasted/v4]

Runs the model on labeled documents (units with a `kind`) cut into <=300-token blocks exactly like the binary,
and reads the residual stream and the MLP neurons at each unit's "?" marker. Reports:
- how the model treats each kind now (share kept at P >= 0.5)
- per layer: held-out AUC of a linear probe (shrunk LDA, documents split in folds) for
  fact-vs-slop (rhetoric, heading, story, cta, question), chrome (boilerplate) vs rest, repeat vs rest
- top single neurons (MLP activations) per task, with layer and AUC
- quiet neurons per layer (smallest 99.9th-percentile |activation| over all positions of training chunks):
  free slots for weight edits
Read-only: forward passes only, nothing is trained or edited.
"""

import json
import random
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ws.model import Yap, chunks  # noqa: E402

SLOP = {"rhetoric", "heading", "story", "cta", "question"}
TASKS = {"fact_vs_slop": (lambda k: k == "fact", lambda k: k in SLOP),
         "chrome_vs_rest": (lambda k: k == "boilerplate", lambda k: k != "boilerplate"),
         "repeat_vs_rest": (lambda k: k == "repeat", lambda k: k != "repeat")}


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(np.float64) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / max(1, len(pos) * len(neg)))


def load_docs(dirs):
    """Labeled documents from the given folders, in order; a document name seen in an earlier folder wins."""
    docs, seen = [], set()
    for d in dirs:
        for p in sorted((ROOT / d).glob("*.json")):
            doc = json.loads(p.read_text())
            if p.stem not in seen and all(u.get("kind") for u in doc["units"]):
                docs.append((p.stem, doc))
                seen.add(p.stem)
    return docs


def collect(y, docs):
    rows = []   # (doc index, kind, p_keep)
    resid, acts = [], []
    for di, (name, doc) in enumerate(docs):
        units, i = doc["units"], 0
        for ch in chunks(y, [u["text"] for u in units], block=300):
            part, i = units[i:i + len(ch)], i + len(ch)
            tap = {}
            p, seq, marks, _ = y.keep_scores(ch, tap=tap)
            m = mx.array(marks)
            resid.append(np.stack([np.array(tap[("resid", l)][m]) for l in range(24)], 1).astype(np.float32))
            acts.append(np.stack([np.array(tap[("mlp_act", l)][m]) for l in range(24)], 1).astype(np.float16))
            rows += [(di, u["kind"], float(pk)) for u, pk in zip(part, p)]
    return rows, np.concatenate(resid), np.concatenate(acts)


def probe_auc(X, yb, groups, shrink=0.2, folds=5):
    """Held-out AUC of a shrunk-LDA direction; documents (not units) are split across folds."""
    g = np.unique(groups)
    random.Random(0).shuffle(g := list(g))
    scores = np.zeros(len(yb))
    for f in range(folds):
        test = np.isin(groups, g[f::folds])
        tr = ~test
        a, b = X[tr & yb], X[tr & ~yb]
        if len(a) < 3 or len(b) < 3 or not test.any():
            continue
        cov = np.cov(np.concatenate([a - a.mean(0), b - b.mean(0)]).T)
        cov = (1 - shrink) * cov + shrink * np.trace(cov) / len(cov) * np.eye(len(cov))
        w = np.linalg.solve(cov, a.mean(0) - b.mean(0))
        scores[test] = X[test] @ w
    return auc(scores[yb], scores[~yb])


def quiet_neurons(y, n_chunks=40, k=8):
    rows = [json.loads(l) for f in sorted((ROOT / "data/claude_train").glob("part_*.jsonl")) for l in open(f)]
    random.Random(1).shuffle(rows)
    mx_abs = np.zeros((24, 3584), np.float32)
    for c in rows[:n_chunks]:
        tap = {}
        y.keep_scores([u["text"] for u in c["units"]], tap=tap)
        for l in range(24):
            a = np.abs(np.array(tap[("mlp_act", l)]))
            mx_abs[l] = np.maximum(mx_abs[l], np.quantile(a, 0.999, axis=0))
    return {l: [(int(n), round(float(mx_abs[l, n]), 4)) for n in mx_abs[l].argsort()[:k]] for l in range(24)}


def main():
    model, out = sys.argv[1], sys.argv[2]
    dirs = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--docs=")),
                ["data/slop_dev", "data/pasted"])
    y = Yap(model)
    docs = load_docs(dirs)
    rows, R, A = collect(y, docs)
    kinds = np.array([k for _, k, _ in rows])
    groups = np.array([d for d, _, _ in rows])
    pk = np.array([p for _, _, p in rows])
    res = {"model": model, "docs": len(docs), "units": len(rows), "kept_share_by_kind": {}, "probe_auc": {}, "top_neurons": {}}
    for k in sorted(set(kinds)):
        sel = kinds == k
        res["kept_share_by_kind"][k] = [int(sel.sum()), round(float((pk[sel] >= 0.5).mean()), 3)]
    for task, (fp, fn) in TASKS.items():
        pos = np.array([fp(k) for k in kinds])
        neg = np.array([fn(k) for k in kinds]) & ~pos
        sel = pos | neg
        if pos.sum() < 5 or neg.sum() < 5:
            continue
        res["probe_auc"][task] = [round(probe_auc(R[sel, l], pos[sel], groups[sel]), 3) for l in range(24)]
        best = []
        for l in range(24):
            a = A[sel, l].astype(np.float32)
            s_pos, s_neg = a[pos[sel]], a[~pos[sel]]
            for n in range(a.shape[1]):
                v = auc(s_pos[:, n], s_neg[:, n])
                best.append((abs(v - 0.5), l, n, v))
        best.sort(reverse=True)
        res["top_neurons"][task] = [(l, n, round(v, 3)) for _, l, n, v in best[:12]]
    res["quiet_neurons"] = quiet_neurons(y)
    Path(out).write_text(json.dumps(res, indent=1))
    print(f"{model}: {res['docs']} docs, {res['units']} units")
    print("kept share by kind (n, share kept):", res["kept_share_by_kind"])
    for task, v in res["probe_auc"].items():
        bl = int(np.argmax(v))
        print(f"{task:15s} probe AUC by layer: best L{bl} {v[bl]}  | L0 {v[0]} L6 {v[6]} L12 {v[12]} L18 {v[18]} L23 {v[23]}")
        print(f"   top neurons (layer, neuron, AUC): {res['top_neurons'][task][:6]}")
    print("quiet neurons L20-23:", {l: res["quiet_neurons"][l][:4] for l in (20, 21, 22, 23)})


if __name__ == "__main__":
    main()
