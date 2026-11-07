"""Toy pilot of the two-stage design on Qwen3.5-0.8B with MLX: keep/drop score per unit + note per kept unit.

Sequence layout (plain text, no new tokens):
    units:
    u1 Hey everyone, welcome back to the blog!?        <- "?" (token 30) closes each unit; at that position
    u2 We moved 12 services from MySQL to Redis in 3 weeks.?    the model predicts " yes" (keep) or " no" (drop)
    u3 ```bash⏎pip install tinyq==1.2.0⏎```?
    notes:
    u2 We moved 12 services from MySQL to Redis in 3 weeks. =>  moved 12 services MySQL -> Redis in 3 weeks
    u3 ```bash⏎pip install tinyq==1.2.0⏎``` =>  =      <- "=" means copy the unit verbatim
Stage 1 is one forward pass over the units. Stage 2: code restates each kept unit, the model writes its note.
Loss = cross-entropy on yes/no at each "?" + cross-entropy on the note tokens. LoRA on the last block only
(the last block is full attention; MLX has no fast backward through the DeltaNet blocks).

Usage: uv run python bench/toy_train.py --train data/toy/train.jsonl --steps 400 --layers 1
"""

import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "qwen35-0.8b-bf16"
OUT = ROOT / "results" / "toy"
YES, NO, Q = 9542, 874, 30
GREEDY = make_sampler(temp=0.0)


def enc(tok, s):
    return tok.encode(s, add_special_tokens=False)


def score_ids(tok, units):
    ids, pos = enc(tok, "units:\n"), []
    for u in units:
        ids += enc(tok, f"u{u['id']} {u['text']}") + [Q]
        pos.append(len(ids) - 1)
        ids += enc(tok, "\n")
    return ids, pos


def restate(tok, u):
    return enc(tok, f"u{u['id']} {u['text']} =>")


def build_train(tok, chunk):
    units = chunk["units"]
    ids, pos = score_ids(tok, units)
    labels = [YES if u["keep"] else NO for u in units]
    ids += enc(tok, "notes:\n")
    tgt = []
    for u in units:
        if u["keep"]:
            ids += restate(tok, u)
            note = enc(tok, f" {u['note']}\n")
            tgt += list(range(len(ids), len(ids) + len(note)))
            ids += note
    return ids, pos, labels, tgt


def hidden(model, ids):
    return model.language_model.model(mx.array(ids)[None])[0]


def logits_at(model, h, idx):
    """Output head only where needed; full T x 248k logits are what used up memory."""
    return model.language_model.model.embed_tokens.as_linear(h[mx.array(idx)])


def loss_fn(model, ids, pos, labels, tgt):
    h = hidden(model, ids[:-1])
    s_loss = nn.losses.cross_entropy(logits_at(model, h, pos), mx.array(labels)).mean()
    n_loss = nn.losses.cross_entropy(logits_at(model, h, [t - 1 for t in tgt]),
                                     mx.array([ids[t] for t in tgt])).mean()
    return s_loss + n_loss, (s_loss, n_loss)


def p_keep(model, tok, units):
    ids, pos = score_ids(tok, units)
    logits = logits_at(model, hidden(model, ids), pos)
    return mx.softmax(logits[:, mx.array([YES, NO])].astype(mx.float32), axis=-1)[:, 0].tolist()


def write_notes(model, tok, units, kept_ids):
    """Returns {unit id: note}. Generation stops at the first newline."""
    ids, _ = score_ids(tok, units)
    ids += enc(tok, "notes:\n")
    notes = {}
    for u in units:
        if u["id"] not in kept_ids:
            continue
        ids += restate(tok, u)
        note = ""
        for r in stream_generate(model, tok, prompt=ids, max_tokens=48, sampler=GREEDY):
            note += r.text
            if "\n" in note:
                break
        note = note.split("\n")[0].strip()
        ids += enc(tok, f" {note}\n")
        notes[u["id"]] = note
    return notes


def render(units, kept_notes):
    """Final output: the note, or the unit itself when the note is '='."""
    return "\n".join(u["text"] if kept_notes[u["id"]] == "=" else kept_notes[u["id"]]
                     for u in units if u["id"] in kept_notes)


def nums(s):
    return sorted(re.findall(r"\d+(?:[.:]\d+)*", s or ""))


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def evaluate(model, tok, chunks, tag):
    model.eval()
    g = defaultdict(lambda: defaultdict(float))
    examples, t_score, t_notes = {}, 0.0, 0.0
    for c in chunks:
        units, s = c["units"], g[c["genre"]]
        t0 = time.time()
        p = p_keep(model, tok, units)
        t_score += time.time() - t0
        kept_ids = [u["id"] for u, pk in zip(units, p) if pk >= 0.5]
        t0 = time.time()
        notes = write_notes(model, tok, units, kept_ids)
        t_notes += time.time() - t0
        for u, pk in zip(units, p):
            s["units"] += 1
            s["keep_ok"] += (pk >= 0.5) == u["keep"]
            if u["keep"] and u["id"] in notes:
                gold_out = u["text"] if u["note"] == "=" else u["note"]
                got_out = u["text"] if notes[u["id"]] == "=" else notes[u["id"]]
                s["notes"] += 1
                s["exact"] += got_out == gold_out
                s["nums_ok"] += nums(got_out) == nums(gold_out)
                s["not_ok"] += ("NOT" in gold_out) == ("NOT" in got_out)
                if u["note"] == "=":
                    s["verbatim_n"] += 1
                    s["verbatim_ok"] += got_out == u["text"]
            if u["keep"] and u["id"] not in notes:
                s["fact_dropped"] += 1
        s["in_tok"] += len(score_ids(tok, units)[0])
        s["out_tok"] += len(enc(tok, render(units, notes))) if notes else 0
        if c["genre"] not in examples:
            examples[c["genre"]] = {"units": units, "p_keep": [round(x, 2) for x in p], "notes": notes,
                                    "output": render(units, notes)}
    rows = {}
    for genre, s in g.items():
        rows[genre] = {"keep_acc": round(s["keep_ok"] / s["units"], 3),
                       "notes_exact": round(s["exact"] / max(1, s["notes"]), 3),
                       "numbers_ok": round(s["nums_ok"] / max(1, s["notes"]), 3),
                       "NOT_ok": round(s["not_ok"] / max(1, s["notes"]), 3),
                       "verbatim_ok": (round(s["verbatim_ok"] / s["verbatim_n"], 3) if s["verbatim_n"] else None),
                       "facts_dropped": int(s["fact_dropped"]),
                       "output_vs_input_tokens": round(s["out_tok"] / s["in_tok"], 2)}
    res = {"tag": tag, "per_genre": rows, "score_ms_per_chunk": round(1000 * t_score / len(chunks)),
           "notes_s_per_chunk": round(t_notes / len(chunks), 2)}
    print(json.dumps(res), flush=True)
    return res, examples


def fit(tok, chunks, max_tokens):
    """Split chunks whose training sequence is longer than max_tokens into halves (ids renumbered).
    Training memory grows with sequence length in the DeltaNet blocks that get gradients."""
    out, todo = [], list(chunks)
    while todo:
        c = todo.pop()
        if len(build_train(tok, c)[0]) <= max_tokens or len(c["units"]) < 2:
            out.append(c)
            continue
        mid = len(c["units"]) // 2
        for part in (c["units"][:mid], c["units"][mid:]):
            todo.append({**c, "units": [{**u, "id": i + 1} for i, u in enumerate(part)]})
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=[str(ROOT / "data/toy/train.jsonl")])
    ap.add_argument("--test", default=str(ROOT / "data/toy/test.jsonl"), help="jsonl, or 'holdout' to split --train")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--layers", type=int, default=1, help="LoRA on the last N blocks")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=800, help="split training chunks longer than this (memory)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = [json.loads(l) for f in args.train for l in open(f) if l.strip()]
    if args.test == "holdout":
        random.Random(1).shuffle(data)
        n_test = max(10, len(data) // 10)
        test, train = data[:n_test], data[n_test:]
    else:
        train, test = data, [json.loads(l) for l in open(args.test)]
    # Every chunk has a different length, so MLX's buffer cache keeps growing: uncapped it reached 12 GB and the
    # Mac swapped; a 2 GB cap made big steps thrash (90 s/step). 8 GB plus a clear every 50 steps holds both off.
    mx.set_cache_limit(8 * 1024 ** 3)
    model, tok = load(str(MODEL))
    train = fit(tok, train, args.max_tokens)
    print(f"train {len(train)} chunks (after splitting long ones), test {len(test)} chunks", flush=True)

    base, base_ex = (None, None) if args.skip_base else evaluate(model, tok, test, "base (untrained)")

    model.freeze()
    linear_to_lora_layers(model, args.layers, {"rank": args.rank, "scale": 10.0, "dropout": 0.0})
    print(f"LoRA params: {sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 1e6:.2f}M", flush=True)
    opt = optim.AdamW(learning_rate=1e-4)
    step_fn = nn.value_and_grad(model, loss_fn)
    # Frozen blocks stay in eval mode (fast DeltaNet kernels, no gradient needed). Blocks with LoRA go to train
    # mode, which makes their DeltaNet use the slower differentiable path.
    model.eval()
    for layer in model.layers[-args.layers:]:
        layer.train()
    (out / "config.json").write_text(json.dumps({"layers": args.layers, "rank": args.rank, "steps": args.steps,
                                                 "train": args.train}))
    random.seed(0)
    t0, log = time.time(), []
    for step in range(1, args.steps + 1):
        ids, pos, labels, tgt = build_train(tok, random.choice(train))
        (loss, (sl, nl)), grads = step_fn(model, ids, pos, labels, tgt)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if step % 50 == 0:
            mx.clear_cache()
        if step % 100 == 0:
            mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
        if step % 10 == 0:
            log.append({"step": step, "loss": round(loss.item(), 4), "score_loss": round(sl.item(), 4),
                        "notes_loss": round(nl.item(), 4), "s": round(time.time() - t0, 1)})
            print(json.dumps(log[-1]), flush=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    (out / "config.json").write_text(json.dumps({"layers": args.layers, "rank": args.rank, "steps": args.steps,
                                                 "train": args.train}))
    tuned, tuned_ex = evaluate(model, tok, test, f"LoRA (last {args.layers} blocks) after {args.steps} steps")
    (out / "report.json").write_text(json.dumps({"base": base, "tuned": tuned, "train_log": log,
                                                 "examples_base": base_ex, "examples_tuned": tuned_ex},
                                                indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
