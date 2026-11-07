"""Trace note generation token by token (MLX copy of the binary): for each generated token, is it in the unit,
and how far behind was the best token that is in the unit? Used to size a copy (grounding) edit.

    from ws.notes_trace import trace
    steps = trace(y, units, keep)      # keep: list of bools (stage-1 decisions to use)
"""

import re

import mlx.core as mx
import numpy as np

NEUTRAL = re.compile(r"^\s*([^\w]*|NOT|NEVER|ONLY|IF|UNLESS|BECAUSE|VS|BUT|w/|w/o|=|->|~|\+|/|;|:|,|\.|'s|s)\s*$")


def unit_vocab(y, unit):
    """Token ids that spell words of the unit (with and without a leading space, lower/upper/title case)."""
    ids = set(y.enc(unit)) | set(y.enc(" " + unit))
    for w in set(re.findall(r"[A-Za-z0-9][\w'.-]*", unit)):
        for v in {w, w.lower(), w.capitalize(), w.upper()}:
            ids |= set(y.enc(v)) | set(y.enc(" " + v))
    return ids


def trace(y, units, keep, max_note_tokens=96):
    """Greedy notes exactly as y.process, recording per step: unit index, chosen token, grounded?, margin to the
    best grounded token (logit chosen - logit best grounded; 0 when the chosen token is grounded)."""
    p, seq, marks, cache = y.keep_scores(units)
    steps, notes = [], {}
    first = True
    for i, u in enumerate(units):
        if not keep[i]:
            continue
        prompt = (y.enc("notes:\n") if first else []) + y.enc(f"u{i + 1} {u} =>")
        first = False
        hn = y.forward(prompt, cache)
        Vs = unit_vocab(y, u)
        V = np.array(sorted(Vs))
        toks, ended = [], False
        for _ in range(max_note_tokens):
            lg = y.logits(hn[-1])
            t = int(mx.argmax(lg).item())
            if t in y.eog:
                break
            pc = y.piece(t)
            if b"\n" in pc:
                y.forward([t], cache)
                ended = True
                break
            txt = pc.decode("utf-8", errors="replace")
            grounded = t in Vs
            neutral = bool(NEUTRAL.match(txt))
            lgn = np.array(lg.astype(mx.float32))
            best_g = int(V[np.argmax(lgn[V])])
            steps.append(dict(unit=i, step=len(toks), tok=t, text=txt, grounded=grounded, neutral=neutral,
                              margin=float(lgn[t] - lgn[best_g]), best_grounded=y.piece(best_g).decode("utf-8", "replace")))
            toks.append(t)
            hn = y.forward([t], cache)
        if not ended:
            y.forward(y.nl, cache)
        notes[i] = y.text(toks).strip()
    return steps, notes, p
