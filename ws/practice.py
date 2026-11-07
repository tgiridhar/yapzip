"""Practice data for weight edits: real Reddit threads (data/threads) rewritten with the author on its own line,
the way forum pages and copy-pasted threads look, so every comment becomes a header unit + a body unit.

Training data always had the author inline ("handle (+3): text"), so standalone header lines are new to the model.
"""

import re
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ws.model import chunks, split_units  # noqa: E402

LINE = re.compile(r"^(↳ )?(\S+) \(\+?(-?\d+)\): (.*)$")

# header styles: (top-level comment, reply). {h} handle, {n} score, {p} parent handle
STYLES = [
    ("{h} · {n} points", "{h} · {n} points · reply to {p}"),
    ("u/{h} · {n} pts", "↳ u/{h} · {n} pts"),
    ("{h} ({n} upvotes)", "{h} replied to {p} ({n} upvotes)"),
    ("Posted by {h}", "Reply from {h} to {p}"),
    ("{h} | {n} points | 3h", "{h} | {n} points | replying to {p}"),
    ("Comment by {h} ({n})", "Reply by {h} ({n})"),
]


def thread_files(split):
    files = sorted((ROOT / "data/threads").glob("*.txt"))
    if split == "analysis":
        return files[:20]
    if split == "heldout":
        return files[20:] + sorted((ROOT / "data/threads/eval").glob("*.txt"))
    raise ValueError(split)


def to_headers(path, style):
    """-> (text, pairs) where pairs lists (header_line, body_line) in order."""
    top, rep = STYLES[style]
    out, pairs, prev = [], [], None
    for line in Path(path).read_text().splitlines():
        m = LINE.match(line.strip())
        if not m:
            if line.strip():
                out.append(line.strip())
            continue
        is_reply, h, n, body = m.groups()
        hdr = (rep if is_reply and prev else top).format(h=h, n=n, p=prev)
        out += [hdr, body]
        pairs.append((hdr, body))
        prev = h
    return "\n".join(out) + "\n", pairs


def practice_chunks(y, split, styles=None):
    """Yields (thread name, style, chunk units, set of header indices in the chunk)."""
    for k, f in enumerate(thread_files(split)):
        st = (k % len(STYLES)) if styles is None else styles[k % len(styles)]
        text, pairs = to_headers(f, st)
        hdrs = {h for h, _ in pairs}
        for ch in chunks(y, split_units(text)):
            idx = {i for i, u in enumerate(ch) if u in hdrs and i + 1 < len(ch)}
            yield f.stem, st, ch, idx


def leaks(header, note):
    """A header note leaks if it carries words that are not in the header (content pulled from other units)."""
    if note is None or note == "=":
        return False
    hw = set(re.findall(r"[a-z0-9_\-]+", header.lower()))
    nw = set(re.findall(r"[a-z0-9_\-]+", note.lower()))
    return len(nw - hw - {"u", "points", "pts", "reply", "to", "by"}) > 0


def notes_tapped(y, units, keep, tap_units, layers=None, hook=None, max_note_tokens=96):
    """Stage 2 of yapzip (as in Yap.process) with stage-1 decisions given by `keep`.
    For units in tap_units, records the residual stream (every layer) and MLP activations (layers) at the
    position that predicts the first note token. Returns (notes, taps)."""
    layers = range(y.n_layers) if layers is None else layers
    p, seq, marks, cache = y.keep_scores(units, hook=hook)
    notes, taps, first = {}, {}, True
    eq = y.single("=") if len(y.enc("=")) == 1 else None
    for i, u in enumerate(units):
        if not keep[i]:
            continue
        prompt = (y.enc("notes:\n") if first else []) + y.enc(f"u{i + 1} {u} =>")
        first = False
        tap = {} if i in tap_units else None
        hn = y.forward(prompt, cache, tap=tap, hook=hook)
        lg = y.logits(hn[-1])
        t = int(mx.argmax(lg).item())
        if tap is not None:
            lp = lg - mx.logsumexp(lg)
            rec = dict(
                resid=np.stack([np.array(tap[("resid", l)][-1]) for l in range(y.n_layers)]),
                act={l: np.array(tap[("mlp_act", l)][-1]) for l in layers},
                lp_eq=float(lp[eq]), top=t, lp_top=float(lp[t]),
            )
            taps[i] = rec
        note, ended = b"", False
        for _ in range(max_note_tokens):
            if t in y.eog:
                break
            pc = y.piece(t)
            if b"\n" in pc:
                note += pc[: pc.index(b"\n")]
                y.forward([t], cache, hook=hook)
                ended = True
                break
            note += pc
            hn = y.forward([t], cache, hook=hook)
            t = int(mx.argmax(y.logits(hn[-1])).item())
        if not ended:
            y.forward(y.nl, cache, hook=hook)
        notes[i] = note.decode("utf-8", errors="replace").strip()
    return p, notes, taps
