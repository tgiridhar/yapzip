"""yapzip's fine-tuned GGUF loaded into MLX, run exactly like native/yapzip.cpp, with activation taps.

Weight-space programming works on this copy: read activations with forward passes, edit tensors in place,
then write the edited tensors back into a GGUF (ws/gguf_write.py) for the native binary.

    from ws.model import Yap
    y = Yap("models/ws/v4c.gguf")
    res = y.process(["[09:14] marcus: payments-api v4.12.0 went out", "lol"])
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "native/llama.cpp/gguf-py"))

from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402
from mlx_lm.models import qwen3_5  # noqa: E402
from mlx_lm.models.base import create_attention_mask, create_ssm_mask  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

BASE = ROOT / "models/qwen35-0.8b-bf16"   # config + tokenizer only; weights come from the GGUF

# GGUF per-layer tensor name -> mlx-lm name (llama.cpp's convert_hf_to_gguf, reversed)
GG2MLX = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
}


def gguf_to_mlx_name(name):
    if name == "token_embd.weight":
        return "language_model.model.embed_tokens.weight"
    if name == "output_norm.weight":
        return "language_model.model.norm.weight"
    _, layer, suffix = name.split(".", 2)
    return f"language_model.model.layers.{layer}.{GG2MLX[suffix]}"


def _bytes_decoder():
    # GPT-2 byte-level BPE: printable unicode char -> raw byte
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class Yap:
    def __init__(self, gguf_path, dtype=mx.float32):
        self.path = str(gguf_path)
        cfg = json.loads((BASE / "config.json").read_text())
        self.model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(cfg))
        self.lm = self.model.language_model
        self.n_layers = len(self.lm.model.layers)
        self.tensors = {}   # GGUF name -> (ReaderTensor)
        weights = []
        reader = GGUFReader(self.path)
        for t in reader.tensors:
            if t.name.startswith(f"blk.{self.n_layers}."):   # MTP layer: not used by yapzip
                continue
            self.tensors[t.name] = t
            a = dequantize(t.data, t.tensor_type).astype(np.float32)
            if t.name.endswith("ssm_a"):
                a = np.log(-a)                 # converter stored -exp(A_log)
            elif t.name.endswith("ssm_conv1d.weight"):
                a = a[:, :, None]              # (channels, kernel) -> (channels, kernel, 1)
            weights.append((gguf_to_mlx_name(t.name), mx.array(a).astype(dtype)))
        self.model.load_weights(weights, strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())

        self.tok = Tokenizer.from_file(str(BASE / "tokenizer.json"))
        self.special = {i for i, t in self.tok.get_added_tokens_decoder().items() if t.special}
        dec = _bytes_decoder()
        vocab = self.tok.get_vocab()
        self.pieces = [b""] * len(vocab)
        for s, i in vocab.items():
            self.pieces[i] = b"" if i in self.special else bytes(dec.get(c, 0) for c in s)
        self.eog = {i for s, i in vocab.items() if s in ("<|im_end|>", "<|endoftext|>", "<|end_of_text|>")}
        self.yes, self.no, self.q = self.single(" yes"), self.single(" no"), self.single("?")
        self.nl = self.enc("\n")
        self.E = self.lm.model.embed_tokens.weight   # tied: also the output head

    # ---- tokens ----
    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids if s else []

    def single(self, s):
        t = self.enc(s)
        assert len(t) == 1, (s, t)
        return t[0]

    def piece(self, t):
        return self.pieces[t]

    def text(self, toks):
        return b"".join(self.pieces[t] for t in toks).decode("utf-8", errors="replace")

    # ---- forward with taps ----
    def forward(self, toks, cache, tap=None, hook=None):
        """Run tokens through the model, appending to `cache`. Returns final-normed hidden states (L, 1024).
        tap: dict to fill with per-layer arrays: tap[("resid", l)] block output, tap[("mlp_in", l)] normed MLP
        input, tap[("mlp_act", l)] swiglu activation (input of down_proj), tap[("mlp_out", l)], tap[("attn_out", l)].
        hook(l, kind, x) -> x may rewrite an activation on the fly (for causal tests; never used for edits)."""
        m = self.lm.model
        h = m.embed_tokens(mx.array(toks)[None])
        fa_mask = create_attention_mask(h, cache[m.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[m.ssm_idx])
        for l, (layer, c) in enumerate(zip(m.layers, cache)):
            x = layer.input_layernorm(h)
            if layer.is_linear:
                r = layer.linear_attn(x, ssm_mask, c)
            else:
                r = layer.self_attn(x, fa_mask, c)
            if hook:
                r = hook(l, "attn_out", r)
            if tap is not None:
                tap[("attn_out", l)] = r[0]
            h = h + r
            x = layer.post_attention_layernorm(h)
            if tap is not None:
                tap[("mlp_in", l)] = x[0]
            mlp = layer.mlp
            act = nn.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
            if hook:
                act = hook(l, "mlp_act", act)
            o = mlp.down_proj(act)
            if hook:
                o = hook(l, "mlp_out", o)
            if tap is not None:
                tap[("mlp_act", l)] = act[0]
                tap[("mlp_out", l)] = o[0]
            h = h + o
            if hook:
                h = hook(l, "resid", h)
            if tap is not None:
                tap[("resid", l)] = h[0]
        return m.norm(h)[0]

    def new_cache(self):
        return self.model.make_cache()

    def logits(self, hn):
        return hn @ self.E.T

    # ---- yapzip, as in native/yapzip.cpp (appended score mode, restate notes mode) ----
    def score_seq(self, units):
        seq = self.enc("units:\n")
        for i, u in enumerate(units):
            seq += self.enc(f"u{i + 1} {u}\n")
        seq += self.enc("keep?\n")
        marks = []
        for i in range(len(units)):
            seq += self.enc(f" u{i + 1}")
            seq.append(self.q)
            marks.append(len(seq) - 1)
        seq += self.nl
        return seq, marks

    def keep_scores(self, units, cache=None, tap=None, hook=None):
        cache = cache if cache is not None else self.new_cache()
        seq, marks = self.score_seq(units)
        hn = self.forward(seq, cache, tap=tap, hook=hook)
        hm = hn[mx.array(marks)]
        d = hm @ (self.E[self.yes] - self.E[self.no])
        p = mx.sigmoid(d)
        mx.eval(p)
        return np.array(p), seq, marks, cache

    def process(self, units, threshold=0.5, max_note_tokens=256, hook=None, keep=None, want_top=False):
        """Returns dict(p, notes, out). keep: optional list of bools overriding stage 1."""
        p, seq, marks, cache = self.keep_scores(units, hook=hook)
        notes = {}
        out = []
        first = True
        for i, u in enumerate(units):
            k = keep[i] if keep is not None else p[i] >= threshold
            if not k:
                continue
            prompt = (self.enc("notes:\n") if first else []) + self.enc(f"u{i + 1} {u} =>")
            first = False
            hn = self.forward(prompt, cache, hook=hook)
            lg = self.logits(hn[-1])
            t = int(mx.argmax(lg).item())
            top = None
            if want_top:
                pr = mx.softmax(lg)
                idx = mx.argsort(-lg)[:3]
                top = [(self.text([int(j)]), float(pr[int(j)])) for j in idx.tolist()]
            note = b""
            toks = []
            ended = False
            for _ in range(max_note_tokens):
                if t in self.eog:
                    break
                pc = self.piece(t)
                if b"\n" in pc:
                    note += pc[: pc.index(b"\n")]
                    self.forward([t], cache, hook=hook)
                    ended = True
                    break
                note += pc
                toks.append(t)
                hn = self.forward([t], cache, hook=hook)
                t = int(mx.argmax(self.logits(hn[-1])).item())
            if not ended:
                self.forward(self.nl, cache, hook=hook)
            s = note.decode("utf-8", errors="replace").strip()
            notes[i] = dict(note=s, toks=toks, top=top)
            out.append(u if s == "=" else s)
        return dict(p=p, notes=notes, out="".join(o + "\n" for o in out))


def split_units(text):
    """stdin -> units, as in native/yapzip.cpp: one per non-empty line; a ``` fenced block is one unit."""
    units, fence, in_fence = [], "", False
    for line in text.split("\n"):
        if line.endswith("\r"):
            line = line[:-1]
        t = line.strip()
        if in_fence:
            fence += "\n" + line
            if t.startswith("```"):
                in_fence = False
                units.append(fence)
            continue
        if t.startswith("```"):
            in_fence, fence = True, t
            continue
        if t:
            units.append(t)
    if in_fence:
        units.append(fence)
    return units


def chunks(yap, units, block=600):
    out, cur, n = [], [], 0
    for u in units:
        k = len(yap.enc(u))
        if cur and n + k > block:
            out.append(cur)
            cur, n = [], 0
        cur.append(u)
        n += k
    if cur:
        out.append(cur)
    return out


def compress(yap, text, **kw):
    res = []
    for ch in chunks(yap, split_units(text)):
        res.append((ch, yap.process(ch, **kw)))
    return res
