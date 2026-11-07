"""Weight edits on the MLX copy of yapzip, and writing them back into a GGUF for the native binary.

An edit is a dict:
  {"layer": 19, "mat": "down", "op": "dir_cols", "cols": [...], "u": (1024,), "alpha": 6.0}
      W[:, cols] -= alpha * u (u^T W[:, cols])   (the column form from "Sparks of Weight-Space Programming":
      the selected neurons' write along unit direction u is scaled by 1 - alpha)
  {"layer": 22, "mat": "down", "op": "set_cols", "cols": [...], "values": (1024, k)}
  {"layer": 22, "mat": "gate"|"up", "op": "set_rows", "rows": [...], "values": (k, 1024)}
  {"layer": 11, "mat": "o", "op": "scale_cols", "cols": [...], "alpha": 1.3}   (scale one attention head's output)
Edited tensors are rounded through Q8_0, so the MLX copy computes with exactly what the binary will load.
"""

import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "native/llama.cpp/gguf-py"))

from gguf import GGMLQuantizationType, GGUFReader  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402

MATS = {"down": ("mlp", "down_proj", "ffn_down"), "gate": ("mlp", "gate_proj", "ffn_gate"), "up": ("mlp", "up_proj", "ffn_up"),
        "o": ("self_attn", "o_proj", "attn_output"), "v": ("self_attn", "v_proj", "attn_v")}   # o, v: full-attention layers


def _module(y, layer, mat):
    sub, attr, _ = MATS[mat]
    return getattr(getattr(y.lm.model.layers[layer], sub), attr)


def gguf_name(layer, mat):
    return f"blk.{layer}.{MATS[mat][2]}.weight"


def q8(w):
    """Round a float matrix through Q8_0 (blocks of 32 along each row), as the GGUF stores it."""
    return dequantize(quantize(np.ascontiguousarray(w, dtype=np.float32), GGMLQuantizationType.Q8_0),
                      GGMLQuantizationType.Q8_0).reshape(w.shape)


def edited_weights(y, edits):
    """-> {(layer, mat): new float32 matrix} computed from the CURRENT (original) weights."""
    out = {}
    for e in edits:
        key = (e["layer"], e["mat"])
        if key not in out:
            out[key] = np.array(_module(y, *key).weight, dtype=np.float32)
        W = out[key]
        if e["op"] == "dir_cols":
            cols = np.asarray(e["cols"], dtype=np.int64)
            u = np.asarray(e["u"], dtype=np.float32)
            u = u / np.linalg.norm(u)
            W[:, cols] -= float(e["alpha"]) * np.outer(u, u @ W[:, cols])
        elif e["op"] == "set_cols":
            W[:, np.asarray(e["cols"])] = np.asarray(e["values"], dtype=np.float32)
        elif e["op"] == "set_rows":
            W[np.asarray(e["rows"]), :] = np.asarray(e["values"], dtype=np.float32)
        elif e["op"] == "scale_cols":
            W[:, np.asarray(e["cols"])] *= float(e["alpha"])
        else:
            raise ValueError(e["op"])
    return out


class Edited:
    """Context manager: apply edits to y for the duration of a block, then restore the original weights.

        with Edited(y, edits):
            y.process(units)
    """

    def __init__(self, y, edits, roundtrip=True):
        self.y, self.edits, self.roundtrip = y, edits, roundtrip
        self.saved = {}

    def __enter__(self):
        for key, W in edited_weights(self.y, self.edits).items():
            m = _module(self.y, *key)
            self.saved[key] = m.weight
            m.weight = mx.array(q8(W) if self.roundtrip else W).astype(m.weight.dtype)
        mx.eval([_module(self.y, *k).weight for k in self.saved])
        return self.y

    def __exit__(self, *exc):
        for key, w in self.saved.items():
            _module(self.y, *key).weight = w
        self.saved = {}


def save(edits, path, meta=None):
    flat = {"meta": json.dumps(dict(meta or {}, n=len(edits)))}
    for k, e in enumerate(edits):
        flat[f"e{k}_spec"] = json.dumps({x: e[x] for x in ("layer", "mat", "op", "alpha") if x in e})
        for x in ("cols", "rows", "u", "values"):
            if x in e:
                flat[f"e{k}_{x}"] = np.asarray(e[x])
    np.savez(path, **flat)


def load(path):
    z = np.load(path)
    n = json.loads(str(z["meta"]))["n"]
    edits = []
    for k in range(n):
        e = json.loads(str(z[f"e{k}_spec"]))
        for x in ("cols", "rows", "u", "values"):
            if f"e{k}_{x}" in z:
                e[x] = z[f"e{k}_{x}"]
        edits.append(e)
    return edits


def apply_npz(y, path):
    """Apply an edit file permanently (for scripts that only ever run the edited model)."""
    ctx = Edited(y, load(path))
    ctx.__enter__()
    return ctx


def write_gguf(y, edits, src, dst):
    """Copy the GGUF src -> dst and overwrite the edited tensors in place (Q8_0, same size)."""
    shutil.copyfile(src, dst)
    reader = GGUFReader(src)
    info = {t.name: t for t in reader.tensors}
    new = edited_weights(y, edits)
    with open(dst, "r+b") as f:
        for (layer, mat), W in new.items():
            t = info[gguf_name(layer, mat)]
            assert t.tensor_type == GGMLQuantizationType.Q8_0, t.tensor_type
            q = quantize(np.ascontiguousarray(W, dtype=np.float32), GGMLQuantizationType.Q8_0)
            assert q.nbytes == t.n_bytes, (q.nbytes, t.n_bytes)
            f.seek(t.data_offset)
            f.write(q.tobytes())
    # check: re-read and compare against the intended (rounded) weights
    r2 = {t.name: t for t in GGUFReader(dst).tensors}
    for (layer, mat), W in new.items():
        got = dequantize(r2[gguf_name(layer, mat)].data, GGMLQuantizationType.Q8_0).reshape(W.shape)
        assert np.allclose(got, q8(W), atol=1e-6), (layer, mat)
    return sorted(gguf_name(*k) for k in new)
