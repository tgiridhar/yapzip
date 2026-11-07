"""Starting weights for Colab training as a small file: colab/v2ws3_start.npz (~90 MB instead of the 812 MB model).
# Note (2026-11-07): the v2/v2ws* models were deleted to free disk; this script reproduces the v2ws3 start delta only if they are restored.
v2 is base Qwen3.5-0.8B plus a merged rank-64 LoRA on the linear layers, stored as Q8_0. So per linear matrix,
v2 - base is rank 64 up to Q8_0 rounding. We keep its top singular directions as fp16 factors (U @ V).
The weight-space edits (v2 -> v2ws3) rewrote a few whole neuron rows/columns in 5 tensors; those go in exactly.
Colab adds U @ V to the base weights it downloads from Hugging Face, then writes the edited rows/columns.

Checks printed: everything outside the linear layers is unchanged from base (so Colab's base copy is right);
each delta really is low-rank (energy kept, residual vs. the delta).

    .venv/bin/python ws/make_start_delta.py
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "native" / "llama.cpp" / "gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

BASE = ROOT / "models" / "qwen35-0.8b-bf16"
V2 = ROOT / "models" / "ws" / "embedded.gguf"
V2WS3 = ROOT / "models" / "ws" / "v2ws3.gguf"
OUT = ROOT / "colab" / "v2ws3_start.npz"
RANK = 64

GG = {"attn_norm.weight": "input_layernorm.weight", "post_attention_norm.weight": "post_attention_layernorm.weight",
      "ffn_down.weight": "mlp.down_proj.weight", "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight",
      "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight", "attn_v.weight": "self_attn.v_proj.weight",
      "attn_output.weight": "self_attn.o_proj.weight", "attn_q_norm.weight": "self_attn.q_norm.weight",
      "attn_k_norm.weight": "self_attn.k_norm.weight", "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
      "attn_gate.weight": "linear_attn.in_proj_z.weight", "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
      "ssm_beta.weight": "linear_attn.in_proj_b.weight", "ssm_out.weight": "linear_attn.out_proj.weight",
      "ssm_norm.weight": "linear_attn.norm.weight", "ssm_dt.bias": "linear_attn.dt_bias", "ssm_a": "linear_attn.A_log",
      "ssm_conv1d.weight": "linear_attn.conv1d.weight"}
LINEAR = ("ffn_down", "ffn_gate", "ffn_up", "attn_q.", "attn_k.", "attn_v.", "attn_output", "attn_qkv", "attn_gate",
          "ssm_alpha", "ssm_beta", "ssm_out")


def hf_name(n):
    if n == "token_embd.weight":
        return "embed_tokens.weight"
    if n == "output_norm.weight":
        return "norm.weight"
    _, layer, suffix = n.split(".", 2)
    return f"layers.{layer}.{GG[suffix]}"


def base_weights():
    idx = json.load(open(BASE / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for shard in sorted(set(idx.values())):
        for k, v in mx.load(str(BASE / shard)).items():
            if k.startswith("language_model.model."):
                out[k[len("language_model.model."):]] = v
    return out


def deq(t):
    return np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32)


def main():
    base = base_weights()
    b = lambda k: np.array(base[k].astype(mx.float32))
    v2 = {t.name: t for t in GGUFReader(str(V2)).tensors if not t.name.startswith("blk.24.")}
    v3 = {t.name: t for t in GGUFReader(str(V2WS3)).tensors if not t.name.startswith("blk.24.")}
    out, worst_other, stats = {}, {}, []
    for name, t in v2.items():
        k, g = hf_name(name), deq(t)
        if not any(s in name for s in LINEAR):
            ref = b(k)
            if ref.shape != g.shape:                      # conv1d is (C, K, 1) in the MLX copy, (C, K) in GGUF
                ref = ref.reshape(g.shape)
            if name.endswith("ssm_a"):
                ref = -np.exp(ref)                        # GGUF stores -exp(A_log)
            kind = name.split(".")[-2] if name.startswith("blk.") else name
            d = float(np.abs(g - ref).max() / (np.abs(ref).max() + 1e-12))
            worst_other[kind] = max(worst_other.get(kind, 0.0), d)
            continue
        W0 = b(k)
        assert W0.shape == g.shape, (name, W0.shape, g.shape)
        delta = g - W0
        dn = float(np.linalg.norm(delta))
        r = min(RANK, *delta.shape)
        if r == min(delta.shape):                         # tiny matrices (16 x 1024): store the delta itself
            out[f"{k}|D"] = delta.astype(np.float16)
            stats.append((name, dn / np.linalg.norm(W0), 1.0, 0.0, 0.0))
            continue
        U, s, Vt = np.linalg.svd(delta, full_matrices=False)
        root = np.sqrt(s[:r])
        Uf, Vf = (U[:, :r] * root).astype(np.float16), (root[:, None] * Vt[:r]).astype(np.float16)
        resid = float(np.linalg.norm(delta - Uf.astype(np.float32) @ Vf.astype(np.float32)))
        out[f"{k}|U"], out[f"{k}|V"] = Uf, Vf
        stats.append((name, dn / np.linalg.norm(W0), float((s[:r] ** 2).sum() / (s ** 2).sum()), resid / dn,
                      float(s[r] / s[r - 1])))
    # the weight-space edits: exact rows (neurons in gate/up) or columns (neurons in down) of v2ws3
    edits = []
    for name, t in v3.items():
        g2, g3 = deq(v2[name]), deq(t)
        if np.array_equal(g2, g3):
            continue
        k, diff = hf_name(name), g3 != g2
        rows, cols = np.where(diff.any(1))[0], np.where(diff.any(0))[0]
        if len(rows) <= len(cols):
            out[f"{k}|rows"], out[f"{k}|rowvals"] = rows.astype(np.int32), g3[rows].astype(np.float32)
            edits.append(f"{name}: rows {rows.tolist()}")
        else:
            out[f"{k}|cols"], out[f"{k}|colvals"] = cols.astype(np.int32), g3[:, cols].astype(np.float32)
            edits.append(f"{name}: {len(cols)} columns {cols.min()}..{cols.max()}")
    np.savez(OUT, **out)

    print("unchanged outside the linear layers? (max |v2 - base| / max |base| per kind)")
    for kind, d in sorted(worst_other.items()):
        print(f"   {kind:22s} {d:.2e}")
    a = np.array([s[1:] for s in stats])
    print(f"{len(stats)} linear matrices: |delta|/|base| median {np.median(a[:, 0]):.3f} (max {a[:, 0].max():.3f}); "
          f"energy in top {RANK}: min {a[:, 1].min():.4f}; residual/|delta| median {np.median(a[:, 2]):.3f} "
          f"(max {a[:, 2].max():.3f}); s[{RANK}]/s[{RANK - 1}] median {np.median(a[a[:, 3] > 0, 3]):.3f}")
    for s in sorted(stats, key=lambda s: -s[3])[:4]:
        print(f"   worst residual: {s[0]:28s} |d|/|W| {s[1]:.3f}  energy {s[2]:.4f}  resid {s[3]:.3f}  s65/s64 {s[4]:.3f}")
    print("edits:", "; ".join(edits))
    print(f"-> {OUT} ({OUT.stat().st_size / 1e6:.0f} MB, {len(out)} arrays)")


if __name__ == "__main__":
    main()
