"""Merge a LoRA trained on Colab (notebook cell 6b: yapzip_lora_v3.npz) into a yapzip GGUF.

    .venv/bin/python ws/merge_lora.py LORA.npz [--src models/ws/v4c.gguf] [--dst models/ws/v3.gguf] [--scale 1.0]

Each touched linear matrix W (Q8_0) becomes Q8_0(dequant(W) + scale * B @ A), with B already scaled by alpha/r;
--scale below 1 merges a weaker version of the same LoRA.
The new tensors are written in place into a copy of src (same size); everything else is copied unchanged.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "native" / "llama.cpp" / "gguf-py"))
sys.path.insert(0, str(ROOT))
from gguf import GGMLQuantizationType, GGUFReader  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402

from ws.make_start_delta import GG  # noqa: E402  (GGUF suffix -> Hugging Face name)

Q8 = GGMLQuantizationType.Q8_0
HF2GG = {v: k for k, v in GG.items()}


def gguf_name(hf):
    """'layers.3.mlp.gate_proj.weight' -> 'blk.3.ffn_gate.weight'"""
    _, layer, rest = hf.split(".", 2)
    return f"blk.{layer}.{HF2GG[rest]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lora")
    ap.add_argument("--src", default=str(ROOT / "models" / "ws" / "v4c.gguf"))
    ap.add_argument("--dst", default=str(ROOT / "models" / "ws" / "v3.gguf"))
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    lora = np.load(a.lora)
    names = sorted({k.rsplit("|", 1)[0] for k in lora.files})
    info = {t.name: t for t in GGUFReader(a.src).tensors}
    shutil.copyfile(a.src, a.dst)
    rel = []
    with open(a.dst, "r+b") as f:
        for hf in names:
            t = info[gguf_name(hf)]
            assert t.tensor_type == Q8, (t.name, t.tensor_type)
            W = np.asarray(dequantize(t.data, Q8), dtype=np.float32)
            delta = a.scale * (lora[f"{hf}|B"].astype(np.float32) @ lora[f"{hf}|A"].astype(np.float32))
            assert delta.shape == W.shape, (hf, delta.shape, W.shape)
            q = quantize(np.ascontiguousarray(W + delta), Q8)
            assert q.nbytes == t.n_bytes, (q.nbytes, t.n_bytes)
            f.seek(t.data_offset)
            f.write(q.tobytes())
            rel.append(np.linalg.norm(delta) / np.linalg.norm(W))
    got = {t.name: t for t in GGUFReader(a.dst).tensors}
    assert all(got[n].n_bytes == info[n].n_bytes for n in info)
    print(f"merged {len(names)} LoRA pairs into {a.dst}: |BA|/|W| median {np.median(rel):.4f}, max {max(rel):.4f}")


if __name__ == "__main__":
    main()
