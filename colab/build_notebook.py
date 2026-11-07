"""Builds yapzip_train_a100.ipynb (run: python3 colab/build_notebook.py)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# yapzip: train Qwen3.5-0.8B as a stream compressor (A100)

Two jobs on any text: **keep/drop** each unit (P(" yes") at a `?` marker per unit, one forward pass) and a terse
**note** per kept unit (`=` means copy verbatim). Runtime → **A100 GPU**, Run all, upload `yapzip_data.zip`
(made by `colab/make_data_zip.py`: training data + the binary's source). ~15 min training, then the GGUF export and
a single `yapzip` binary with the model inside, which the last cell tests and downloads.

**Continue from the mainline weights instead** (v2ws3 = v2 + weight-space edits): upload `yapzip_data_v3.zip` (made by
`ws/make_train_v3.py`). It carries `start/v2ws3_start.npz` (v2ws3 as a low-rank delta on base Qwen, cell 4b) and
`data/extra_train.jsonl` (AI-written articles, pasted web pages with their chrome, repeat blocks, Medium posts). The
notebook then trains gently (lr 5e-5, 2 epochs) and cell 6 downloads only the LoRA (~90 MB), merged on the Mac by
`ws/merge_lora.py`. Cells 7-8 are not needed in that mode.
""")

code(r"""
# 1. Install
!pip -q install -U "transformers>=5.20" "peft>=0.21" accelerate kernels "flash-linear-attention>=0.5" safetensors sentencepiece
!pip -q uninstall -y torchao   # Colab's torchao 0.10 breaks new PEFT; not needed
""")

code(r"""
# 2. Data + binary source: upload yapzip_data.zip
import json, os, random, zipfile
from pathlib import Path

DATA = Path("/content/data")
if not (DATA / "claude_train.jsonl").exists():
    from google.colab import files
    zipfile.ZipFile(next(iter(files.upload()))).extractall("/content")
print(sorted(p.name for p in DATA.iterdir()), sorted(p.name for p in Path("/content/native").iterdir()))
""")

code(r"""
# 3. Config + split (the first 300 chunks hold the fixed 30-chunk test set; later chunks only ever train)
#    CONTINUE: the upload carries the mainline weights (yapzip_data_v3.zip): start from them (cell 4b), train gently.
#    V2NOTES: data/claude_train_v2notes.jsonl (the notes v2 was trained on, before the strict relabel) is uploaded:
#    keep/drop trains on those labels plus the extra chunks.
#    notes_kl (continue mode): notes are not trained on labels at all. At every note token the model must match the
#    starting model (v2ws3, LoRA switched off): KL to its next-token distribution. Keep/drop moves; note writing stays.
#    Flags (empty files in /content/data): from_base = ignore the start weights; notes_kl = pin notes to the start.
CONTINUE = Path("/content/start/v2ws3_start.npz").exists() and not (DATA / "from_base").exists()
V2NOTES = (DATA / "claude_train_v2notes.jsonl").exists()
NOTES_KL = CONTINUE and (DATA / "notes_kl").exists()
CFG = dict(model_id="Qwen/Qwen3.5-0.8B", max_tokens=2048, lora_r=64, lora_alpha=128, lora_dropout=0.05,
           lr=5e-5 if CONTINUE else 2e-4, epochs=2 if CONTINUE else 3, grad_accum=8, warmup_ratio=0.05,
           toy_mix=0.15, small_mix=0.2, holdout=30, seed=0, notes_kl=NOTES_KL,
           tag=("v4c" if CONTINUE else "v4base") + ("_kl" if NOTES_KL else "") + ("_v2notes" if V2NOTES else ""))
claude = [json.loads(l) for l in open(DATA / ("claude_train_v2notes.jsonl" if V2NOTES else "claude_train.jsonl"))]
toy = [json.loads(l) for l in open(DATA / "toy_train.jsonl")]
extra_v3 = [{**json.loads(l), "no_notes": V2NOTES} for l in open(DATA / "extra_train.jsonl")] if (DATA / "extra_train.jsonl").exists() else []
rng = random.Random(CFG["seed"])
base, extra = claude[:300], claude[300:]
rng.shuffle(base)
test, train_chunks = base[:CFG["holdout"]], base[CFG["holdout"]:] + extra
# Tiny inputs (a single line, 2-3 lines) must work too: add windows of 1-3 units cut from training chunks.
def window(c):
    k = rng.choice([1, 1, 2, 3])
    s = rng.randrange(max(1, len(c["units"]) - k + 1))
    return {**c, "units": [{**u, "id": i + 1} for i, u in enumerate(c["units"][s:s + k])]}
small = [window(rng.choice(train_chunks)) for _ in range(int(len(train_chunks) * CFG["small_mix"]))]
n_toy = int(len(train_chunks) * CFG["toy_mix"] / (1 - CFG["toy_mix"]))
train_chunks += small + rng.sample(toy, min(n_toy, len(toy))) + extra_v3
print(f"{'continue from v2ws3' if CONTINUE else 'from base Qwen'}{', v2-era notes' if V2NOTES else ''}: {len(train_chunks)} train chunks "
      f"({len(small)} 1-3 line windows, {n_toy} template, {len(extra_v3)} extra), {len(test)} held-out, lr {CFG['lr']}")
""")

code(r"""
# 4. Model + sequence format (must match native/yapzip.cpp: markers appended after the chunk, notes restate the unit)
import math, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer

torch.manual_seed(CFG["seed"])
tok = AutoTokenizer.from_pretrained(CFG["model_id"])
model = AutoModelForImageTextToText.from_pretrained(CFG["model_id"], dtype=torch.bfloat16, device_map="cuda")
lm, head = model.model.language_model, model.lm_head
enc = lambda s: tok.encode(s, add_special_tokens=False)
YES, NO, Q = enc(" yes")[0], enc(" no")[0], enc("?")[0]


def build(chunk):
    units = chunk["units"]
    ids = enc("units:\n")
    for u in units:
        ids += enc(f"u{u['id']} {u['text']}\n")
    ids += enc("keep?\n")
    pos = []
    for u in units:
        ids += enc(f" u{u['id']}") + [Q]
        pos.append(len(ids) - 1)
    ids += enc("\nnotes:\n")
    labels, tgt = [YES if u["keep"] else NO for u in units], []
    for u in units:
        if u["keep"]:
            ids += enc(f"u{u['id']} {u['text']} =>")
            note = enc(f" {u['note']}\n")
            tgt += range(len(ids), len(ids) + len(note))
            ids += note
    return ids, pos, labels, tgt


def fit(chunks):
    out, todo = [], list(chunks)
    while todo:   # halve chunks whose sequence is too long (ids renumbered)
        c = todo.pop()
        if len(build(c)[0]) <= CFG["max_tokens"] or len(c["units"]) < 2:
            out.append(c); continue
        mid = len(c["units"]) // 2
        for part in (c["units"][:mid], c["units"][mid:]):
            todo.append({**c, "units": [{**u, "id": i + 1} for i, u in enumerate(part)]})
    return out


train_set = fit(train_chunks)
print(f"{len(train_set)} training sequences")

# 4b. CONTINUE: start from the yapzip mainline weights (v2ws3), not from base Qwen. v2 = base + a merged rank-64
#     LoRA, so v2 - base ships as low-rank factors U @ V per linear matrix; the weight-space edits (v2 -> v2ws3)
#     ship as exact neuron rows/columns. Built by ws/make_start_delta.py.
if CONTINUE:
    import numpy as np
    start = np.load("/content/start/v2ws3_start.npz")
    sd = lm.state_dict()                               # shares storage with the parameters
    cuda = lambda a: torch.from_numpy(a).to("cuda")
    moved, edited = 0, []
    with torch.no_grad():
        for key in start.files:
            name, part = key.rsplit("|", 1)
            if part == "U":
                delta = cuda(start[key]).float() @ cuda(start[f"{name}|V"]).float()
            elif part == "D":
                delta = cuda(start[key]).float()
            else:
                continue
            W = sd[name]
            assert W.shape == delta.shape, (name, W.shape, delta.shape)
            W.copy_((W.float() + delta).to(W.dtype))
            moved += 1
        for key in start.files:                        # edits last: they overwrite whole rows / columns
            name, part = key.rsplit("|", 1)
            if part == "rows":
                sd[name][cuda(start[key]).long()] = cuda(start[f"{name}|rowvals"]).to(sd[name].dtype)
                edited.append(name)
            elif part == "cols":
                sd[name][:, cuda(start[key]).long()] = cuda(start[f"{name}|colvals"]).to(sd[name].dtype)
                edited.append(name)
    print(f"v2ws3 start: {moved} linear matrices moved from base Qwen to v2, edits written in {edited}")
""")

code(r"""
# 5. LoRA on every language-model layer + a fast held-out check (one forward pass per chunk, no generation)
from peft import LoraConfig, get_peft_model

TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)"
model = get_peft_model(model, LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
                                         lora_dropout=CFG["lora_dropout"], target_modules=TARGETS))
params = [p for p in model.parameters() if p.requires_grad]
print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.1f}M")


def forward(ids, pos, labels, tgt):
    h = lm(input_ids=torch.tensor([ids[:-1]], device="cuda")).last_hidden_state[0]
    s_logits = head(h[pos]).float()
    t = torch.tensor(tgt, device="cuda")
    n_loss = F.cross_entropy(head(h[t - 1]).float(), torch.tensor([ids[i] for i in tgt], device="cuda")) if tgt else torch.zeros((), device="cuda")
    return s_logits, F.cross_entropy(s_logits, torch.tensor(labels, device="cuda")), n_loss


def forward_kl(ids, pos, labels, tgt):
    # keep/drop: cross-entropy on the labels; notes: KL to the starting model (LoRA off) at every note token
    x = torch.tensor([ids[:-1]], device="cuda")
    h = lm(input_ids=x).last_hidden_state[0]
    s_logits = head(h[pos]).float()
    s_loss = F.cross_entropy(s_logits, torch.tensor(labels, device="cuda"))
    if not tgt:
        return s_logits, s_loss, torch.zeros((), device="cuda")
    t = torch.tensor(tgt, device="cuda") - 1
    with torch.no_grad(), model.disable_adapter():
        teacher = F.log_softmax(head(lm(input_ids=x).last_hidden_state[0][t]).float(), -1)
    student = F.log_softmax(head(h[t]).float(), -1)
    return s_logits, s_loss, F.kl_div(student, teacher, log_target=True, reduction="batchmean")


@torch.no_grad()
def heldout():
    model.eval()
    ys, ps, nl = [], [], []
    for c in test:
        ids, pos, labels, tgt = build(c)
        s_logits, _, n_loss = forward(ids, pos, labels, tgt)
        ps += s_logits[:, [YES, NO]].softmax(-1)[:, 0].tolist()
        ys += [l == YES for l in labels]
        if tgt: nl.append(n_loss.item())
    model.train()
    pos_, neg_ = [p for p, y in zip(ps, ys) if y], [p for p, y in zip(ps, ys) if not y]
    auc = sum((a > b) + 0.5 * (a == b) for a in pos_ for b in neg_) / max(1, len(pos_) * len(neg_))
    return {"keep_acc": round(sum((p >= 0.5) == y for p, y in zip(ps, ys)) / len(ys), 3),
            "keep_auc": round(auc, 3), "notes_loss": round(sum(nl) / max(1, len(nl)), 3)}


print("before training:", heldout())
""")

code(r"""
# 6. Train
opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=0.0)
total = math.ceil(len(train_set) * CFG["epochs"] / CFG["grad_accum"])
warm = max(1, int(total * CFG["warmup_ratio"]))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
model.train()
step, t0, hist, best = 0, time.time(), [], (float("inf"), -1, None)
for epoch in range(CFG["epochs"]):
    order = list(range(len(train_set)))
    random.Random(epoch).shuffle(order)
    for i, k in enumerate(order):
        _, s_loss, n_loss = (forward_kl if CFG.get("notes_kl") else forward)(*build(train_set[k]))
        w = 0.0 if train_set[k].get("no_notes") and not CFG.get("notes_kl") else 1.0   # no_notes: keep/drop only
        ((s_loss + w * n_loss) / CFG["grad_accum"]).backward()
        hist.append((s_loss.item(), n_loss.item()))
        if (i + 1) % CFG["grad_accum"] == 0 or i == len(order) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                r = hist[-25 * CFG["grad_accum"]:]
                print(f"epoch {epoch} step {step}/{total}  keep/drop {sum(a for a, _ in r) / len(r):.3f}  notes {'KL' if CFG.get('notes_kl') else ''} {sum(b for _, b in r) / len(r):.4f}  {time.time() - t0:.0f}s")
    h = heldout()
    print(f"=== epoch {epoch} held-out: {h}")
    if h["notes_loss"] < best[0] or CFG.get("notes_kl"):   # best notes epoch (notes overfit after ~2); KL mode: last
        best = (h["notes_loss"], epoch, {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k})
model.load_state_dict(best[2], strict=False)
print(f"using epoch {best[1]} (held-out notes loss {best[0]})")

# 6b. Download only the LoRA (fp16, B pre-scaled by alpha/r). CONTINUE: ws/merge_lora.py merges it into v2ws3.gguf;
#     from base: merge it into the base model (ws/merge_lora.py --base) or run cells 7-8 for a full GGUF.
if True:
    import numpy as np
    from google.colab import files
    lora_out, scale = {}, CFG["lora_alpha"] / CFG["lora_r"]
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in mod.lora_A:
            key = name.split("language_model.", 1)[1] + ".weight"
            lora_out[key + "|A"] = mod.lora_A["default"].weight.detach().float().cpu().numpy().astype(np.float16)
            lora_out[key + "|B"] = (mod.lora_B["default"].weight.detach().float().cpu() * scale).numpy().astype(np.float16)
    lora_file = f"/content/yapzip_lora_{CFG.get('tag', 'v3')}.npz"
    np.savez(lora_file, **lora_out)
    print(len(lora_out) // 2, "LoRA pairs ->", lora_file, round(os.path.getsize(lora_file) / 1e6), "MB")
    files.download(lora_file)
""")

code(r"""
# 7. Merge LoRA + export Q8_0 GGUF
import shutil
from huggingface_hub import hf_hub_download

merged_dir = "/content/yapzip-merged"
model.merge_and_unload().save_pretrained(merged_dir, safe_serialization=True)
tok.save_pretrained(merged_dir)
for f in ["preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"]:
    try: shutil.copy(hf_hub_download(CFG["model_id"], f), merged_dir)
    except Exception as e: print("skip", f, e)
!git clone -q --depth 1 --branch b11803 https://github.com/ggml-org/llama.cpp /content/llama.cpp 2>/dev/null || true
!pip -q install gguf sentencepiece protobuf
!PYTHONPATH=/content/llama.cpp/gguf-py python /content/llama.cpp/convert_hf_to_gguf.py {merged_dir} --outtype q8_0 --outfile /content/yapzip-qwen3.5-0.8b-q8_0.gguf 2>&1 | tail -1
""")

code(r"""
# 8. Build one yapzip binary (llama.cpp CUDA sm_80, static, model inside), test it, download it
!cmake -S /content/native -B /content/yapzip_build -DLLAMA_DIR=/content/llama.cpp -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release -DYAPZIP_MODEL=/content/yapzip-qwen3.5-0.8b-q8_0.gguf > /content/cmake.log 2>&1 || tail -20 /content/cmake.log
!cmake --build /content/yapzip_build --target yapzip -j 12 2>&1 | tail -1
import subprocess
from google.colab import files
BIN = "/content/yapzip_build/yapzip"
oos = json.load(open(DATA / "oos.json"))
for case in oos + [{"name": "filler", "units": ["blaah blaah blaaah ..meh mehhh eh yap yap yap. secret=123 . bleheheh meh meh"]}]:
    src = "\n".join(case["units"]) + "\n"
    out = subprocess.run([BIN], input=src, capture_output=True, text=True).stdout
    print(f"=== {case['name']}: {len(src.split())} -> {len(out.split())} words\n{out.strip()}")
shutil.copy(BIN, "/content/yapzip-linux-x86_64-cuda")
files.download("/content/yapzip-linux-x86_64-cuda")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                                    "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).resolve().parent / "yapzip_train_a100.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", out, len(cells), "cells")
