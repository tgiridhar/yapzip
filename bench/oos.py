"""Run the toy-trained Qwen 0.8B LoRA on out-of-sample inputs and write a side-by-side with Opus.

Inputs: data/oos.json (hand-written, not from the toy templates) + the 54-line offsite chat test file.
Output: results/oos.md
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.toy_train import (MODEL, OUT, ROOT, linear_to_lora_layers, load, p_keep, render,  # noqa: E402
                             write_notes)
from yapzip.split import ntok  # noqa: E402


def main():
    adapter = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    label = sys.argv[2] if len(sys.argv) > 2 else "toy LoRA"
    cfg = json.loads((adapter / "config.json").read_text()) if (adapter / "config.json").exists() else {"layers": 1, "rank": 16}
    cases = json.loads((ROOT / "data/oos.json").read_text())
    offsite = [l.strip() for l in (ROOT / "data/raw/chat_offsite.txt").read_text().splitlines() if l.strip()]
    cases.append({"name": "chat: offsite planning (54 lines, realistic test file)", "genre": "chat",
                  "units": offsite, "opus": (ROOT / "data/claude/chat_offsite.txt").read_text().strip()})

    model, tok = load(str(MODEL))
    model.freeze()
    linear_to_lora_layers(model, cfg["layers"], {"rank": cfg["rank"], "scale": 10.0, "dropout": 0.0})
    model.load_weights(str(adapter / "adapters.safetensors"), strict=False)
    model.eval()

    md = [f"# Out-of-sample: Qwen 0.8B ({label}) vs Opus", "",
          f"Qwen3.5-0.8B + LoRA on the last {cfg['layers']} blocks ({cfg.get('steps', '?')} steps, data: "
          f"{', '.join(Path(t).name for t in cfg.get('train', ['toy templates']))}). None of these inputs were in "
          "training. Opus = compression written by Claude Opus. Sizes are estimated tokens.", ""]
    for c in cases:
        units = [{"id": i + 1, "text": t} for i, t in enumerate(c["units"])]
        p = p_keep(model, tok, units)
        kept = [u["id"] for u, pk in zip(units, p) if pk >= 0.5]
        notes = write_notes(model, tok, units, kept)
        qwen = render(units, notes)
        src = "\n".join(c["units"])
        md += [f"## {c['name']}", "",
               f"Input {ntok(src)} tok · Qwen {ntok(qwen)} tok ({ntok(qwen) / ntok(src):.0%}) · "
               f"Opus {ntok(c['opus'])} tok ({ntok(c['opus']) / ntok(src):.0%})", "",
               "**Input** (P(keep) from Qwen, ✓ = kept)", "", "```text"]
        md += [f"{pk:.2f} {'✓' if pk >= 0.5 else ' '} {t}" for t, pk in zip(c["units"], p)]
        md += ["```", "", f"**Qwen 0.8B ({label})**", "", "```text", qwen or "(nothing kept)", "```", "",
               "**Opus**", "", "```text", c["opus"], "```", ""]
        print(f"done: {c['name']}", flush=True)
    out_md = ROOT / "results" / (sys.argv[3] if len(sys.argv) > 3 else "oos.md")
    out_md.write_text("\n".join(md))
    print("wrote", out_md)


if __name__ == "__main__":
    main()
