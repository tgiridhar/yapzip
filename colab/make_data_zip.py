"""Pack yapzip_data.zip for the Colab notebook: training data + the binary's source.

data/claude_train.jsonl = data/claude_train/part_*.jsonl in order (+ data/threads/real_train.jsonl with --real)
data/toy_train.jsonl, data/oos.json; native/yapzip.cpp, CMakeLists.txt, embed.sh.
Usage: python3 colab/make_data_zip.py [--real] [OUT.zip]   (default OUT: colab/yapzip_data.zip; with --real the
default is data/threads/yapzip_data.zip, which is gitignored because it holds real Reddit text)
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    real = "--real" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--real"]
    out = Path(args[0]) if args else ROOT / ("data/threads/yapzip_data.zip" if real else "colab/yapzip_data.zip")
    parts = sorted((ROOT / "data/claude_train").glob("part_*.jsonl"))
    train = "".join(p.read_text() for p in parts)
    if real:
        train += (ROOT / "data/threads/real_train.jsonl").read_text()
    with zipfile.ZipFile(ROOT / "colab/yapzip_data.zip") as old:   # template chunks come from the original zip
        toy = old.read("data/toy_train.jsonl")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data/claude_train.jsonl", train)
        z.writestr("data/toy_train.jsonl", toy)
        z.write(ROOT / "data/oos.json", "data/oos.json")
        for f in ["yapzip.cpp", "CMakeLists.txt", "embed.sh"]:
            z.write(ROOT / "native" / f, f"native/{f}")
    print(f"{out}: {train.count(chr(10))} chunks")


if __name__ == "__main__":
    main()
