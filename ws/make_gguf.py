"""Write an edit file into a copy of the GGUF (only the edited Q8_0 tensors are rewritten, in place).

    python ws/make_gguf.py results/ws/edit.npz models/ws/v2ws.gguf [--src models/ws/v4c.gguf]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ws.edits import load, write_gguf  # noqa: E402
from ws.model import Yap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edit")
    ap.add_argument("out")
    ap.add_argument("--src", default=str(ROOT / "models/ws/v4c.gguf"))
    a = ap.parse_args()
    y = Yap(a.src)
    names = write_gguf(y, load(a.edit), a.src, a.out)
    print("rewrote", names, "->", a.out)


if __name__ == "__main__":
    main()
