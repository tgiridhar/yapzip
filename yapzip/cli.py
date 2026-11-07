"""yapzip: stdin -> blocks of numbered lines -> Qwen -> checked fact lines -> stdout, one block at a time."""

import argparse
import json
import sys
import time

from . import backend, check, prompt, stages
from .split import ntok, units


def blocks(lines, budget: int):
    """Group units into blocks of about `budget` tokens, yielding each as soon as it is full."""
    cur, size = [], 0
    for u in units(lines):
        t = ntok(u.text) + 3
        if cur and size + t > budget:
            yield cur
            cur, size = [], 0
        cur.append(u)
        size += t
    if cur:
        yield cur


def main():
    ap = argparse.ArgumentParser(description="Compress chatty text into short fact lines.")
    ap.add_argument("--block", type=int, default=int(__import__("os").environ.get("YAPZIP_BLOCK", "400")), help="tokens per block (default 1500)")
    ap.add_argument("--stats", help="append per-block stats as JSON lines to this file")
    ap.add_argument("--no-check", action="store_true", help="print model lines without the number check")
    ap.add_argument("--mode", choices=["two", "one"], default="two",
                    help="two: keep/drop scores then per-line rewrite (default); one: single prompt per block")
    ap.add_argument("--rewrite", choices=["block", "line"], default="block",
                    help="two-stage rewrite: one call over the kept lines of a block (default) or per line")
    ap.add_argument("--keep", type=float, default=0.55, help="keep a line if P(KEEP) >= this (two-stage)")
    args = ap.parse_args()

    stats_f = open(args.stats, "a") if args.stats else None
    recent: list[str] = []
    t_start = time.time()
    for n, blk in enumerate(blocks(iter(sys.stdin.readline, ""), args.block)):
        if args.mode == "two":
            t0 = time.time()
            lines, st, kept = [], {"lines": 0, "fallback": 0, "dropped": 0, "scores": []}, []
            for u in blk:
                p = stages.p_keep(u.text)
                st["scores"].append([u.id, round(p, 4)])
                if p >= args.keep:
                    kept.append(u.text)
                else:
                    st["dropped"] += 1
            st["t_score_s"] = round(time.time() - t0, 3)
            if args.rewrite == "block" and kept:
                text = "\n".join(kept)
                reply, timings = backend.complete(prompt.messages(text, "\n".join(recent[-5:])),
                                                  max_tokens=max(128, int(ntok(text) * 0.8)))
                lines, cst = check.process(reply, kept)
                st.update(lines=cst["lines"], fallback=cst["fallback"], repeats=cst["repeats"],
                          raw=reply, gen_tps=timings.get("predicted_per_second"))
            else:
                for line in kept:
                    note, _ = stages.rewrite(line)
                    st["lines"] += 1
                    if not args.no_check and check.nums(note) - check.nums(line):
                        st["fallback"] += 1
                        note = line
                    lines.append(note)
            recent += lines
            sys.stdout.write("".join(l + "\n" for l in lines))
            sys.stdout.flush()
            if stats_f:
                stats_f.write(json.dumps({"block": n, "units": len(blk), "wall_s": round(time.time() - t0, 3),
                                          "since_start_s": round(time.time() - t_start, 3),
                                          "prompt_tps": None, **{"gen_tps": None, **st}}) + "\n")
                stats_f.flush()
            continue
        text = "\n".join(u.text for u in blk)
        t0 = time.time()
        reply, timings = backend.complete(prompt.messages(text, "\n".join(recent[-5:])),
                                          max_tokens=max(256, int(ntok(text) * 0.7)))
        src = [u.text for u in blk]
        if args.no_check:
            lines, st = [l for l in reply.splitlines() if l.strip()], {}
        else:
            lines, st = check.process(reply, src)
        sys.stdout.write("".join(l + "\n" for l in lines))
        sys.stdout.flush()
        recent += lines
        if stats_f:
            stats_f.write(json.dumps({
                "block": n, "units": len(blk), "in_tok_est": ntok(text), "out_tok_est": ntok("\n".join(lines)),
                "wall_s": round(time.time() - t0, 3), "since_start_s": round(time.time() - t_start, 3),
                "prompt_tok": timings.get("prompt_n"), "prompt_tps": timings.get("prompt_per_second"),
                "gen_tok": timings.get("predicted_n"), "gen_tps": timings.get("predicted_per_second"),
                "raw": reply, **st}) + "\n")
            stats_f.flush()


if __name__ == "__main__":
    main()
