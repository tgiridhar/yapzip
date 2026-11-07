"""The prompt Qwen sees for each block: telegraph-style notes (STE-like: short, literal, no filler)."""

import os

SYSTEM = """Rewrite the text as telegraph notes for another AI. Keep facts, drop talk.
Rules:
- One fact per line. Max 12 words per line.
- Drop greetings, thanks, small talk, jokes, sarcasm, reactions ("lol", "+1", "this"), sign-offs.
- No articles (a, an, the). No filler ("I think", "basically", "just", "honestly").
- Copy names, numbers, dates, prices and versions exactly as written.
- Write NOT for every negation. Write IF / ONLY IF for conditions.
- Write -> for a change. If a value is corrected later, write old -> new.
- Start with the speaker's name when it matters: "dana: ...".
- Never add anything that is not in the text."""

EXAMPLE_IN = """[09:02] ana: morning all!
[09:03] ana: the elevator repair is moved to Thursday the 14th, not Tuesday. it will be out from 8am to about 4pm
[09:05] raj: ugh again
[09:06] raj: will the service elevator work? i have a sofa delivery that day
[09:08] ana: no, both are off. the company said they can't split the job
[09:10] lee: same, my delivery is thursday too
[09:11] raj: ok I'll move mine to Friday. $40 to reschedule, great
[09:12] ana: thanks for understanding 🙏
[09:12] lee: +1"""

EXAMPLE_OUT = """ana: elevator repair Tuesday -> Thursday 14th, out 8am to about 4pm
ana: service elevator also off; company can NOT split job
raj, lee: deliveries Thursday
raj: moves delivery to Friday, $40 to reschedule"""


FEWSHOT = os.environ.get("YAPZIP_FEWSHOT", "1") != "0"


def messages(block_text: str, context: str = "") -> list[dict]:
    user = block_text if not context else f"(notes so far:\n{context})\n\n{block_text}"
    shots = [{"role": "user", "content": EXAMPLE_IN},
             {"role": "assistant", "content": EXAMPLE_OUT}] if FEWSHOT else []
    return [{"role": "system", "content": SYSTEM}, *shots, {"role": "user", "content": user}]
