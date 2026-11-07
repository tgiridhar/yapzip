# Blind judging: yapzip outputs

You grade compressed versions of a text. A downstream AI will read ONLY the compressed version instead of the original, so what matters is: are the facts still there, are they still true to the original, and is the filler gone.

Each packet (`<name>.packet.md`) has the ORIGINAL TEXT, sometimes KEY FACTS and QUESTIONS, and 2 or 3 OUTPUTS labelled A, B, C. The outputs come from different systems in random order. You don't know which is which; don't guess.

For each packet:

1. If the packet has no KEY FACTS, first write your own list from the original: every concrete fact a reader would need (numbers, names, dates, times, decisions, conditions, negations, who said what, commands/code/config values), as F1..Fn. Short items, 8-25 of them. Use the same list for every output.
2. For each output, label every fact:
   - `kept`: present and correct (wording may differ; exact values must match).
   - `partial`: present but missing a qualifier that changes meaning (a condition, a negation, a date, a hedge).
   - `wrong`: present but changed (wrong number/name/date, negation flipped, wish/guess/rumor stated as fact, attributed to the wrong person).
   - `missing`: not there.
3. List `errors`: anything in the output that is NOT true to the original: invented details, wrong values, flipped negations, hedges dropped ("hopefully X" written as "X"), misattribution, code/command/config text altered. Quote the output line and say why. Don't list omissions here (those are `missing`).
4. Count `noise`: output lines that carry no fact at all (filler, chatter, greetings, reactions that should have been dropped).
5. `code_intact`: if the original has code, commands, JSON, config lines, table rows or log lines, are they reproduced exactly in the output? `yes`, `no` (quote what broke in errors), or `n/a`.
6. If the packet has QUESTIONS, answer each one using ONLY that output (pretend you never read the original), then mark `correct` against the gold answer.
7. Rank the outputs best-first for the downstream-AI use case, with a one-sentence reason. This is a compressor: every word costs. Facts kept and correct matter most, then no errors, then brevity; but an output that keeps nearly the same facts (within ~5% of the list) in clearly fewer words ranks higher, and noise lines count against it.

Write one JSON file per packet next to it, `<name>.verdict.json`:

```json
{
  "fact_list": ["F1 ...", "F2 ..."],
  "A": {"facts": {"F1": "kept", "F2": "missing"}, "errors": [{"quote": "...", "why": "..."}], "noise": 0, "code_intact": "n/a", "qa": {"Q1": {"answer": "...", "correct": true}}},
  "B": {"...": "..."},
  "ranking": ["B", "A"],
  "why": "one sentence"
}
```

`fact_list` is only needed when you wrote the list yourself. `qa` is `{}` when there are no questions. Be strict and consistent across outputs. Check that each file parses with `python3 -m json.tool <file>`.
