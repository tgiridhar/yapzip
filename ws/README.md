# Weight-space edits

A research log. Paths under `results/`, `models/` and `data/` (other than `data/STYLE.md` and `data/sample_train.jsonl`) are local files: the training data, eval inputs and outputs are not published.

Fixing yapzip's fine-tuned Qwen3.5-0.8B by reading and writing its weights directly. The rules follow "Sparks of Weight-Space Programming" (Takehi et al., MIT CSAIL, Nov 2026): forward passes and weight inspection only, with no gradients, no optimizer and no fitting weights to a loss. Edits are explicit row or column writes, checked against a random-direction control and held-out data before they go into a GGUF.

## Tools

| File | What |
|---|---|
| `model.py` | Loads a yapzip GGUF into MLX. It dequantizes Q8_0 and reverses llama.cpp's conversion (`A_log = log(-ssm_a)`; norms already carry the +1). It runs both stages exactly like `native/yapzip.cpp`, with taps on the residual stream, MLP input, activation and output, and attention output. |
| `edits.py` | Edits as data (`dir_cols`: `W[:, cols] -= a·u·uᵀW[:, cols]`; `set_rows`, `set_cols`). The `Edited(y, edits)` context manager applies them, with a Q8_0 round trip so MLX sees what the binary will load. `write_gguf` patches the edited tensors in place in a GGUF copy. |
| `practice.py` | Practice data: the real threads in `data/threads`, rewritten with the author on its own line in 6 header styles. 20 threads for analysis, 15 held out. |
| `regress.py` | The notebook's 30 held-out chunks: keep/drop accuracy and AUC, teacher-forced notes loss. |
| `run_eval.py` | Runs the 14 eval inputs through the MLX copy and saves per-unit records (P(keep), note). |
| `author_lines.py` | Edit 1, below: `detect` (forward passes) and `edit` (build the edit file). |
| `make_gguf.py` | Writes an edit file into a GGUF copy. |

The MLX copy matches the binary. On the 14 eval inputs, 11 outputs are byte-identical and 3 differ only at near-tied tokens. P(keep) agrees to within 0.005.

The v2 model was extracted from the then-current binary, now `build/yapzip-legacy` (section `__YAPZIP,__gguf`), to `models/ws/embedded.gguf`:

```bash
python3 - <<'EOF'
off, size = 9551872, 0x3063bea0          # from: otool -l build/yapzip-legacy | grep -A8 __gguf
with open("build/yapzip-legacy", "rb") as f, open("models/ws/embedded.gguf", "wb") as o:
    f.seek(off); o.write(f.read(size))
EOF
```

## Edit 1: author lines get `=` notes

**Problem.** In training, the author was always inline: `handle (+3): text`. When an author line stands alone (`ravi_s · 36 points · reply to karen_b_dairy`, `From: Priya <...>`), v2 sometimes writes a note for it that pulls text from a neighbouring post. That produced three of v2's 6–7 errors in each of judge runs 3, 5 and 6:
- karen's question appeared under ravi;
- walt's pizza-party jab appeared under marisol;
- dtrenchard's "never paid for mine either" appeared under greg.

**Mechanism, from forward passes and weight reading:**
- Projecting every `ffn_down` column through the output embedding finds neurons that write almost nothing but `=` tokens. The two that track the decision are **L20 n58** and **L22 n274**.
- On author lines, their gate pre-activation decides the note. It is +3.4 / +3.8 when the note is a clean `=`, +2.0 / +2.5 when the line gets rewritten (the case that leaks), and −0.6 / −0.7 on ordinary post text.
- On the forum eval, every leaky author line had these two neurons under-firing.
- The MLP input at the note position also carries a near-perfect "this unit is an author or metadata line" direction. It is a plain difference of means, author lines minus post text, with held-out AUC 0.997.

**Edit.** Add `0.25 × detector` to the gate rows of the two neurons. That is 2 rows × 1,024 weights, in `blk.20.ffn_gate` and `blk.22.ffn_gate`. Author lines now open the copy-verbatim path as wide as the clean cases do. Files: `results/ws/author_detector.npz`, `results/ws/edit_author_lines.npz`.

```bash
python ws/author_lines.py edit results/ws/author_detector.npz results/ws/edit_author_lines.npz --beta 0.25
python ws/make_gguf.py results/ws/edit_author_lines.npz models/ws/v2ws.gguf
build/yapzip -m models/ws/v2ws.gguf < chat.txt
```

**Checks (MLX).**

| | v2 | edited | random direction, same size |
|---|---|---|---|
| forum eval: kept author lines that leak text | 5 of 10 | **0 of 10** | 5 (6 at 3× size) |
| forum eval: lowest P(`" ="`) on a kept author line | 0.009 | 0.67 | 0.007 |
| held-out chunks: keep accuracy / AUC | 0.910 / 0.9665 | 0.910 / 0.9666 | |
| held-out chunks: notes loss (nats/token) | 1.5233 | 1.5231 | |
| held-out real threads: author lines copied verbatim | 346 of 472 | 400 of 474 | |
| held-out real threads: author-line notes with words not in the line | 2 | 3 | |
| held-out real threads: post text copied verbatim | 30 of 661 | 34 of 661 | |
| held-out real threads: keep/drop flips | | 5 of 2,630 (max ΔP 0.04) | |

**Binary plus blind judges** (`results/eval/v2ws`, `results/eval/judge7`). The binary's output changed on 8 of the 14 eval inputs; the other 6 are byte-identical. On those 8, four Claude judges compared v2 and the edit blind:

| | v2 | edited |
|---|---|---|
| key facts kept | 88% | 88% |
| errors (invented, misattributed, flipped) | 7 | **4** |
| noise lines | 26 | 25 |
| output size vs input | 51% | 49% |
| questions answered from the output | 40/40 | 40/40 |
| ranked first | 4 | 2 |

- **What the edit removed:** the three misattributions above, plus two duplicate lines. The forum output is 14% shorter (333 vs 389 words).
- **What it cost:**
  - One keep decision flipped in the forum. The `tobias_r · 58 points` header was a coin flip (P(keep) 0.501 in v2, 0.493 edited), so the edited model drops it. Tobias's aisle-9 post now sits under nell's header, a misattribution by position. The judges did not count it, and they did not count v2's equivalent either (ravi's numbers sit under tobias's header in v2). Counting it, the forum goes from 5 errors to 3, not 2.
  - The same post lost "took an hour and a half", the only fact any judge labelled differently across the 8 files.
  - The `secret=123` filler line is now copied whole.
  - The four files v2 "won" tie on facts and errors and were decided by a word or two of wording.
- **How strong the evidence is:**
  - The forum eval is where the bug was found and the fix was checked, so its error drop is in-sample.
  - On the held-out real threads the leak barely occurs (2 author-line notes with foreign words before the edit, 3 after). There is no out-of-sample evidence yet that the edit reduces errors, only that it does not hurt keep/drop or notes loss.
  - On the notebook's held-out chunks, whether a note starts with `=` matches the label slightly less often (93.0% vs 94.6%, 2 of 129 units).
- **On long AI-written articles, yapzip's main use, the edit doesn't help** (`results/eval/judge8`, 8 articles). Articles have no standalone author lines, so there is nothing for it to fix.
  - 5 of the 8 outputs are byte-identical to v2.
  - Judges: facts kept 88% (v2) vs 87% (edited); errors 37 vs 36; average rank 2.12 vs 2.25, where lower is better.
  - On the recipe, the edit copies two filler-laden ingredient bullets verbatim, drops "zest of" from the lemons line, and loses "cool completely" before freezing.
- **Errors both still make:**
  - big_pete's post tagged `Dana:`;
  - `this` → `no`;
  - "only if" → `UNLESS` in the meeting;
  - Tom's sarcasm stated as sincere.

**Edit 1 alone, by the project's priority order** (no hallucination > compression > code and exact content > facts), across all 22 judged inputs:

| | v2 | edited |
|---|---|---|
| hallucinations (invented, wrong speaker, flipped meaning): chats and threads / articles | 7 / 12 | 5 / 12 |
| output size vs input: chats and threads / articles | 51.8% / 33.1% | 50.8% / 33.0% |
| code, commands and table rows exact | all | all |
| facts kept (judges): chats and threads / articles | 88% / 88% | 88% / 87% |

- **The edit fixed 4 hallucinations:** the three forum misattributions, and the recipe's "1 tbsp → 2 tsp" correction written as "1 → 2 tsp".
- **It introduced 2:** tobias's post under nell's name, and "zest of 2 lemons" written as "2 lemons".
- **How these were counted:** from the judges' notes, leaving out dropped hedges and digit formatting. Both models share those.

## Edit 2: drop in-block slop (v2ws2 = v2 + Edit 1 + Edit 2)

**Problem.** On long AI-written articles, v2 keeps generic rhetoric, headings, questions, boilerplate and story lines at P(keep) ≈ 1.0, e.g. "Control or trust." and "It's a gap in trust.". The notes it writes for those lines are also where many of its inventions happen. In the transit test article, three of v2's five meaning errors sat in such lines: "Residents: buses may or may not arrive", an expert claim stated as fact, and two groups of residents merged.

Repeats across blocks (Key Takeaways, FAQ) are out of scope: yapzip is a block-level streaming tool, like gzip.

**Data.**
- **Dev articles:** 12 long AI-style articles written in-session and labeled unit by unit. They live in `data/slop_dev/` (see `ws/dev_data.py`) and are kept apart from the 8 test articles.
- **Other genres:** 1,201 kept units from 92 genres of training chunks, excluding the 30 held-out chunks, serve as the "content" side.
- **Harness:** `ws/dev_eval.py` runs the real binary and scores every unit against its label.

**Mechanism, from forward passes on the layer-23 MLP input at each `?` marker:**
- **Slop direction.** A closed-form discriminant between in-block slop and content: the mean difference, whitened by the pooled covariance with 0.2 shrinkage. No optimizer is involved.
  - The threshold is the 99.5th percentile of content scores.
  - A plain mean difference also flagged narrative lines in fiction, questions in chats and legal clauses: keep accuracy on the 30 held-out chunks fell from 0.910 to 0.877. Whitening fixed that.
- **Marker direction.** Markers minus every other position, including note writing.
- **Bias carrier.** The mean marker direction varies only 5% across markers, so it carries the bias the layer doesn't have.

**Edit.** Two near-dead layer-23 neurons, n2664 and n2416, are rewired into a step. They were near-dead on dev: max |activation| 0.25, against a typical 1.76.
- gate = slop score minus threshold, with the threshold carried by the marker directions;
- up = the marker direction;
- down = a push toward " no", with opposite signs for the two neurons.

Neuron B is neuron A shifted 2 units higher, so A − B is a fixed drop of about 12 in the keep logit for flagged units, and nothing below the threshold. Layer 23 is the last layer, so only the keep scores change. The edit writes 2 gate rows, 2 up rows and 2 down columns.

```bash
python ws/keep_edit.py --base-edit results/ws/edit_author_lines.npz collect results/ws/keep_feats12_e1.npz --other 23
python ws/keep_edit.py edit results/ws/keep_feats12_e1.npz results/ws/edit_slop_step.npz --content results/ws/train_feats_e1.npz --shrink 0.2 --K 12
python ws/make_gguf.py results/ws/edit_combined.npz models/ws/v2ws2.gguf      # Edit 1 + Edit 2
build/yapzip < article.txt      # mainline since 2026-11-06: v2ws2 is the model built into build/yapzip
```

**Checks before the test set:**

| | v2 + Edit 1 | + Edit 2 |
|---|---|---|
| Dev, one article held out at a time (12 articles): slop kept | 184 | 50 |
| … facts kept | 182 | 181 ("We're five people, fully bootstrapped.", P(keep) was 0.50) |
| … code units kept | 27 | 27 |
| 3 rhetoric-heavy articles never used (LinkedIn, Medium, newsletter): slop kept | 120 | 35 |
| 30 held-out training chunks (all genres): keep accuracy | 0.910 | 0.907 (1 flip: a legal boilerplate clause) |
| Notes changed with keep decisions held fixed | | 2 of 157 (near-tied words) |
| 12 dev articles, real binary, v2 vs v2ws2: output size / hallucination flags | 32.8% / 101 | 26.6% / 82 |

**Test set: blind judges, by the project's priority order.** The judges saw `results/eval/judge9a` (articles) and `judge9b` (chats, emails, threads). None of these files were used to build Edit 2. 15 of the 22 inputs changed; the other 7 outputs are byte-identical to v2.

| | v2 | v2ws2 | ideal (Opus, 9 files) |
|---|---|---|---|
| meaning-changing errors (invented, wrong speaker, flipped) | 24 | **16** | 0 |
| minor errors (dropped hedge words, digits for number words) | 72 | 72 | 16 |
| output size, all 8 articles / all 14 chats, emails, threads | 33.1% / 51.8% | **30.9% / 49.5%** | |
| noise lines | 52 | **22** | 0 |
| code, commands and tables reproduced exactly | 5/5 | 5/5 | |
| key facts kept | 88% | 89% | 99% |
| questions answered from the output alone | 93/94 | 93/94 | |
| average rank (1 = best; judged by the priority order) | 2.13 | **1.93** | 1.22 |

- **Where the 8 fewer meaning errors come from:**
  - **Transit article, 5 → 2.** Edit 2 dropped three slop lines whose notes were wrong.
  - **Forum thread, 9 → 6.** This is Edit 1, and the forum was used to build Edit 1.
  - **Recipe, 2 → 0.** A near-tie change fixed "1 tbsp → 2 tsp".
- **Biggest size cuts:** the LinkedIn post (314 → 188 words) and the transit article (453 → 324).
- **Still wrong in both:** the 16 remaining meaning errors sit in fact lines both models keep:
  - "less frequent" for "more frequent";
  - "both opposed new routes";
  - Wi-Fi 6 given the 6 GHz band;
  - "UNLESS" for "only if";
  - sarcasm stated as fact;
  - "Dana:" on big_pete's post;
  - a garbled insomnia condition.
- **Gain is smaller on test than on dev:** most of the test articles' extra length is Key Takeaways and FAQ repeats across blocks, which are out of scope.

## Round 2: tried, not adopted

| Idea | Result | Why not adopted |
|---|---|---|
| **Keep author lines in threads** (step pair pushing "keep", reusing Edit 1's detector) | v2ws2 drops 1,054 of 1,371 author lines on the practice threads, and 478 of those are lines whose post is kept, so the post reads as the previous poster's. A push of +20 rescues 335 of the 387 risky lines on the held-out threads. | The same push also keeps 425 author lines whose post is dropped, which become empty noise lines. An author line's keep score is unrelated to its post's (correlation 0.0), and the keep stage decides each line before seeing the next one's decision. |
| **Retune the slop detector with 3 more rhetoric-heavy articles** (op-ed, sales emails, policy explainer) | The current edit already cuts slop on them from 93 to 30 lines with all 46 facts kept, unseen. Retuned on 15 articles, the held-out result is the same: 30. | Accuracy on the 30 held-out training chunks drops from 0.907 to 0.899. |
| **Copy boost:** scale the layer-11/15 copy heads (×1.3, ×1.6) | Notes already take 92–95% of their words from the source, and the boost doesn't change that. Hallucination flags rise from 7/2 to 10/10 at ×1.6, and 2 keep decisions flip. | The wrong notes come from how a sentence gets compressed (which words are dropped, which relations are rewritten), not from weak copying. |

Readable output is done in the binary, not the weights. There is one bullet per note, with the note's `; ` facts kept together, and long points wrap with a hanging indent in a terminal. It only adds the bullet and line breaks.

## Trap baseline (round 2)

`data/traps/` holds 225 short documents, each with one target line built to provoke a single kind of meaning flip (15 kinds, 15 traps each). Each trap carries exact checks: words the note must keep, and words whose presence means the meaning flipped. Run them with `ws/trap_eval.py TAG [--model X.gguf]`; it calls the real binary and takes about 1 minute per model.

| | v2 | v2ws2 |
|---|---|---|
| traps passing every check | 201 / 225 | 201 / 225 |
| identical notes | | 220 / 225 |
| real meaning changes, read by hand | ~9 | ~9 (the same ones) |

- **Most of the 24 failed checks are harmless:** a check that was too strict ("mouse" vs "mice", "10-20%" vs "10%"), or a detail dropped without changing the meaning.
- **The ~9 real ones are mostly about modality,** i.e. the speaker's stance toward a statement:
  - sarcasm read as sincere ("Oh great, a fourth sign-off step… that'll definitely speed things up" → "will speed things up");
  - a wish turned into a forecast ("fingers crossed the grant comes through" → "grant likely through");
  - a tag question stated as fact;
  - because/despite garbled ("keeping Dovetail despite a 9% higher price, because of the lower defect rate" → "price higher BECAUSE defect rate");
  - two subtle word swaps ("worse" → "worst", "trouble staying asleep" → "trouble sleeping").
- **No trap tripped a flip word** such as "less" or "unless". On short texts the models rarely invert facts; the errors in long articles come from compressing long paragraphs.

**Eval tools after the switch (2026-11-06).** `ws/dev_eval.py`, `ws/trap_eval.py` and `bench/slop_eval.py` call `build/yapzip`, which is now the mainline: v2ws2, blocks of 300 by default. They add `--flat` automatically. The v2 numbers above came from the legacy binary at blocks of 600. To reproduce them, pass `--model models/ws/embedded.gguf` (and `--block 600` where the tool has it).

## Edit 2b: page chrome (v2ws3 = v2 + Edit 1 + Edit 2b, the mainline)

**Problem.** Articles pasted from a browser keep bits of the page around the text. On a pasted Medium article, the mainline kept "Oct 26, 2026" and "33" (a clap count) at P(keep) ≈ 1. Edit 2's detector had never seen this kind of chrome.

**Data.** `ws/make_web_chrome.py` writes five pasted-page dev documents (Medium, Substack, a news site, LinkedIn, a dev blog). Their chrome is labelled boilerplate: bylines, dates, counts, read times, captions, buttons, footers. Short real facts sit right next to the chrome ("Monthly cost dropped to $2,900."), so the detector learns "page chrome", not "short line".

**Edit.** Same two-neuron step as Edit 2. The detector is rebuilt from 17 dev documents plus the all-genre training content, with step 24 and λ 30.

**Checks.**

| | Edit 2 (v2ws2) | Edit 2b (v2ws3) |
|---|---|---|
| 17 dev docs, one held out at a time: slop + chrome kept | 215 → 65 (step 12) | 215 → 66 |
| … facts kept | 202 → 200 | 202 → 200 (the same two borderline short lines) |
| 30 held-out training chunks (all genres): keep accuracy | 0.907 | 0.907 |
| notes changed with keep decisions held fixed (5 docs) | | 1 of 168 ("finished successfully" → "succeed") |
| pasted Medium article (never used): "Oct 26, 2026" / "33" | kept / kept | dropped (0.22) / dropped (0.0002) |
| trap set, 225 | 201 pass | 201 pass; notes identical on 219 |

- **Why the step is 24:** at 24 the date drops reliably; at 12 it only got partway.
- **Why λ is 30:** with λ 10, step 24 changed 8 of 168 notes while they were being written. λ 30 keeps the gates shut during note writing. The cost is a noisier threshold, which leaves more slop behind (66 kept vs 40); hallucinations come first.

```bash
python ws/make_web_chrome.py
python ws/keep_edit.py --base-edit results/ws/edit_author_lines.npz collect results/ws/keep_feats_web_e1.npz --other 23 --names dev_web_medium,dev_web_substack,dev_web_news,dev_web_linkedin,dev_web_devblog
python ws/keep_edit.py edit results/ws/keep_feats17_e1.npz results/ws/edit_slop_step_web.npz --content results/ws/train_feats_e1.npz --shrink 0.2 --K 24 --lam 30
python ws/make_gguf.py results/ws/edit_combined_v3.npz models/ws/v2ws3.gguf
```

## Round 3: fine-tuning on top of v2ws3 (Colab A100), not adopted

The question: does a short LoRA fine-tune from the mainline weights, with new article data, beat v2ws3? It doesn't, yet.

**Setup.**
- Start weights: `colab/v2ws3_start.npz`, built by `ws/make_start_delta.py`. v2 minus base Qwen is a rank-64 LoRA delta, so it ships as low-rank factors (83 MB instead of the 812 MB model). The rows and columns the edits wrote ship exactly.
- Colab's held-out check before training matched the Mac exactly: notes loss 1.595 per chunk on both.
- New data: `data/extra_train.jsonl` holds 338 chunks:
  - the 20 dev articles;
  - 5 pasted web articles, labelled, kept local (not published);
  - 95 repeat blocks and 95 controls.
- Training: 2 epochs at lr 5e-5. Only the LoRA is downloaded, then `ws/merge_lora.py` merges it into v2ws3.gguf.

**A finding about the training notes.** v2 was trained on the notes from before the strict relabel (commit 4932a28), not on the current ones. v2ws3's held-out notes loss is 0.80 against the old notes and 1.60 against the current ones. The models later trained from base on the strict notes (v5, v6) hallucinated more (judge5: 27 errors vs v2's 6; judge6: 15 vs 7). The old notes rebuild from git: `data/claude_train/part_*.jsonl` at `4932a28^`.

| | v2ws3 | v3a | v3b | v3c |
|---|---|---|---|---|
| notes trained on | (start) | current strict notes | v2-era notes | KL to v2ws3 at every note token |
| keep/drop trained on | | current labels + extra | v2-era labels + extra | v2-era labels + extra |
| test articles: key facts / size | 95% / 32% | 87% / 29% | 95% / 29% | 91% / 31% |
| traps (285): ok / flips | 259 / 0 | 253 / 2 | 251 / 4 (1 real: "as long as" → UNLESS) | 258 / 1 (a checker artifact: "NOT caused by database failover") |
| yap.txt words (of 2,000) | 940 | 737 | 615 | 729 |
| blind judges, 9 docs (judge10): meaning errors | 11 | | 14 | |
| … facts kept / partial | 96% / 3% | | 89% / 10% | |

- **v3a** learned the strict note style and lost facts the way v5 and v6 did. One example: "Achterberg cannot back $15 million out of the capital fund…" became "both explained their reasoning". It also changed "must" to "should".
- **v3b** kept the v2 style, but training on labels still moved how notes are written. The judges found more meaning errors than v2ws3, and on yap.txt it compressed too hard (17 facts kept, 7 partial, vs 23 and 2).
- **v3c** pinned note writing to v2ws3 by KL, and its notes stayed as safe as v2ws3's (traps 258, leak traps 60/60). But keep/drop over-learned and dropped real facts (docker article: 17/20 vs 20/20), for one point of size.
- **Not fixed by any of them:** in yap.txt, the note for one line copies the next line into itself. That is a note-stage leak, and v3a hid it only by dropping the line.

The notebook keeps all three modes (`colab/build_notebook.py`): continue from v2ws3, V2NOTES, and notes_kl. The LoRAs were deleted to free disk space; `ws/merge_lora.py` merges a downloaded LoRA into a GGUF.
