# yapzip

A gzip-style stream compressor for text headed into an LLM. It drops the parts that say nothing and rewrites the rest as short telegraph notes, keeping numbers, names, negations and conditions. Code, JSON, tables and logs are copied verbatim.

```
cat article.txt | yapzip > notes.txt
```

It runs locally: a fine-tuned **Qwen3.5-0.8B** in llama.cpp (Metal on a Mac, CUDA on Linux), one chunk at a time, flushing output as it goes.

## Install

macOS (Apple Silicon) with Homebrew. This repository is its own tap:

```
brew tap tgiridhar/yapzip https://github.com/tgiridhar/yapzip
brew trust tgiridhar/yapzip          # Homebrew 7 refuses formulae from third-party taps until trusted
brew install yapzip
yapzip -h                            # full help: options, output format, tips for scripts and agents
```

`yapzip` builds from source (llama.cpp b11803, a few minutes) and pulls in `yapzip-model`, which downloads the model (775 MB, release `model-v4c`) to `share/yapzip/yapzip.gguf`, where the binary finds it. The same commands work with Homebrew on Linux, which builds a CPU-only binary (untested). For an NVIDIA GPU, build from source with `-DGGML_CUDA=ON` (see **Build**).

## How it works

stdin is split into units (one per non-empty line; a fenced ``` block is one unit) and packed into chunks of about 600 tokens. Each chunk gets two stages:

1. **Keep/drop, without generating text.** The units, then one `?` marker per unit, go through the model in a single forward pass. The model's probability of ` yes` vs ` no` at each marker decides keep or drop. The markers come after the whole chunk, so a unit is judged with later replies in view.
2. **Notes.** For each kept unit the model writes a telegraph note, or `=` meaning "copy this unit verbatim". Everything is appended to the same context, so nothing is read twice.

```
units:
u1 Hey everyone, welcome back!
u2 We moved 12 services from MySQL to Redis in 3 weeks.
u3 ```bash⏎pip install tinyq==1.2.0⏎```
keep?
 u1? u2? u3?
notes:
u2 We moved 12 services ... =>  moved 12 services MySQL -> Redis in 3 weeks
u3 ```bash⏎pip install ...``` =>  =
```

The keep/drop scores are read through the normal output head (no new tokens or heads), so any runtime that exposes logits can run it.

## Results so far

| Setup | Keep/drop (held-out) | Notes |
|---|---|---|
| Untuned Qwen3.5-0.8B, one prompt per block | n/a | copies lines, loses about 1/3 of key facts, invents some |
| Untuned, per-line yes/no probability | AUC 0.92 | n/a |
| LoRA, last 2 blocks, on Mac (MLX), 300 Claude-written chunks | ~90% accuracy | still loops and mixes up lines out of sample |
| LoRA r=64 on all 24 layers, A100, same data (5 epochs) | 94.6% accuracy, AUC 0.982 | 0.4% of notes have a number not in the source; out of sample close to Opus on short inputs; no merging of repeated points yet |
| Same, plus 150 hard-case chunks (filler around a fact, unfenced code, long posts, hedges, multi-sentence turns) | **93.2% accuracy, AUC 0.986** | blind judges below; this is the released model |

The native binary with the trained Q8_0 GGUF on an M3 Pro (Metal) gives the same notes as the Python eval, at about 1–3 s per short input including model load, and 9.4 s for a 54-line chat.

Blind judging of the native binary, first model vs current model (Claude judges see the original and unlabeled outputs; Opus references included where we have them). 14 test inputs: chats, emails, a forum thread, a meeting, a blog, docs with code, mixed code/prose, filler lines:

| | First model | Current model | Opus |
|---|---|---|---|
| Key facts kept | 89% | 91% | 98% (6 inputs) |
| Facts changed (wrong number, flipped condition, wrong speaker) | 9 | 3 | 0 |
| Errors in output | 13 | 7 | 0 |
| Questions answered from the output alone | 46/48 | 48/48 | |
| Code/data reproduced exactly | 3/4 | 4/4 | 2/2 |
| Ranked best | 2 | 8 | 4 |

On a 3,900-word comment thread the current model kept 64% of facts vs 51% (it no longer drops long posts or turns wishes into facts), at 1,850 words vs 1,210. Remaining errors: misattributed replies in nested forum threads, one "only if" written as "unless", sarcasm read literally, a long post copied whole instead of compressed, and `key=value` inside a filler line sometimes copied with the filler. `bench/judge_instructions.md` is the judge prompt.

### Long AI-written articles (the main use case)

The test set is 8 long-winded AI-style articles, 15,200 words in all, written in-session:
- an SEO Wi-Fi how-to;
- a Python asyncio explainer with code;
- a recipe blog with a life story;
- a LinkedIn thought-leader post;
- a product launch post;
- a pasted chatbot answer about Docker;
- a sleep listicle;
- an AI-rewritten local news story.

Each has an answer key (167 facts and 62 questions in all) and an ideal compression. The articles, answer keys and outputs are not published.

The traps are negations, conditions, hedges, corrections, numbers in words, similar numbers, and Key Takeaways/FAQ sections that repeat the body. `bench/slop_eval.py` runs the binary on them and checks the answer keys automatically.

| | Current model | Ideal compression (Opus) |
|---|---|---|
| Output size vs input | 33% | 23% |
| Key facts kept / partial | 88% / 10% | 99% / 1% |
| Errors (judges' count; most of the ideal's are dropped hedge words) | 37 | 14 |
| Lines that only repeat earlier output (Key Takeaways, FAQ) | 72 | 0 |
| Lines with no fact (rhetoric, headings, disclaimers) | 35 | 0 |
| Questions answered from the output alone | 61/62 | 62/62 |
| Code blocks reproduced exactly | 4/4 | 3/4 |

**Facts survive, the slop doesn't go.**
- **Errors that change meaning:** about a dozen of the 37. Examples:
  - two no-votes merged under an invented shared reason;
  - "more frequent buses" written as "less frequent";
  - "1 tbsp → 2 tsp" written as "1 → 2 tsp", which reads as an increase;
  - plain Wi-Fi 6 credited with the 6 GHz band.
- **Repeats:** each ~600-token chunk is compressed with a cleared context, so a Key Takeaways or FAQ section can't be recognized as a repeat of earlier chunks.
- **Rhetoric:** the keep stage scores lines like "It's a gap in trust." at P(keep) ≈ 1.0, so raising `--threshold` doesn't help.
- **Training data:** it has few articles like these. Its median unit is 17 words; these articles' paragraphs run 40–150.

**Mainline model: `models/ws/v4c.gguf`, built into `build/yapzip`.** v4c is v2ws3 (below) plus a LoRA trained on relabeled data (every training note rechecked for meaning errors and recompressed, see `data/STYLE.md`; 19 pasted web articles added). Against v2ws3: 0 meaning flips in the 285 traps (same), output 26% vs 32% of the input on the 8 test articles and 43% vs 47% on a pasted blog post, code blocks 17/17 exact, key facts 91% vs 95% (mostly reworded, a few secondary details cut). Older model versions were deleted to free disk space; v4c is the only one kept.

**Previous mainline: `models/ws/v2ws3.gguf`.** v2ws3 is v2ws2 plus page-chrome removal (bylines, dates, like/clap counts, captions, share buttons, footers), still done only in the weights. The numbers below were measured on v2ws2. It is v2 plus two weight-space edits made without training (see `ws/README.md`), and it beats v2 on the test set:
- **Meaning-changing errors:** 24 → 16.
- **Noise lines:** 52 → 22.
- **Output size:** 33.1% → 30.9% on articles, 51.8% → 49.5% on chats, emails and threads.
- **Unchanged:** code (5/5) and facts kept (88–89%).

`build/yapzip` is the mainline binary. The local build loads `models/ws/v4c.gguf` (`-DYAPZIP_DEFAULT_MODEL`); build with `-DYAPZIP_MODEL=...` to embed the model for a standalone file. It has the new layout, streaming, summary and safety features (see **Use**).

## Layout

| Path | What |
|---|---|
| `native/yapzip.cpp`, `native/CMakeLists.txt` | the binary (C++ on the llama.cpp API) |
| `colab/yapzip_train_a100.ipynb` | training notebook: LoRA, eval, GGUF export, builds and runs the binary |
| `data/STYLE.md`, `data/sample_train.jsonl` | the note style guide and a 4-chunk sample of the training format (the training data itself is not published) |
| `data/toy.py` | template generator (chat, threads, stories, blogs, docs with code) |
| `Formula/` | Homebrew formulas: this repository is its own tap |
| `LICENSE`, `NOTICE`, `licenses/` | MIT license for this code; licenses and notices for Qwen3.5-0.8B (Apache-2.0) and llama.cpp (MIT) |
| `bench/` | Mac experiments (MLX training, out-of-sample comparison), the blind-judge eval (`judge_packets.py`, `judge_instructions.md`, `judge_score.py`) and the article deslop test (`slop_eval.py`) |
| `ws/` | weight-space edits: reading and editing the model's weights directly, no training (see `ws/README.md`) |
| `yapzip/` | the first Python prototype (prompted, via llama-server) |

## Train

Open `colab/yapzip_train_a100.ipynb` in Colab with an A100 runtime and run all cells. Upload a data zip when cell 2 asks: `colab/make_data_zip.py` builds one from your own labelled chunks in the `data/sample_train.jsonl` format. Training takes about 12 minutes. The notebook evaluates on held-out chunks, prints out-of-sample comparisons against Opus, exports `yapzip-qwen3.5-0.8b-q8_0.gguf`, and builds and runs the binary. Settings worth trying: `score_mode` (appended / inline), `notes_mode` (restate / id), `full_finetune`.

## Build

The binary links llama.cpp statically:

```bash
git clone --depth 1 --branch b11803 https://github.com/ggml-org/llama.cpp native/llama.cpp
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release                                   # Mac (Metal)
cmake -S native -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release   # Linux + NVIDIA
cmake --build build --target yapzip -j
```

### One file with the model inside

Pass `-DYAPZIP_MODEL=/path/to/model.gguf` to CMake and the GGUF goes into the binary. On macOS it goes into a `__YAPZIP,__gguf` Mach-O section, which keeps the code signature valid. On Linux it is appended with a trailer (`native/embed.sh`). The binary then runs with no `-m`:

```bash
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release -DYAPZIP_MODEL=$PWD/models/ws/v4c.gguf   # mainline model
cmake --build build --target yapzip -j
otool -L build/yapzip      # only system frameworks
cat chat.txt | build/yapzip > short.txt
```

On an M3 Pro the result is one 829 MB file. The first run takes ~30 s while macOS compiles the Metal shaders and reads the file cold; after that it takes ~1 s per short input.

## Use

```
yapzip [-m MODEL.gguf] [--block 300] [--threshold 0.5] [--keep-only] [--flat] [--wrap N] [-v | -vv] < input > output
```

Notes are laid out as points by default: one bullet per note, with the note's `; `-separated facts kept together on that line. In a terminal, long points wrap with a hanging indent at the terminal width. When the output goes to a file or a pipe, no line breaks are added. `--wrap N` sets the width, and `--flat` prints plain notes without bullets. The layout only adds the bullet and line breaks; it never changes words. Verbatim units (code, commands, tables) are left as they are.

**Streaming and blocks.**
- **Input** is read as it arrives. A block is compressed once about 300 tokens of whole lines have arrived (about 225 words, `--block N`), or when the input ends.
- **Output:** each note is printed the moment it is written. On a 2,500-word article the first note appears after about 1 s and the rest follow line by line.
- **Stats:** nothing goes to stderr by default except problems (a block copied through after an error). `-v` prints stats at the end:
  ```
  yapzip: 424 -> 170 words (40%), 504 -> 237 tokens (47%)
  yapzip: 15 units, 10 kept (67%), 2 blocks of <=300 tokens
  yapzip: 3.4 s (reading 0.3 s, notes 2.8 s, 80 note tokens/s); model qwen35 0.8B Q8_0, embedded, 99 GPU layers
  ```
  `-vv` adds debug output: per-block stats, the model's top-3 first note tokens, llama.cpp logs. `-q` is still accepted; quiet is the default.
- **Block size:** 300 by default (training chunks were ~150-600 tokens). Measured on v2:
  - On 15 dev articles, blocks of 300, 450, 600 and 900 tokens took the same time and dropped no facts.
  - Their serious hallucination flags (added negations, foreign speakers, unsupported conditions) did not differ beyond noise.
  - 600 and 900 gave the smallest output: 26.3% vs 27.2–27.3% for 300 and 450.
  - Much larger blocks (1,500–3,000) roughly doubled the hallucination flags.

Watch the shell when you pipe text in: inside double quotes, `$90` is expanded as a shell variable before yapzip sees it. Use single quotes, a file or `pbpaste`.

`--keep-only` skips notes and prints the kept units verbatim, so nothing can be invented. `--score-mode` and `--notes-mode` must match the settings the model was trained with.

## License

The yapzip code is under the [MIT License](LICENSE). It comes with **absolutely no warranty**: use it at your own risk. Its notes are lossy and can be wrong, so check anything important against the source.

The model is a fine-tuned version of [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (Alibaba Cloud) and is distributed under the Apache License 2.0 ([`licenses/Qwen3.5-0.8B-LICENSE`](licenses/Qwen3.5-0.8B-LICENSE)). [`NOTICE`](NOTICE) lists what was changed. The binary links [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT, [`licenses/llama.cpp-LICENSE`](licenses/llama.cpp-LICENSE)).

The training data and the eval inputs and outputs are not published. `data/sample_train.jsonl` shows the training format.
