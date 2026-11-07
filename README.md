# yapzip

yapzip makes text shorter before an LLM or a person reads it. It removes filler and writes each useful line as a short note. It copies code, commands and tables exactly.

A fine-tuned Qwen3.5-0.8B model does the work on your computer, in llama.cpp. yapzip sends no data over the network.

## Examples

A blog post:

````
$ cat post.txt
Hey everyone, welcome back to the blog!
In today's fast-paced world, keeping your infrastructure lean has never been more important.
Last quarter we moved 12 services from MySQL to Redis in 3 weeks, and p99 latency fell from 180 ms to 40 ms.
The final cutover is NOT happening Friday; it moves to Monday 9am, unless QA finds a blocker.
```bash
pip install tinyq==1.2.0
```
Thanks for reading, and don't forget to like and subscribe!

$ yapzip < post.txt
- moved 12 services MySQL -> Redis in 3 weeks; p99 latency 180 ms -> 40 ms
- final cutover NOT Friday; Monday 9am UNLESS QA finds blocker
```bash
pip install tinyq==1.2.0
```
````

A team chat:

````
$ cat chat.txt
thanks everyone!!
[10:02] dana: payments-api 4.12 rolled back, 502s gone; root cause was the keepalive change
[10:03] li: nice
```
$ kubectl rollout undo deploy/payments-api -n prod
```

$ yapzip < chat.txt
- [10:02] dana: payments-api 4.12 rolled back, 502s gone; root cause keepalive change
```
$ kubectl rollout undo deploy/payments-api -n prod
```
````

One line:

```
$ echo 'the deploy is NOT going out Friday, it moves to Monday 9am' | yapzip
- deploy NOT out Friday -> Monday 9am
```

## Install

On a Mac with Apple Silicon, type:

```
brew tap tgiridhar/yapzip https://github.com/tgiridhar/yapzip
brew trust tgiridhar/yapzip
brew install yapzip
```

Homebrew 7 does not load formulas from a third-party tap until you trust the tap. `brew install` compiles yapzip and llama.cpp, then downloads the model (812 MB).

On Linux, the same commands compile a binary that uses only the CPU. For an NVIDIA GPU, compile from the source code (refer to [Build](#build)).

## Use

```
yapzip [options] < input.txt > notes.txt
pbpaste | yapzip
```

Put one message, comment or paragraph on each line. yapzip writes each note when it is ready, so the first note comes after approximately 1 second.

| Option | Function |
|---|---|
| `--flat` | Write plain notes, one on each line, with no bullets. Use this option in scripts. |
| `--keep-only` | Write the useful lines without change. The model writes no text, so it cannot add errors. |
| `--threshold P` | Keep a line when its keep probability is P or more (default 0.5). A lower value keeps more lines. |
| `--wrap N` | Wrap notes at N columns. The default is the terminal width. Output to a file or a pipe does not wrap. |
| `-m FILE` | Use a different model file. |
| `-v` | At the end, write word counts, token counts and time to stderr. |

`yapzip -h` shows all options, the note format and tips for scripts and agents.

CAUTION: In double quotes, the shell changes `$90` before yapzip reads the text. Use single quotes, a file or `pbpaste`.

## Note format

| Mark | Meaning |
|---|---|
| `NOT`, `NEVER`, `IF`, `ONLY IF`, `UNLESS`, `BECAUSE`, `BUT` | Words that change the meaning, in capital letters |
| `->` | Result, change or correction (`old -> new`) |
| `~` | Approximately |
| `+` | And |
| `w/`, `w/o` | With, without |
| `asks: ...?` | A question |
| `per X:` | A statement that X made |

## How it works

yapzip divides the input into units. A unit is one line, or one code block from ` ``` ` to ` ``` `. yapzip puts the units into blocks of approximately 300 tokens. The model does two steps on each block:

1. Keep or remove. The model reads the block, then one `?` marker for each unit. At each marker, the probability of ` yes` sets the decision. This step writes no text.
2. Write notes. For each kept unit, the model writes one note, or `=` to copy the unit exactly.

## Limits

- The notes are lossy. The model can change a meaning, for example when it removes "only". Check important facts in the source.
- yapzip compresses each block alone. Thus a summary at the end of a long article can repeat facts from earlier blocks.
- A code block must start at the start of a line. After a fence in the middle of a line, such as ` dana: ``` `, yapzip can copy the next lines as code.

## Build

```
git clone --depth 1 --branch b11803 https://github.com/ggml-org/llama.cpp native/llama.cpp
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target yapzip -j
```

For an NVIDIA GPU, add `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80` to the first `cmake` command.

yapzip looks for the model in this sequence:

1. The file given with `-m`.
2. The file in `$YAPZIP_MODEL`.
3. A model inside the binary. To put the model in the binary, add `-DYAPZIP_MODEL=/path/to/model.gguf` to `cmake`.
4. The file set with `-DYAPZIP_DEFAULT_MODEL` at build time.
5. `../share/yapzip/yapzip.gguf`, relative to the binary. Homebrew installs the model there.

## Train

`colab/yapzip_train_a100.ipynb` trains the model with LoRA on a Colab A100 GPU. The training data is not published, so the notebook needs your own data. `data/sample_train.jsonl` shows the format, and `data/STYLE.md` gives the note rules. `ws/README.md` is the research log for the model.

## License

The yapzip code has the [MIT License](LICENSE). There is no warranty: you use yapzip at your own risk.

The model is a modified version of [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) by Alibaba Cloud. It has the [Apache License 2.0](licenses/Qwen3.5-0.8B-LICENSE), and [NOTICE](NOTICE) lists the changes. The binary contains [llama.cpp](https://github.com/ggml-org/llama.cpp), which has the [MIT License](licenses/llama.cpp-LICENSE).
