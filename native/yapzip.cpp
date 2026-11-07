// yapzip: a gzip-style stream compressor for text headed into an LLM.
//
//   cat chat.txt | yapzip > short.txt                      (model embedded in the binary)
//   cat chat.txt | yapzip -m other-model.gguf > short.txt  (or load a model file)
//
// Which model: -m, else $YAPZIP_MODEL, else a model embedded in the executable, else an installed model file:
// the build's YAPZIP_DEFAULT_MODEL (Homebrew: $(brew --prefix)/share/yapzip/yapzip.gguf), then
// share/yapzip/yapzip.gguf next to the executable's bin/ directory.
//
// The model can live inside the executable: on macOS in a __YAPZIP,__gguf section added at link time
// (-sectcreate, keeps the code signature valid); elsewhere appended to the file with a 16-byte trailer
// ("YZGGUF01" + little-endian offset), see native/embed.sh. Either way the GGUF is read straight from the
// executable file at its offset.
//
// stdin is split into units (one per non-empty line; a fenced ``` block is one unit), packed into chunks of
// about --block tokens, and each chunk goes through the fine-tuned model in two stages:
//   1. keep/drop: one forward pass over the chunk; the model's P(" yes") vs P(" no") at a "?" marker per unit
//   2. notes: for each kept unit the model writes a telegraph note, or "=" meaning "copy the unit verbatim"
// The token layout matches the training notebook exactly (colab/yapzip_train_a100.ipynb).
// Output for a chunk is flushed before the next chunk is read.

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/ioctl.h>
#include <unistd.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <mach-o/getsect.h>
#endif

static std::string self_path() {
#ifdef __APPLE__
    char buf[PATH_MAX];
    uint32_t n = sizeof(buf);
    if (_NSGetExecutablePath(buf, &n) != 0) return "";
    char real[PATH_MAX];
    return realpath(buf, real) ? std::string(real) : std::string(buf);
#else
    return "/proc/self/exe";
#endif
}

// Returns the executable opened and positioned at the start of an embedded GGUF, or nullptr if there is none.
static FILE * open_embedded_model() {
    std::string path = self_path();
    if (path.empty()) return nullptr;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return nullptr;
#ifdef __APPLE__
    const struct section_64 * s = getsectbyname("__YAPZIP", "__gguf");
    if (s && s->size > 0 && fseeko(f, (off_t) s->offset, SEEK_SET) == 0) return f;
#endif
    // Appended model: last 16 bytes are "YZGGUF01" followed by the GGUF's offset (uint64, little endian).
    unsigned char tail[16];
    if (fseeko(f, -16, SEEK_END) == 0 && fread(tail, 1, 16, f) == 16 && memcmp(tail, "YZGGUF01", 8) == 0) {
        uint64_t off = 0;
        for (int i = 7; i >= 0; i--) off = (off << 8) | tail[8 + i];
        if (fseeko(f, (off_t) off, SEEK_SET) == 0) return f;
    }
    fclose(f);
    return nullptr;
}

// Installed model files a build without an embedded model looks for, in order.
static std::vector<std::string> default_model_paths() {
    std::vector<std::string> v;
#ifdef YAPZIP_DEFAULT_MODEL
    v.push_back(YAPZIP_DEFAULT_MODEL);
#endif
    std::string invoked;   // the path as invoked (e.g. a Homebrew bin/ symlink), and the resolved executable
#ifdef __APPLE__
    char buf[PATH_MAX];
    uint32_t n = sizeof(buf);
    if (_NSGetExecutablePath(buf, &n) == 0) invoked = buf;
#else
    char buf[PATH_MAX];
    ssize_t k = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (k > 0) { buf[k] = 0; invoked = buf; }
#endif
    for (const std::string & p : {invoked, self_path()}) {
        size_t s = p.rfind('/');
        if (s != std::string::npos) v.push_back(p.substr(0, s) + "/../share/yapzip/yapzip.gguf");
    }
    return v;
}

struct Args {
    std::string model;
    std::string score_mode = "appended";  // appended | inline   (must match training)
    std::string notes_mode = "restate";   // restate | id        (must match training)
    int block = 300;                      // unit tokens per chunk (training chunks were ~150-600; 300 chosen 2026-11-06)
    float threshold = 0.5f;               // keep a unit when P(keep) >= threshold
    bool keep_only = false;               // skip notes: print kept units verbatim (extractive mode)
    bool points = true;                   // readable layout: one bullet per note (its "; " facts stay on that point)
    int wrap = -1;                        // wrap points to this width; -1: the terminal's width, none for files/pipes
    int verbose = 0;                      // 0: nothing on stderr but problems; 1 (-v): stats + info; 2 (-vv): debug
    bool scores = false;                  // print P(keep) per unit to stderr
    int n_gpu_layers = 99;
    int max_note_tokens = 256;           // a note ends at its newline; this only stops a runaway
    int n_ctx = 8192;
};

static bool g_debug = false;

static void log_cb(enum ggml_log_level level, const char * text, void *) {
    if (g_debug || level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

static void usage(FILE * out) {
    fputs(R"HELP(yapzip - compress text before an LLM (or a person) reads it. Drops filler, writes terse notes for the rest.

USAGE
  yapzip [options] < input.txt > notes.txt
  cat article.txt | yapzip
  pbpaste | yapzip -v

WHAT IT DOES
  Reads text on stdin and splits it into units: one per non-empty line; a ``` fenced block is one unit.
  A local model (fine-tuned Qwen3.5-0.8B, runs on the Mac GPU, no network) decides keep or drop for each unit:
    dropped: filler, greetings, reactions, hype/slop, page chrome (bylines, post dates, like/clap counts,
             share/subscribe/footer text), headings with no content, recaps of facts already in the block
    kept:    facts, numbers, names, IDs, dates, decisions, instructions, warnings, questions that matter, code
  Each kept unit becomes one short telegraph note. Code, commands, config, JSON/YAML, tables, logs and error
  text are copied exactly. Notes keep the input's order. Output is lossy on purpose (notes, not a summary):
  numbers, names, IDs and every negation, condition and hedge are kept; minor details may be cut.

OUTPUT
  stdout  one note per kept unit, as a bullet ("- note"); --flat gives plain lines. Verbatim units
          (code, tables) are printed exactly as in the input, without a bullet. Streams as it goes.
  Note conventions:
    NOT NEVER IF "ONLY IF" UNLESS BECAUSE BUT VS   caps mark meaning-critical words; never dropped
    ->  result / change / then (a correction: "old -> new")      ~  approx      +  and      w/ w/o  with/without
    "asks: ...?"  a question        "per X:"  an attributed claim        "name:" / "[time] name:"  speaker
  stderr  nothing by default except problems (a block that fails is copied through verbatim, never lost).
  Exit status 0 on success; 1 on a bad option or when no model is found.

OPTIONS
  -m, --model FILE  model GGUF. Default: $YAPZIP_MODEL, else the model embedded in this binary, else the
                    installed share/yapzip/yapzip.gguf (Homebrew: $(brew --prefix)/share/yapzip/yapzip.gguf)
  --flat            plain notes, one per line, no bullets or wrapping (best for scripts and agents)
  --wrap N          wrap bullets at N columns (default: the terminal width; never when piped)
  --block N         tokens per block, about 0.75 words each (default 300). The model sees one block at a time:
                    repeats across blocks are kept; much larger blocks are slower and less accurate
  --threshold P     keep a unit when P(keep) >= P (default 0.5). Lower keeps more, higher keeps less
  --keep-only       no notes: print the kept units verbatim (extractive)
  --scores          print P(keep) for every unit on stderr: "[score] 0.97<TAB>unit"
  -v                stats on stderr at the end: words and tokens in -> out, units kept, blocks, time, model
  -vv               debug on stderr: per-block stats, the model's first-token choices, llama.cpp logs
  -ngl N            model layers on the GPU (default 99 = all)
  --score-mode M, --notes-mode M   advanced; must match how the model was trained (appended, restate)
  -h, --help        this help

TIPS FOR AGENTS AND SCRIPTS
  - Input is plain text. Keep line breaks: one message, comment or paragraph per line works best.
    Chats, threads, emails, articles, docs, logs and pasted web pages (with their chrome) all work.
  - Use --flat to parse the output: each output line is one kept unit (or a verbatim code/table unit).
  - Code inside ``` fences is always copied exactly. Fence code you need preserved.
  - Typical output is 25-45% of the input for articles (padded blogs shrink most; dense news least).
  - Speed on Apple Silicon: about 1 s per 100 words (a 2,000-word article takes 15-20 s).
  - Safe to chain: yapzip output can be fed to yapzip again.
  - Shell quoting: use single quotes or a file; inside double quotes "$90" is expanded by the shell.

EXAMPLES
  $ echo 'the deploy is NOT going out Friday, it moves to Monday 9am' | yapzip
  - deploy NOT out Friday -> Monday 9am

  $ yapzip --flat < standup.txt         (input: "thanks everyone!!" / a status line / a fenced kubectl command)
  [10:02] dana: payments-api 4.12 rolled back, 502s gone; root cause keepalive change
  ```
  $ kubectl rollout undo deploy/api -n prod
  ```

  $ cat article.txt | yapzip -v > notes.txt
  yapzip: 2000 -> 853 words (43%), 2635 -> 1282 tokens (49%)        (on stderr)
)HELP", out);
}

static std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static size_t ulen(const std::string & s) {   // display columns, counting UTF-8 code points
    size_t n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}

// One point, word-wrapped at `width` with a hanging indent so continuation lines sit under the text, not the bullet.
static std::string wrap_point(const std::string & lead, const std::string & text, int width) {
    size_t w = width > 0 ? (size_t) width : 0, ind = ulen(lead);
    if (w < ind + 20 || ind + ulen(text) <= w) return lead + text + "\n";
    std::string out, line = lead;
    size_t col = ind;
    bool fresh = true;
    for (size_t i = 0; i <= text.size();) {
        size_t j = text.find(' ', i);
        if (j == std::string::npos) j = text.size();
        std::string word = text.substr(i, j - i);
        i = j + 1;
        if (word.empty()) continue;
        if (!fresh && col + 1 + ulen(word) > w) {
            out += line + "\n";
            line = std::string(ind, ' ');
            col = ind;
            fresh = true;
        }
        if (!fresh) { line += ' '; col++; }
        line += word;
        col += ulen(word);
        fresh = false;
    }
    return out + line + "\n";
}

// Points layout for one note: "- note", the note's own "; " facts kept together on that point, wrapped with a
// hanging indent when a width is set (terminal). Only adds the bullet and line breaks: words are never changed.
static std::string as_points(const std::string & note, int width) {
    std::string p = trim(note);
    if (p.rfind("- ", 0) == 0 || p.rfind("* ", 0) == 0) p = trim(p.substr(2));   // the note already had a bullet
    return p.empty() ? "\n" : wrap_point("- ", p, width);
}

class Compressor {
  public:
    explicit Compressor(const Args & a) : a_(a) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = a.n_gpu_layers;
        if (!a.model.empty()) {
            model_ = llama_model_load_from_file(a.model.c_str(), mp);
            if (!model_) die("could not load model " + a.model);
        } else {
            embedded_ = open_embedded_model();  // kept open: the model's weights are mapped from this file
            if (embedded_) {
                model_ = llama_model_load_from_file_ptr(embedded_, mp);
                if (!model_) die("could not load the embedded model");
            } else {
                for (const std::string & p : default_model_paths())
                    if (access(p.c_str(), R_OK) == 0) { a_.model = p; break; }
                if (a_.model.empty())
                    die("no model found: pass -m MODEL.gguf, set YAPZIP_MODEL, or install one "
                        "(brew install yapzip-model)");
                model_ = llama_model_load_from_file(a_.model.c_str(), mp);
                if (!model_) die("could not load model " + a_.model);
            }
        }
        vocab_ = llama_model_get_vocab(model_);
        n_vocab_ = llama_vocab_n_tokens(vocab_);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = a.n_ctx;
        cp.n_batch = 2048;
        cp.n_ubatch = 512;
        ctx_ = llama_init_from_model(model_, cp);
        if (!ctx_) die("could not create context");
        n_batch_ = (int) cp.n_batch;
        batch_ = llama_batch_init(n_batch_, 0, 1);

        yes_ = single(" yes");
        no_ = single(" no");
        q_ = single("?");
        nl_ = enc("\n");
    }

    ~Compressor() {
        llama_batch_free(batch_);
        llama_free(ctx_);
        llama_model_free(model_);
        if (embedded_) fclose(embedded_);
    }

    int count(const std::string & s) { return (int) enc(s).size(); }

    // Compress one chunk of units; returns the output text for the chunk.
    std::string process(const std::vector<std::string> & units) {
        auto t0 = std::chrono::steady_clock::now();
        llama_memory_clear(llama_get_memory(ctx_), true);
        n_pos_ = 0;
        next_ = 0;
        p_.clear();
        out_.clear();

        // Stage 1: units + one "?" marker per unit, one pass, P(keep) read at every marker.
        std::vector<llama_token> seq = enc("units:\n");
        std::vector<int> marks;
        if (a_.score_mode == "inline") {
            for (size_t i = 0; i < units.size(); i++) {
                append(seq, enc("u" + std::to_string(i + 1) + " " + units[i]));
                seq.push_back(q_);
                marks.push_back((int) seq.size() - 1);
                append(seq, nl_);
            }
        } else {
            for (size_t i = 0; i < units.size(); i++) append(seq, enc("u" + std::to_string(i + 1) + " " + units[i] + "\n"));
            append(seq, enc("keep?\n"));
            for (size_t i = 0; i < units.size(); i++) {
                append(seq, enc(" u" + std::to_string(i + 1)));
                seq.push_back(q_);
                marks.push_back((int) seq.size() - 1);
            }
            append(seq, nl_);
        }
        std::vector<float> p = feed_scores(seq, marks);
        p_ = p;
        auto t1 = std::chrono::steady_clock::now();
        if (a_.scores)
            for (size_t i = 0; i < units.size(); i++) fprintf(stderr, "[score] %.4f\t%s\n", p[i], units[i].c_str());

        std::string & out = out_;                  // the content (notes, verbatim units) without layout, for the summary
        auto emit = [&](const std::string & shown, const std::string & content) {   // stream each note as it is written
            fwrite(shown.data(), 1, shown.size(), stdout);
            fflush(stdout);
            out += content;
        };
        int kept = 0, gen = 0;
        bool first = true;
        for (size_t i = 0; i < units.size(); i++) {
            next_ = i;                             // units before i are done (written or dropped)
            if (p[i] < a_.threshold) {
                // code is never dropped: a fenced block the model scored low is copied verbatim instead
                if (units[i].rfind("```", 0) == 0) emit(units[i] + "\n", units[i] + "\n");
                continue;
            }
            kept++;
            if (a_.keep_only) {
                emit(units[i] + "\n", units[i] + "\n");
                continue;
            }
            // Stage 2: note for a kept unit, appended to the same context (nothing is re-read).
            std::vector<llama_token> prompt = first ? enc("notes:\n") : std::vector<llama_token>{};
            first = false;
            if (a_.notes_mode == "restate") append(prompt, enc("u" + std::to_string(i + 1) + " " + units[i] + " =>"));
            else append(prompt, enc("u" + std::to_string(i + 1) + ":"));
            want_top_ = a_.verbose >= 2;   // -vv: print the model's top-3 first note tokens ("=" means copy verbatim)
            llama_token t = feed_argmax(prompt);
            want_top_ = false;
            if (a_.verbose >= 2) {
                fprintf(stderr, "[note] u%zu first token:", i + 1);
                for (auto & [tok, pr] : top_) fprintf(stderr, " '%s' %.2f", piece(tok).c_str(), pr);
                fprintf(stderr, "\n");
            }
            std::string note;
            bool ended = false;
            for (int g = 0; g < a_.max_note_tokens; g++, gen++) {
                if (llama_vocab_is_eog(vocab_, t)) break;
                std::string pc = piece(t);
                size_t nl = pc.find('\n');
                if (nl != std::string::npos) {
                    note += pc.substr(0, nl);
                    feed_argmax({t});
                    ended = true;
                    break;
                }
                note += pc;
                t = feed_argmax({t});
            }
            if (!ended) feed_argmax(nl_);
            note = trim(note);
            if (note == "=") emit(units[i] + "\n", units[i] + "\n");   // verbatim (code, tables): never reformatted
            else emit(a_.points ? as_points(note, a_.wrap) : note + "\n", note + "\n");
        }
        auto t2 = std::chrono::steady_clock::now();
        auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
        stats.units += units.size();
        stats.kept += kept;
        stats.note_tokens += gen;
        stats.score_ms += ms(t1 - t0);
        stats.note_ms += ms(t2 - t1);
        if (a_.verbose >= 2) {
            fprintf(stderr, "[yapzip] chunk: %zu units, %d kept, %zu scoring tokens in %.0f ms, %d note tokens in %.0f ms\n",
                    units.size(), kept, seq.size(), ms(t1 - t0), gen, ms(t2 - t1));
        }
        next_ = units.size();
        return out;
    }

    // After process() threw: what this block already wrote, and the units it had not reached yet. Those are
    // copied verbatim (they cannot be wrong), skipping units stage 1 already dropped.
    const std::string & partial() const { return out_; }
    struct Stats { long units = 0, kept = 0, note_tokens = 0; double score_ms = 0, note_ms = 0; } stats;
    std::string model_desc() const {
        char buf[128];
        llama_model_desc(model_, buf, sizeof buf);
        return std::string(buf) + (embedded_ ? ", embedded" : ", " + a_.model);
    }
    std::string fallback(const std::vector<std::string> & units) const {
        std::string rest;
        for (size_t k = next_; k < units.size(); k++)
            if (p_.empty() || p_[k] >= a_.threshold || units[k].rfind("```", 0) == 0) rest += units[k] + "\n";
        return rest;
    }

  private:
    size_t next_ = 0;          // progress through the current block
    std::vector<float> p_;     // its stage-1 scores (empty until stage 1 finished)
    std::string out_;          // its content written so far
    Args a_;
    FILE * embedded_ = nullptr;
    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    llama_batch batch_{};
    int n_vocab_ = 0, n_batch_ = 0, n_pos_ = 0;
    llama_token yes_ = 0, no_ = 0, q_ = 0;
    std::vector<llama_token> nl_;
    bool want_top_ = false;
    std::vector<std::pair<llama_token, float>> top_;

    [[noreturn]] static void die(const std::string & msg) {
        fprintf(stderr, "yapzip: %s\n", msg.c_str());
        exit(1);
    }

    static void append(std::vector<llama_token> & a, const std::vector<llama_token> & b) { a.insert(a.end(), b.begin(), b.end()); }

    std::vector<llama_token> enc(const std::string & s) {
        if (s.empty()) return {};
        int n = -llama_tokenize(vocab_, s.c_str(), (int) s.size(), nullptr, 0, false, false);
        std::vector<llama_token> t(std::max(n, 1));
        int got = llama_tokenize(vocab_, s.c_str(), (int) s.size(), t.data(), (int) t.size(), false, false);
        if (got < 0) die("tokenize failed");
        t.resize(got);
        return t;
    }

    llama_token single(const std::string & s) {
        auto t = enc(s);
        if (t.size() != 1) die("expected '" + s + "' to be one token");
        return t[0];
    }

    std::string piece(llama_token t) {
        char buf[256];
        int n = llama_token_to_piece(vocab_, t, buf, sizeof(buf), 0, false);
        if (n < 0) {
            std::string big(-n, '\0');
            llama_token_to_piece(vocab_, t, big.data(), -n, 0, false);
            return big;
        }
        return std::string(buf, n);
    }

    // Decode tokens in slices of n_batch; `want` lists indices whose logits are needed.
    // `on_logits(index, logits)` is called for each wanted index while its logits are valid.
    template <typename F>
    void feed(const std::vector<llama_token> & toks, const std::vector<int> & want, F on_logits) {
        size_t w = 0;
        for (size_t s = 0; s < toks.size(); s += n_batch_) {
            int n = (int) std::min<size_t>(n_batch_, toks.size() - s);
            batch_.n_tokens = n;
            for (int i = 0; i < n; i++) {
                batch_.token[i] = toks[s + i];
                batch_.pos[i] = n_pos_ + i;
                batch_.n_seq_id[i] = 1;
                batch_.seq_id[i][0] = 0;
                batch_.logits[i] = 0;
            }
            size_t w0 = w;
            while (w < want.size() && want[w] < (int) (s + n)) batch_.logits[want[w++] - s] = 1;
            // never exit mid-stream: the caller copies the rest of the block verbatim and carries on
            if (n_pos_ + n > a_.n_ctx) throw std::runtime_error("block too long for the model's context");
            if (llama_decode(ctx_, batch_) != 0) throw std::runtime_error("decode failed");
            n_pos_ += n;
            for (size_t k = w0; k < w; k++) on_logits(want[k], llama_get_logits_ith(ctx_, want[k] - (int) s));
        }
    }

    std::vector<float> feed_scores(const std::vector<llama_token> & toks, const std::vector<int> & marks) {
        std::vector<float> p(marks.size(), 0.0f);
        size_t m = 0;
        feed(toks, marks, [&](int, const float * lg) {
            p[m++] = 1.0f / (1.0f + std::exp(lg[no_] - lg[yes_]));  // softmax over {yes, no}
        });
        return p;
    }

    llama_token feed_argmax(const std::vector<llama_token> & toks) {
        llama_token best = 0;
        feed(toks, {(int) toks.size() - 1}, [&](int, const float * lg) {
            best = (llama_token) (std::max_element(lg, lg + n_vocab_) - lg);
            if (!want_top_) return;
            double z = 0;
            for (int k = 0; k < n_vocab_; k++) z += std::exp((double) (lg[k] - lg[best]));
            std::vector<int> idx(n_vocab_);
            for (int k = 0; k < n_vocab_; k++) idx[k] = k;
            std::partial_sort(idx.begin(), idx.begin() + 3, idx.end(), [&](int x, int y) { return lg[x] > lg[y]; });
            top_.clear();
            for (int k = 0; k < 3; k++) top_.push_back({idx[k], (float) (std::exp((double) (lg[idx[k]] - lg[best])) / z)});
        });
        return best;
    }
};

int main(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(stderr); exit(1); }
            return argv[++i];
        };
        if (s == "-m" || s == "--model") a.model = next();
        else if (s == "--block") a.block = std::stoi(next());
        else if (s == "--threshold") a.threshold = std::stof(next());
        else if (s == "--keep-only") a.keep_only = true;
        else if (s == "--points") a.points = true;
        else if (s == "--flat") a.points = false;
        else if (s == "--wrap") a.wrap = std::stoi(next());
        else if (s == "--score-mode") a.score_mode = next();
        else if (s == "--notes-mode") a.notes_mode = next();
        else if (s == "-ngl") a.n_gpu_layers = std::stoi(next());
        else if (s == "-v") a.verbose = std::max(a.verbose, 1);
        else if (s == "-vv") a.verbose = 2;
        else if (s == "-q") a.verbose = 0;   // kept for old scripts: quiet is the default now
        else if (s == "--scores") a.scores = true;
        else if (s == "-h" || s == "--help") { usage(stdout); return 0; }
        else { fprintf(stderr, "yapzip: unknown option %s\n\n", s.c_str()); usage(stderr); return 1; }
    }
    if (a.wrap < 0) {   // wrap to the terminal; never add line breaks to output going into a file or a pipe
        a.wrap = 0;
        if (isatty(STDOUT_FILENO)) {
            struct winsize ws {};
            if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) a.wrap = ws.ws_col;
            else if (const char * c = getenv("COLUMNS")) a.wrap = atoi(c);
            else a.wrap = 100;
        }
    }
    if (a.model.empty())   // no -m: $YAPZIP_MODEL, else embedded / installed (resolved when the model loads)
        if (const char * e = getenv("YAPZIP_MODEL"); e && *e) a.model = e;
    g_debug = a.verbose >= 2;
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    auto t_start = std::chrono::steady_clock::now();
    Compressor z(a);
    std::vector<std::string> chunk;
    int chunk_tok = 0;
    long in_words = 0, in_tok = 0, out_words = 0, out_tok = 0, blocks = 0;
    auto words = [](const std::string & s) {
        long n = 0;
        bool in = false;
        for (unsigned char c : s) {
            bool sp = c == ' ' || c == '\n' || c == '\t' || c == '\r';
            if (!sp && !in) n++;
            in = !sp;
        }
        return n;
    };
    auto put = [&](const std::string & s) {
        fwrite(s.data(), 1, s.size(), stdout);
        fflush(stdout);
    };
    auto flush = [&]() {   // a block is processed once ~a.block tokens of input have arrived (or at end of input);
        if (chunk.empty()) return;   // its notes stream to stdout from inside process()
        std::string out;
        try {
            out = z.process(chunk);
        } catch (const std::exception & e) {   // never die mid-stream: copy the rest of the block and go on
            std::string rest = z.fallback(chunk);
            put(rest);
            out = z.partial() + rest;
            fprintf(stderr, "yapzip: %s; copied the rest of the block verbatim\n", e.what());
        }
        out_words += words(out);
        out_tok += z.count(out);
        blocks++;
        chunk.clear();
        chunk_tok = 0;
    };
    const int max_unit = (a.n_ctx - 1024) / 2;   // a unit plus its restatement must fit the model's context
    auto add_unit = [&](const std::string & u) {
        int n = z.count(u);
        in_words += words(u);
        in_tok += n;
        if (n > max_unit) {                      // too big for the model: pass it through untouched
            flush();
            put(u + "\n");
            out_words += words(u);
            out_tok += n;
            if (a.verbose) fprintf(stderr, "yapzip: a %d-token line is too long for the model; copied verbatim\n", n);
            return;
        }
        if (!chunk.empty() && chunk_tok + n > a.block) flush();
        chunk.push_back(u);
        chunk_tok += n;
    };

    std::string line, fence;
    bool in_fence = false;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::string t = trim(line);
        if (in_fence) {
            fence += "\n" + line;
            if (t.rfind("```", 0) == 0) { in_fence = false; add_unit(fence); }
            continue;
        }
        if (t.rfind("```", 0) == 0) { in_fence = true; fence = t; continue; }
        if (!t.empty()) add_unit(t);
    }
    if (in_fence) add_unit(fence + "\n```");   // input ended inside a code block: close it so readers see valid code
    flush();
    if (a.verbose && in_words > 0) {
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        const auto & st = z.stats;
        fprintf(stderr, "yapzip: %ld -> %ld words (%.0f%%), %ld -> %ld tokens (%.0f%%)\n",
                in_words, out_words, 100.0 * out_words / in_words, in_tok, out_tok, in_tok ? 100.0 * out_tok / in_tok : 0.0);
        fprintf(stderr, "yapzip: %ld units, %ld kept (%.0f%%), %ld block%s of <=%d tokens\n", st.units, st.kept,
                st.units ? 100.0 * st.kept / st.units : 0.0, blocks, blocks == 1 ? "" : "s", a.block);
        fprintf(stderr, "yapzip: %.1f s (reading %.1f s, notes %.1f s, %.0f note tokens/s); model %s, %d GPU layers\n", secs,
                st.score_ms / 1000, st.note_ms / 1000, st.note_ms > 0 ? st.note_tokens / (st.note_ms / 1000) : 0.0,
                z.model_desc().c_str(), a.n_gpu_layers);
    }
    llama_backend_free();
    return 0;
}
