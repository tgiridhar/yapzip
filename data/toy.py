"""Synthetic toy data for a generic text compressor: chat logs, comment threads, stories, blog posts, docs.

Each chunk is one genre, split into units (a line, a comment, a sentence, a paragraph, or a whole code block).
Per unit the gold label is keep/drop, and for kept units a note:
  - a telegraph rewrite (names, numbers, timestamps, negations copied exactly), or
  - "=" meaning "copy this unit verbatim" (code, JSON, YAML, tables, CLI output).
Labels are right by construction because every template has a hand-written target.
"""

import json
import random
from pathlib import Path

NAMES = ["ana", "raj", "lee", "priya", "marcus", "dana", "tomas", "hana", "sam", "elena", "wei", "jo", "nina", "omar"]
HEROES = ["Mara", "Tobin", "Elsa", "Kade", "Ines", "Rook", "Yara", "Bram"]
TOWNS = ["Harrow", "Eastmere", "Vell", "Duncross", "Oakhollow", "Saltby"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MONTHS = ["Jan", "Feb", "March", "April", "May", "June", "Sept", "Oct", "Nov"]
THINGS = ["elevator repair", "offsite", "launch", "release", "team lunch", "audit", "server migration", "board review"]
ITEMS = ["lodge", "van", "venue", "catering", "conference room", "flights"]
SVCS = ["orders-db", "checkout-api", "auth-service", "billing-worker", "search-api"]
DOCS = ["permission form", "invoice", "timeline", "contract", "expense report"]
DBS = ["MySQL", "Postgres 16", "MongoDB", "DynamoDB", "SQLite", "Redis"]
PKGS = ["fastcache", "tinyq", "logpipe", "vaultd", "edgeproxy"]
VARS = ["CACHE_TTL", "MAX_WORKERS", "RETRY_LIMIT", "POOL_SIZE", "BATCH_SIZE"]
FILL_PRE = ["", "", "", "ok so ", "honestly ", "lol ", "um ", "so ", "fyi "]
FILL_POST = ["", "", "", " 🙏", "!!", " lol", ", sorry", " fyi", " 😅"]
CHAT_JUNK = ["morning all!", "lol", "+1", "same", "thanks!", "haha nice", "ugh", "this", "👍", "ok", "brb coffee",
             "anyone watch the game last night", "my cat just walked on my keyboard", "sorry, mic issues", "gm gm",
             "yes from me", "nice work everyone", "lol sure", "that's wild", "agreed", "🎉🎉", "can you all hear me?",
             "omg", "classic", "haha", "sounds good", "great, love that for us 🙄", "who's bringing donuts", "ty"]
STORY_JUNK = ["The wind howled through the empty streets.", "Somewhere far off, a dog barked.",
              "She wondered, not for the first time, what the point of it all was.", "The tea had gone cold hours ago.",
              "It was the kind of evening that made everything feel slower.", "He smiled, though it never reached his eyes.",
              "Rain tapped softly against the shutters.", "The fire crackled and spat.", "Nobody spoke for a long while.",
              "The stars were bright and indifferent."]
BLOG_JUNK = ["Hey everyone, welcome back to the blog!", "In today's post I want to talk about something that's been on my mind.",
             "Before we dive in, grab a coffee ☕", "If you enjoyed this, don't forget to subscribe!",
             "Let me know what you think in the comments.", "This was a wild ride, honestly.",
             "I know, I know, another post about databases.", "Anyway, enough rambling.", "Stay tuned for part two!"]
DOC_JUNK = ["This guide walks you through the setup.", "Configuration is easy!", "Let's get started.",
            "The following section describes the options.", "Here is an example:", "That's it, you're done!",
            "We hope you enjoy using it.", "Read on to learn more."]


def r(lo, hi):
    return random.randint(lo, hi)


def noisy(text):
    return random.choice(FILL_PRE) + text + random.choice(FILL_POST)


def office_fact():
    """A generic workplace fact (text, telegraph note), used by chat and comment threads."""
    d, d2 = random.sample(DAYS, 2)
    n, m, x, y, p = r(20, 400), r(5, 300), r(15, 900), r(15, 900), r(2, 60)
    hh, mm = r(7, 18), random.choice(["00", "15", "30", "45"])
    mon, dom = random.choice(MONTHS), r(1, 28)
    who, t, it, svc, doc = random.choice(NAMES).capitalize(), random.choice(THINGS), random.choice(ITEMS), \
        random.choice(SVCS), random.choice(DOCS)
    v = f"v{r(1, 6)}.{r(0, 20)}.{r(0, 9)}"
    room = r(100, 499)
    return random.choice([
        (f"the {t} is moved to {d} the {dom}th, not {d2}", f"{t} moved to {d} {dom}th, NOT {d2}"),
        (f"budget is ${x} per head, hard cap. do not go over unless {who} signs off",
         f"budget ${x}/head hard cap; NOT over unless {who} signs off"),
        (f"I'll book the {it}. deposit is {p}% due by {mon} {dom}", f"books {it}; deposit {p}% due {mon} {dom}"),
        (f"the {it} costs ${x} a year now, up from ${y}", f"{it} ${x}/year, up from ${y}"),
        (f"we have {n} open tickets, up from {m} last week", f"{n} open tickets, up from {m} last week"),
        (f"error rate on {svc} is {p}% since {hh}:{mm}", f"{svc} error rate {p}% since {hh}:{mm}"),
        (f"{v} of {svc} went out at {hh}:{mm}", f"{svc} {v} deployed {hh}:{mm}"),
        (f"do NOT restart {svc} until the backup finishes", f"do NOT restart {svc} until backup finishes"),
        (f"the {t} meeting is at {hh}:{mm} on {d}, room {room}", f"{t} meeting {d} {hh}:{mm}, room {room}"),
        (f"only {m} of {n + m} people answered the survey", f"{m} of {n + m} answered survey"),
        (f"I can't make it {d}, I'm out until {mon} {dom}", f"out until {mon} {dom}; NOT available {d}"),
        (f"{who} approved the {it} for ${x}", f"{who} approved {it}, ${x}"),
        (f"actually it's {d}, not {d2}. my bad", f"correction: {d2} -> {d}"),
        (f"please send the {doc} by {d}, otherwise we lose the {it}", f"send {doc} by {d}, else lose {it}"),
        (f"we only sign if legal approves the {doc}", f"sign ONLY IF legal approves {doc}"),
    ])


def chat_chunk():
    units, minute = [], r(0, 9 * 60)
    for _ in range(r(8, 16)):
        minute += r(0, 4)
        tag = f"[{8 + minute // 60:02d}:{minute % 60:02d}] {random.choice(NAMES)}:"
        if random.random() < 0.4:
            units.append((f"{tag} {random.choice(CHAT_JUNK)}", None))
        else:
            text, note = office_fact()
            units.append((f"{tag} {noisy(text)}", f"{tag} {note}"))
    return units


def thread_chunk():
    units = []
    for i in range(r(6, 12)):
        user = random.choice(NAMES) + random.choice(["_k", "79", "_dev", "", "_x"])
        tag = f"{'↳ ' if i and random.random() < 0.5 else ''}{user} (+{r(1, 120)}):"
        if random.random() < 0.4:
            units.append((f"{tag} {random.choice(CHAT_JUNK)}", None))
        else:
            text, note = office_fact()
            units.append((f"{tag} {noisy(text)}", f"{tag} {note}"))
    return units


def story_chunk():
    a, b = random.sample(HEROES, 2)
    t1, t2 = random.sample(TOWNS, 2)
    n, k, m = r(2, 28), r(5, 90), r(2, 12)
    mon = random.choice(MONTHS)
    facts = [
        (f"{a} left {t1} on the {n}th of {mon} with {k} silver coins and a borrowed horse.",
         f"{a} left {t1} {mon} {n}th with {k} silver coins, borrowed horse"),
        (f"{b} told {a} that the bridge at {t2} had collapsed, so they could not go north.",
         f"{b} told {a}: bridge at {t2} collapsed -> can NOT go north"),
        (f"{b} agreed to guide them for {k} gold, but only as far as {t2}.",
         f"{b} guides for {k} gold, ONLY as far as {t2}"),
        (f"On the {m}th night, {a} discovered that the map was a forgery.", f"night {m}: {a} found map was forgery"),
        (f"{a} was not the heir after all; {b} was.", f"{a} NOT heir; {b} is heir"),
        (f"They reached {t2} {m} days later, with {m + 1} of the {m + 4} horses still alive.",
         f"reached {t2} {m} days later; {m + 1} of {m + 4} horses alive"),
        (f"The innkeeper at {t1} refused to sell them food unless they paid {k} coins up front.",
         f"{t1} innkeeper: no food unless {k} coins up front"),
    ]
    random.shuffle(facts)
    units = []
    for f in facts[:r(3, 6)]:
        units.append(f)
        for _ in range(r(0, 2)):
            units.append((random.choice(STORY_JUNK), None))
    return units


def blog_chunk():
    n, k, x, y, p, cost = r(3, 40), r(2, 12), r(200, 900), r(20, 190), r(5, 60), r(500, 9000)
    d1, d2 = random.sample(DBS, 2)
    v = f"{r(1, 9)}.{r(0, 9)}"
    facts = [
        (f"We moved {n} services from {d1} to {d2} in {k} weeks.", f"moved {n} services {d1} -> {d2} in {k} weeks"),
        (f"p99 latency dropped from {x}ms to {y}ms after the switch.", f"p99 latency {x}ms -> {y}ms after switch"),
        (f"The migration cost about ${cost} in extra compute, which we did not budget for.",
         f"migration cost about ${cost} extra compute, NOT budgeted"),
        (f"Do not run VACUUM FULL on a replica; it locks every table.", "do NOT run VACUUM FULL on replica: locks every table"),
        (f"Version {v} fixed the memory leak, but only on Linux.", f"version {v} fixed memory leak, ONLY on Linux"),
        (f"Our bill went down {p}% in the first month.", f"bill down {p}% first month"),
    ]
    random.shuffle(facts)
    units = [(random.choice(BLOG_JUNK), None)]
    for f in facts[:r(3, 5)]:
        units.append(f)
        if random.random() < 0.5:
            units.append((random.choice(BLOG_JUNK), None))
    return units


def doc_chunk():
    pkg, var, svc = random.choice(PKGS), random.choice(VARS), random.choice(SVCS)
    port, n, k, x, p = r(3000, 9999), r(2, 600), r(1, 30), r(10, 900), r(1, 9)
    v = f"{r(1, 4)}.{r(0, 20)}.{r(0, 9)}"
    prose = [
        (f"Set {var} to {n} to cache responses for {k} minutes.", f"{var}={n} caches responses {k} minutes"),
        (f"The default port is {port}; change it with --port.", f"default port {port}; change with --port"),
        (f"Requires Python 3.{r(9, 13)} or newer.", None),  # note filled below
        (f"Do not set {var} to 0 in production; it disables retries.", f"do NOT set {var}=0 in production: disables retries"),
        (f"The rate limit is {n} requests per minute per API key.", f"rate limit {n} requests/min per API key"),
    ]
    code = [
        f"```bash\npip install {pkg}=={v}\n{pkg} serve --port {port}\n```",
        json.dumps({"service": svc, "replicas": k, "timeout_ms": x}),
        f"cache:\n  ttl: {n}\n  size_mb: {x}\n  evict: lru",
        f"| endpoint | p99 ms | errors |\n|---|---|---|\n| /login | {x} | {p}% |\n| /search | {n} | {k}% |",
        f"$ {pkg} status\nworkers: {k} running, {p} idle\nqueue depth: {n}",
    ]
    random.shuffle(prose)
    random.shuffle(code)
    units = [(random.choice(DOC_JUNK), None)]
    for i in range(r(2, 4)):
        text, note = prose[i]
        if note is None:
            note = text.replace("Requires", "requires").rstrip(".")
        units.append((text, note))
        if random.random() < 0.4:
            units.append(("Here is an example:", None))
        units.append((code[i], "="))
    return units


GENRES = {"chat": chat_chunk, "thread": thread_chunk, "story": story_chunk, "blog": blog_chunk, "docs": doc_chunk}


def chunk(genre=None):
    genre = genre or random.choice(list(GENRES))
    raw = GENRES[genre]()
    return {"genre": genre, "units": [{"id": i + 1, "text": t, "keep": n is not None, "note": n}
                                      for i, (t, n) in enumerate(raw)]}


def main(n_train=600, n_test_per_genre=6, seed=0):
    random.seed(seed)
    out = Path(__file__).resolve().parent / "toy"
    out.mkdir(exist_ok=True)
    with open(out / "train.jsonl", "w") as f:
        for _ in range(n_train):
            f.write(json.dumps(chunk(), ensure_ascii=False) + "\n")
    with open(out / "test.jsonl", "w") as f:
        for g in GENRES:
            for _ in range(n_test_per_genre):
                f.write(json.dumps(chunk(g), ensure_ascii=False) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
