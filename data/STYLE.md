# yapzip labeling style: zip it

yapzip compresses a text stream for another AI that reads ONLY the output. Every output token must carry signal. Each unit (one line = one message, comment, paragraph or code block) is either dropped, or kept with one terse telegraph note. A note of exactly "=" means "copy the unit verbatim".

Be stingy. When in doubt, drop. When writing, cut.

## Drop (keep=false)

- greetings, thanks, sign-offs, apologies, "brb", emoji-only
- jokes, puns, memes, sarcasm with no fact inside, roleplay, banter
- reactions and agreement: "lol", "this", "same", "+1", "congrats", "wow", "so true", "came here to say this"
- opinions, vibes and venting with no specific reason or fact ("this company sucks", "love it")
- rhetorical questions; questions nobody answers that aren't the topic of the thread
- repeats: a point already kept from someone else, or the same person saying it again (keep the first or fullest)
- meta: "edit: typo", "thanks for the gold", "following", "RemindMe", lead-ins like "Here's the thing:", "Let's dive in", headings with no content

## Keep

Facts, numbers, prices, dates, times, names, places, versions, specs; decisions; instructions and steps; conditions; warnings; corrections; outcomes; firsthand specifics ("mine failed after 2y"); claims and advice with a specific point; the question that defines a thread; questions answered later; code, commands, config, data, log lines.

## Note format

- **Tag first, shortened.** Chat: keep the timestamp and name: `[09:15] dana:`. Threads: handle only, drop the vote count, keep `↳` for replies: `↳ Witty-Walrus-753:`. Emails/letters: once per message, `From Priya, 10 Mar:` (never drop who sent what). Meetings: `HASSAN:`.
- **Telegraph.** No articles (a/an/the), no auxiliaries (is/are/was/will/has been), no subject pronouns unless needed to tell who, no filler (really, just, basically, honestly, actually, kind of). Verbs short and present: "books", "moved", "fails".
- **Symbols:** `->` change/result/then, `~` approx, `+` and, `/` or, `w/` `w/o`, `@` at, `#` number.
- **Numbers exactly as written.** Copy every number, amount, time, date, duration and unit in the unit's own form: "an hour and a half" stays "hour and a half", "2.9 billion" stays "2.9 billion", "$1,890" stays "$1,890". Never convert units, round, abbreviate a number, or compute anything (no totals, differences, per-month figures).
- **Markers in caps (never drop):** `NOT`, `NEVER`, `ONLY IF`, `IF`, `UNLESS`, `BECAUSE`, `VS`, `BUT`.
- **Modality, short but kept:** `maybe X`, `thinks X`, `claims X`, `rumor: X`, `hopes X`, `per <source>: X`, `asks: X?`. Never turn a wish, guess, joke or sarcasm into a fact.
- **Corrections:** `old -> new` (`Feb 7 -> Feb 9`).
- **Several facts:** separate with `; `.
- **Short form = cut words, not rewrite them.** Drop filler, articles and auxiliaries; keep the unit's own words for everything that carries meaning. Common short forms are fine (w/, w/o, ->, NOT, IF), but don't swap in synonyms or rephrase: the reader must see the speaker's words.
- **Exact:** names, numbers, IDs, versions, code, commands, error text. URLs: keep if the URL itself is the info, else domain only.
- **Length:** as short as it can be without losing a fact, condition, negation, hedge or speaker. Prose note ~1/3 of the unit or less. A long comment becomes a fact list; stay under ~25 words unless it really holds more distinct facts.
- **"=" only** for code/commands/config/JSON/table rows/log lines, or exact wording that matters (a quote, an error message, a riddle). Never for prose; prose always gets a note.
- **Never add** anything not in the unit. Don't resolve "he/it/they" by guessing. Don't merge units.

## Examples

`[09:15] dana: lol ok so. I'll book the lodge. deposit is 30% due by Feb 7, if it's never paid the lodge releases the rooms. ugh adulting`
→ `[09:15] dana: books lodge; deposit 30% due by Feb 7; IF never paid -> lodge releases rooms`

`blaah blaah blaaah ..meh mehhh eh yap yap yap. secret=123 . bleheheh meh meh`
→ `secret=123`

`↳ Witty-Walrus-753 (+3): My first MSP boss told me never to give a customer my personal number and honestly best advice ever. I keep the work VOIP app on my phone with office hours set so after hours I only see the voicemail transcript in email the next day. I ignore anyone without an active ticket, even during work hours, except a few VIPs. Company trains everyone to ignore out-of-band requests and leadership backs us.`
→ `↳ Witty-Walrus-753: per first MSP boss: NEVER give customers personal #; work VOIP app w/ office hours -> after-hours = voicemail transcript next day; ignores callers w/o ticket except few VIPs; company trains + backs ignoring out-of-band requests`

`kuh_riss (+34): Hopefully the person with the genius idea of daily counts got the axe too`
→ `kuh_riss: hopes person w/ daily counts idea got the axe`

`HASSAN: I'll approve both contractors, but only if we push the Zendesk upgrade to Q3.`
→ `HASSAN: approves both contractors ONLY IF Zendesk upgrade pushed to Q3`

`From: Priya Shah, Mon 10 Mar 2025 — Hi Dan, thanks for the quick turnaround! Could we ask for 15% instead of 12%? We've been with you seven years and never paid late. Best, Priya`
→ `From Priya Shah, Mon 10 Mar 2025: asks 15% instead of 12%; with you seven years, NEVER paid late`

`↳ lucky_lantern60 (+212): lmao that watermelon is smaller than my will to live`
→ drop

`$ kubectl rollout undo deploy/api -n prod`
→ `=`

## Quality bar (relabel v4)

Telegraph and short, but a reader who sees ONLY the note must get the same facts, with nothing false. Check every kept note against its unit:

- **Every fact-bearing number, ID, date, amount, version, name and place in the unit is in the note**, in the unit's own form. Ticket and order IDs, card endings, error codes and PR numbers all count. Exception: page or thread chrome, such as vote counts like `(+33)`.
- **Keep the role a number plays.** Write `card ends 2259`, not just `2259`. Write `logged 14 Feb, before renewal`, not `pre-renewal`.
- **Keep every qualifier that changes meaning:** negation (`NOT`, `NEVER`, `no`), condition (`IF`, `ONLY IF`, `UNLESS`), cause (`BECAUSE`), contrast (`BUT`), hedge (`maybe`, `thinks`, `per X`, `rumor`), and who said or did what. If the unit is a question, the note is a question (`asks: …?`).
- **Readable, not symbol soup.** Symbols go only where the list above allows, with spaces around them. Never glue tokens together: `v4.12.0 to prod at 09:02`, not `v4.12.0@09:02`. Don't reorder the facts into a puzzle.
- **No new words with new meaning.** Cutting is fine; substituting synonyms, interpreting, summarizing away a detail ("both explained their reasoning"), or resolving a pronoun by guessing is not. Standard abbreviations are fine (NDA, ToS, min, w/, approx).
- **Code, commands, config, JSON, YAML, tables, logs, error text and fenced blocks get `=`.** Never rewrite them. Inline code inside prose stays exact inside the note.
- **Keep or drop:** keep any unit holding a fact a reader would want. Drop filler: greetings, reactions, slop, hype, recaps that repeat an earlier kept fact, page chrome (bylines, dates of posts, read times, clap or like counts, "Follow", captions, share buttons, footers).

Bad → good:
- `Player: original email is wolfgang.reichert@mailbox.example, card ends 2259. i'm in Hamburg, never been to Bucharest lol`
  - bad: `Player: wolfgang.reichert@mailbox.example; 2259; Hamburg`
  - good: `Player: original email wolfgang.reichert@mailbox.example; card ends 2259; in Hamburg, NEVER been to Bucharest`
- `[Internal note — Rosa K.] Marcus, she's on Team Annual, not Monthly. Annual renewal date was 1 March and CXL-55031 was logged 14 Feb, before renewal. Per policy 4.2 this is a full refund. Please correct.`
  - bad: `Rosa K. (internal): Annual NOT Monthly; cancelled pre-renewal -> full refund per policy 4.2`
  - good: `Rosa K. (internal): Marcus, she's Team Annual NOT Monthly; renewal 1 March; CXL-55031 logged 14 Feb, before renewal -> full refund per policy 4.2; asks to correct`
- `[09:14] marcus: heads up, payments-api v4.12.0 went out to prod at 09:02 and we're seeing 502s on /v2/charge, like 3-4% of requests`
  - bad: `[09:14] marcus: payments-api v4.12.0@09:02; /v2/charge 3-4% 502s`
  - good: `[09:14] marcus: payments-api v4.12.0 to prod at 09:02; 502s on /v2/charge, ~3-4% of requests`

## Compression (relabel v4): every fact, no other words

Faithful does NOT mean wordy. After every fact, number, qualifier and speaker is in, cut everything else:
- discourse and filler: "was thinking that", "yes,", "sure", "so", "honestly", "just", "basically", "I think" (keep `thinks` only when the hedge matters)
- process verbs around the fact: "will put up change to set X to 30s" → `will set X 30s`; "we're seeing 502s" → `502s`
- restated context the tag already gives, and articles, auxiliaries, pronouns
- Targets, tag aside: chat line ≤ ~50% of the unit; prose paragraph ≤ ~35%; slop-heavy article paragraph ≤ ~25% (the fact survives, the fluff doesn't)

Over-long → right:
- `was thinking that, BUT most of ~2.9 billion objects are thumbnails under 128KB + intelligent tiering never moves those; lifecycle rule moving ONLY originals/ to STANDARD_IA after 60d gets ~same savings`
  → `most of ~2.9 billion objects thumbnails under 128KB, intelligent tiering never moves those; lifecycle ONLY originals/ to STANDARD_IA after 60d -> ~same savings`
- `lag back to 0.4s; will put up change to set max_standby_streaming_delay to 30s; NOT touching hot_standby_feedback BECAUSE it bloats primary`
  → `lag back to 0.4s; will set max_standby_streaming_delay 30s; NOT touching hot_standby_feedback BECAUSE bloats primary`

Tags: keep the unit's own tag order and timestamp as written (`rosa (11:30):`, `[2026-03-04 10:14] sam.k:`); only strip `<>` around IRC names (`<kfox>` → `kfox:`). Bot or alert messages get a note, unless they are pure log/code text (`=`). A question with no content whose answer stands alone is dropped; keep it when the answer needs it ("repo private?" → "yes private").

## Priority order (user's, binding; overrides "every fact" above when they conflict)

1. **Nothing false.** No flipped or dropped negation, condition, hedge or speaker; no invented or substituted words that change meaning; questions stay questions.
2. **Short.** Compression beats completeness: a secondary detail may go if it costs words.
3. **Code exact.** Code, commands, config and error text: `=` or exact inline.
4. **Facts.** Keep the facts a reader would act on: the decision, the number that matters, the ID that names the thing (PO 7781-B, PAY-3317), the date and time of the event.

What to cut first: context and process words ("once fixed", "as planned", "via console", "per audit log", "in progress", "repro'd", "release", "eng ticket"); the year when it's obvious; the second mention of anything; politeness.

Compact forms are fine and preferred: `7d`, `Feb`, `Mon`, `~`, `w/`, `->`, `+`, `auths`, `25µm`. Spell a number differently only if the meaning is identical.

- good: `Eilidh: retry failed captures w/ same Idempotency-Key; auths valid 7d -> none of 1,900 orders should be lost`
- good: `From Owen, 6 Feb: asks: PO 7781-B still ships 11th?; granola run 16th, can't push`
- good: `Internal: 2026.05.05-r3 treats JPY as 2-decimal; PAY-3317; rollback ETA 04:30 UTC`
