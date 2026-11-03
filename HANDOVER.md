# Handover: vLLM ROCm reviewer simulator

Last worked on: 2026-11-03. This file is the starting point for whoever picks the project up next. `README.md` is the user-facing guide; this file covers how the project is built, how it got here, what's half-done and where the traps are.

## 1. What it is, in one paragraph

A local tool that predicts what AMD/ROCm reviewers will say about a vLLM pull request *before* they say it. You paste a PR link. The page pulls the PR from GitHub, runs about 40 deterministic checks built from comments ROCm reviewers made repeatedly (**Pre-flight**), and can then have Claude write the review a specific reviewer would likely leave (**Review as**). The reviewer knowledge lives in 13 persona files built from 1,156 `rocm`-labelled vLLM PRs (Aug 2025 to Oct 2026): 9,276 human review comments plus every approval, changes-requested and merge event.

## 2. Run it

```
python server.py                      # http://127.0.0.1:8765/
# or open http://127.0.0.1:8765/?pr=58769 to load a PR straight away
```

Requirements: Python 3.8+ (stdlib only, no pip installs). For Review as / Reply, the Claude Code CLI (`claude`) must be on PATH and logged in. Without it, those modes show a prompt to copy into Claude instead.

`.claude/launch.json` holds the same command, for previewing in Claude Code's browser pane.

## 3. Repository map

```
index.html            Single-file web app: HTML + CSS + JS + an embedded JSON data blob (~290 KB)
server.py             Stdlib HTTP server: serves the page, POST /api/claude streams `claude -p`, GET /api/health
reviewers/
  _common.md          Rules every ROCm reviewer applies (hard blockers, evidence, output format)
  <handle>.md x13     One persona per reviewer (see section 5)
REVIEW_PROMPT.md      Prompt template for "Review as" (also usable by hand in Claude)
REPLY_PROMPT.md       Prompt template for "Reply to review"
data/
  lists/*.json        Raw: PR lists per monthly search window (scraped)
  prs/<n>.json        Raw: every scraped PR's comments, inline review comments and review states
  nums.txt            PRs selected for fetching in the last refresh
  quotes.json         Derived: every substantive comment per reviewer
  derived.json        Derived: the numbers behind each persona's data-derived profile
  outcomes.json       Derived: per-file history (past PR count, days to merge, rounds, reviewers, top asks)
  drift.md            Derived: reviewer theme shifts, last 60 days vs earlier
tools/
  static_checks.py    CLI pre-flight. The reference implementation of the rules (more complete than the JS)
  outcome.py          CLI: per-file history for a diff, from data/outcomes.json
  evidence.py         CLI: serve / lm_eval / bench commands + result tables to paste
  scrape_list_prs.py  Lists rocm PRs via the GitHub search API (HTML fallback); no token needed
  scrape_fetch_prs.py Fetches each PR's comments/reviews -> data/prs/<n>.json (HTML pages, or the API if GITHUB_TOKEN is set)
  derive_personas.py  Rebuilds the auto sections of reviewers/*.md + data/{quotes,derived,outcomes}.json + data/drift.md
  refresh.py          Runs list (monthly windows) -> stale check -> fetch -> derive -> build_html
requirements.txt      beautifulsoup4 + lxml (scraper HTML mode only)
  build_html.py       Re-embeds reviewers/*.md, _common.md, REPLY_PROMPT.md, outcomes.json into index.html
```

## 4. How the pieces fit

```
GitHub --scrape_*.py--> data/lists, data/prs  (raw data, committed; see section 6)
                                        |
                                 derive_personas.py
                                        v
            reviewers/*.md (auto sections) + data/outcomes.json + data/drift.md
                                        |
                                  build_html.py
                                        v
                    index.html  <script id="data" type="application/json">
                                        |
 browser: Load PR --> api.github.com (CORS, unauthenticated) --> title/body/diff/comments
          Pre-flight --> JS rules, in the page
          Review as  --> buildPrompt(persona + _common + PR) --> POST /api/claude --> claude -p (streamed)
```

### index.html internals (all in one inline `<script>`)
- `D` = parsed data blob: `{reviewers, personas, common, reply, outcomes}`.
- **Load PR** (`loadPR`, `parseRef`, `ghGet`, `ghAll`): accepts a full URL, `59280`, `#59280` or `owner/repo#n`. It makes about 5 GitHub API calls per PR (PR JSON, `.diff`, review comments, reviews, issue comments). If the diff is too large it falls back to `/files`. It drops bots and `/ci run`-style command comments, tags the author's comments, guesses model/TP/GPU/kind from the PR body, then runs the pre-flight.
- **Pre-flight** (`preflight()`): a JS port of `tools/static_checks.py`. Each finding is `{sev: BLOCK|FIX|NOTE, where, msg, who}`. It also shows "most likely to comment first" and per-file history from `D.outcomes`.
- **LLM modes** (`one()`, `buildPrompt()`, `replyPrompt()`): the model call is abstracted as `sample(prompt, {signal, onText})`. Three backends, tried in order: claude.ai's `window.claude.use('sample')` when hosted there, `localSample` (POST `/api/claude`) when `server.py` reports `claude: true`, otherwise the copy-prompt fallback.
- Full panel = one call per reviewer (13), then a merge call that produces a single table.
- Form state persists in `localStorage` key `rocm-review-form`.

### server.py
- `claude -p --output-format stream-json --verbose --include-partial-messages --tools "" --no-session-persistence --strict-mcp-config`. The prompt goes in on stdin and only `text_delta` text is streamed to the browser.
- Runs in an empty temp dir so no CLAUDE.md or project settings leak into reviews. No tools, no MCP.
- When the browser presses Stop, the dropped connection kills the subprocess.
- Binds to 127.0.0.1 only. There's no auth, so don't expose it.

## 5. Persona file anatomy (`reviewers/<handle>.md`)

1. YAML front matter: handle, role, activity counts, focus, verdict style.
2. **Hand-written** sections: who they are, what they expect before reading the diff, what they check (code / tests / flags / platform), how they write, verdict rules, a checklist in their voice.
3. `## Real review comments from this reviewer (verbatim, with PR numbers)`: **auto-generated**.
4. `## Data-derived profile (computed <date>)`: **auto-generated**. Covers time to first comment / approval / merge, the paths they comment on, the models they name, who they escalate to, their own PRs' style, 60-day drift, approval triggers, changes-requested triggers and things they let slide.

`derive_personas.py` keeps everything above section 3 and regenerates 3 and 4. **Edit only the hand-written part by hand.**

Reviewers: tjtanaa, AndreasKaratzas, dllehr-amd, shen-shanshan, simondanielsson, Rohan138, hongxiayang, fxmarty-amd, JartX, Fangzhou-Ai, gshtras, mawong-amd, kliuae.

## 6. Data collection: how the reviewer data is scraped (method and rules)

Everything in `reviewers/*.md` (auto sections), `data/outcomes.json` and `data/drift.md` comes from this pipeline. The raw data it collects is committed under `data/lists/` and `data/prs/`, so you can re-derive without scraping, or scrape only what changed.

### Run it

```
python -m pip install -r requirements.txt           # beautifulsoup4 + lxml, only needed for the HTML fallback
python tools/refresh.py --since 2025-08-01          # list -> fetch -> derive -> build_html; resumable, safe to re-run
```

Optional but recommended: a GitHub token makes fetching exact and about 5x faster. Create a fine-grained personal access token with **public repositories, read-only** access and no other permissions:

```
$env:GITHUB_TOKEN = "<token>"        # PowerShell
export GITHUB_TOKEN=<token>          # bash
```

Re-running for the latest data is the same command. Finished months are reused from `data/lists/`, the current month and open PRs are re-listed, and any PR updated since its cached copy is re-fetched. A first full run without a token takes about 1–2 hours; later runs take minutes.

### Step 1: list PRs (`tools/scrape_list_prs.py`)

- **Source:** the GitHub search API, `GET /search/issues`. It needs no token (10 requests/minute; 30 with a token). If the API fails it falls back to scraping `github.com/vllm-project/vllm/pulls?q=...` HTML, with retries, because GitHub sometimes serves throttled pages with no data.
- **Queries,** built in `refresh.py` for every calendar month from `--since` to today:
  - `is:pr is:merged label:rocm merged:<month>` -> `data/lists/merged-YYYY-MM.json`
  - `is:pr is:closed is:unmerged label:rocm closed:<month>` -> `closed-YYYY-MM.json`
  - `is:pr is:open label:rocm created:<month>` -> `open-YYYY-MM.json`
  - `is:pr is:open label:rocm created:<since` -> `open-older.json`
- **Why monthly:** GitHub search returns at most 1,000 results per query, and `label:rocm` has several thousand PRs (on 2026-11-03: 2,294 merged and 1,317 closed-unmerged since Aug 2025, plus 840 open). The original dataset used single queries and was silently capped. If a window ever hits 1,000, `refresh.py` prints a warning; split that window further.
- **Record per PR:** `num, title, author, assoc, created, updated, merged, closed, state, labels, comments, query`.

### Step 2: choose PRs to fetch (`refresh.py`)

- Drop PRs whose author ends in `[bot]`.
- Keep PRs with 1 or more conversation comments. The search count excludes inline review comments, so a stricter filter loses reviewed PRs.
- **Stale check:** if a PR's `updated` date is newer than its cached `data/prs/<n>.json`, the cached file is deleted and re-fetched.
- The final list is written to `data/nums.txt`.

### Step 3: fetch each PR's review history (`tools/scrape_fetch_prs.py`)

Two modes, chosen automatically:

| | HTML mode (no token) | API mode (`GITHUB_TOKEN` set) |
|---|---|---|
| Source | `github.com/vllm-project/vllm/pull/<n>`, plus up to 12 "load more" timeline fragments | `pulls/<n>`, `pulls/<n>/comments`, `issues/<n>/comments`, `pulls/<n>/reviews` |
| Inline comments | Most of them; file paths can be shortened to `...` | All, with full paths |
| Review states | Parsed from timeline text ("approved these changes", "requested changes", "dismissed") | Exact |
| Extra | `resolved`, `outdated`, `minimized` flags (heuristic) | `outdated` (`position is null`), diffstat, files changed |
| Fragility | Breaks if GitHub changes its HTML | Stable |

Both modes run 3 workers, retry with backoff, skip PRs already cached, and write the same schema:

```
data/prs/<n>.json = {
  "meta":     {"num", "title", "files_changed", "diffstat", "merged_by"?},
  "comments": [{"author", "time", "body", "type": "comment"|"review_comment", "minimized", "file"?, "outdated"?, "resolved"?, "suggestion"?}],
  "reviews":  [{"author", "state": "approved"|"changes_requested"|"dismissed", "time"}]
}
```

### Step 4: derive the personas (`tools/derive_personas.py`)

Rules applied to the raw data, for each reviewer in `R`:

- **Excluded:** bot accounts (`BOTS` set, plus `*[bot]`), `/ci` and `/amd-ci` commands, and the reviewer's own PRs.
- **Verbatim quotes:** comments of at least 70 characters, with quoted text removed, not starting with lgtm / thanks / done / ok / etc. Inline review comments rank first, then longer ones; at most 3 per PR and 30 per reviewer.
- **Time to first comment:** the reviewer's first comment minus the PR's created date (day precision). **First comment to approval** and **approval to merge** are medians across PRs.
- **Approval triggers (data part):** the reviewer's last comment before their first approval (up to 2 minutes after it), longer than 25 characters and not just "LGTM".
- **Changes-requested triggers:** their comments within 3 hours of their first changes-requested review.
- **Nits:** comments that start with "nit".
- **Themes and drift:** each comment is tagged by the regex table `TH` (accuracy, e2e perf, blast radius, init not forward, platform guard, tests, reuse, regression, flags, aiter pin, comments cleanup, graph/warmup/invariance, rebase/precommit). Drift compares the share of comments per theme in the last 60 days against earlier ones. It's reported when there are at least 15 recent comments and the change is 5 points or more.
- **Models named:** the `MODELS` regex. Extend it when new model families appear.
- **Escalates to:** `@mentions`, excluding the reviewer, the PR author and bots.
- **Outcomes per file** (`data/outcomes.json`): keyed by file basename, using files that received inline comments. Records PR count, merged/closed, median days to merge, median human comments, median **rounds** (distinct days on which non-authors commented), top reviewers, top themes, and 3 recent merged examples. Only files touched by 2 or more PRs are included.
- **Hand-curated lines are preserved:** `Pet models / configs to pre-test`, `Their own PRs look like`, the bullets under `### Approval triggers` and under `### Things they let slide`. The script carries them over from the existing file, so edit them by hand in the persona file.

### Step 5: rebuild the app (`tools/build_html.py`)

Re-embeds every persona file, `_common.md`, `REPLY_PROMPT.md` and `data/outcomes.json` into `index.html`.

### After a refresh

1. `git diff reviewers/ data/drift.md`: read what changed for each reviewer.
2. Update the **hand-written** sections of a persona when drift shows a real change in what that reviewer asks for (e.g. a new recurring demand).
3. If a new person starts reviewing many ROCm PRs, add them (see "How to extend").
4. Commit `data/` along with the persona changes, so the next person can re-derive without scraping.

### Adjusting the scrape

- **A different time range:** `--since YYYY-MM-DD`.
- **A different label or repo:** the queries live in `refresh.py`, and `REPO` is set in `scrape_list_prs.py` and the URLs in `scrape_fetch_prs.py`.
- **Re-derive only, no network:** `python tools/derive_personas.py && python tools/build_html.py`.
- **Force a re-fetch of one PR:** delete `data/prs/<n>.json` and re-run.

## 7. History: what was done when

1. *(Earlier chat on claude.ai)* Built the persona files, prompts, CLI tools and a claude.ai-hosted version of `index.html`. Then added the data-derived profile fields, static checks, evidence generator, outcome lookup, reply assistant and the refresh pipeline.
2. *(2026-10-06, Claude Code)* **Made the app work locally.** The shipped `index.html` was broken. Its `<style>` block was cut off mid-rule, so the whole body was parsed as CSS and the page was blank. An unguarded `claude.use()` call also threw before any button handlers were attached. I wrote the missing CSS, guarded the claude.ai runtime, added the copy-prompt fallback and the "Load example" button, and added `build_html.py`.
3. *(2026-10-06)* **"Just give the URL."** Added Load PR from GitHub, `?pr=` deep links, `server.py` with `claude -p` streaming, the elapsed-time indicator and a README quick start. Verified end to end on PR #58769: the tjtanaa review flagged the 5-shot / conc-64 GSM8K run, matching the real review.
4. *(2026-11-03)* **Handover prep and data refresh.** Added `encoding="utf-8"` to every `open()` call in the tools (non-ASCII diffs crashed on Windows) and fixed the scrapers' curl decoding on Windows. Found that the raw scrape cache from the original build never left the claude.ai sandbox, and that its single search queries had been capped at 1,000 results. Rebuilt the pipeline: the search API with monthly windows, an API fetch mode when a token is set, retries for throttled pages, and re-fetching of stale PRs. Upgraded `derive_personas.py` so it computes every field the persona files show (it previously dropped approval-to-merge, models named and nits), keeps the hand-curated lines, and regenerates `data/outcomes.json`, which no script produced before. Re-scraped all `label:rocm` PRs since 2025-08-01, committed the raw data, regenerated the personas, wrote this file and pushed to GitHub.

## 8. Open decision (agreed direction, not built yet)

The owner questioned whether the **Evidence commands** and **Reply to review** tabs are needed. The proposal on the table:

- Keep **Pre-flight** and **Review as**, as one flow: load the PR, see the pre-flight, press Review as.
- Fold the useful part of Evidence into the pre-flight. When it flags "no GSM8K" or "no launch command", show the exact `lm_eval` / `vllm serve` command under that finding (the logic is already in `evidence()` in `index.html` and `tools/evidence.py`).
- Remove Reply to review, or demote it to a small link for after a review round.

Confirm with the owner before deleting anything.

## 9. Known issues and traps

- **Two implementations of the rules.** `tools/static_checks.py` (47 `add()` calls) and the JS `preflight()` in `index.html` are maintained separately and have already drifted. Python-only checks: the JIT/hiprtc check in `.cu`/`.hip` files, plus the repo-aware checks (DCO sign-off, branch behind main, new test missing from `.buildkite/test-amd.yaml`, pinned AITER tag from `docker/Dockerfile.rocm_base`). Options: generate the JS rules from a shared JSON rule table, or have `server.py` expose `/api/preflight` that runs the Python version.
- **Scraping is fragile in HTML mode.** Without a token, PR pages are parsed from GitHub's HTML, which changes without notice; GitHub also throttles anonymous page loads. If a refresh fetches PRs with 0 comments or the log shows `no data page`, set `GITHUB_TOKEN` to switch to API mode. Inline file paths in HTML-mode data are sometimes shortened to `...`. `derive_personas.py` refuses to run on an empty cache, so a failed scrape can't wipe the personas.
- **GitHub rate limit in the browser.** Unauthenticated requests are limited to 60 an hour, which is about 12 PR loads. Adding an optional token (or a server-side `/api/pr` that reads `GITHUB_TOKEN`) would fix it.
- **Review latency.** A large PR produces about 60k characters of prompt, and the first text takes 1–3 minutes. Full panel makes 14 sequential calls. Running them in parallel or using a faster model (`server.py` accepts `{"model": ...}`) would help.
- **The heuristics are regexes.** `inForward` / `in_forward_context` guess hot paths from the enclosing `def` name. Expect false positives and negatives; the severity labels are judgement calls.
- **Personas go stale.** Reviewer habits shift (see the drift notes inside each persona). Re-run the refresh every few weeks and check `data/drift.md`.
- **The claude.ai-hosted copy** published from the earlier chat was not updated and may carry the same cut-off `<style>` bug.
- **`index.html` is large and hand-patched.** Edit the HTML/CSS/JS freely, but leave the `<script id="data">` blob alone; regenerate it with `python tools/build_html.py`.

## 10. How to extend

- **Add a pre-flight rule:** add it to `tools/static_checks.py` `run()` *and* to `preflight()` in `index.html` (search `add('FIX'`). Give it a severity, a location, a message quoting the reviewer's real wording, and who asks for it. Test with the CLI and with "Load example".
- **Add a reviewer:** create `reviewers/<handle>.md` with front matter (`handle`, `display_name`, `focus`, ...) and the hand-written sections, add the handle to `R` at the top of `tools/derive_personas.py`, then run `python tools/build_html.py`. New persona files are picked up automatically and labelled from `display_name` and `focus`.
- **Refresh data:** `python tools/refresh.py --since <date>`, which ends by running `build_html.py`. See section 6 for the full method.

## 11. Test checklist (there is no automated test suite)

```
for f in tools/*.py server.py; do python -m py_compile "$f"; done
python tools/static_checks.py --diff <some.diff> --body "x" --title "[ROCm][Perf] t"   # exit 1 when there are BLOCK findings
python tools/outcome.py --diff <some.diff>
python tools/evidence.py --model moonshotai/Kimi-K3 --tp 8 --gpu gfx950 --kind perf
python server.py   then in the browser:
  - "Load example" -> 4 BLOCK / 10 FIX / 3 NOTE
  - ?pr=58769       -> loads, pre-flight runs, no console errors
  - Review as tjtanaa -> streams a review (1-3 min)
  - Stop             -> cancels cleanly
```

## 12. Backlog ideas (from the original design notes, not built)

- Duplicate detection: search open `label:rocm` PRs for kernel/function names in the diff.
- Resolve every `aiter.*` symbol in the diff against the pinned AITER tag (clone at the tag and grep).
- A feedback loop on your own PRs: record which predictions came true and which comments you still got.
- A nightly refresh job that flags new recurring phrases per reviewer.
- A Claude Code skill (`.claude/skills/rocm-review/SKILL.md`) so `/rocm-review <pr>` works inside a vLLM checkout. The README has a draft.

## 13. Sensitivity note

The persona files name real reviewers and quote their public GitHub comments verbatim. They are useful for preparing your own PRs, but keep the repo private, and don't present simulated reviews as something the named person said.
