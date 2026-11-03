# vLLM ROCm reviewer personas

Review a vLLM PR the way a specific AMD/ROCm reviewer would, before you open it.

Built from 1,156 `rocm`-labelled vLLM PRs (Aug 2025 - Oct 2026): 9,276 human review comments, every approval, changes-requested and merge event. Each persona file contains what that reviewer asks for before reading the diff, what they check in code and tests, how they write, their verdict rules, a checklist in their voice, and 10-35 verbatim comments with PR numbers.

## Quick start

```
python server.py
```

Open http://127.0.0.1:8765/ and paste a PR link (or just the number, e.g. `59280`). The page pulls the title, description, diff and review comments from GitHub and runs the pre-flight. Then:

- **Review as** - pick a reviewer (or Full panel) and press Review. Runs through your local Claude Code CLI (`claude -p`), no API key needed.
- **Evidence commands** - serve / lm_eval / bench commands, with model and TP guessed from the PR body.
- **Reply to review** - the thread is already filled; add what you changed and press Draft replies.

Shortcut: `http://127.0.0.1:8765/?pr=59280` loads a PR directly. GitHub allows about 12 PR loads an hour without sign-in. Opening `index.html` without the server still works: pre-flight and evidence run in the page, and the LLM modes give you a prompt to copy.

## Files

| File | What it is |
| --- | --- |
| `index.html` | The review tool: paste a PR link, everything else is fetched from GitHub; pre-flight, review as any reviewer, evidence commands, reply drafts |
| `server.py` | Serves the page and runs the LLM modes through your Claude Code CLI |
| `tools/build_html.py` | Re-embeds the persona files and outcomes into `index.html` (run automatically by `refresh.py`) |
| `reviewers/_common.md` | Rules every ROCm reviewer applies: hard blockers, evidence for perf and bugfix PRs, code and test rules, output format |
| `reviewers/tjtanaa.md` | TJ - architecture gatekeeper, CODEOWNER on all ROCm paths |
| `reviewers/AndreasKaratzas.md` | Andreas - AMD CI owner, tests and tolerances, most merges |
| `reviewers/dllehr-amd.md` | Doug - attention/MLA/MoE, blast radius, sweeps, both GPUs |
| `reviewers/shen-shanshan.md` | Shanshan - DeepSeek-V4, real deployment shapes, plausibility |
| `reviewers/simondanielsson.md` | Simon - thorough inline perf review, warmup/capture, reuse |
| `reviewers/Rohan138.md` | Rohan - duplicates, AITER bumps, fold helpers into main path |
| `reviewers/hongxiayang.md` | Hongxia - MiniMax/Kimi recipes, common-path gating, both archs |
| `reviewers/fxmarty-amd.md` | Felix - quantization oracles and checkpoint compliance |
| `reviewers/JartX.md` | JartX - RDNA3, KV-cache quant accuracy |
| `reviewers/Fangzhou-Ai.md` | Fangzhou - DSv4/MiniMax kernels, InferenceX 8k/1k evidence |
| `reviewers/gshtras.md` | Gregory - cross-model fusion impact, pattern passes (rarely active) |
| `reviewers/mawong-amd.md` | Matthew - multi-stream, allocator, overlap handling |
| `reviewers/kliuae.md` | kliuae - Kimi-K3 KDA, how a model-specific ROCm PR should look |
| `REVIEW_PROMPT.md` | The prompt template the tool uses, plus panel mode and `gh` commands |
| `REPLY_PROMPT.md` | Reply-assistant prompt: turns a review thread into per-comment replies, a re-verification list and the ping to post |
| `tools/static_checks.py` | Deterministic pre-flight on a diff + PR body: 40 rules mapped to real reviewer asks (BLOCK / FIX / NOTE), DCO and rebase check when run inside the repo, exit 1 on blockers |
| `tools/evidence.py` | Prints the exact `vllm serve`, lm_eval (numshot 30, conc 128/256) and `vllm bench serve` sweep commands plus the result tables to paste |
| `tools/outcome.py` + `data/outcomes.json` | For every file in your diff: how past PRs on it went (days to merge, rounds, who reviewed, top asks, example PRs) |
| `tools/refresh.py`, `scrape_*.py`, `derive_personas.py` | Re-scrape GitHub (no token), rebuild quotes and the data-derived profile in every persona, write `data/drift.md` with reviewer shifts |

## Each persona file now has

1. Hand-written: who they are, what they expect before reading the diff, what they check (code, tests, flags, platform), how they write, verdict rules, a checklist in their voice.
2. Auto-derived from the data: PRs commented on, median time to first comment / to approval / to merge, files and paths they comment on, models they name, who they escalate to, their own PRs' style, a 60-day drift of their themes, **approval triggers** (the last ask before each LGTM), **changes-requested triggers** (what they said when they formally blocked), **things they let slide** (their own nits).
3. Verbatim: 10-35 real review comments with PR numbers.

## Workflow that catches things before reviewers do

```
gh pr diff <n> --repo vllm-project/vllm > pr.diff
gh pr view <n> --repo vllm-project/vllm --json body -q .body > pr.md
python tools/static_checks.py --diff pr.diff --body pr.md --title "<title>" --repo /path/to/vllm   # fix every BLOCK and FIX
python tools/outcome.py --diff pr.diff                                                              # see who will show up and what they asked last time
python tools/evidence.py --model <hf id> --tp 8 --gpu gfx950 --kind perf                           # run these, paste the tables
# then the LLM review: index.html "Review as" (or REVIEW_PROMPT.md in Claude Code), start with the predicted first commenter
# after the review round: index.html "Reply to review" (or REPLY_PROMPT.md)
```

## Use in Claude Code (recommended for real PRs)

1. Copy `reviewers/` into your vLLM checkout, e.g. `.claude/skills/rocm-review/reviewers/`.
2. Add a `SKILL.md` next to it:

```
---
name: rocm-review
description: Review the current branch or a PR number as a named vLLM ROCm reviewer (tjtanaa, AndreasKaratzas, dllehr-amd, simondanielsson, ...). Use before opening or updating a ROCm PR.
---
1. Run `python tools/static_checks.py --diff <diff> --body <body> --title <title> --repo .` and fix or list every BLOCK/FIX.
2. Run `python tools/outcome.py --diff <diff>` and name the likely first reviewer.
3. Read reviewers/_common.md, then reviewers/<handle>.md for the requested reviewer (default: the likely first reviewer).
4. Follow REVIEW_PROMPT.md. If the user says "panel", run every persona and merge.
5. If the user pastes a review thread, follow REPLY_PROMPT.md instead.
Get the PR with `gh pr view <n> --json title,body,url` and `gh pr diff <n>`, or `git diff main...HEAD`.
```

3. In Claude Code: `/rocm-review 59280 as simondanielsson` or `/rocm-review panel`.

## Keeping the personas current

Re-run the scrape every few weeks: list `is:pr label:rocm` PRs, pull the review threads, and re-extract each reviewer's comments. The "Real review comments" sections are the part that drifts; the checklists change slowly.

Who merges matters more than who comments: in this window AndreasKaratzas and tjtanaa merged about 80% of human-merged ROCm PRs; dllehr-amd and shen-shanshan most of the rest. simondanielsson and Rohan138 cannot merge but their approval is what those committers wait for on perf PRs.
