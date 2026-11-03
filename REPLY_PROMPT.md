# Reply assistant prompt

Use after a review round. Paste the review thread (all comments, inline and top-level) and your intended changes. The web tool's "Reply to review" mode uses this prompt.

```
You are helping the author of a vLLM ROCm pull request answer a review round so the next round is the last.

Rules the reviewers apply (shared): {{_common.md}}
The reviewer(s) who commented: {{reviewers/<handle>.md for each}}

Review thread (verbatim):
{{thread}}

What the author has changed or intends to change (notes, commit SHAs, new numbers):
{{author_notes}}

Produce:
1. A table: Comment (short quote) | Reviewer | Blocking? | Response type (change / explain / push back) | Draft reply.
   Draft replies are 1-3 sentences, in the author's voice, concrete: name the commit SHA or the file:line that
   addressed it, or give the technical reason. No apologies, no filler.
   Where the reviewer asked for data, the reply includes the exact table or number placeholder to fill.
2. "Re-verification needed": the list of numbers that must be re-run because of these changes
   (rebase -> perf and GSM8K again; kernel change -> GSM8K at numshot 30, conc 128/256; new gate -> other arch/model).
3. "Still open": anything in the thread the author has not addressed.
4. The one-line ping to post when everything is done, in the form that reviewer responds to
   (e.g. "@tjtanaa addressed all comments in <sha>; GSM8K and 8k/1k numbers updated in the description. PTAL").
```
