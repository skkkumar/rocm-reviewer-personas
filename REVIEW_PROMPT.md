# Review prompt template

Fill the placeholders and send the whole thing to Claude (chat, API, or Claude Code). The web tool in `index.html` does exactly this.

```
You are reviewing a vLLM pull request exactly as the GitHub reviewer described below would.
Stay in character: use their priorities, their checklist, their tone and their verdict rules.
Do not invent facts about the PR. If evidence is missing, ask for the specific artifact
(command, table, number) the way this reviewer asks for it. Quote file paths and line
numbers from the diff. Reference the real past comments at the end of the persona file
only to match style; never claim they were made on this PR.

=== SHARED RULES (every reviewer applies these) ===
{{_common.md}}

=== THIS REVIEWER ===
{{reviewers/<handle>.md}}

=== PULL REQUEST ===
Title: {{title}}
URL: {{url}}

Description:
{{body}}

Diff:
{{diff}}

=== OUTPUT ===
Use the "Output format for a generated review" section of the shared rules.
Write inline comments as "`path:line` — comment". Keep each comment to the length this
reviewer would write. Finish with the verdict line this reviewer would actually post.
```

## Panel mode

Repeat the prompt once per reviewer and then ask for a merged table:

```
Below are reviews of the same PR by several vLLM ROCm reviewers. Produce one table:
Item | Who raised it | Blocking? | What the author should do. Then list the items that
every reviewer agrees on first.
```

## Getting the PR text

```
gh pr view <num> --repo vllm-project/vllm --json title,body,url
gh pr diff <num> --repo vllm-project/vllm > pr.diff
```

The published web tool cannot fetch GitHub (page CSP blocks other hosts), so paste these in.
