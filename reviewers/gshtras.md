---
handle: gshtras
display_name: Gregory Shtrasberg (gshtras)
role: Committer, AMD area owner (ROCm/AMD GPU integration); not in CODEOWNERS
activity: 20 substantive comments, 8 approvals in this window (largely inactive on reviews)
focus: cross-model impact of fusions, pattern-matching passes over hand fusions, RDNA/APU arch coverage, verifying on MI300/MI355/gfx1100 himself
verdict_style: brief; verifies on hardware and reports ("Verified the functionality on MI300, MI355 and gfx1100")
---

# Reviewer persona: gshtras (Gregory)

## Who this is

Long-time ROCm maintainer, now rarely active on reviews. When he reviews, he asks about the other models that share a fusion path, prefers `torch.compile` pattern matching over hand-written fusion branches, and checks arch lists for APUs (gfx115x). Do not wait on him; request tjtanaa or dllehr-amd.

## What Gregory asks

- "Do you happen to have some test results (lm_eval) with other models that can use this fusion path? Would DSR1 and Kimi-K2 be affected?"
- "Can this be done through a pattern matching mechanism?"
- "Should the clone_elimination pass be disabled unconditionally for any model when aiter is on? Or does it only affect specific models or backends?"
- "My only concern here is gfx115x which is not a Radeon. Should it also be included?"
- Whether a shell flag (`-e`, `-o pipefail`) is actually needed given how commands are composed.
- Pre-commit before a cherry-pick to a release branch.

## Review checklist in Gregory's voice

- [ ] lm_eval on the other models that share this path?
- [ ] Could a pattern-matching pass replace the hand fusion?
- [ ] Is the pass disable scoped to the models/backends that need it?
- [ ] Arch list covers APUs and RDNA variants?

## Real review comments from this reviewer (verbatim, with PR numbers)

- **#41394** ([Kernel][ROCm] Native W4A16 kernel for AMD RDNA3 (gfx1100) — fp16 + bf16) — `rocm.py`: "My only concern here is gfx115x which is not a Radeon. Should it also be included?"
- **#38272** ([ROCm][CI] Unsetting arch completely) — `run-amd-test.sh`: "Do you have an example of a $commands value that needs either -e of -o pipefail? From what I can see, every instance passed here is using &&"
- **#41816** ([ROCm] Disable AITER allreduce fusion for HIP graph replay) — PR thread: "What impact would removing these other passes (it's only the reduce+rms fusion that breaks accuracy almost across the board) have on the performance of LLama/Qwen models?"
- **#41825** ([ROCm][Perf] Fix RMSNorm+Quant fusion for gfx950 (non-fnuz)) — PR thread: "Do you happen to have some test results (lm_eval) with other models that can use this fusion path? Would DSR1 and Kimi-K2 be affected?"
- **#36893** ([Feature] Kvcache Int8 per-token  scale on TRITON_ATTN continue of #34327 thanks) — PR thread: "Verified fp8_per_token and fp8_per_group from this commit on MI300 with gsm8k. The scores are nearly the same as w/o quantizing the cache"
- **#40035** ([ROCm] torch 2.11 + rocclr profiler hotfix for ROCm 7.2 + aiter v0.1.12.post2) — PR thread: "CI run with the new base image: https://buildkite.com/vllm/amd-ci/builds/7898/summary"
- **#41816** ([ROCm] Disable AITER allreduce fusion for HIP graph replay) — PR thread: "Should the clone_elimination pass be disabled unconditionally for any model when aiter is on? Or does it only affect specific models or backends?"

## Data-derived profile (computed from the PR set, 6 Oct 2026)

- **PRs commented on (not own):** 10
- **Median time to first comment after PR open:** 1.7 days
- **Median time from first comment to approval:** 0 days
- **Median time from approval to merge:** 0.6 days
- **Where they comment (paths):** `vllm/platforms/rocm.py` (1), `vllm/model_executor/layers` (1), `.buildkite/scripts/hardware_ci` (1)
- **Files they comment on most:** `rocm.py` (1), `layernorm.py` (1), `run-amd-test.sh` (1)
- **Models named in their comments:** Kimi-K2 (1), gpt-oss (1), DS3.2 (1), Llama (1)
- **Pet models / configs to pre-test:** DSR1, Kimi-K2, gpt-oss
- **Escalates to (people they @mention):** @zyongye (1). Pre-tag the first one when your diff is outside this reviewer's area.
- **Their own PRs look like:** Short (1 own PRs in the set; 0 mention GSM8K, 0 include a serve/bench command)

### Approval triggers (what flipped them to LGTM)

- Verified on MI300, MI355 and gfx1100 himself
