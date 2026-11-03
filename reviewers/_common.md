# Common vLLM ROCm review rules (inherited by every reviewer persona)

Source: 1,156 `rocm`-labelled vLLM PRs (Aug 2025 - Oct 2026), 9,276 human review comments, plus the repo's contributing guide, PR template and `pr-checklist` skill. Every persona file assumes these rules; the persona adds its own emphasis and voice on top.

## How to use this file in a review

1. Read the PR title, description and diff.
2. Run the "Hard blockers" list first. Any hit is a blocking comment.
3. Run the "Evidence" list. Missing evidence is a blocking comment phrased as a request for the specific artifact (command, table, number).
4. Run the "Code rules". Each hit is an inline comment on the exact line.
5. Then apply the persona's own checklist and voice.
6. End with a verdict: `Approve`, `Approve pending <one item>`, `Request changes`, or `Needs a core reviewer` (when shared CUDA/core code is touched).

## Hard blockers (any one of these blocks)

- A commit without `Signed-off-by` (DCO).
- Pre-commit / ruff / mypy failure.
- `from vllm.platforms.rocm import ...` at module top level, or any ROCm-only import not guarded by `current_platform.is_rocm()`.
- `envs.VLLM_*` read inside `forward()` or any per-step path (issue #17067). Must be cached in `__init__` or in `rocm_aiter_ops` as a classmethod.
- `try/except` used to select a kernel or detect an AITER feature at runtime.
- Version checks or attribute probes against AITER internals. vLLM pins one AITER version and promises no forward/backward compatibility.
- A kernel that is not in the AITER version pinned in `docker/Dockerfile.rocm_base`.
- A test file with no `@pytest.mark.skipif(not current_platform.is_rocm(), ...)` (and an AITER-availability skip when it needs AITER).
- A new test that no `.buildkite/test-amd.yaml` test group runs.
- A change to a shared CUDA/ROCm path (model file, `fused_moe/`, `quantization/`, `compilation/`, `v1/worker`, `v1/core`) that alters CUDA behaviour without gating and without a core reviewer tagged.
- JIT-building C++/HIP kernels at runtime. Everything compiles at build time through CMake into `_rocm_C`.
- Benchmark scripts, READMEs, or debug prints left in the diff.
- A PR body that is the empty template.

## Evidence every ROCm perf PR must carry

- The exact `vllm serve` command with every env var, GPU (gfx942 / gfx950), TP/DP/EP sizes, and AITER version.
- End-to-end serving numbers (`vllm bench serve`), baseline vs PR, at 2-3 ISL/OSL/concurrency points, with TTFT and TPOT. A kernel microbenchmark alone is not enough.
- GSM8K via lm_eval, `--num_fewshot 30`, concurrency 128 (256 for DeepSeek-V4-class models), baseline vs PR, pasted in lm_eval's own table layout (Tasks / Version / Filter / n-shot / Metric / Value / Stderr).
- Whether `VLLM_ROCM_USE_AITER=1` (and other recipe flags) were on during the runs.
- The second architecture (MI300X vs MI355X) tested, or an explicit code gate that excludes it.
- TP, EP, DCP, DP-attention, MTP: each tested or rejected at init with a clear error.

## Evidence every ROCm bugfix PR must carry

- Root cause in the description (what was wrong, why, how found).
- A reproducer command and the after state (GSM8K before/after when accuracy was involved).
- A test that fails without the fix, in an existing AMD test group.
- A perf check that the fix did not regress (especially if it adds `.contiguous()`, a copy, a sync, or a wider dtype).
- Which other models and platforms share the fixed path, and which were tested.

## Code rules (inline comments)

- Decide once in `__init__`: arch, AITER availability, dtype, shapes, kernel pointer. Forward only branches on attributes.
- AITER ops: `_impl`, `_fake`, `direct_register_custom_op` and flags live in `vllm/_aiter_ops.py`; expose `is_<feature>_enabled()` classmethods refreshed by `refresh_env_variables()`.
- No new `VLLM_ROCM_*` env flags when `--moe-backend`, `--attention-backend`, config inference, or an existing flag can express it. If a kernel always wins, make it the default with no knob.
- MoE backend selection goes through `fused_moe/oracle/*.py`, not ad-hoc code in a quant scheme.
- Model-specific ROCm code lives in `vllm/models/<model>/amd/` (RFC #42770). Copy the shared file there and edit, rather than adding `if is_rocm()` branches to the shared file.
- Arch checks use CDNA/RDNA helpers (`on_gfx950()`, CDNA family helpers), not hard-coded strings in shared code. In HIP use `#if defined(__gfx942__)` for fnuz, not `!defined(__gfx950__)`.
- Reuse existing helpers: `get_fp8_min_max()`, `WorkspaceManager`, `set_manual_seed` (vLLM's), `_can_use_aiter_fused_ar_rms`, oracle conversion functions. No sibling copies; add a parameter to the existing one.
- Keep the unfused and torch fallback paths when adding a fused path.
- Only instantiate arch-specific objects on that arch; then remove the `is None` checks.
- Delete comments that explain history, the old behaviour, or reference the PR. That text belongs in the PR description.
- No wider `except` to make a crash go away.
- Vectorise reads/writes in HIP kernels.
- Lazy-import heavy optional modules (triton kernels, tilelang, flydsl) where used.

## Test rules

- Parametrise with the real model's shapes (block size, num_heads, head_dim, topk) and several token counts.
- Tolerances derived from dtype/arithmetic, and the test run 10+ times for stability.
- The test must exercise the new path, not fall through to a fallback. A mocked dispatch test must assert something that would fail.
- Prefer extending an existing test file / test group over a new one. Merge sibling test files.
- Use vLLM's `set_manual_seed`, `torch.testing.assert_close`, `current_platform.is_cuda_alike()` helpers.
- Keep TP sizes in tests at 1 or 2.

## Process rules

- Title: `[ROCm][Perf]`, `[Bugfix][ROCm]`, `[ROCm][<Model>]`, etc. Template sections Overview / Claims / Validation / Details filled.
- Branch from a feature branch; rebase, never merge main after CI is green; `/ci run` requires zero commits behind main.
- One CI run at a time; `/amd-ci run nightly` for AITER or library bumps.
- Reply to every comment with a change or a reason; name the commit.
- Re-run perf and GSM8K after any rebase or tuning change.
- PRs over ~250 changed lines for a bugfix, or ~500 for a feature without an RFC, get asked to split.
- Search open `label:rocm` PRs first; 36% of closed ROCm PRs were superseded or duplicates.
- AI-assisted PRs: attribute (`Co-authored-by: Claude`) and keep a human in the loop; PRs that read as unreviewed agent output are closed.

## Output format for a generated review

```
## Summary verdict
<Approve | Approve pending ... | Request changes | Needs core reviewer> - one sentence why.

## Blocking
- [file:line or section] <comment in the reviewer's voice>

## Should fix before merge
- ...

## Nits
- ...

## Questions for the author
- ...

## What is missing from the PR description
- ...
```
