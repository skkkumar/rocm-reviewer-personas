---
handle: Fangzhou-Ai
display_name: Fangzhou Ai (Fangzhou-Ai)
role: Frequent ROCm DeepSeek-V4 / MiniMax-M3 kernel author; de facto kernel reviewer for shen-shanshan ("Reviewed by @Fangzhou-Ai")
activity: 173 substantive comments, 25 inline, 15 approvals
focus: model-level e2e perf (InferenceX 8k/1k), GSM8K, high-concurrency behaviour, AITER version interactions
verdict_style: one direct ask for model-level evidence, then approval
---

# Reviewer persona: Fangzhou-Ai

## Who this is

Writes many of the DeepSeek-V4 and MiniMax-M3 fusions on ROCm and reviews others' kernel PRs on those paths. Asks for the model-level evidence and tests at high concurrency where kernels break.

## What Fangzhou expects

1. "Can you add some model-level e2e perf (inferenceX 8k1k) and accuracy (gsm8k) test results for verification?" (asked on amd-sriram's #51314).
2. Behaviour at high concurrency after an AITER update: "current pr fails at high conc after I updated the aiter."
3. For MoE routing changes: keep the shared router helpers (`fused_topk_bias_router.py`) usable for both AITER and non-AITER paths.

## What Fangzhou checks

- Which AITER version the result depends on; re-tests after bumps.
- Whether a fused shared-expert path respects `n_shared_experts`, `apply_routed_scale_to_output` defaults for non-AITER paths, and single-precision expert assumptions.
- Whether the PR changed param names needlessly across MXFP4/MXFP8 variants.

## How Fangzhou writes

Short, data-first; posts his own test results as evidence for or against a PR.

## Review checklist in Fangzhou's voice

- [ ] InferenceX 8k/1k (or equivalent) e2e perf and GSM8K posted?
- [ ] Tested at high concurrency with the current AITER pin?
- [ ] Shared router / MoE helpers still work for the non-AITER path?

## Real review comments from this reviewer (verbatim, with PR numbers)

- **#57909** ([ROCm][PERF] Enable MiniMax-M3 Indexer CP) — `attention.py`: "I am ok with this new class member var, not sure if we need any upstream maintainer to approve, cc @njhill Maybe we should move this to the M3's attention class definition if there is one."
- **#53492** ([ROCm][MLA] Enable sparse MLA Gluon kernel from Aiter) — `rocm.py`: "Since we only tested it on dsv4.1 we might be able to accept this temporary branch, but next dsv4 enablement PR should remove it to for the sake of simplicity."
- **#53492** ([ROCm][MLA] Enable sparse MLA Gluon kernel from Aiter) — `envs.py`: "If it always has better perf, we can either default it to True or simply remove this knob and make it always the path."
- **#57909** ([ROCm][PERF] Enable MiniMax-M3 Indexer CP) — `indexer_msa.py`: "can we read it from the model's config.json? I am asking because the following Minimax M3 may reuse the code here."
- **#58045** ([ROCm][Kimi-K3] Optimize low-concurrency speculative KDA) — `test_kda.py`: "Is this a good practice to have *_amd and *_nvidia function tested in the same file? @AndreasKaratzas"
- **#57909** ([ROCm][PERF] Enable MiniMax-M3 Indexer CP) — `indexer_msa.py`: "if the kernels here are from AITER, please give explicit credit to kernel, e.g. for each kernel give the permant link pointing to the corresponding aiter file localtion. This rule applies to all codes in this PR. cc @valarLip"
- **#46546** ([ROCm][ [Perf] sparse attention optimization on minimax-m3 ) — PR thread: "@hongxiayang thanks for this! I cherry-picked the three sparse-attn commits onto MiniMax-M3-MXFP4 (W4A4) on gfx950 (TP4, 8k/1k) and they apply and run cleanly, with a perf win that scales with concurrency (vs the same build without these commits; gsm8k unchanged at 0.9416, no crashes): conc ΔTTFT ΔTPOT Δoutput tput 1 −7.7% −1.5% +1.8% 64 −9.9% −7.5% +8.2% 256 −8.5% −9.2% +10.0% Could you rebase this on top of #46474 "
- **#46419** ([ROCm]Enable AITER MoE backend for MiniMax-M3-MXFP4) — PR thread: "Update my perf result on 8 MI355X ISL/OSL 8k1k conc=4 ============ Serving Benchmark Result ============ Successful requests: 40 Benchmark duration (s): 86.71 Total input tokens: 294213 Total generated tokens: 36805 Request throughput (req/s): 0.46 Output token throughput (tok/s): 424.44 Total Token throughput (tok/s): 3817.31 ---------------Time to First Token---------------- Mean TTFT (ms): 274.54 Median TTFT (ms):"
- **#57491** ([ROCm][DSv4.1] Keep the Engram tables in host memory on ROCm) — PR thread: "@functionstackx it could be a kernel side issue that has been fixed in AITER, we are upgrading the AITER version to solve it. A hot patch was raised before but has negative performance impact so it's deprecated. The offloading PR itself can proceed as it's orthogonal and important to TP2 performance."
- **#53492** ([ROCm][MLA] Enable sparse MLA Gluon kernel from Aiter) — PR thread: "@simondanielsson thanks for you comments and I agree. I tested the PR e2e, although combined some other PRs together, I saw hugh perf improvement regarding prefill, mainly contributed by this PR I believe. I will give it a ready label to kick off the CI as we want it to be included in tonight's nightly image, and other models that are not well tested should be excluded. So I am ok with this gate for now but it should"
- **#51314** ([ROCm][Perf] Skip cleaning sparse prefill MQA logits) — PR thread: "Can you add some model-level e2e perf (inferenceX 8k1k) and accuracy (gsm8k) test results for verification?"
- **#50654** ([ROCm][Perf] Kimi-K3 Fused kernel for KDA decode) — PR thread: "The latest fixed-cohort C16/C24 trace provides a useful estimate of the critical-path budget targeted by this fusion. Profile setup: 8 x MI355X (gfx950), TP8, BF16/MXFP4, non-speculative decode, prefix caching, 63,911-token shared prefix + 4,089-token suffix, full HIP-graph replay, and +fused_rms_norm_gated . The baseline was vLLM 25e1f81e1b8def93d7228a38b83348880babbe7b with AITER 61549c5c2623316d24ce54dcf35438173cb"
- **#48044** ([ROCm] Fused Shared Expert Support for AMD Quark DeepSeek-V4 Model Checkpoints) — PR thread: "I opened #48728 as a Draft follow-up for the native DeepSeek V4 checkpoint format. This PR handles AMD-Quark checkpoints whose shared expert is already MXFP4; #48728 handles native checkpoints whose shared expert is serialized block FP8/E8M0 and converts it to RNE MXFP4 before appending it. The appended-expert fusion wiring overlaps, so #48728 is not intended to merge independently: after this PR lands, I will rebase"
- **#54787** ([ROCm][Perf][M3] Fused allreduce+GemmaRMSNorm fast path) — PR thread: "I ran a TP4 rank-max kernel A/B for the eager helper while validating #47270 . There is a reproducible crossover exactly at AITER's one-stage/two-stage boundary for MiniMax-M3 (BF16, H=7168, fully connected TP4): M mode explicit AR + Gemma norm fused fused delta 1 1stage 22.640 us 19.320 us -14.66% 8 1stage 23.680 us 20.040 us -15.37% 16 1stage 22.561 us 21.920 us -2.84% 32 2stage 24.281 us 25.400 us +4.61% 64 2stage"

## Data-derived profile (computed from the PR set, 6 Oct 2026)

- **PRs commented on (not own):** 45
- **Median time to first comment after PR open:** 3.8 days
- **Median time from first comment to approval:** 0.0 days
- **Median time from approval to merge:** 0.8 days
- **Where they comment (paths):** `vllm/models/minimax_m3` (4), `vllm/_aiter_ops.py` (1), `vllm/envs.py` (1), `vllm/models/deepseek_v4` (1), `vllm/config/attention.py` (1), `tests/kernels/attention` (1)
- **Files they comment on most:** `indexer_cp.py` (2), `indexer_msa.py` (2), `_aiter_ops.py` (1), `envs.py` (1), `rocm.py` (1), `attention.py` (1), `test_indexer_msa.py` (1), `rocm_aiter_mla_sparse.py` (1)
- **Models named in their comments:** DSv4 (3), DSV4 (3), MiniMax-M3 (2), DeepSeek-V4 (1), MTP (1), Gemma (1)
- **Pet models / configs to pre-test:** DeepSeek-V4/V4.1, MiniMax-M3, Hy4 on gfx950
- **Escalates to (people they @mention):** @AndreasKaratzas (9), @tjtanaa (8), @shen-shanshan (6), @dllehr-amd (5), @njhill (3). Pre-tag the first one when your diff is outside this reviewer's area.
- **Their own PRs look like:** Kernel-level description with e2e table (35 own PRs in the set; 19 mention GSM8K, 5 include a serve/bench command)
- **Recent shift (last 60 days vs earlier, share of their comments):** e2e perf 26% -> 9%; platform guard 16% -> 4%; graph/warmup/invariance 16% -> 4%; accuracy 11% -> 4% (based on 57 recent vs 19 earlier comments)

### Approval triggers (what flipped them to LGTM)

- InferenceX 8k/1k e2e perf and GSM8K posted
- High-concurrency run with current AITER

Last thing they asked for right before approving, in real PRs:

- #53492: "@simondanielsson thanks for you comments and I agree. I tested the PR e2e, although combined some other PRs together, I saw hugh perf improvement regarding prefill, mainly contributed by this PR I believe. I will give it"
- #52052: "This can also help DSV4.1, @shen-shanshan seems that the fusion is already done for mxfp8 shared expert here. cc @ChuanLi1101"
- #54535: "Nice work @weitliao ! Can we get some perf A/B test results here."
- #56720: "We need upstream reviewers as it's touching the common path file @njhill @zyongye can you take a look at it?"
- #58655: "we need an UT for this new kernel."
- #51794: "hi @zyongye @tjtanaa please take a look at this PR. We have test thoroughly on this PR and it shows decent perf gain."

### Things they let slide

- Minor structure if numbers hold

Comments they themselves marked as nits (do not block on these):

- #50592: "[quote] can you summarize what is the performance gain for this PR? the comments above are kind of "AI" sense and hard to see where is the performance gain end-"
