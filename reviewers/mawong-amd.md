---
handle: mawong-amd
display_name: Matthew Wong (mawong-amd)
role: Reviewer; frequent author of multi-stream, cumem and CI fixes
activity: 63 substantive comments, 15 inline, 8 approvals
focus: multi-stream shared experts, allocator/cumem teardown, timing clashes between PRs
verdict_style: short, acknowledges overlap ("Unfortunate timing clash; #46203 was merged with substantially the same changes")
---

# Reviewer persona: mawong-amd (Matthew)

## Who this is

Author of the multi-stream shared-experts tuning and several allocator fixes. As a reviewer he is pragmatic about overlap and closes his own PRs when another landed first. Expects benchmark re-validation when his tuning is changed (simondanielsson asked him for exactly that on #56098).

## What Matthew checks

- Multi-stream gating conditions (DP vs TP combinations) and the bench-serve evidence behind them.
- Allocator and teardown ordering on ROCm (MemPool release, double-free of allocation handles).
- Whether a workaround belongs in the ROCm stack rather than vLLM (tjtanaa asked him to dissect a cumem issue).

## Review checklist in Matthew's voice

- [ ] Multi-stream conditions validated with bench serve for TP, DP and DP+TP?
- [ ] Teardown ordering safe on ROCm?
- [ ] Is another PR already fixing this?

## Real review comments from this reviewer (verbatim, with PR numbers)

- **#49309** ([ROCm][CI] Use explicit wvSplitKrc skinny-GEMM test tolerance for bf16 (gfx950)) — `test_rocm_skinny_gemms.py`: "Regarding the analysis: it occurred to me that the Xavier initialization done here is not strictly correct, because A is Xavier normal initialized only if each element is drawn from an independent normal distribution with mean 0 and variance 2/(n+k) (*or 2/k might suffice). Xavier uniform initialized only if each element is drawn from an independent uniform distribution on [-r, r] , where r = sqrt(6/(n+k)) *or sqrt(6"
- **#43022** ([ROCm][CI] Stabilize sleep-mode memory release) — `cumem_allocator.cpp`: "I'm trying to understand the logic here. It seems the issue here is that hipMemAddressReserve called with an non-trivial alignment (i.e. not 0 or 1) might fail even where hipMemAddressReserve might return a properly aligned virtual memory reservation if it was called with alignment = 0. Is that right? If so, then shouldn't the code here read something like this: Call cuMemAddressReserve with requested alignment. If s"
- **#43022** ([ROCm][CI] Stabilize sleep-mode memory release) — `cumem_allocator.cpp`: "I don't believe cuMemAddressFree is needed if the cuMemAddressReserve fails, but assuming cuMemAddressFree follows standard free semantics (i.e. it's a no-op on a nullptr) it should be harmless here. We should still probably return a HIP_ERROR_NOT_SUPPORTED though: ultimately, the aligned memory allocation failed regardless if the free succeeded or not."
- **#46904** ([ROCm][CI] Refresh ROCm base images when docker rocm_base changes) — `build-test-image.sh`: "If I'm not wrong this should just be a bare return : don't want to mask the exit code of the previous IMAGE_TAG="" bash .buildkite/scripts/ci-bake-rocm.sh test-rocm-ci-with-artifacts"
- **#51692** ([ROCm][Perf] Add bpreshuffled blockscaled fp8 GEMM) — PR thread: "Was this PR tested on DeepSeek V3/V4 TP8? I'm seeing garbage output on the former, and it refuses to initialize on the latter (it hits the not implemented apply_block_scaled_mm ). Note: Testing on main now is slightly complicated because #56228 changed the signature of the sparse-attention indexer call (a separate breakage), but the results stand after patching that."
- **#48847** ([ROCm][CI] Loosen block-FP8 fused MoE test tolerance for large-K shapes) — PR thread: "Not opposed to increasing tolerances per se, but I think we can pin down some of the sources of numerical divergence more carefully. I looked into this test group a while back and found the following sources of divergence on ROCm: Torch reference vs Triton/HIP FP8 group quant kernel. For a given tensor A , Torch calculates the scale s as A.abs().max().clamp(min=eps).float() / 448.0 . Then it calculates A / s elementw"
- **#52033** ([Perf][ROCm] Dual-stream decode with hipgraphs) — PR thread: "#53712 should cut down on the cost of some hipStreamWaitEvent , i.e. when a given stream needs to sync with another (PyTorch level: torch.cuda.Stream.wait_stream() ). So the performance heuristic here can be updated (although it does not need to be done in this PR). The performance numbers are certainly different now however. In particular, some choices that used to cost (e.g. adding copies to avoid the buffer aliasi"
- **#53182** ([ROCm] Ship rocprofiler-sdk 1.3.2 in Dockerfile.rocm_base to fix torch.profiler ) — PR thread: "Concern: we might also want to sync https://github.com/vllm-project/vllm/blob/7f6432cc0daf97c5b9a2df141641c595a766ff40/.buildkite/scripts/rocm/refresh-base-image.sh so it doesn't keep rebuilding the rocm_base image (and therefore also the ci_base image). cc @AndreasKaratzas"
- **#46203** ([Bugfix][ROCm] Fix cumem sleep and teardown) — PR thread: "Overall looks good pending rebase and possibly a nit on correcting HandleType on ROCm. I think we should also be able to enable and gate the AMD: Basic Correctness test group in this PR: I've tested this PR locally and it now passes all the previously-failing CuMem allocator tests in tests/basic_correctness/test_mem.py ."
- **#57132** ([ROCm][Bugfix] Revert #56433 + #51692 to fix accuracy breakdown for DeepSeek-V4) — PR thread: "No objections to the revert. I did test gsm8k on quite a few models and configs after #56433 (which was itself a hotfix) but I'm guessing something else has come in since that has broken it again. I was already a bit worried in terms of brittleness of the contract that #51692 and #56433 jointly introduce (namely: needing to declare wants_transposed_act_scale and preshuffles_weight ) among other things so the right mo"

## Data-derived profile (computed from the PR set, 6 Oct 2026)

- **PRs commented on (not own):** 18
- **Median time to first comment after PR open:** 4.8 days
- **Median time from first comment to approval:** 5.1 days
- **Median time from approval to merge:** 0.9 days
- **Where they comment (paths):** `csrc/cumem_allocator.cpp` (3), `.buildkite/scripts/rocm` (1), `tests/kernels/quantization` (1), `vllm/models/deepseek_v4` (1)
- **Files they comment on most:** `cumem_allocator.cpp` (3), `build-test-image.sh` (1), `test_rocm_skinny_gemms.py` (1), `rocm.py` (1)
- **Models named in their comments:** Mixtral (1), DSv4 (1), DSV4 (1), Qwen3 (1), Qwen3.5 (1)
- **Pet models / configs to pre-test:** Qwen3.5-35B-A3B multi-stream, cumem teardown tests
- **Escalates to (people they @mention):** @AndreasKaratzas (1). Pre-tag the first one when your diff is outside this reviewer's area.
- **Their own PRs look like:** Explains the race or ordering issue, links the failing build (11 own PRs in the set; 2 mention GSM8K, 0 include a serve/bench command)

### Approval triggers (what flipped them to LGTM)

- Bench-serve evidence for TP, DP, DP+TP

Last thing they asked for right before approving, in real PRs:

- #48847: "Not opposed to increasing tolerances per se, but I think we can pin down some of the sources of numerical divergence more carefully. I looked into this test group a while back and found the following sources of divergenc"
- #46944: "One alternative approach: refer to https://github.com/vllm-project/vllm/pull/46658/changes for utility functions that can check if 2 FP8 values are 1 ULP apart. We can perhaps set a reasonable metric here to allow a smal"
- #49309: "Regarding the analysis: it occurred to me that the Xavier initialization done here is not strictly correct, because A is Xavier normal initialized only if each element is drawn from an independent normal distribution wit"
- #43022: "I'm trying to understand the logic here. It seems the issue here is that hipMemAddressReserve called with an non-trivial alignment (i.e. not 0 or 1) might fail even where hipMemAddressReserve might return a properly alig"

### Things they let slide

- Overlap with his own PRs

Comments they themselves marked as nits (do not block on these):

- #46203: "Overall looks good pending rebase and possibly a nit on correcting
HandleType
on ROCm. I think we should also be able to enable and gate the AMD: Basic Correctn"
