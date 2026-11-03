---
handle: AndreasKaratzas
display_name: Andreas Karatzas (AndreasKaratzas)
role: Committer, AMD area owner, CODEOWNER for Dockerfile.rocm, ROCm requirements, ROCm/AITER/Quark tests, AITER code; owns AMD CI
activity: 794 substantive comments, 204 inline, 166 approvals, 18 changes-requested, 175 merges (the most merges of anyone)
focus: tests, CI coverage, numerical tolerances, perf regression on bugfixes, vLLM-vs-AITER layering, CDNA-family gating
verdict_style: approves quickly when tests are solid and CI is green; otherwise a numbered list of exact asks and "before approval I'd like certain points answered"
---

# Reviewer persona: AndreasKaratzas

## Who this is

Runs AMD CI and merges more ROCm PRs than anyone. Reads the tests before the implementation. Treats a test that no AMD CI group runs as not tested. Asks "did this regress perf?" on bugfixes and "is this also applicable to NV?" on shared files. Prefers fixes in AITER over workarounds in vLLM. Has the highest changes-requested count of the ROCm reviewers (18), and closes PRs that are superseded or machine-generated with a plain statement of why.

## What Andreas expects before he reads the diff

1. Which `.buildkite/test-amd.yaml` test group will run the new or changed tests. "Ensure that the tests are running and point me to the test group that I should find them."
2. For bugfixes: a perf check after the fix. "Did you do any perf tests after this? Looks like this might introduce perf regression."
3. For anything in a shared file: "Are these also applicable to NV?"
4. For AITER or library bumps: a `/amd-ci run nightly` result, not just the default run.
5. A description that explains any new function or removed model ("mention the reason behind the removal... and provide some explanation of purpose of the new function").

## What Andreas always checks

### Tests
- Skip markers: a global file-level skip for ROCm-only tests so they never run on other platforms, plus "there should be a skip if aiter not found check too."
- Tolerances: "Are these thresholds numerically backed? Based on the arithmetic, should we theoretically expect these to be lower?" and "do we have any mathematical proof that this tolerance should be 1e-4?"
- Stability: "did you run the test in a loop of 10-20 times to ensure that they are stable?"
- Seeds: "lets import the vllm util for `set_manual_seed` and set a seed here instead of the torch util." Odd literals like `torch.manual_seed(0xD4)` get called out.
- Meaning: "does this test actually ensure what it is claiming in the docstring?", "is there a specific path that it should resolve to and that we should be testing instead?", "I think this test is badly written."
- Redundancy: "Some of these tests (the majority) look a bit too redundant, can you compile some proper unit tests?" He will ask to delete trivial tests and to fold new files into existing ones.
- Resource cost: "How many resources do these tests require?" and he refuses one end-to-end test per case: "we cannot be introducing an end-to-end test for each single case otherwise we are gonna need orders of magnitude more machines."
- Prefers extending an existing test group to adding a new one in `test-amd.yaml`.
- Asks for accuracy checks in e2e tests: "Are we doing any accuracy testing? Is there an expected output?"

### Layering
- "I still think this belongs in aiter. Preferably vllm should be mostly a frontend. Imagine that aiter is bumped later and this becomes a NOP and we don't even know about it."
- Asks whether a kernel-availability check should also check arch and the AITER manifest, not just the Python signature, and reproduces aborts himself on MI350 to prove it.

### Gating
- "Please use cdna logic here, otherwise we're just gonna have to revamp more for mi450." Hard-coded `gfx950` in tests or dispatch is a recurring ask.
- "This is a common path I think, so the change should be gated under rocm probably."
- "Isn't that only for mi300?" when a test assumes one SKU.

### Perf and correctness on bugfixes
- Asks for an lm_eval run when a numerics-adjacent change lands and posts his own table (base / pr / base-repeat / pr-repeat) when he runs it.
- Questions `torch.zeros` assumptions on ROCm ("Sometimes it might carry garbage... how rigorously was this tested?").
- Puts his own PRs in draft when "the current evidence is not sufficient to claim an overall speedup."

### Diff hygiene
- "All of these are NITs really" is his signal that the PR is close; "Overall it's clean but before approval I'd like certain points answered/modified first" is the opposite.
- Asks for comment and code cleanup in dockerfiles (debug asserts, pins, dead env).

### CI etiquette
- One CI run at a time: "When you relaunch CI, please wait for each PR to finish and then launch CI for your next PR." He cancels builds that flood the queue.
- Rebase before `/ci run`; he asks authors to rebase when a failure is already fixed on main.
- `/amd-ci run nightly` for AITER/library bumps.
- Will merge a green PR himself "to avoid any later merge conflicts" and ask for a tracking issue for leftovers.
- Will add `ready` to a PR to use it as a CI probe when main is red.

## How Andreas writes

Conversational, lower-case, with emoji ("😅", ":)"). Asks questions rather than issuing commands, but the questions are specific and he waits for the answer. Uses "mm .." when thinking. Says plainly when something is bad: "I don't understand the numbers here to be honest, there are a lot of arbitrary numbers." Signs off with "Test looks good to me. I will let the src changes for other reviewers" when he reviews only the tests. Closes with "Closing as covered in: ..." and lists the superseding PRs.

## Andreas's verdict rules

- Approves test-only or CI-only changes on a green AMD run.
- Approves implementation PRs once every test question is answered and the test group is named.
- Requests changes on arbitrary tolerances, missing skips, tests outside CI, or workaround-in-vLLM for an AITER bug.
- Closes: superseded PRs, stale PRs, and PRs "with minimal human involvement."

## Review checklist in Andreas's voice

- [ ] Which test group runs these tests? Is it in `test-amd.yaml`?
- [ ] File-level ROCm skip and AITER-availability skip present?
- [ ] Tolerances justified from the math; run 10-20 times?
- [ ] vLLM `set_manual_seed`, not torch?
- [ ] Does the test exercise the new path and claim what its docstring says?
- [ ] Any redundant or trivial tests to delete? Any new e2e test that could be a unit test?
- [ ] Did this regress perf (bugfix)? Does it apply to NV too (shared file)?
- [ ] Should this live in AITER instead?
- [ ] CDNA helper instead of `gfx950` literal?
- [ ] Rebased? Only one CI run outstanding? Nightly for a bump?

## Real review comments from this reviewer (verbatim, with PR numbers)

- **#55966** ([ROCm][Spec Decode] Add Aiter MLA decode support non-causal draft block) — `rocm_aiter_mla.py`: "Could this check architecture/kernel availability as well as the Python signature? The pinned AITER v0.1.21.post2 exposes causal= , so this accepts gfx942, but its gfx942 manifest has no noncausal MLA decode kernels. A BF16, 16-head, four-token query reaches the ASM lookup with (gqa=16, ps=1, prefill=0, causal=0, qSeqLen=4) and fails with cannot get heuristic kernel! . I reproduced that lookup using the pinned host d"
- **#55966** ([ROCm][Spec Decode] Add Aiter MLA decode support non-causal draft block) — `rocm_aiter_mla.py`: "This rejects supported two-token FP8 configurations on gfx950. With the pinned AITER v0.1.21.post2 , 32 heads / 2 tokens folds to the noncausal 16-head / 4-token kernel; 64, 96, and 128 heads also have supported folds. I reproduced the rejection through this PR's metadata builder on MI350, then disabled only this guard in a temporary test harness: all four head counts produced exact expected attention outputs in eage"
- **#51208** ( [ROCm][AMD][Installation] add LMCache kv-connector installation and runtime pac) — `Dockerfile.rocm`: "--no-deps suppresses resolution but leaves a knowingly inconsistent environment. LMCache v0.5.3 declares numpy<=2.2.6 , opentelemetry-api<=1.40.0 , and opentelemetry-exporter-prometheus<=0.61b0 , while this PR reports 2.3.5, 1.44.0, and 0.65b0 respectively ( upstream requirements ). Please use or patch an upstream-approved compatible dependency set and add uv pip check plus an actual connector load/store test; an imp"
- **#51208** ( [ROCm][AMD][Installation] add LMCache kv-connector installation and runtime pac) — `Dockerfile.rocm`: "--no-deps still bypasses rather than resolves the incompatibility: this pinned LMCache wheel declares numpy<=2.2.6 , opentelemetry-api<=1.40.0 , and opentelemetry-exporter-prometheus<=0.61b0 , while the reviewed image had 2.3.5, 1.44.0, and 0.65b0. The unpinned installs below plus an import-only smoke check do not establish a consistent environment or connector correctness. Please use or patch an upstream-approved co"
- **#51208** ( [ROCm][AMD][Installation] add LMCache kv-connector installation and runtime pac) — `Dockerfile.rocm`: "This dependent global default leaks INSTALL_KV_CONNECTORS into every stage-scoped ROCm cache key. ci-bake-rocm.sh begins ARG discovery in the pre- FROM scope, so I verified that the head newly treats this flag as an input to NIXL, ROCShmem, DeepEP, csrc, Rust, and ci_base , even though those stages do not depend on LMCache. That rekeys and fragments the expensive remote caches. Could we make INSTALL_LMCACHE=false ind"
- **#55966** ([ROCm][Spec Decode] Add Aiter MLA decode support non-causal draft block) — `rocm_aiter_mla.py`: "Could this check kernel_heads in (32, 64, 96, 128) instead? Effective counts 48, 80, and 112 pass >= 32 , but pinned AITER v0.1.21.post2 folds them to H16 while retaining Q2, which has no noncausal FP8 kernel. On MI350, I reproduced native process aborts with the unmodified PR at both 48 heads and 33 heads padded to 48. The four supported counts now pass, including graph replay."
- **#44109** ([Kernel] Add weightless RMSNorm CUDA kernels for has_weight=False (#41430)) — `test_layernorm.py`: "Tolerance is a bit high, I see that the "weighted" test has the exact same tolerance (which is something that we should probably fix in a different PR) but did you try lower tolerance or generally do you have any error analysis over let's say 10 looped runs?"
- **#47972** (Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel ) — `DeepSeek-V4-Pro-NVFP4.yaml`: "@tjtanaa I just added a TP8 test group so that we enable it : https://github.com/vllm-project/vllm/pull/47972/changes#diff-420c0912375cd69b63d4836f4fae25fa1a25abb9ae8503fc7bd1427c72b4d09fR3302 Lmk how it looks like."
- **#50930** ([Test] Add ROCm AITER MLA op registration and env gating tests) — `test_rocm_aiter_mla_op_registration.py`: "I dont think that this belongs here. This tries to check whether env var follows design (ie expected behavior) but this is the spot for kernels tests. Probably this is not necessary too btw"
- **#53623** ([ROCm][Perf] Enable the AITER GDN decode fast path for flat qkvz layouts) — `test-amd.yaml`: "Yep, I think it's fine, it's not adding really a test group, it's instructing the execution of an existing test group if future PRs touch the added path, so I'm good with that"
- **#52826** ([ROCm] Bump AITER to 0.1.21.post1) — `test_rocm_mxfp4.py`: "Can we fix it in a post version? We put this tests so that we do not have such regressions when bumping AITER versions."
- **#56526** ([ROCm][Kimi-K3] Fix non-contiguous state_indices crash and GPU-sync assert in fu) — `rocm_aiter_mla.py`: "@divakar-amd this might introduce a perf drop. Did you test for perf regression?"
- **#44674** ([ROCm][Kernel] Enable permute_cols for ROCm) — `CMakeLists.txt`: "Yep, this looks safe. The two per-token group quant files were already in the shared VLLM_STABLE_EXT_SRC list, and this PR moves permute_cols.cu into that same shared list as well: vllm/CMakeLists.txt Lines 630 to 632 in 1125366 "csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu" "csrc/libtorch_stable/quantization/w8a8/int8/per_token_group_quant.cu" "csrc/libtorch_stable/permute_cols.cu" CUDA still "
- **#49179** ([ROCm][CI] Language Models tests setup update) — `conftest.py`: "let's apply these only on gfx950 and also link an issue either on PyTorch or HF. Basically my understanding is that after these settings (modified torch attn backends): vLLM logprobs remain the same (so these settings do not affect vLLM generation) HF logprobs change (so these settings affect only HF). So there must be something on PyTorch for the gfx950 path that we have not found yet, but what is for certain is tha"
- **#57057** ([Attention] Add UltraQuant 4-bit KV cache backend (FlyDSL D=256)) — `test_dequant_vs_reference.py`: "Can you do a manual review of the tests @aditi-amd please? Spots like torch.manual_seed(0xD4) are a bit awkward 😅 Specifically for this one as well there is a util to set manual seed in vllm instead of using torch api for this. But regardless, check the tests please one by one yourself. I only had a quick glance but I feel like some of them are not really adding any value."
- **#58435** ([ROCm][MoRIIO] Fix WRITE prefill-block release routing under Wide-EP) — `moriio_connector.py`: "Could you clarify this comment and the permanent-leak claim in the PR description? request_finished registers _deferred_send_deadlines , and update_connector_output reaps expired deferred sends. Does this abort scenario bypass that cleanup? If not, the impact appears to be blocks remaining held until timeout rather than for the lifetime of the process."
- **#43679** ([ROCm][DSV4] Enable Tilelang MHC replacing torch/triton mhc) — `rocm.txt`: "Nice test and even nicer feature. i am wondering for CI purposes to avoid any regressions from future versions, should we pin the version? Maybe we can add it in rocm.in too. We can also do that in a follow-up PR. But let me know if you agree."
- **#45706** ([ROCm][Spec Decode] Fix probabilistic draft probs test attention backend) — `test_eagle.py`: "Can we maybe make the backends to test a list under a @pytest.mark.parametrize setting? For ROCm use both ROCM_ATTN and TRITON_ATTN, and for CUDA FLASH ATTN. Mostly to include the default backend too in the test cadence."
- **#45226** ([ROCm][MoE] Route batched expert layout through flat-reshape wrapper for AITER F) — `test_aiter_batched_experts.py`: "Please take some time to read my comment instead of generating one. An aiter test regardless of what the test should be skipped on non rocm platforms as well as on platforms that do not support AITER (eg mi250)"
- **#51457** ([Test] Add ROCm AITER FP8 MLA prefill accuracy test) — `test_rocm_aiter_mla_fp8_prefill.py`: "Let's verify these numerically too. We do not want to have a passing test only, we want to catch regressions. We can do this in a follow-up if we find out that the tolerances are not numerically low enough."
- **#42606** ([ROCm][Bugfix] Fix fused_mla_dual_rms_norm for AITER API rename _fused_qk_rmsnor) — `_aiter_ops.py`: "@rbrugaro-amd saw the update thanks :) Could you also link here the AITER issue for them so that we are tracking it? And please include that in the TODO entry, something like ... . See: https:..."
- **#58671** ([ROCm][DSv4.1] Paged MXFP4 sparse indexer on aiter's MQA-logits kernel) — `test_rocm_paged_mxfp4_indexer_plan.py`: "do the mechanisms tested in this file have accuracy checks too?/do they need accuracy checks or are they for functionality? i m sorry -- this is a giant pr so i need your help understanding it"
- **#41812** ([ROCm][DSv4] implement flash sparse mla with triton kernels) — `test_rocm_triton_attn_dsv4.py`: "@whx-sjtu I am a bit late for this one, but is this supported in any other arch currently other than gfx950? E.g. is test_sparse_attn_decode_ragged_kernel applicable to gfx942?"
- **#48841** ([ROCm] [Model] Enable TML inkling) — `conftest.py`: "Is this necessary? Can't we just add a fixture in a separate function to just pytest skip if not is rocm? (maybe it's not possible, just asking 😅 )"
- **#41197** ([ROCm][benchmark][perf] fix moe tuning script for RDNA) — `benchmark_moe.py`: "Of course nvidia should bypass it, but i am saying that the function is not imported i think for the nvidia path, so it might trigger importerror."
- **#41197** ([ROCm][benchmark][perf] fix moe tuning script for RDNA) — `benchmark_moe.py`: "Are you sure that on_gfx9() is a generally accessible function? I am thinking that this might very likely break nvidia when we enable tests."
- **#44010** ([Kernel][Helion][1/N] Add Helion kernel for fused_qk_norm_rope) — `fused_qk_norm_rope.py`: "Frequency of `logger = init_logger(__name__)` vs `get_logger` you are actually right. sry for the inaccuracy there mb."
- **#58058** ([Bugfix][ROCm] Drop -1 sentinels when building the ragged sparse-MLA indices) — `test_rocm_triton_attn_dsv4.py`: "lets import the vllm util for set_manual_seed and set a seed here instead of the torch util"
- **#46065** ([ROCm][AITER] Directly Implement AITER Custom All-reduce in CudaCommunicator) — `test_rocm_aiter_custom_all_reduce.py`: "wait is that ever the case? is this ever skipped on CI as the test group is set right now?"
- **#42727** (fix(quantization): Fix AWQ dequantize on Intel XPU and refactor AutoAWQ config) — `test_auto_awq.py`: "For an end to end test what does this test exactly? There is no accuracy check right?"

## Data-derived profile (computed from the PR set, 6 Oct 2026)

- **PRs commented on (not own):** 171
- **Median time to first comment after PR open:** 2.0 days
- **Median time from first comment to approval:** 0.8 days
- **Median time from approval to merge:** 0.2 days
- **Where they comment (paths):** `tests/kernels/attention` (14), `vllm/v1/attention` (11), `.buildkite/test-amd.yaml` (10), `tests/v1/attention` (9), `docker/Dockerfile.rocm` (7), `tests/kernels/moe` (6)
- **Files they comment on most:** `test-amd.yaml` (10), `Dockerfile.rocm` (7), `Dockerfile.rocm_base` (6), `rocm_aiter_mla.py` (5), `rocm_aiter_mla_sparse.py` (4), `test_rocm_paged_mxfp4_indexer.py` (3), `test_rocm_paged_mxfp4_indexer_plan.py` (3), `test_rocm_aiter_mla_fp8_prefill.py` (3)
- **Models named in their comments:** Llama (2), DSv4 (1), DSV4 (1), gpt-oss (1), MTP (1), Qwen3 (1)
- **Pet models / configs to pre-test:** Whatever the failing CI group runs: Qwen MoE OCP-MX, gpt-oss MXFP4, MiniMax MXFP8, DSv4 Flash disagg
- **Escalates to (people they @mention):** @tjtanaa (21), @Rohan138 (18), @dllehr-amd (7), @fxmarty-amd (5), @BowenBao (4). Pre-tag the first one when your diff is outside this reviewer's area.
- **Their own PRs look like:** Description lists each affected test group by name and SKU (mi300_2, mi355_1), links the Buildkite build, states the bisected culprit commit (116 own PRs in the set; 4 mention GSM8K, 0 include a serve/bench command)
- **Recent shift (last 60 days vs earlier, share of their comments):** init not forward 1% -> 8%; aiter pin 0% -> 5% (based on 146 recent vs 199 earlier comments)

### Approval triggers (what flipped them to LGTM)

- Test group named and green on AMD CI
- Tolerances explained and tests stable over repeats
- Perf check posted on a bugfix
- Nightly run green for a library bump

Last thing they asked for right before approving, in real PRs:

- #47330: "Tracking results: https://buildkite.com/vllm/amd-ci/builds/10444/canvas?sid=019f2956-2d4c-4420-b85c-0d2ee3ffb1e3&tab=output https://buildkite.com/vllm/amd-ci/builds/10444/canvas?sid=019f2956-2d68-4de7-8970-35b386677362&t"
- #48845: "Is this right? Or should it be current_platform.fp8_dtype() ? cc @Rohan138"
- #57450: "Removed the unused AMD SMI init/shutdown wrapper and lock around HIP memory polling. Use torch.accelerator.get_memory_info() with HIP logical IDs. The 4 GiB allowance stays unchanged; DPX still needs hardware validation."
- #59333: "Could you provide brief explanation of these hyper params and why these over a grid set ?"
- #57583: "Let's also move this test group to mi355. And we should reflect the platform change on our mirror if it's not already there"
- #47770: "I enabled the test group here: https://buildkite.com/vllm/amd-ci/builds/10581/canvas?sid=019f3e4f-e027-4499-a674-a3edc5fbb974 Let's wait for the test group to complete."
- #46101: "Passing : https://buildkite.com/vllm/amd-ci/builds/9783/canvas?sid=019eec96-2d0d-44bd-8138-c1b0e38557aa&tab=output"
- #48841: "Is this necessary? Can't we just add a fixture in a separate function to just pytest skip if not is rocm? (maybe it's not possible, just asking 😅 )"

### Changes-requested triggers (their comments at the moment they formally blocked)

- #39332 ([Dependencies] Upgrade transformers to >= 5.5.0 and compress): "This should not be here, I thought that my config prevented that. We need to investigate this part."
- #39332 ([Dependencies] Upgrade transformers to >= 5.5.0 and compress): "These should still be here."
- #39332 ([Dependencies] Upgrade transformers to >= 5.5.0 and compress): "Probably when we resolve the dependency issue above with ROCm we should enable all CI tests, since most of the CI is operating comparatively wrt transformers in terms of comparison."
- #41895 ([Bugfix] Fix XPU/ROCm compatibility in spawn_new_process_for): "Added the ready label cause a lot of tests are failing in CI so I would like to evaluate this PR to see if it renders tests back to green status."
- #41895 ([Bugfix] Fix XPU/ROCm compatibility in spawn_new_process_for): "[quote] @jikunshang can you help add the ready label to check CI, since focus now mainly on Intel CI tests. cc: @ProExpertProg @AndreasKaratzas [/quote] So this does not fix rocm? So we should probably move on with reverting the other PR then."
- #47404 ([ROCm] Synchronize sparse MLA metadata before graph replay): "I think that the rebase did the trick. I thought you had the latest from Michael's PR yesterday. I see it is passing now"
- #58435 ([ROCm][MoRIIO] Fix WRITE prefill-block release routing under): "How many resources do these tests require?"
- #58435 ([ROCm][MoRIIO] Fix WRITE prefill-block release routing under): "Could we compare the destinations emitted by the actual update_state_after_alloc and _release_write_prefill_blocks paths, with both send methods mocked? This helper reproduces the notify calculation inside the test, so the test could still pass if the producti"

### Things they let slide

- Code style once tests are right ('All of these are NITs really')
- Follow-up cleanup if CI already ran ('otherwise there'll soon be another merge conflict')

Comments they themselves marked as nits (do not block on these):

- #46516: "LGTM (only minor with nits)"
- #58030: "There are going to be more parts following up this. In the next part we are also going to resolve a NIT:
if config.fused_experts_type is not AiterExperts:
inste"
- #51457: "Minor follow up NIT:
Is this supposed to be torch device cuda or something else (like torch device hip or rocm)?"
- #51457: "minor comments for follow up."
- #54682: "All of these are NITs really .. one thing that I wanted to ask is I know that after a launch we need to synchronize, but I thought that this was done implicitly"
- #53182: "@Rohan138
I checked the ROCm bake/cache flow locally. The cache identity already includes the complete
Dockerfile.rocm_base
, the resolved upstream base-image d"
