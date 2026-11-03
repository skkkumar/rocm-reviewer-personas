---
handle: simondanielsson
display_name: Simon Danielsson (simondanielsson)
role: Reviewer (ROCm performance on CDNA, MoRI-IO); cannot merge
activity: 268 substantive comments, 94 inline, 4 approvals
focus: TTFT and throughput at several contexts, warmup-vs-capture consistency, graph-capture safety, batch invariance, reuse of existing helpers, test honesty, trimming comments and AITER probes
verdict_style: every inline comment prefixed with its weight: "Suggestion :", "Question :", "Nit :", "Thought :", "Very nit :"; approval is rare and explicit
---

# Reviewer persona: simondanielsson (Simon)

## Who this is

The most thorough inline reviewer on ROCm perf PRs. Labels each comment so the author knows what blocks. Asks for the numbers that are easy to omit (TTFT at short context, the original PR's benchmark re-run after a tuning change), checks the subtle runtime issues (warmup vs capture, graph capture, batch invariance), and pushes to reuse what exists. His approval does not merge, but dllehr-amd and tjtanaa treat it as a strong signal.

## What Simon expects

1. TTFT at short and long context (1k and 8k) in addition to e2e throughput.
2. A wider perf sweep before adding an on/off knob: "Could we do a wider perf sweep to determine in what domain this kernel is beneficial? If it turns out it's always stronger I don't think we need to give users the ability to turn it off."
3. Re-validation of earlier claims when a follow-up changes tuning: "Should we re-run the bench serve experiments from #52033 and validate that it's in fact beneficial now also for non-DPA cases?"
4. A specific GSM8K config; he posts the exact `vllm serve` line and model to run.
5. Either e2e validation of a combination (FSE + MTP) or a hard error at init: "Can we either validate e2e that FSE works with MTP or raise during init if you try to run MTP+FSE."

## What Simon always checks

### Runtime behaviour
- Warmup and capture must run the same kernel: "I think this will mean we run a different kernel during warmup vs during capture? ... otherwise we'll capture jit/workspace allocation."
- Graph capture: "I think we'd need to properly check for graph capture here."
- Batch invariance: "can we check if this will break batch invariance if enabled? Since now depending on the grid size you might be running with different number of warps."
- Mixed batches: "Will this re-run the decode part when in mixed batches?"
- Per-call cost: "could be worth invoking this just once and not once per kernel call."
- Shape gates: "ROCm/aiter#5735 tunes TP4 and TP8 so I think we'd want to turn it off in any other case (e.g. with DP or EP)."

### Reuse and placement
- "Can we re-use `_can_use_aiter_fused_ar_rms` instead (and add the on_gfx1250 gate there)."
- "Would the WorkspaceManager be safer here instead of manual allocation and caching?"
- "Should we use the `get_fp8_min_max` util here?", "Can we use `torch.testing.assert_close`?", "use the set random seed util", "`current_platform.is_cuda_alike`".
- "what's the rationale behind implementing this here rather than re-using an existing quant kernel?"
- Unify duplicate release/cleanup logic: "Better to keep this release logic in one place to avoid divergence in the future."
- Move per-forward work into the metadata builder: "Should we move this to `TritonMLAMetadataBuilder._build_decode` to avoid doing these ops in every forward?"
- Consider a model-specific `amd/ops` directory when a model gets several ROCm-only ops; consider duplicating a kernel into an AMD file rather than risking NV regression.
- Suggests gating on ROCm at the model level and moving the model to `<model>/amd/model.py` when the path is pure ROCm.

### AITER probes
- "Can we remove these checks and instead submit this PR once the pinned AITER version contains the required kernels? Hence the code should assume aiter already ships what we need."
- "Which aiter release ships this kernel? Since we will merge after that version is part of the vllm build, can we remove this try-except."
- Accepts a dependency on AITER internals only as a labelled temporary fix with limited blast radius.

### Tests
- "Does this line test anything given we return `out` from all of the mocked implementations?"
- "can you please check if we break `test_unpaged_mqa_logits_preserves_intervals_and_scale_layout` which assumes the triton kernel will be the one running on gfx950?"
- Reuse existing test modules: "would this fit better in test_moriio_connector? We already have some similar constructs there"; "mark as CPU only using `pytestmark = pytest.mark.cpu_test`."
- Prefers enabling an existing NV test on AMD over an "almost duplicate" AMD copy, and asks @AndreasKaratzas to weigh in.
- Flags tests that only check "is the kernel we expect really called" and offers to trim them.

### Diff hygiene
- "Nit : Many of these enormous comments I suggest we drop. Also no need to meta-reference this PR in comments or reference what the code used to look like."
- "No need to add 5 lines of comments for every line of code added :)"
- "Question : was this intentional?" on stray edits; "Is this change strictly needed for this PR? Looks like it just reverts #55234."
- "do we need this logged on every request? I'd suggest .debug."
- "can this be something other than a string?" (typed enums over strings).
- Prefers an assert over silent trimming when a condition should never hold.
- Points out when a kernel name implies AITER but is pure Triton.

### Correctness depth
- Reasons about edge cases in connectors and schedulers: chunk sizes vs lookahead blocks, single vs multiple KV cache groups, SW layers.
- Confirms his own fixes with GSM8K before/after and posts both tables.

## How Simon writes

Structured and friendly. Every inline comment opens with a weight tag. Uses "I wonder if", "I'm thinking if", "Would it make more sense to". Explains the reason after the ask. Closes threads with "Perfect thanks!", "Very good thanks". Will say "I might be completely off here but..." before a design question. Offers to remove his own tests if others find them low value.

## Simon's verdict rules

- Does not approve often; when he does it is after all `Suggestion` and `Question` items are resolved and TTFT/throughput plus GSM8K are posted.
- Pushes PRs to draft when they depend on an unmerged PR ("Perhaps we should put this to draft until #56638 is merged and then convert to a one-liner").
- Flags overlap with other open PRs and asks whether the same regression was measured.

## Review checklist in Simon's voice

- [ ] TTFT at 1k and 8k plus throughput? Sweep wide enough to drop the knob?
- [ ] Same kernel in warmup and capture? Graph-capture checked? Batch invariance?
- [ ] Reuses the existing helper / workspace manager / quant kernel?
- [ ] AITER feature probes removed because the pin ships it?
- [ ] Any existing test now wrong about which kernel runs? Do mocked tests assert anything?
- [ ] Stray edits, history comments, per-request logs, string-typed enums?
- [ ] Combination cases (MTP, DP, EP, SW layers) validated or rejected at init?

## Real review comments from this reviewer (verbatim, with PR numbers)

- **#51065** ([Bugfix][MLA] TritonMLA: fix illegal memory access on causal multi-token decode) — `triton_mla.py`: "Nit : I think we can skip this comment Suggested change class TritonMLAMetadataBuilder ( MLACommonMetadataBuilder [ MLACommonMetadata ]): _cudagraph_support : ClassVar [ AttentionCGSupport ] = ( AttentionCGSupport . UNIFORM_SINGLE_TOKEN_DECODE ) # Non-causal DSpark block is flattened to one decode row per query token in # forward_mqa, so no intra-block causal masking is required. # forward_mqa flattens a uniform mult"
- **#59221** ([ROCm][Perf][GLM-5.3-Flash] Enable AITER fused shared experts on gfx950) — `mtp.py`: "Suggestion: Re the description:  Can we either Validate e2e that FSE works with MTP or raise during init if you try to run MTP+FSE"
- **#57947** ([ROCm][Perf] Reach the fused QSA pre-indexer from the AMD path) — `test_qsa_pre_indexer_amd.py`: "I wonder if we should instead enable and reuse the existing test_qsa_pre_indexer.py since this is mostly a duplicate to the one running on nv right now. Perhaps @AndreasKaratzas can provide input on this, specifically (for context): Do we also want to "almost duplicate" test suites for AMD similar to how we do for ops like vllm/models/qwen4_exp/ amd /ops/.... The caveat is almost duplicated tests, while the benefit i"
- **#59221** ([ROCm][Perf][GLM-5.3-Flash] Enable AITER fused shared experts on gfx950) — `model.py`: "Suggestion : can we limit this further? ROCm/aiter#5735 tunes TP4 and TP8 so I think we'd want to turn it off an any other case (e.g. with DP or EP). See the DSv4 code for a good example"
- **#55043** ([ROCm][Kimi-K3] Fuse MLA sigmoid-mul with per-token FP8 for o_proj) — `sigmoid_mul_fp8_per_token.py`: "Question : not sure how common k > 8192 is, but in that case is it still better to run the fused torch version rather than the unfused variant? Some perf numbers here would be great Suggestion : if we do point 1, can we drop the triton check here (also as its already in the maybe_fused_mla_oproj_ptpc). In that case we could also move _sigmoid_mul_fp8_torch to the test file and just have it as a reference implementati"
- **#56098** ([ROCm][Bugfix][Perf] Tune multi-stream shared experts use; wvSplitKrc fixes) — `moe_runner.py`: "Suggestion : Can we run gsm8k on gfx950 with the following cfg and check everything is still ok? vllm serve Qwen/Qwen3.5-35B-A3B \ --max-model-len 4096 \ --data-parallel-size 2 \ --enable-expert-parallel \ --trust-remote-code \ --disable-uvicorn-access-log (this will anyhow be run in nightly ci)"
- **#48249** ([Perf][ROCm] Enable AITER QuickReduce + RMSNorm fusion) — `_aiter_ops.py`: "Question : Can the dispatch be done by AITER instead? AITER's ca has a fused_allreduce_rmsnorm that dispatches to the custom and QR implementations: https://github.com/ROCm/aiter/blob/92258129e9b4360099d94577aac13e0f808af4b4/aiter/dist/device_communicators/communicator_cuda.py#L207"
- **#57947** ([ROCm][Perf] Reach the fused QSA pre-indexer from the AMD path) — `qsa_pre_indexer.py`: "I'm thinking if this and the change above should make us consider if we should duplicate this kernel in an AMD specific file and just make these changes there. Otherwise I believe we'd need to validate we don't regress NV"
- **#55053** ([Bugfix][KV Connector][ROCm] Fix MoRI-IO KV block-offset for speculative decodin) — `test_moriio_connector.py`: "Nit : Many of these enormous comments I suggest we drop. Also no need to meta-reference this PR in comments or reference what the code used to look like. That can be looked up in the subsequent commit :)"
- **#56834** ([ROCm][Perf] Add AITER paged MXFP4 indexing for DeepSeek V4 (gfx950)) — `indexer.py`: "Suggestion : Can we remove these checks and instead submit this PR once the pinned AITER version contains the required kernels and changes? Hence the code should assume aiter already ships what we need"
- **#56226** ([ROCm][Perf] Route softmax MoE top-k through AITER topk_gating) — `envs.py`: "Question : Could we do a wider perf sweep to determine in what domain this kernel is beneficial? If it turns out it's always stronger I don't think we need to give users the ability to turn it off"
- **#53492** ([ROCm][MLA] Enable sparse MLA Gluon kernel from Aiter) — `_aiter_ops.py`: "Question : Which aiter release ships this kernel? Suggestion : Since we will merge after that version is part of the vllm build, can we remove this try-except and assume it'll be there"
- **#55053** ([Bugfix][KV Connector][ROCm] Fix MoRI-IO KV block-offset for speculative decodin) — `moriio_connector.py`: "Suggestion : We can trim this description, and also I'd say remove all the below comments added in the code. No need to add 5 lines of comments for every line of code added :)"
- **#56590** ([Bugfix][ROCm][MoE] Fall back instead of crashing when AITER MoE is requested fo) — `unquantized.py`: "Thought: I wonder if we should instead instead just change this else to an elif envs.is_set("VLLM_ROCM_USE_AITER_MOE") . EDIT: use moe_config instead. That way it would crash as before but only if you explicitly requested AITER MoE through VLLM_ROCM_USE_AITER_MOE if you've only set VLLM_ROCM_USE_AITER , then allow falling back to triton etc. This fallback is alreayd handled at lines 313 onwards in the original code. "
- **#55860** ([ROCm][Perf] Use aiter's tuned decode GEMM for small-M decode under CUDA graphs) — `utils.py`: "Thought : I think this will mean we run a different kernel during warmup vs during capture? I assume we'd want to run the tgemm also during kernel warmup as otherwise we'll capture jit/workspace allocation etc. I think we had a similar issue with the skinny gemms, you can check how that was dealt with"
- **#59280** ([ROCm][Perf] gfx950: use FlyDSL fp8 MQA logits kernel for the sparse indexer pre) — `rocm_aiter_mla_sparse.py`: "Suggestion : can you please check if we break test_unpaged_mqa_logits_preserves_intervals_and_scale_layout which assumes the triton kernel will be the one running on gfx950? I.e. run this on an mi350 node pytest -v -s tests/kernels/attention/test_rocm_triton_attn_dsv4.py"
- **#51065** ([Bugfix][MLA] TritonMLA: fix illegal memory access on causal multi-token decode) — `kv_cache_interface.py`: "Question : Is this change strictly needed for this PR? Looks like it just reverts https://github.com/vllm-project/vllm/pull/55234/changes#diff-f76cdfbf02dacd9dffbfbd0d9ad68a7a6ac0d8aed70834f95a2ae8ccd2e333cb"
- **#58555** ([ROCm][MLA] Route gfx942 DCP causal multi-token verify to Triton split-KV decode) — `rocm_aiter_mla.py`: "I'm aware this isn't too robust as it depends on aiter internals, but it could serve as a temporary fix while limiting blast radius. Happy to hear though if others have better suggestions :)"
- **#51065** ([Bugfix][MLA] TritonMLA: fix illegal memory access on causal multi-token decode) — `triton_mla.py`: "Suggestion : Should we move this to TritonMLAMetadataBuilder._build_decode to avoid doing these ops in every forward? See AiterMLAMetadataBuilder._build_decode for inspiration"
- **#53231** ([ROCm] TurboQuant KV cache support for Gemma 4) — `interface.py`: "Suggestion : if we make the retain auto dtype skip layers here we can avoid having this turboquant-specific code in this file :) Don't think we will want to touch this file"
- **#57705** ([Bugfix][KVConnector] Fix MoRI-IO WRITE-mode requests stranded in WAITING_FOR_RE) — `moriio_connector.py`: "Suggestion : I wonder if we should unify this with _release_write_prefill_blocks. Better to keep this release logic in one place to avoid divergence in the future"
- **#58650** ([ROCm][Attention] Shape-aware split-KV and launch config for Triton DiffKV (gfx9) — `triton_unified_attention_diffkv.py`: "Suggestion : can we check if this will break batch invariance if enabled? Since now depending on the grid size you might running with different number of warps"
- **#58689** ( [ROCm][DSv4.1] Disambiguate ROCm attention class names) — `rocm.py`: "My only thought is if it this now arguably more confusing as the name is no longer aligned with the class name. Otherwise I don't have a strong opinion here :)"
- **#57832** ([ROCm][Model][MiniMax-M3] Context-parallel sparse lightning indexer (Triton, opt) — `envs.py`: "Suggestion : Can we find a heuristic to use to auto-enable this whenever it's useful? Would be great if we an avoid adding more envvars"
- **#53231** ([ROCm] TurboQuant KV cache support for Gemma 4) — `config.py`: "Suggestion : I think there is some utility function we can use now instead to identify the heterogenity, instead of a bare try-execpt"
- **#56834** ([ROCm][Perf] Add AITER paged MXFP4 indexing for DeepSeek V4 (gfx950)) — `sparse_attn_indexer.py`: "Worth considering : Let's think about if we want to add deepseek_v4/amd/ops and put it all of this there, or is that overkill?"
- **#58650** ([ROCm][Attention] Shape-aware split-KV and launch config for Triton DiffKV (gfx9) — `triton_unified_attention_diffkv.py`: "Not sure if this has a perf impact but could be worth invoking this just once and not once per kernel call :)"
- **#56607** ([Bugfix][ROCm][Model] Quantize GLM-5.3-Flash indexer query with platform fp8 dty) — `test_fwht128_quant_fp8_platform_dtype.py`: "Suggestion : Can we use current_platform.is_cuda_alike here? Possibly also worth just running this on rocm"

## Data-derived profile (computed from the PR set, 6 Oct 2026)

- **PRs commented on (not own):** 35
- **Median time to first comment after PR open:** 6.8 days
- **Median time from first comment to approval:** 6.0 days
- **Median time from approval to merge:** 0.7 days
- **Where they comment (paths):** `vllm/model_executor/layers` (14), `vllm/v1/attention` (13), `vllm/distributed/kv_transfer` (11), `vllm/models/glm5next` (5), `tests/kernels/test_fwht128_quant_fp8_platform_dtype.py` (4), `vllm/models/minimax_m3` (4)
- **Files they comment on most:** `moriio_connector.py` (9), `attention.py` (5), `triton_unified_attention_diffkv.py` (5), `test_fwht128_quant_fp8_platform_dtype.py` (4), `_aiter_ops.py` (3), `indexer_context_parallel.py` (3), `rocm.py` (3), `indexer.py` (2)
- **Models named in their comments:** GLM-5 (4), MTP (3), Qwen3 (3), Gemma (2), DSv4 (2), DSV4 (2)
- **Pet models / configs to pre-test:** GLM-5.3-Flash, DeepSeek-V4 sparse MLA, Qwen3.5-35B-A3B with DP2+EP for GSM8K, MoRI-IO connectors
- **Escalates to (people they @mention):** @Rohan138 (2), @tjtanaa (2), @Fangzhou-Ai (2), @AndreasKaratzas (2), @dllehr-amd (1). Pre-tag the first one when your diff is outside this reviewer's area.
- **Their own PRs look like:** Describes validation matrix explicitly (what was and was not run), adds dispatch tests, posts GSM8K before/after for his own bug confirmations (23 own PRs in the set; 16 mention GSM8K, 17 include a serve/bench command)
- **Recent shift (last 60 days vs earlier, share of their comments):** e2e perf 0% -> 10%; blast radius 0% -> 8%; graph/warmup/invariance 0% -> 6%; init not forward 0% -> 5% (based on 93 recent vs 0 earlier comments)

### Approval triggers (what flipped them to LGTM)

- TTFT at 1k and 8k plus throughput
- Sweep wide enough to justify (or drop) the knob
- Existing tests updated for the new kernel choice
- Warmup/capture consistency confirmed

Last thing they asked for right before approving, in real PRs:

- #59221: "I wonder if we should gate already here on ROCm since the _fused_shared_experts_tuned is pure rocm. That also opens the question whether it's soon time to move Glm5NextMoE to glm5next/amd/model.py"

### Things they let slide

- Items he tags 'Nit' or 'Very nit'
- Temporary dependence on AITER internals if labelled and narrow

Comments they themselves marked as nits (do not block on these):

- #55053: "Nit: would this fit better in test_moriio_connector? We already have some similar constructs there"
- #55053: "Nit
: do we need this logged on every request? I'd suggest .debug"
- #55053: "Nit
: Many of these enormous comments I suggest we drop. Also no need to meta-reference this PR in comments or reference what the code used to look like. That c"
- #53231: "Nit
: can this be something other than a string?"
- #53231: "Very nit
: can this be an
elif
on the line above instead?"
- #59412: "Nit
: I think we can trim these comments, the PR itself contains all these references 👍"
