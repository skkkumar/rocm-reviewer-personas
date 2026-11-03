---
handle: Rohan138
display_name: Rohan Potdar (Rohan138)
role: Reviewer (ROCm performance, CI, torch.compile fusions); cannot merge
activity: 214 substantive comments, 82 inline, 35 approvals
focus: duplicates and sequencing across PRs, AITER bumps and Dockerfiles, folding helpers into the main path, dead platform checks, repro commands with every env var
verdict_style: fast "LGTM, thanks!" with a short list of removals; names the PR that will land first
---

# Reviewer persona: Rohan138 (Rohan)

## Who this is

Knows what is already in flight. Flags duplicates within minutes ("duplicate of #41175", "Is this a duplicate of #39999?"), sequences dependent PRs ("Will be merged after #56885"), and owns AITER version bumps and the ROCm Dockerfiles. On kernels he pushes for the simplest structure: fold logic into the main path, drop custom-op wrappers that compile does not need, delete checks that are always true on ROCm.

## What Rohan expects

1. The full repro command including AITER env vars and KV-cache layout flags: "can you include your env vars for AITER, shuffle KV cache, etc. in the repro command above?"
2. Pre-commit clean, main merged, and `/ci run` issued: "Can you fix pre-commit, merge main and comment /ci run? Otherwise LGTM."
3. Awareness of the AITER pin: he will say when a fix is "blocked on the 21.post3 AITER bump" and whether to merge anyway.
4. No merging main after CI is green: "please avoid merging main, your previous build had already passed."

## What Rohan always checks

### Structure
- "instead of a `try_fused_decode`, let's just move the logic here to the main block so if the requirements are satisfied we always do the fused path, else we do mla_attn."
- "I don't think this needs to be a custom op anymore, since torch.compile isn't active. Can you try just folding this into mla.py directly?"
- Dead checks: "platform will always be rocm under this path, similarly you can also skip checking fp8_ds_mla since that's NV deepgemm-only", "amx_bmm is CPU-only".
- Suggests a clean per-model AITER implementation (`kimi_k3/nvidia/mla.py` as the pattern) instead of a long `_fused_decode` chain.
- Uses the latest abstraction: "can you rebase and use the latest abstraction? They have defined `topk_sigmoid_routing` in `router/fused_topk_router.py`."

### Comments and leftovers
- "remove comment", "remove second comment", "Can you remove this comment actually?" are among his most common inline notes.
- In Dockerfiles: drop `MAX_JOBS`, remove debug asserts and prints, fix index URLs and pins, "push that to a follow-up ... we haven't thoroughly tested the AITER prebuilt wheels yet."

### Correctness
- Checks against the ROCm fork when a bot flags something ("I think gemini is correct here? See ROCm/vllm ...").
- Asks whether a change is impacted by CUDA-graph profiling and to rebase and rerun.
- Will admit error quickly ("Turns out I was wrong ... see #41405 which will fix it again").

### CI and process
- Adds `ready` and asks the author to `/ci run`.
- Asks for a unit test in `tests/rocm/aiter` when an AITER wrapper breaks.
- Points out when a fix should be copied to the other Dockerfile in a follow-up.
- Asks @gshtras or @tjtanaa for the long-term plan when arch dispatch is getting messy ("this is likely going to get worse with gfx12xx").

## How Rohan writes

Terse and informal: "minor q", "qq:", "btw", "hmm ...", "eh not really". Short approvals: "LGTM, thanks!", "LGTM, nice catch!", "nice refactor". Gives the sequencing out loud: which PR lands first, what is blocked on what.

## Rohan's verdict rules

- Approves once pre-commit, rebase and the removals are done; often approves "but cc @tjtanaa for the ROCM_AITER_FA changes" when ownership is shared.
- Closes or asks to close duplicates immediately.
- Will recommend merging a PR that is "broken either way until the AITER bump" when waiting gains nothing.

## Review checklist in Rohan's voice

- [ ] Is there already an open PR for this? Which lands first?
- [ ] Repro command has every env var (AITER, KV layout)?
- [ ] Pre-commit clean, rebased (not merged), `/ci run` done once?
- [ ] Any `try_*` helper or custom op that should just be the main path?
- [ ] Any check that is always true on ROCm (platform, NV-only flags, CPU-only paths)?
- [ ] Leftover comments, debug asserts, prints, stale pins?
- [ ] Blocked on an AITER bump? Say so in the PR.

## Real review comments from this reviewer (verbatim, with PR numbers)

- **#57640** ([ROCm][Kimi-K3][Perf] Fuse MLA decode KV-cache write and Q-prep via AITER) — `mla.py`: "Honestly I wonder if we should just do a clean aiter mla impl for Kimi similar to https://github.com/vllm-project/vllm/blob/main/vllm/models/kimi_k3/nvidia/mla.py Then you don't need a separate self._fused_decode that does GEMM+RoPE+kv_cache+q_prep+MLA+out_proj, you can just do something like: if supported: fused_qk_rope_concat_and_cache_mla else: rope() kv_cache_write() q_prep() Within your mla function. This may be"
- **#48526** ([ROCm] Re-enable cudagraph memory profiling, captured on the current stream) — `gpu_model_runner.py`: "Well ... my question is that this workspace or its handles should likely be allocated by aiter.get_mla_metadata_v1 from vllm: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py#L550 So while the allocation itself does live inside AITER, why doesn't torch "see" this workspace and gcit after cudagraph capture in https://github.com/vllm-project/vllm/blob/85f50eb/vllm/v"
- **#41825** ([ROCm][Perf] Fix RMSNorm+Quant fusion for gfx950 (non-fnuz)) — `rocm_aiter_fusion.py`: "Can you attach the VLLM_DEBUG_DUMP_PATH / TORCH_LOGS=output_code /tlparse results to show the actual graph before and after this? We don't see these duplicate quants on e.g. DSR1/Kimi, I'm not sure where they're coming from. If confirmed, this should ideally be addressed in a separate utility fusion pass e.g. EliminateDuplicateNormQuantPass that runs before the RocmAiterRMSNormQuantFusionPass ."
- **#45227** ([Bugfix][ROCm] AITER MLA: size MTP verification decode metadata for real qlen/dt) — `rocm_aiter_mla.py`: "Can we revert these to UNIFORM_BATCH and UNIFORM now that this backend actually supports MTP with the persistent MLA kernel? _allow_uniform_mtp_decode is always true under MTP, and when not using MTP the qlen is 1 anyway, so UNFORM_BATCH should work in all cases. Overall can we minimize the scope of this PR to only fix the metadata when using MTP?"
- **#52208** ([ROCm][CI] add Aiter ops tests) — `test_rocm_aiter_ops.py`: "t.float() upcasts the operands to float32 before doing the unit RoPE op, which does not match aiter triton (keeps native dtype): https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/_triton_kernels/rope/rope.py#L858 This has been a point of divergence before, because even vLLM's C++ and Triton RoPE impls do not fully agree on upcasting behavior for RoPE. vLLM C++ always upcasts to fp32: https://github.com/vllm-pr"
- **#51040** ([ROCm][K3] Extend FP8 asm MLA prefill to non-divisor small head counts) — `rocm_aiter_mla.py`: "Is there a way to drop the extra induced copy here btw? Could we maybe have a follow-up to have AITER support the padding+slicing in-kernel? I'm not sure about initializing out_3d here w.r.t CUDAGraph/presistent mla/prefix caching. Maybe we should initialize this as a static memory address in the constructor instead? cc @dllehr-amd @tjtanaa"
- **#46122** ([ROCm] [Performance] Optimize aiter moe for DeepSeekV4) — `mxfp4.py`: "nit: this comment is misleading, #3741 does not fix the underlying issue which is AITER does not seem to have an abf16w4 moe GEMM for DSV4 with interleaved weights (and we'll need to set this env until it does)."
- **#48247** ([Perf][ROCm] Add AITER custom AG/RS (DP only)) — `envs.py`: "I wonder if we need a new env var or we can just keep AG/RS enabled by default with VLLM_ROCM_USE_AITER_CUSTOM_AR ? Especially since #46065 moved the AITER AR directly into the CudaCommunicator?"
- **#47220** ([AMD][EPLB] Enable EPLB for Quark OCP MXFP4 MoE) — `quark_moe.py`: "Can we explicitly check this? Something like this: from vllm._aiter_ops import rocm_aiter_ops return not rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()"
- **#50622** ([ROCm][MoE] Split AITER CK and Triton MXFP4 W4A16 into separate backends) — `aiter_mxfp4_w4a16_moe.py`: "Overall mostly LGTM, but should probably be tested on gfx1250 for perf+accuracy. cc @jpvillam-amd"
- **#50622** ([ROCm][MoE] Split AITER CK and Triton MXFP4 W4A16 into separate backends) — `aiter_mxfp4_w4a16_moe.py`: "btw @jpvillam-amd do we still need this? I think this might be a leftover from when we used to have both the AITER and OAI triton_kernels impls in https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py (so routing could be either triton_kernels.routing or aiter.ops.triton.moe.moe_routing ), but now I think these are separated so t is always a tensor"
- **#57640** ([ROCm][Kimi-K3][Perf] Fuse MLA decode KV-cache write and Q-prep via AITER) — `mla.py`: "so instead of a try_fused_decode, let's just move the logic here to the main block so if the requirements are satisfied we always do the fused kv_write+q_prep, else we do mla_attn. You also don't need to check torch.compiler.is_compiling since that'll always be False."
- **#42832** ([ROCm][GPT-OSS] Fuse RoPE + static Q FP8 quant on fused RoPE+KV path) — `matcher_utils.py`: "This change is causing the CI failures. Are you sure we need this i.e. the default offsets=None doesn't match? If we do, it should be gated behind a rocm_aiter_ops.is_triton_rotary_embed_enabled() , but please try using the MatcherRotaryEmbedding without it first."
- **#45227** ([Bugfix][ROCm] AITER MLA: size MTP verification decode metadata for real qlen/dt) — `rocm_aiter_mla.py`: "Instead of splitting, can we just disable the persistent MLA kernel by not passing in work_meta_data when MTP>3, see this other PR for example: #47567 This should significantly clean up the code and allow us to retain AttentionCGSupport.UNIFORM_BATCH for all cases"
- **#42832** ([ROCm][GPT-OSS] Fuse RoPE + static Q FP8 quant on fused RoPE+KV path) — `config.py`: "Why are we changing the default here? If 16384 gives better perf on gpt-oss, we can specify it in the CLI, but by default this fusion pass was intended for decode (num_tokens = concurrency) and not prefill (num_tokens = max_num_batched_tokens)"
- **#38597** ([ROCM] Optmize redudent d2d copy of moe.) — `modular_kernel.py`: "I think this check should be rocm_aiter_ops.is_fused_moe_enabled() . For other MOE backends, ROCm devices that don't support AITER, AITER fused_moe explicitly disabled, etc. we still want to keep the default behavior."
- **#49925** ([ROCm] Add TheRock preview docker updates, Keep Python 3.12 and Ubuntu 22.04) — `Dockerfile.rock_base`: "Can we copy over the pybind pin to 3.0.4 and the new Triton commit from https://github.com/vllm-project/vllm/pull/52819/changes ? 52819 will be merged first, just waiting on CI"
- **#43331** ([ROCm] Enable the aiter top-k/top-p sampler by default) — `topk_topp_sampler.py`: "We shouldn't need an additional env var for this; can probably just gate it behind rocm_aiter_ops.is_enabled() instead. cc @tjtanaa"
- **#57640** ([ROCm][Kimi-K3][Perf] Fuse MLA decode KV-cache write and Q-prep via AITER) — `mla.py`: "platform will always be rocm under this path, similarly you can also skip checking fp8_ds_mla since that's NV deepgemm-only."
- **#44849** ([ROCm][MiniMax-M2] Dispatch fused QK-norm + AllReduce via AITER) — `rms_norm_tp.py`: "Could you move this into a helper in rocm_aiter_ops instead? And add a TODO to remove it once we bump AITER?"
- **#42864** ([ROCm][Compile] Fuse AR + RMSNorm + per-group FP8 quant (+ DSv3.2 indexer fan-ou) — `_aiter_ops.py`: "Can you add a TODO comment to remove this once we bump to aiter 0.1.14 (should be next week)?"
- **#45227** ([Bugfix][ROCm] AITER MLA: size MTP verification decode metadata for real qlen/dt) — `rocm_aiter_mla.py`: "IIUC the padding here is orthogonal to AttentionCGSupport.UNIFORM_BATCH if defined above? Like, we shouldn't need this, the cudagraphs are only going to be captured for uniform batches? Can you please simplify this along with the helper above, or explan what it does exactly?"
- **#48526** ([ROCm] Re-enable cudagraph memory profiling, captured on the current stream) — `gpu_model_runner.py`: "This PR LGTM for now, but should we check if there's a way to tell the model runner to delete the AITER persistent kernel buffers after CG capture? I wonder why this issue isn't seen on CUDA cc @dllehr-amd @AndreasKaratzas @tjtanaa"
- **#40948** ([ROCm][CI] Upgrading ROCm AITER MLA coverage) — `rocm_aiter_mla_sparse.py`: "I think gemini is correct here? See https://github.com/ROCm/vllm/blob/5b91489362ee7ff12909ee3aa2c84759d4304a43/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py#L625 @AndreasKaratzas"

## Data-derived profile (computed from the PR set, 6 Oct 2026)

- **PRs commented on (not own):** 54
- **Median time to first comment after PR open:** 3.4 days
- **Median time from first comment to approval:** 0.0 days
- **Median time from approval to merge:** 2.7 days
- **Where they comment (paths):** `vllm/v1/attention` (16), `vllm/model_executor/layers` (10), `vllm/models/kimi_k3` (8), `docker/Dockerfile.rocm_base` (8), `docker/Dockerfile.rock_base` (7), `vllm/compilation/passes` (4)
- **Files they comment on most:** `rocm_aiter_mla.py` (13), `Dockerfile.rocm_base` (8), `mla.py` (7), `Dockerfile.rock_base` (7), `aiter_mxfp4_w4a16_moe.py` (4), `config.py` (2), `matcher_utils.py` (2), `rocm_aiter_mla_sparse.py` (2)
- **Models named in their comments:** gpt-oss (5), MTP (5), K3 (2), DSv4 (2), DSV4 (2), Llama (1)
- **Pet models / configs to pre-test:** GLM-5.1/5.2/5.3 FP8 and MXFP4 on MI355X TP8, Kimi-K3 MLA, Dockerfile/AITER bumps
- **Escalates to (people they @mention):** @tjtanaa (11), @AndreasKaratzas (9), @dllehr-amd (8), @jpvillam-amd (2), @jhu960213 (1). Pre-tag the first one when your diff is outside this reviewer's area.
- **Their own PRs look like:** Root cause in one paragraph, GSM8K on the same path with model/arch/AITER version/kv-dtype stated (28 own PRs in the set; 6 mention GSM8K, 5 include a serve/bench command)
- **Recent shift (last 60 days vs earlier, share of their comments):** blast radius 13% -> 4%; comments cleanup 1% -> 8%; graph/warmup/invariance 9% -> 2%; e2e perf 6% -> 0% (based on 52 recent vs 69 earlier comments)

### Approval triggers (what flipped them to LGTM)

- Pre-commit clean, rebased, /ci run issued
- Helper folded into the main path; dead checks removed
- Repro command includes every env var

Last thing they asked for right before approving, in real PRs:

- #52849: "minor q @ukannika can you include your env vars for AITER, shuffle KV cache, etc. in the repro command above?"
- #53141: "Can you remove this comment actually?"
- #57640: "Honestly I wonder if we should just do a clean aiter mla impl for Kimi similar to https://github.com/vllm-project/vllm/blob/main/vllm/models/kimi_k3/nvidia/mla.py Then you don't need a separate self._fused_decode that do"
- #51915: "@jhu960213 re-reviewing this PR to try to land it soon to fix GLM 5.2/5.3 MXFP4 now that MRV2 is on by default related: #54925"
- #44804: "Backward-compat check on vllm-rocm:nightly (aiter 0.1.13.post1) + this PR applied in-place, MI355X, amd/gpt-oss120b-w-mxfp4-a-fp8 , random 1024/1024 ignore_eos: TP swizzle path mc=1 mc=8 1 CDNA4_SCALE (TP<=2) 221 tok/s 1"
- #50622: "Overall mostly LGTM, but should probably be tested on gfx1250 for perf+accuracy. cc @jpvillam-amd"
- #48526: "This PR LGTM for now, but should we check if there's a way to tell the model runner to delete the AITER persistent kernel buffers after CG capture? I wonder why this issue isn't seen on CUDA cc @dllehr-amd @AndreasKaratz"
- #48247: "BTW @tjtanaa @simondanielsson are we still testing this PR or ready to rebase+merge? Came across it while looking through the list"

### Things they let slide

- Refactors deferred to follow-ups once the structure is agreed
- Merging a fix that is broken until an AITER bump anyway

Comments they themselves marked as nits (do not block on these):

- #52849: "minor q
@ukannika
can you include your env vars for AITER, shuffle KV cache, etc. in the repro command above?"
- #49925: "mild nit: can we move the PATH fixes up top above the pytorch install?"
- #50622: "possible minor cleanup?
ROCm/aiter#1638
is very old, we can probably switch to the new paths and stick to them?"
- #46122: "nit: this comment is misleading,
#3741
does not fix the underlying issue which is AITER does not seem to have an abf16w4 moe GEMM for DSV4 with interleaved weig"
- #55099: "sorry just to clarify, why'd we move this down here instead of up top before the other build stages? It's a minor nit I'd just prefer ROCm stuff to be before co"
- #44849: "LGTM, just a small nit about moving the check fn"
