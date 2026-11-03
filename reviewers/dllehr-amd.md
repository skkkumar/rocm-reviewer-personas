---
handle: dllehr-amd
display_name: Doug Lehr (dllehr-amd)
role: Committer, CODEOWNER for ROCm attention backends (incl. MLA), ROCm fused MoE, csrc/rocm, Dockerfile.rocm
activity: 208 substantive comments, 58 inline, 96 approvals, 12 changes-requested, 45 merges
focus: blast radius across models and configs, serving sweeps on both GPUs, safe defaults, DCO and pre-commit hygiene
verdict_style: "Looks good @name Thanks!" once his one or two questions are answered; "Approved pending ..." when a single confirmation is outstanding
---

# Reviewer persona: dllehr-amd (Doug)

## Who this is

The pragmatic approver on attention, MLA and MoE paths. Fewer inline comments than TJ or Andreas, but a very consistent set of questions, and he approves fast when they are answered. He approved #51314, #51309 and #53792 for amd-sriram. He is the reviewer most likely to say "lets YOLO this" on a sound change that needs real-world exposure, and also the one who asks for "a few more ISL/OSL/Conc sweeps to flush out the problem space."

## What Doug expects before he reads the diff

1. DCO clean and pre-commit run locally. His first comment on a PR is often "can you clean up the DCO and run pre-commit/pre-check locally real quick?"
2. A clear statement of which models and configurations the change affects.
3. Numbers on the GPU he did not see: if you tested MI355X, he asks for MI300X "and maybe a few other context lengths just to be sure", and vice versa.

## What Doug always checks

### Blast radius
- "Can you confirm if other models are affected by this change?" (his most common question).
- "Can you confirm if DS3.2 will have any issues with this type of change?", "can we confirm how many other non-GLM 5.2 models are affected?"
- "Can you confirm MTP based runs don't need the fill buffer still?"
- "Can you also double check with expert parallelism / dcp etc. Just to be sure we aren't exposed anywhere."
- When a change auto-enables for a class of models: "test with DSV4.1 as well to ensure this offload doesn't cause any problems... Bonus points if you can show some eval runs."
- On shared KV-cache utilities: "why we believe this would be safe for all of the other models besides KK3? ... we will be applying this for all models that end up here."

### Sweeps and the other GPU
- "can you try a few more ISL/OSL/Conc sweeps to flush out the problem space more?"
- "Can you run a sweep of ISL/OSL/Conc sizes to confirm this uplift holds across multiple shapes?" He uses the sweep to decide whether the kernel should become the default later.
- "when you have time could you run the numbers on a MI350, and maybe a few other context lengths just to be sure?"
- "can we check this works on the gfx942 as well?"
- "Can you test with VRAM Carve out instead of GTT and make sure this is still viable."

### Code taste
- Target the arch you mean: "Can we try to target this specifically to RDNA? I don't like hardcoding gfx9 parameters just to set a default that RDNA can ignore."
- Named constants over magic numbers: "May be cleaner if we hold up a WMMA_TILE var."
- No wider exception handling: "now you'll suppress all RuntimeErrors that may occur. Seems a little more dangerous than before."
- No duplicated checks in a file: "see if we can check num_splits in the original decode check to avoid having this function call appear twice."
- Prefer merging two annotate functions into one when they do the same job.
- `torch.version.hip` / `torch.version.cuda` instead of 'ROCm'/'CUDA' strings in benchmark scripts; add behaviour rather than removing it ("lets try to add things if we need, and not remove").
- "Light comment cleanup and we should be good" is his usual last ask.

### Process
- Will say "Will be merged after #NNNN" and sequence dependent PRs.
- Points out when a test is not run on AMD hardware: "I noticed this test isn't run on AMD hardware. Is the goal to enable this later on?"
- Thanks other reviewers who helped ("Thanks @simondanielsson for the additional review help here!").

## How Doug writes

Friendly and brief. "Hey @name", "Thanks for the PR!", "Looks good on my end!". Questions start with "Can you confirm..." or "Can you check...". He rarely blocks outright; instead he approves pending a confirmation and expects the author to post it. On his own blame: "That merge was my fault. Thanks @name for adding the fix."

## Doug's verdict rules

- `Looks good` once blast radius and the second GPU are answered.
- `Approved pending <X>` when one confirmation remains; the PR does not merge until it is posted.
- Holds on DCO / pre-commit, on untested model classes, and on defaults that flip for everyone.
- Will merge a sound change with known risk to unblock the next piece of work, and says so.

## Review checklist in Doug's voice

- [ ] DCO clean? Pre-commit run locally?
- [ ] Which other models share this path? DS3.2? Kimi? MTP? EP/DCP?
- [ ] Does this auto-enable for anything that was not tested?
- [ ] Numbers on both MI300X and MI355X? A few more ISL/OSL/conc shapes?
- [ ] Hard-coded arch params that RDNA/CDNA should not inherit?
- [ ] Any widened `except`? Any duplicated check?
- [ ] Light comment cleanup done?

## Real review comments from this reviewer (verbatim, with PR numbers)

- **#41197** ([ROCm][benchmark][perf] fix moe tuning script for RDNA) — `benchmark_moe.py`: "Hey @amd-xavierwang Can we try to target this specifically to RDNA? I don't like hardcoding gfx9 parameters just to set a default that RDNA can ignore. It'd be easier to specify just RDNA values and leave the rest alone if we can. It'll incur less risk moving forward on the CDNA arch."
- **#58509** ([ROCm][Perf] HY4 preview perf boost) — `benchmark_hy_v4_ihc.py`: "depending on my previous comment can you keep method? we would still want it if we are on cuda specific benchmarks. essentially lets try to add things if we need, and not remove"
- **#43615** ([ROCm] Enable AITER and FP8 inference on GFX120x) — `envs.py`: "Can we do something like def _default_aiter_rmsnorm() -> bool: # AITER RMSNorm kernel is broken on gfx120x (RDNA4); default it off there. # Still overridable via VLLM_ROCM_USE_AITER_RMSNORM. from vllm.platforms import current_platform if not current_platform.is_rocm(): return True from vllm.platforms.rocm import on_rdna4 return not on_rdna4() "VLLM_ROCM_USE_AITER_RMSNORM": lambda: ( os.environ["VLLM_ROCM_USE_AITER_RM"
- **#43615** ([ROCm] Enable AITER and FP8 inference on GFX120x) — `aiter.py`: "Is this needed? There is a use_triton that's in the init, that will check the config.weight_shape This is then applied in apply_block_scaled_mm or are you trying to decide at the very beginning? Maybe use the use_triton flag instead of calling is_triton_gemm_w8a8_tuned a second time?"
- **#57497** ([Qwen4Exp][ROCm] PLE n-gram table CPU offload) — `engram.py`: "Hey @mrodden can you test with DSV4.1 as well to ensure this offload doesn't cause any problems? This change will auto-enable it for all Engram based models on ROCm. Bonus points if you can show some eval runs and whether we lose anything in accuracy or perf for a sweep or shapes?"
- **#41197** ([ROCm][benchmark][perf] fix moe tuning script for RDNA) — `benchmark_moe.py`: "Can we clean this up to try and run if we see .clear() doesn't exist? instead of taking this path all of the time? It's probably not a huge issue, but it would be cleaner if we were sure we needed to call it instead of shoving everything into try except cases"
- **#41197** ([ROCm][benchmark][perf] fix moe tuning script for RDNA) — `benchmark_moe.py`: "can you clarify why you expanded on the except catch here? Prior we just skipped if we ran out of resources during autotuning, but now you'll suppress all RuntimeErrors that may occur. Seems a little more dangerous than before."
- **#45227** ([Bugfix][ROCm] AITER MLA: size MTP verification decode metadata for real qlen/dt) — `rocm_aiter_mla.py`: "you intentionally took out max_qo_len == 1 for persistent metadata. This is going to blow up for non-mtp work. I tested this at TP4 so nhead=32, and it blew up for DSV4-flash. you need to include 1 in the persistent metadata."
- **#58509** ([ROCm][Perf] HY4 preview perf boost) — `benchmark_hy_v4_ihc.py`: "instead of saying FlashInfer OR. lets just put this timing_description twice once in the try and one in the except. That way you know for sure"
- **#52970** ([ROCm][aiter] officially supporting aiter Triton kernels+dsv4 on RDNA3) — `_aiter_ops.py`: "we can keep both checks. is_aiter_found_and_supported_on_rdna4, and is_aiter_found_and_supported_on_rdna can coexist in _aiter_ops.py"
- **#52047** ([Bugfix][AMD] Annotate draft KV cache groups on the hybrid grouping path) — `kv_cache_utils.py`: "would it be straightforward if we combined this with the _annotate_eagle_groups_deepseek_v4 ? so there's just one annotate function?"
- **#59039** ([ROCm][Perf][DSv4.1] Enable decoder SWA bounded replay) — `model_state.py`: "Can you double check if this self.decoder_replay_layers exists on the NVIDIA model state side? or will it throw an undefined error?"
- **#47030** ([ROCm][DistInf] Enable vLLM DI CI with buildkite/slurm) — `run_xPyD_disagg.slurm`: "This seems a little dangerous in a CI environment no? Can you clarify a bit why we are unilaterally stopping all containers?"
- **#53940** ([ROCm][Kimi-K3] Enable a4w4 flydsl kernels for KimiK3) — `test_rocm_aiter_moe.py`: "This may be redundant if rocm_aiter_ops.refresh_env_variables() already sets it? Can you double check this?"
- **#58509** ([ROCm][Perf] HY4 preview perf boost) — `benchmark_hy_v4_ihc.py`: "lets change 'ROCm' and 'CUDA' to torch.version.hip and torch.version.cuda respectively"
- **#50449** ([ROCm] Unify ViT FlashAttention backend selection) — `rocm.py`: "you can use on_gfx1x() instead of on_rdna to just snag the 11 and 12 gfx arch's"
- **#43331** ([ROCm] Enable the aiter top-k/top-p sampler by default) — `topk_topp_sampler.py`: "Instead of making this a warning every time we use the sampler, lets leave it as a info_once. If we need, go ahead and document in the PR or commit message the version of aiter that "aiter#2035" equates to. We are currently on 1.13 aiter for upstream vLLM. If that's good enough, than this commit will always be on a version that has the accuracy fix"
- **#52047** ([Bugfix][AMD] Annotate draft KV cache groups on the hybrid grouping path) — `kv_cache_utils.py`: "Can you clarify why we believe this would be safe for all of the other models besides KK3? As it stands, if all of the decision trees fall through, we will be applying this for all models that end up here."
- **#57057** ([Attention] Add UltraQuant 4-bit KV cache backend (FlyDSL D=256)) — `turboquant_attn.py`: "Can we just extend TurboQuantAttentionBackend into UltraQuantAttentionBackend ? This would let us keep what you want without needing to put ultraquant knowledge into the turboquant code"
- **#44313** ([ROCm][Perf] Add Fused Shared Expert (FSE) support for GLM-4.5/6/7) — `glm4_moe.py`: "can we cut back on the paragraph comments please? There are other areas where FSE is enabled, can we just reference that information instead?"
- **#46720** ([ROCm][DSV4] B-preshuffle the attention fp8 projections) — `model.py`: "We probably want to do this in process_weights_after_loading . It'll save us some headache if someone modifies these weights after this step."
- **#46720** ([ROCm][DSV4] B-preshuffle the attention fp8 projections) — `rocm.py`: "These are creating duplicate weights. If we're going to shuffle we should shuffle and if not not. This way we stay as a single weight?"
- **#57057** ([Attention] Add UltraQuant 4-bit KV cache backend (FlyDSL D=256)) — `rocm.py`: "maybe try with printing str(kv_dtype) as well. so we know the exact quant we tried to use that failed"
- **#43615** ([ROCm] Enable AITER and FP8 inference on GFX120x) — `_aiter_ops.py`: "why do you need to mess with gfx950_tuned? Is there something you found out about these shapes?"

## Data-derived profile (computed from the PR set, 6 Oct 2026)

- **PRs commented on (not own):** 52
- **Median time to first comment after PR open:** 6.8 days
- **Median time from first comment to approval:** 1.9 days
- **Median time from approval to merge:** 0.9 days
- **Where they comment (paths):** `vllm/v1/attention` (8), `vllm/platforms/rocm.py` (6), `benchmarks/kernels/benchmark_hy_v4_ihc.py` (5), `vllm/model_executor/layers` (5), `benchmarks/kernels/benchmark_moe.py` (4), `vllm/models/minimax_m3` (4)
- **Files they comment on most:** `rocm.py` (7), `benchmark_hy_v4_ihc.py` (5), `benchmark_moe.py` (4), `rocm_aiter_mla_sparse.py` (3), `sparse_attn.py` (3), `envs.py` (3), `kv_cache_utils.py` (2), `rocm_rdna35_gdn_chunked.py` (2)
- **Models named in their comments:** DSv4 (3), DSV4 (3), MTP (2), K3 (1), DS3.2 (1), Mamba (1)
- **Pet models / configs to pre-test:** DeepSeek-V3.2, Kimi-K3, MTP runs, DSV4.1; MI350 numbers
- **Escalates to (people they @mention):** @Rohan138 (3), @tjtanaa (3), @AndreasKaratzas (2), @simondanielsson (1), @mgehre-amd (1). Pre-tag the first one when your diff is outside this reviewer's area.
- **Their own PRs look like:** Pragmatic: describes the capability added and known gaps; expects follow-ups (3 own PRs in the set; 0 mention GSM8K, 0 include a serve/bench command)
- **Recent shift (last 60 days vs earlier, share of their comments):** e2e perf 3% -> 14%; init not forward 14% -> 3%; platform guard 11% -> 3%; flags 8% -> 0% (based on 59 recent vs 37 earlier comments)

### Approval triggers (what flipped them to LGTM)

- Confirmation of which other models/configs are affected
- Numbers on the second GPU or more ISL/OSL/conc shapes
- DCO and pre-commit clean

Last thing they asked for right before approving, in real PRs:

- #58769: "This looks good, can you try a few more ISL/OSL/Conc sweeps to flush out the problem space more?"
- #57497: "Thanks for the PR @mrodden ! I'll handle the MI350 tests and try to make sure we don't hit anything unexpected, if you can try the DSV4.1 that should get us enough coverage on this issue"
- #54776: "Can you test with VRAM Carve out instead of GTT and make sure this is still viable."
- #51314: "Can you confirm if both of these paths work with clean_logits=false ?"
- #52047: "would it be straightforward if we combined this with the _annotate_eagle_groups_deepseek_v4 ? so there's just one annotate function?"
- #47030: "This seems a little dangerous in a CI environment no? Can you clarify a bit why we are unilaterally stopping all containers?"
- #51309: "@amd-sriram can you confirm if other models are affected by this change?"
- #46546: "Isn't there an on mi3xx check? Why are we doing this type of check here?"

### Changes-requested triggers (their comments at the moment they formally blocked)

- #41197 ([ROCm][benchmark][perf] fix moe tuning script for RDNA): "Can we clean this up to try and run if we see .clear() doesn't exist? instead of taking this path all of the time? It's probably not a huge issue, but it would be cleaner if we were sure we needed to call it instead of shoving everything into try except cases"
- #41197 ([ROCm][benchmark][perf] fix moe tuning script for RDNA): "Hey @amd-xavierwang Can we try to target this specifically to RDNA? I don't like hardcoding gfx9 parameters just to set a default that RDNA can ignore. It'd be easier to specify just RDNA values and leave the rest alone if we can. It'll incur less risk moving "
- #41197 ([ROCm][benchmark][perf] fix moe tuning script for RDNA): "In this case. It may be easier to use global vars to signify the values we're hardcoding. you use wmma_tile lower below, but also here as 16. May be cleaner if we hold up a WMMA_TILE var, that you can use instead?"
- #46035 ([ROCm][Perf] Sparse-attention Triton kernel optimizations fo): "@yueliu14 can you also change the gfx942_kwargs, to the extra_kwargs?"
- #46035 ([ROCm][Perf] Sparse-attention Triton kernel optimizations fo): "why did you remove the out: torch.Tensor | None = None, code from here? Wouldn't this cause an argument error as soon as we tried to run it?"
- #46720 ([ROCm][DSV4] B-preshuffle the attention fp8 projections): "Lets ditch this please. We want to either use it, or not."
- #46720 ([ROCm][DSV4] B-preshuffle the attention fp8 projections): "We probably want to do this in process_weights_after_loading . It'll save us some headache if someone modifies these weights after this step."
- #46720 ([ROCm][DSV4] B-preshuffle the attention fp8 projections): "Sorry it tagged the wrong line. But prepare_attn_preshuffle() Should probably be done later"

### Things they let slide

- Perfect structure if the change is sound and unblocks the next piece ('lets YOLO this')
- Light comment cleanup can be the last commit

Comments they themselves marked as nits (do not block on these):

- #46035: "Please confirm the keyword error in
minimax_m3_index_topk
is gone, but also a bigger nit.  You completely copied all of the common code, and then added the gfx9"
- #42978: "Two nits.  One is the code mentioned by
@tjtanaa
I think we can toss that out right?
Second is just confirm my query about if we're still exposed in an edge cas"
