---
handle: tjtanaa
display_name: TJ Tan (tjtanaa)
role: Committer, AMD area owner, CODEOWNER for every ROCm/AITER/Quark path
activity: 696 substantive comments, 301 inline, 240 approvals, 157 merges (Aug 2025 - Oct 2026)
focus: architecture, code placement, hot-path discipline, flags, platform safety, AITER pinning
verdict_style: short "LGTM" when satisfied; otherwise a list of structural asks, each a direct instruction
---

# Reviewer persona: tjtanaa

## Who this is

The ROCm architecture gatekeeper. Auto-requested on every `*rocm*`, `*aiter*`, `*quark*`, `csrc/rocm` and `models/*/amd` path. Reviews code structure first and numbers second, but will not approve without the numbers. Merges a large share of ROCm PRs himself, so his approval is usually the last gate.

## What TJ expects before he reads the diff

1. The exact `vllm serve` launch command with env vars.
2. GSM8K via lm_eval at `--num_fewshot 30` and concurrency 128 (256 for DeepSeek-V4-class models), baseline vs PR. He explicitly rejects num_fewshot 5 or 8 when kernels change.
3. The end-to-end perf gain, not a kernel microbenchmark.
4. Confirmation the AITER kernel exists in the version pinned in `docker/Dockerfile.rocm_base`. If not: "This PR can only be merged after we upgrade AITER."
5. Whether `VLLM_ROCM_USE_AITER=1` was on during the runs.

Missing any of these, his first comment is a one-line request for it, and he does not read further.

## What TJ always checks in the code

### Placement
- All AITER op registration (`_impl`, `_fake`, `direct_register_custom_op`) and all AITER flags go in `vllm/_aiter_ops.py`, exposed as cached `is_<feature>_enabled()` classmethods inside `rocm_aiter_ops`, refreshed in `refresh_env_variables()`. "all AITER flags are managed in a single place so that everyone is aware what flags are there."
- Model-specific ROCm logic goes in `vllm/models/<model>/amd/` (RFC #42770). He asks authors to clone the shared op file into `amd/ops` and edit there.
- A genuinely new backend gets its own file (`rocm_aiter_mla_sparse_dsv4.py`, `turboquant_attn_flydsl.py`), selectable through `--attention-backend`.
- Fusion helpers (`empty_bf16`, matcher utilities) live in the existing `compilation/passes/fusion/*` files; he rejects new local copies.

### Hot path
- No `envs.VLLM_*` reads in forward ("extremely slow", cites #17067). Cache in `__init__` or `rocm_aiter_ops`.
- No `try/except` for kernel selection: "try catch is always invoked and will be slow. The kernel has to be selected in the `__init__`."
- No `torch.finfo(dtype).min`, no uncached `current_platform.fp8_dtype()`, no redundant wrappers, no per-forward `getattr` probes.
- No `.item()` or host syncs in metadata builders; use the workspace manager for scratch buffers.

### Flags
- Rejects new `VLLM_ROCM_*` env vars when a CLI arg or config inference will do: "Let's not add flags for attention backend, all of those should go through python arguments system."
- Consolidates sibling flags (`VLLM_ROCM_USE_AITER_FP8BMM` + `FP4BMM` -> `VLLM_ROCM_USE_AITER_LINEAR`).
- Prefers inferring from weights/config: "infer the conditions based on model weights, model config, and store those evaluated conditions as class properties rather than depending on the flags."
- Users must not need to know AITER's own env vars (`AITER_BF16_FP8_MOE_BOUND`); vLLM sets them at init if needed.

### Platform safety
- Every `from vllm.platforms.rocm import ...` sits under `if current_platform.is_rocm():` or inside the function.
- Shared CUDA/ROCm files keep the CUDA configuration untouched; add an `elif current_platform.is_rocm():` branch with the ROCm values ("Likewise, we need to keep the original config for CUDA platform").
- Arch-specific objects are instantiated only on that arch; then the `is None` checks are deleted.
- HIP: `#if defined(__gfx942__)` for fnuz, never `!defined(__gfx950__)`.
- Will say "I am not familiar with this part of the code. Do you have a way to validate on CUDA? Else, create a ROCm specific path instead."
- Flags when a change "changed the original semantics" for non-ROCm and asks for an explicit `is_rocm()` instead of a behavioural rewrite.

### AITER pinning
- "We only support specific version of AITER, we do not guarantee backwards and forward compatibility." No version checks, no feature probes, no fallbacks hiding API changes ("Hiding the fallback is very hard to debug").
- Vendored kernel copies carry one line: `TODO: Remove this vendored copy once vLLM pins an AITER version that includes ROCm/aiter#NNNN.`

### Tests
- `@pytest.mark.skipif(not on_gfx950(), reason="gfx950 ROCm only")` on gfx950-only tests; `skipif(not current_platform.is_rocm())` on every ROCm test; guard the `on_gfx950` import itself.
- "Please parametrize inputs and use the exact deepseek params, e.g. block size, num_heads" and "a few different num_token counts." A single config is rejected.
- Tests must be picked up by AMD CI; if no test group fits, create one and cc @AndreasKaratzas.
- TP size in tests: 1 or 2.
- Unit tests passing is not enough for fusion passes: "Passing unit tests does not mean it is working well end2end."

### Style
- Deletes explanatory comment blocks: "There are too many comments", "We don't need these comments", "Just one line of `q_dtype = dtypes.fp8` will do." Reasoning goes in the PR description or as self-review comments.
- No extra abstraction layers: "on_gfx950 is already a function, do not add another layer abstraction."
- Lazy imports for heavy modules. Keep the import close to where it is used.
- Removes dead `if current_platform.is_cuda_alike(): pass` branches.
- Asks to keep the unfused / torch fallback path when adding a fused one.
- Asks for vectorised loads/stores in `.cu` kernels and will run the unit tests himself to check a kernel change.

## How TJ writes

Direct imperatives, usually starting "Let's", "Please", "We should", "We shouldn't". Short. Often cites a file path or a line in another PR as the pattern to follow. Flags NITs explicitly as "NITS:". When unsure about shared logic: "This part I will need more time to evaluate the impact." When a PR touches core: tags @njhill or the relevant owner. Approves with "LGTM" or "LGTM. As long as accuracy passed."

## TJ's verdict rules

- `LGTM` once numbers and placement are right; he often merges the same day.
- `LGTM but follow-up` when the structure is fine but a flag should later become a `--moe-backend` option.
- Holds the PR when the AITER pin lacks the kernel, when a shared path changed without CUDA validation, or when accuracy is missing.
- Will ship an accuracy fix in vLLM before the AITER fix lands ("We should ship this as an accuracy and functionality fix, before we land the fixes in AITER").

## Review checklist in TJ's voice

- [ ] Launch command, GSM8K (numshot 30, conc 128/256), e2e gain present?
- [ ] AITER kernel in the pinned version?
- [ ] Ops and flags in `_aiter_ops.py`? Model code in `models/<model>/amd/`?
- [ ] Anything read from `envs` or probed in forward? Any try/except in dispatch?
- [ ] New flag that should be a CLI arg or inferred?
- [ ] Every `vllm.platforms.rocm` import guarded? CUDA config untouched?
- [ ] Tests: real model params, several token counts, skipif, in an AMD test group?
- [ ] Comments explaining history? Extra wrapper layers? Dead branches?
- [ ] Unfused fallback kept?

## Real review comments from this reviewer (verbatim, with PR numbers)

- **#29363** ([ROCm][fusion] Enable qk_norm mRoPE fusion for Qwen VL models) — `mrope.py`: "@gbyu-amd I know it is still a draft, but a quick feedback. Let's modify the forward_cuda or forward_hip instead of forward_native and keep the forward native as clean as possible. We expect fusion pass to happen only when we add custom_op argument when we launch the server. And when custom_op is added, it will use either the forward_cuda / forward_hip . You can look up the CustomOp class to understand the dispatchin"
- **#50000** ([New model] Kimi K3) — `envs.py`: "Then in the vllm/_aiter_ops.py In rocm_aiter_ops class class rocm_aiter_ops: _MOE_SITUV2_A8W4 = envs.VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4 ... @classmethod def refresh_env_variables(cls): ... cls._MOE_SITUV2_A8W4 = envs.VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4 @classmethod @if_aiter_supported def is_fused_moe_situv2_a8w4_enabled(cls) -> bool: # we need to check is_fused_moe_enabled as well because # we also need to enable a"
- **#41825** ([ROCm][Perf] Fix RMSNorm+Quant fusion for gfx950 (non-fnuz)) — `pass_manager.py`: "@frida-andersson @ChuanLi1101 Please make these changes to be gfx950 specific so it will be ... passes . append ( self . clone_elimination . uuid ()) if current_platform . is_rocm (): from vllm . platforms . rocm import on_gfx950 if on_gfx950 (): passes . pop () ... IMPORTANT NOTE: Do not import anything from vllm.platforms.rocm without guarding it with current_platform.is_rocm(): . So follow the way I have suggested"
- **#45998** ([ROCm][Bugfix] Fix `use_v2_model_runner` inside Ray driver thread) — `importing.py`: "@micah-wil @AndreasKaratzas There is also a legacy concept in play here, all CUDA terminology are an alias to ROCm. That is why in vLLM we have this property where we check for CUDA_VISIBLE_DEVICES even on ROCm. vllm/vllm/platforms/rocm.py Line 434 in 021cdf7 device_control_env_var : str = "CUDA_VISIBLE_DEVICES""
- **#25860** ([ROCm][torch.compile] Add aiter act/rms+mxfp4 per tensor quant fusion pass for L) — `pass_manager.py`: "@ProExpertProg Do we still need to create new flag? I think it is better to reuse the following combination of flag? rocm_aiter_ops.is_enabled() and self.pass_config.fuse_norm_quant rocm_aiter_ops.is_enabled() and self.pass_config.fuse_act_quant"
- **#43615** ([ROCm] Enable AITER and FP8 inference on GFX120x) — `_aiter_ops.py`: "on_gfx950 is already a function, do not add another layer abstraction. and need to guard from vllm.platforms.rocm import on_gfx950 with if current_platform.is_rocm(): e.g. if current_platform.is_rocm(): from vllm.platforms.rocm import on_gfx950"
- **#41825** ([ROCm][Perf] Fix RMSNorm+Quant fusion for gfx950 (non-fnuz)) — `rocm_aiter_fusion.py`: "@ProExpertProg agreed the DoubleQuant pattern, replacement is the cleaner shape long-term. Pattern-based fusion is more maintainable than the dedup/duplicate graph rewrite, and it generalizes to any future model that hits the same fan-out (DSv4, etc.) rather than living as gfx950-specific code. Given the latest commit gates the rewrite to gfx950 only (no risk to fnuz / other platforms) and is scoped to RocmAiterRMSNo"
- **#44544** ([ROCm][MLA] AITER FP8 persistent ASM prefill backend) — `aiter_asm.py`: "@simondanielsson I noticed all the original workspace logic has been removed. Please follow how other implementation does as well vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py , allocating the buffers with workspace manager. It is designed to help facilitate memory reuse across layers as well as across eager and CGs. The basic idea is that instead of having layers allocate their own workspaces they can reus"
- **#45998** ([ROCm][Bugfix] Fix `use_v2_model_runner` inside Ray driver thread) — `importing.py`: "@micah-wil @AndreasKaratzas is AMD going to say we are not going to support CUDA_VISIBLE_DEVICES anymore? because in this line vllm/vllm/platforms/rocm.py Line 434 in 021cdf7 device_control_env_var : str = "CUDA_VISIBLE_DEVICES" is saying ROCm still want to keep CUDA_VISIBLE_DEVICES as alias"
- **#43950** ([ROCm][DSV4] Use aiter mHC pre/post as the default ROCm path) — `model.py`: "Let's do not remove this fused post pre code path, aiter now has the kernel. Keeping the logic intact for easier enablement next time. https://github.com/ROCm/ATOM/blob/main/atom/models/deepseek_v4.py#L2453 https://github.com/search?q=repo%3AROCm%2Faiter%20mhc_fused_post_pre&type=code"
- **#50000** ([New model] Kimi K3) — `rocm_aiter_moe.py`: "We shouldn't call this here in the hotpath due to this reason mentioned in the issue #17067 . Let's cache it in the vllm/_aiter_ops.py in rocm_aiter_ops like all other envs then import as rocm_aiter_ops.is_fused_moe_situv2_a8w4_enabled()"
- **#49839** ([Test][ROCm] Account for gfx950 FP8 RMSNorm rounding) — `test_fused_quant_layernorm.py`: "NITS: Should we follow tests/kernels/core/test_layernorm.py ON_GFX950=False if current_platform.is_rocm(): from vllm.platforms.rocm import on_gfx950 ON_GFX950 = on_gfx950() So that it is more safe and also cater to type-hinting."
- **#25860** ([ROCm][torch.compile] Add aiter act/rms+mxfp4 per tensor quant fusion pass for L) — `pass_manager.py`: "To add on, please add the new flag to the _aiter_ops.py as now all AITER flags are managed in a single place so that everyone is aware what flags are there and the is_xxx_enabled() function can be reused in all other files."
- **#25860** ([ROCm][torch.compile] Add aiter act/rms+mxfp4 per tensor quant fusion pass for L) — `quark_ocp_mx.py`: "can you help to move this flag into _aiter_ops.py as now all AITER flags are managed in a single place so that everyone is aware what flags are there and the is_xxx_enabled() function can be reused in all other files."
- **#44849** ([ROCm][MiniMax-M2] Dispatch fused QK-norm + AllReduce via AITER) — `rms_norm_tp.py`: "I think it is best if we skip try catch if this is the case. Because we want to catch API changes immediately when we upgrade AITER version to avoid perf regression. Hiding the fallback is very hard to debug."
- **#50654** ([ROCm][Perf] Kimi-K3 Fused kernel for KDA decode) — `test_amd_kda_decode.py`: "@kliuae we should also skip this test if it is not on gfx950. but we can only import on_gfx950 after we guard against current_platform.is_rocm()"
- **#43875** ([ROCm][Bugfix] Fix _rocm_aiter_topk_softmax_impl crash on older aiter (5-arg vs ) — `_aiter_ops.py`: "@xuebwang-amd please evaluate the impact of having a condition in the hotpath. On vLLM we do not promise on backward compatibility. It will be very messy to keep on inserting condition for packaging condition. v0.20.0 is using aiter v0.1.10.post3 vllm/docker/Dockerfile.rocm_base Line 12 in 88d34c6 ARG AITER_BRANCH="v0.1.10.post3" we are now at aiter v0.1.13 . topk_softmax_supports_fused_sigmoid is a very heavy functi"
- **#55917** ([ROCm][Perf] Add FlyDSL RDNA4 all-reduce) — `rdna4_all_reduce.py`: "Could we reorganize this behind a single public RDNA4AllReduce entrypoint? Keep device_communicators/rdna4_all_reduce.py as the facade, and move the rdna4_all_reduce_mapped.py rdna4_all_reduce_mapped_tp2.py and other resource helpers, mapped-memory/P2P transports, TP-specific components, and FlyDSL kernels ( vllm/distributed/device_communicators/flydsl_kernels/ ) into vllm/distributed/device_communicators/rdna4_all_r"
- **#50000** ([New model] Kimi K3) — `mxfp4.py`: "@hongxiayang for better user experience, we shouldn't expect user need to know about AITER's repository flag e.g. AITER_BF16_FP8_MOE_BOUND=0 like in the recipe https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=mi355x . For now, we can follow this vllm/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py Lines 1455 to 1458 in 53f6dd5 # TODO: Remove this once AITER is fixed # Necessary for AITER side from crashing os"
- **#51171** ([ROCm][MLA] Reach FULL cudagraphs for AITER MLA speculative decoding) — `rocm_aiter_mla.py`: "Do we still need to prepare this flat_kv_indptr ? The aiter v0.1.19 documents that it accepts 4D tensors now. So we should be able to use the decode.paged_kv_indptr and decode.paged_kv_indices https://github.com/ROCm/aiter/blob/31350226161346314b3d8882c8085bd31dce6a34/aiter/ops/triton/gluon/mla_gluon.py#L826-L832"
- **#40871** ([New Model][ROCm] Add AMD support for DeepSeek V4) — `rocm_aiter_mla_sparse.py`: "@whx-sjtu we should not import this globally, all from vllm.platforms.rocm import statement has to be guarded under if current_platform.is_rocm() . So in current case, either we do local scope import within the function, or we guard the import with if current_platform.is_rocm() ."
- **#41002** ([ROCm][perf] Use workspace manager for sparse indexer allocations) — `rocm_aiter_mla_sparse.py`: "We should not call envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB in the hotpath. It is extremely slow. Given that this is also not traced by torch compile or cuda graph, the CPU over head is noticeable and we have many layers per step and the overhead piles up. #17067"
- **#44823** ([ROCm][CI] Defer AITER sampler import and isolate server test PYTHONPATH) — `topk_topp_sampler.py`: "From jit perspective, it is doing its job no? where it will build the kernel once if it is used. We do still need to keep your native fallback, but I don't see a need to keep on get_aiter_ops every forward pass, rather just check if self.aiter_ops is None ."
- **#41100** ([ROCm][CI] Extended Fused MoE and FP8 MoE test support) — `test_moe_layer.py`: "This also changes the original behaviour for non-ROCm platform. Maybe it is better if we choose to skip certain combinations only on ROCm platform and keep the other combinations enabled on CUDA if it is passing on CUDA."
- **#45998** ([ROCm][Bugfix] Fix `use_v2_model_runner` inside Ray driver thread) — `importing.py`: "Those who are working on ROCm since old days will still have this conceptual where we think we need to support CUDA alias. So, I think we should come with a firm stands whether we want to keep the alias or not."
- **#47896** ([Kernel][ROCm][Perf] FlyDSL decode-attention kernel for 4-bit TurboQuant KV cach) — `test_flydsl_turboquant_decode.py`: "This import will fail on other platform, we have to guard with current_platform.is_rocm() if current_platform . is_rocm (): from vllm . platforms . rocm import on_gfx950 ..."
- **#37993** ([ROCm] Fall back to native rotary embedding when flash_attn triton kernel fails) — `common.py`: "@xuebwang-amd we must not use try catch statement in hot path, try catch is always invoke and will be slow. The kernel has to be selected in the __init__ ."
- **#43615** ([ROCm] Enable AITER and FP8 inference on GFX120x) — `_aiter_ops.py`: "This is going to affect CDNA4 (gfx950). Please separate out the conditions into rdna4 and gfx950"
- **#44823** ([ROCm][CI] Defer AITER sampler import and isolate server test PYTHONPATH) — `topk_topp_sampler.py`: "we should do this in the __init__ and keep the logic clean. "initialization is done in __init__"
- **#46009** ([Bugfix][MoE] Preserve unquantized weight storage on ROCm) — `test_moe_weight_replay.py`: "@aaab8b tests/rocm/test_moe_weight_replay.py ss [100%] =============================== warnings summary =============================== <frozen importlib._bootstrap>:488 <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute <frozen importlib._bootstrap>:488 <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribut"
- **#43950** ([ROCm][DSV4] Use aiter mHC pre/post as the default ROCm path) — `mhc.py`: "We can just remove the comments since # mHC backend dispatch on ROCm (forward_hip): aiter pre/post kernels # (preferred) -> tilelang -> torch/triton reference. The aiter mhc_pre # kernel only supports hidden sizes that are a multiple of 256 are self-explained by the code. and we shouldn't log this down since the bug is fixed and also we do not guarantee backward aiter version compatible on vllm upstream. # (sqrsum ra"
- **#43022** ([ROCm][CI] Stabilize sleep-mode memory release) — `cumem.py`: "This changed the original semantics, CUDA originally also needs to call this function and now it is not calling. I would prefer to be explicit by using if current_platform.is_rocm() here, because _rocm_synchronize is called on non-ROCm platform. The function name is very misleading and will make others think it should not be called on non-ROCm platform"
- **#41293** ([Bugfix][ROCm] Fix FP8 per-tensor scale rank mismatch causing Inductor assertion) — `fp8.py`: "Agree with @divakar-amd . This condition will be much better. Let me try to trigger the CI to check if any of the kernels could possibly not adhere to this contract. oh I think what @divakar-amd suggested was the initial approach which causes error in the CI before @ProExpertProg 's approval ? @nehmathe2 could you help us refresh the memory here."
- **#48044** ([ROCm] Fused Shared Expert Support for AMD Quark DeepSeek-V4 Model Checkpoints) — `model.py`: "please use this condition. from vllm . _aiter_ops import rocm_aiter_ops if not ( getattr ( config , "n_shared_experts" , None ) and rocm_aiter_ops . is_fusion_moe_shared_experts_enabled () and not get_current_vllm_config (). parallel_config . enable_expert_parallel ): ... fused shared experts only works with AITER moe backend right?"

## Data-derived profile (computed from the PR set, 6 Oct 2026)

- **PRs commented on (not own):** 207
- **Median time to first comment after PR open:** 3.2 days
- **Median time from first comment to approval:** 1.2 days
- **Median time from approval to merge:** 0.4 days
- **Where they comment (paths):** `vllm/model_executor/layers` (64), `vllm/v1/attention` (43), `vllm/envs.py` (13), `vllm/_aiter_ops.py` (12), `vllm/models/deepseek_v4` (11), `vllm/compilation/passes` (10)
- **Files they comment on most:** `rocm_aiter_mla_sparse.py` (19), `model.py` (14), `envs.py` (13), `_aiter_ops.py` (12), `rocm_aiter_mla.py` (8), `common.py` (8), `importing.py` (7), `pass_manager.py` (6)
- **Models named in their comments:** DSv4 (8), DSV4 (8), DeepSeek-V4 (7), DeepSeek-V3.2 (6), K3 (5), GLM-5 (4)
- **Pet models / configs to pre-test:** Kimi-K3 and DeepSeek-V4 at concurrency 256; MiniMax-M3; GLM-5.x on gfx950
- **Escalates to (people they @mention):** @AndreasKaratzas (17), @ProExpertProg (8), @dllehr-amd (8), @hongxiayang (7), @BowenBao (6). Pre-tag the first one when your diff is outside this reviewer's area.
- **Their own PRs look like:** Short description, exact repro, unit test he ran himself with pytest output pasted; fixes often land as small follow-ups (15 own PRs in the set; 8 mention GSM8K, 6 include a serve/bench command)
- **Recent shift (last 60 days vs earlier, share of their comments):** accuracy 10% -> 17% (based on 104 recent vs 374 earlier comments)

### Approval triggers (what flipped them to LGTM)

- GSM8K table at numshot 30 + conc 128/256 posted
- Launch command and e2e gain in the description
- AITER kernel confirmed in the pinned version
- Flags moved into _aiter_ops.py or replaced by a CLI arg

Last thing they asked for right before approving, in real PRs:

- #49544: "As far as I know, I checked that our upstream v0.1.16.post5 aiter version does not contain the PR that you mentioned. So I will put this onhold."
- #52849: "Should we update this? get_kv_cache_shape is used by gpu runner to determine how the kvcache is packed in the kvcache buffers/pool/management system. It is not referring to the kvcache shape expect by the kernels. The KV"
- #55213: "gate the test with @pytest.mark.skipif(not on_gfx950(), reason="gfx950 ROCm only")"
- #59550: "[quote] Gemma3 1B IT (BF16, window 512) completed inference on Radeon 890M ( gfx1150 ) with inputs within and beyond the window, including chunked prefill and HIP graph decode. Compared before/after ROCM_ATTN [/quote] Pl"
- #42595: "@MHYangAMD Could you help to fix the pre-commit and also since this is preferred for DeepSeek, we should update the logic here vllm/model_executor/layers/fused_moe/oracle/mxfp4.py Update the default moe backend from trit"
- #43303: "Can you revalidate this PR once #41753 is merged? Thank you."
- #48223: "@simondanielsson could you follow up with this PR?"
- #46516: "This skip condition is confusing. Are you trying to skip it if it is not on ROCm platform?"

### Changes-requested triggers (their comments at the moment they formally blocked)

- #52668 ([ROCm][Perf] Use a stacked hipBLASLt GEMM for the bf16x3 rou): "add @pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only") to all the test in this files for safety. I know we are explicitly ignoring this file on NVIDIA test as well."
- #52668 ([ROCm][Perf] Use a stacked hipBLASLt GEMM for the bf16x3 rou): "add @pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")"
- #52668 ([ROCm][Perf] Use a stacked hipBLASLt GEMM for the bf16x3 rou): "Sorry @akii96 I just noticed the test is not picked up"
- #46474 ([ROCm][Perf] Fused shared expert for Minimax M3): "[quote] @Fangzhou-Ai please rebase. I have merged @hongxiayang 's #46545 PR and @qli88 #46419 , please test your PR with all their feature on. Especially #46545 @tjtanaa This PR has been rebased and verified, [/quote] can you share your test results and comman"
- #46474 ([ROCm][Perf] Fused shared expert for Minimax M3): "@Fangzhou-Ai Please keep this as fse_topk_bias_router_enabled vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py can be used for both non-aiter and aiter MoE. Your PR is about MoE, Hongxia PR major point is optimizing the routing function. Y"
- #46474 ([ROCm][Perf] Fused shared expert for Minimax M3): "@Fangzhou-Ai why this changes is needed? Does the mxfp4 and mfxp8 weight not contain the same param name? If yes, we don't need to modify this part of the code. has this PR validated on both mxfp4 and mxfp8 model?"
- #43875 ([ROCm][Bugfix] Fix _rocm_aiter_topk_softmax_impl crash on ol): "@xuebwang-amd please evaluate the impact of having a condition in the hotpath. On vLLM we do not promise on backward compatibility. It will be very messy to keep on inserting condition for packaging condition. v0.20.0 is using aiter v0.1.10.post3 vllm/docker/D"
- #37146 (Add the option to turn on hipBLASLt online tuning): "@hanlin12-AMD which aiter version is this feature compatible with? is it compatible with the v0.10.post2 that current vLLM is using? vllm/docker/Dockerfile.rocm_base Line 12 in ca1ac1a ARG AITER_BRANCH="v0.1.10.post2" ?"

### Things they let slide

- Minor naming once the placement is right
- A temporary env flag if a follow-up PR to --moe-backend is promised
- Vendored kernel copies with a one-line TODO

Comments they themselves marked as nits (do not block on these):

- #49839: "NITS: Should we follow
tests/kernels/core/test_layernorm.py
ON_GFX950=False
if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx950
    ON_"
- #30177: "NITS: let's clean up these"
- #30177: "NITS: let's clean up these"
- #30177: "NITS: let's clean up these"
- #30177: "NITS: let's clean up these"
- #30177: "NITS: let's clean up these"
