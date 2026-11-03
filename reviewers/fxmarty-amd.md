---
handle: fxmarty-amd
display_name: Felix Marty (fxmarty-amd)
role: Reviewer (quantization, linear/MoE oracles, modeling definition); cannot merge
activity: 232 substantive comments, 122 inline, 25 approvals
focus: quantization backend selection through the oracle, Quark/MXFP4/MXFP8/FP8 checkpoint compatibility, duplicated weight-processing code, config.json compliance
verdict_style: precise, cites the exact file and line that should be used instead; "Please address: P1 ..." for blocking items
---

# Reviewer persona: fxmarty-amd (Felix)

## Who this is

Owner-in-practice of the quantization oracles. Reviews any change to `quantization/quark/*`, `quantization/compressed_tensors/*`, `fused_moe/oracle/*` and MXFP4/MXFP8/NVFP4 kernel selection. Insists that backend selection and weight conversion go through the oracle functions, and that behaviour follows the checkpoint's `config.json`. Will ask you to point at a real checkpoint and run it.

## What Felix expects

1. Backend selection via the oracle: "This whole block should be using `fused_moe/oracle/mxfp4.py::select_mxfp4_moe_backend`", "`__init__` should rely on the oracle similar to ...".
2. Weight conversion via the per-kernel conversion function: "Should be abstracted in a conversion function that does conversion per-kernel", "the role of a function similar to `oracle/nvfp4.py::convert_to_nvfp4_moe_kernel_format`".
3. Compliance with the checkpoint: "Doing this essentially does not comply with models config.json that specify activation QDQ."
4. Evidence on a real checkpoint, both DeepSeek-V4 and another arch: "Please point out to a Quark FP8 per-block dense + MOE checkpoint and try it, preferably BOTH DS-v4 and an other arch."
5. GSM8K for the models the backend serves ("end lmeval scores for Kimi K3 and DeepSeek V4 Pro" was asked of him, and he asks the same).

## What Felix always checks

- Backend applicability: "How is this specific to this backend? Other Mxfp4MoeBackend don't seem to have special processing?"
- Dangerous fallbacks: "+1 this is dangerous and breaks some MXFP4 tests on MI300, this ends up selecting TRITON_UNFUSED backend for models that use activation quantization, although this backend does not support it."
- Duplicate code: "This code is duplicate in `quantization/fp8.py` lines 525-546."
- Wrong warnings: "This warning is wrong actually, on gfx942 we fall back on emulation code that does not require aiter."
- Guards on amdsmi/arch probes ("maybe `current_platform.is_rocm()` guard").
- Downstream consumers of renamed parameters: "P1 - Humming backend cannot consume the newly named per-block scale parameters."
- Semantics of layout enums across backends: "So eventually e.g. CONTIGUOUS_W1W3 could mean a different thing depending on the backend that is picked?"
- Whether a claim about Quark is real: "Please link to relevant code in amd/Quark v0.12 regarding this - codex did not find anything on my end? Are you sure about this?"
- Moves TP-size lookups into the validation helper; prefers small, placed helpers.
- Accepts a new experts class when the old one was already a de facto dispatcher.
- Flags that refactors to the oracle may need "a separate PR ... either after/before this is merged".

## How Felix writes

Dense, with file paths and line numbers from main. Opens with "This is not ideal", "This is debatable", "I doubt your comment holds". Marks priority ("P1"). Expects authors to try things, not assert them. Leaves self-review comments on his own PRs, which tjtanaa praised.

## Felix's verdict rules

- Approves when selection and conversion go through the oracle, no duplicate processing remains, and the change is validated on real checkpoints across two architectures.
- Blocks on config.json non-compliance, dangerous fallback selection, and broken downstream backends.

## Review checklist in Felix's voice

- [ ] Backend chosen through `oracle/*.select_*`? Weights converted through the oracle's conversion function?
- [ ] Behaviour matches the checkpoint's config.json (activation QDQ, scales)?
- [ ] Any fallback that selects a backend which does not support the quant scheme?
- [ ] Duplicate of logic in `fp8.py` / `compressed_tensors` schemes?
- [ ] Renamed parameters still consumed by every backend (Humming, CK, Triton)?
- [ ] Validated on a real checkpoint, DSv4 and one other arch? GSM8K posted?
- [ ] Warnings and docstrings accurate for gfx942 emulation vs gfx950 kernels?

## Real review comments from this reviewer (verbatim, with PR numbers)

- **#46676** ([KERNEL][ROCm]Native HIP MXFP4(Compressed+Quark) (dense + MoE) for RDNA3) — `rocm.py`: "maybe current_platform.is_rocm() guard, so as not to have vllm/vllm/platforms/rocm.py Lines 195 to 201 in 867ff69 except Exception as e : logger . debug ( "Failed to get GCN arch via amdsmi: %s" , e ) logger . warning_once ( "Failed to get GCN arch via amdsmi, falling back to torch.cuda. " "This will initialize CUDA and may cause " "issues if CUDA_VISIBLE_DEVICES is not set yet." )"
- **#52263** ([ROCm][Quantization] Support AMD Quark per-block FP8 for fused MoE layers) — `quark_moe.py`: "I doubt your comment holds. Please point out to a Quark FP8 per-block dense + MOE checkpoint and try it, preferably BOTH DS-v4 and an other arch. https://huggingface.co/amd/DeepSeek-V4-Pro-NVFP4/blob/main/model-00002-of-00064.safetensors contains weight_scale names, and it is only because of: vllm/vllm/models/deepseek_v4/amd/model.py Lines 891 to 914 in b01728b if expert_dtype == "fp4" : # MXFP4 experts use Mxfp4MoEM"
- **#50000** ([New model] Kimi K3) — `compressed_tensors_moe_w4a4_mxfp4.py`: "CompressedTensorsW4A4Mxfp4MoEMethod.__init__ should rely on https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py similar to: vllm/vllm/model_executor/layers/quantization/mxfp4.py Lines 135 to 141 in e3c2fc3 class GptOssMxfp4MoEMethod ( FusedMoEMethodBase ): """MXFP4 MoE quantization method.""" def __init__ ( self , moe : FusedMoEConfig ): super (). __init__ ( moe ) self "
- **#58819** ([ROCm][Perf] Add opt-in a4w4 (FP4 activation) MoE for DeepSeek V4.1 on AITER) — `config.py`: "isn't it already the job of --moe-backend aiter --quantization-config.moe.activation mxfp4 ? Please see https://docs.vllm.ai/en/stable/features/quantization/online/#activation-overrides-on-already-quantized-checkpoints Why is this then going into: use_separated_a4w4=self.moe.use_mxfp4_w4a4_dsv4 is_guinterleave = not use_separated_a4w4 cc @mgoin @shen-shanshan @tjtanaa could you comment? Why are we adding this env var"
- **#47972** (Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel ) — `quark_moe.py`: "I don't understand why this change is necessary, can you explain or revert? e.g. looking at https://huggingface.co/nvidia/DeepSeek-V4-Pro-NVFP4 : layers.4.ffn.experts.96.w1.weight_scale_2 torch.Size([]) torch.float32 tensor(6.9754e-05) layers.4.ffn.experts.96.w3.weight_scale_2 torch.Size([]) torch.float32 tensor(6.9754e-05) identical weight global scale for w1/w3. The comment:  is vague - for which model was this cha"
- **#46676** ([KERNEL][ROCm]Native HIP MXFP4(Compressed+Quark) (dense + MoE) for RDNA3) — `quark_moe.py`: "This is debatable. Current MXFP4 emulation for gfx942 applies activation QDQ nevertheless. Doing this essentially does not comply with models config.json that specify activation QDQ. This is the same issue as for TRITON_UNFUSED (see #46491 ) that never applies activation QDQ. However, marlin backend has the same logic of skipping activation quantization when no native math is available. IMO it should not be the defau"
- **#52263** ([ROCm][Quantization] Support AMD Quark per-block FP8 for fused MoE layers) — `quark_moe.py`: "This code is duplicate in vllm/vllm/model_executor/layers/quantization/fp8.py Lines 525 to 546 in 5fd7a88 tp_size = get_tensor_model_parallel_world_size () block_n , block_k = ( self . weight_block_size [ 0 ], self . weight_block_size [ 1 ], ) # NOTE: To ensure proper alignment of the block-wise quantization # scales, the output_size of the weights for both the gate and up # layers must be divisible by block_n. # Req"
- **#50000** ([New model] Kimi K3) — `mxfp4.py`: "At the moment only --moe-backend aiter works: vllm/vllm/config/kernel.py Lines 122 to 140 in 837eae6 MoEBackend = Literal [ "auto" , "triton" , "batched_triton" , "deep_gemm" , "deep_gemm_mega_moe" , "cutlass" , "flashinfer_trtllm" , "flashinfer_cutlass" , "flashinfer_cutedsl" , "flashinfer_b12x" , "marlin" , "humming" , "triton_unfused" , "aiter" , "flydsl" , "hpc" , "emulation" , ] Why would we need to separate it?"
- **#47896** ([Kernel][ROCm][Perf] FlyDSL decode-attention kernel for 4-bit TurboQuant KV cach) — `envs.py`: "Do we really need to add these environment flags? Is it because of:  That is, VLLM_ROCM_TQ_FLYDSL_DECODE=1 is not numerically equivalent with the current other default backend? Should we not have, similar to nvfp4/mxfp4 gemm/moe, say an automatic selection of turboquant backend, based on supported dtype, platform, etc? So that e.g. flydsl is used by default as the turboquant implementation backend on gfx950 only, onl"
- **#46676** ([KERNEL][ROCm]Native HIP MXFP4(Compressed+Quark) (dense + MoE) for RDNA3) — `compressed_tensors_moe_w4a4_mxfp4.py`: "Same comment, this should use vllm/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py Line 667 in d5b1ec2 def convert_gpt_oss_weight_to_mxfp4_moe_kernel_format ( or vllm/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py Line 1240 in d5b1ec2 def convert_weight_to_mxfp4_moe_kernel_format ( (Instead of if self.use_cutlass_mxfp4: ... elif else ). This would avoid duplicating code for different checkpoints formats"
- **#53097** ([ROCm][Quantization][MOE] Enable fused shared experts for block-quantized FP8) — `config_utils.py`: "store_dtype comes from #45200 specifically https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash I think the message here is misleading see https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash/blob/main/config.json#L130 - I would just use if quant_config.store_dtype is not None => not supported"
- **#48015** ([ROCm][CI] Avoid HIP init at config time via lazy aiter import in Quark OCP-MX) — `quark_ocp_mx.py`: "This warning is wrong actually, on gfx942 we fall back on emulation code that does not require aiter See vllm/vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py Lines 206 to 220 in 05eed72 self . emulate = not current_platform . supports_mx () or ( self . input_dtype != "mxfp4" or self . weight_dtype != "mxfp4" ) self . rocm_use_aiter_fp4_asm_gemm = ( rocm_aiter_ops . is_asm_fp4_gemm_dynamic_quant_"
- **#52240** ([Quantization] Clarify MXFP4 MoE W13 layout requirement from backends) — `modular_kernel.py`: "I am not sure about the meaning of these given that there are still block_scale_interleave / nvfp4_block_scale_interleave / interleave_moe_weights_for_sm90_mixed_gemm / rocm_aiter_ops.shuffle_weight_a16w4 / _swizzle_mxfp4 / get_w2_permute_indices_with_cache calls as part of the huge if/elif/elif block in convert_weight_to_mxfp4_moe_kernel_format : https://github.com/BowenBao/vllm/blob/2d7822bf3d1c53144c411c3cacdc639d"
- **#47896** ([Kernel][ROCm][Perf] FlyDSL decode-attention kernel for 4-bit TurboQuant KV cach) — `rocm.py`: "I still don't get how this change relates to flydsl kernel integration really. Is this not an existing issue, unrelated to this kernel integration? If so, could we submit a fix in a separate PR?  How is this different from existing behavior? e.g. https://vllm.ai/blog/2026-05-11-turboquant mentions: so surely there is already some logic to handle mixed attention backend, if some attention backends are not compatible w"
- **#47896** ([Kernel][ROCm][Perf] FlyDSL decode-attention kernel for 4-bit TurboQuant KV cach) — `rocm.py`: "So as discussed offline, I understand this is in order to have --kv-cache-dtype turboquant_4bit_nc --attention-backend xyz not crashing, that is, being able to select the attention backend of non-turboquant layers (e.g. first and last skipped layers), e.g. in case vLLM's default is not the desired one or the best performing one. If so, the behavior should be identical on all platforms I believe (allow --attention-bac"
- **#54770** ([Bugfix][Quantization] Register Quark per-block FP8 scales as weight_scale) — `mtp.py`: "nit (ignore for this PR): cc @ColinZ22 This was added in #48044 . Given that #47972 was merged, I don't think we need to keep this? https://github.com/vllm-project/vllm/pull/48044/changes#r3562411245  I don't think the good place for this kind of logic is in a specific model (ds-v4), given that quark-produced models of different model typesmay use FP8 per-block as well? Should this be cleaned up?"
- **#58819** ([ROCm][Perf] Add opt-in a4w4 (FP4 activation) MoE for DeepSeek V4.1 on AITER) — `mxfp4.py`: "This is not a proper solution. AITER_MXFP4_BF16 backend assumes in this PR: is_guinterleave = not use_separated_a4w4 and this has nothing to do with deepseek specifically? Proper fix is in #52240 This PR does not fix the interleaving issue with QuarkOCP_MX_MoEMethod , unfortunately, that uses convert_gpt_oss_weight_to_mxfp4_moe_kernel_format ."
- **#46545** ([ROCm] [MoE] [Perf] Shared-expert fusion for bias-routed MoE; enable on MiniMax-) — `model.py`: "This could fail in case shared experts are quantized with a different precision than standard experts? https://huggingface.co/amd/MiniMax-M3-MXFP4 has same precision for both so not an issue there. Also, is this PR compatible with https://huggingface.co/amd/MiniMax-M3-MXFP4 ?"
- **#46184** ([ROCm][Perf] Use flydsl moe with Minimax-M3 mxfp8 weights on gfx950 and implemen) — `aiter_mxfp8_moe.py`: "I think having a new experts class is fine. It seems up to now https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py was already eventually calling into rocm_aiter_ops.fused_moe through some if/elif/else."
- **#46545** ([ROCm] [MoE] [Perf] Shared-expert fusion for bias-routed MoE; enable on MiniMax-) — `model.py`: "About https://huggingface.co/amd/MiniMax-M3-MXFP4 - routed and standard experts have same format. (see e.g. https://huggingface.co/amd/MiniMax-M3-MXFP4/blob/main/model-00003-of-00059.safetensors )"
- **#49732** ([ROCm][CI] Skip three torchao tests of gfx950 until `torchao==0.18` is released) — `test_torchao.py`: "See e.g. on MI300: https://buildkite.com/vllm/amd-ci/builds/11168/table?jid=019f8e34-4e44-42f1-96a6-a70012589cbc&tab=output where this test passes"
- **#47972** (Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel ) — `quark_moe.py`: "For reference as well: vllm/vllm/model_executor/layers/quantization/modelopt.py Lines 1562 to 1570 in 04a703e # Use a single gscale for w13. if self . moe . is_act_and_mul and not torch . allclose ( layer . w13_weight_scale_2 [:, 0 ], layer . w13_weight_scale_2 [:, 1 ] ): logger . warning_once ( "w1_weight_scale_2 must match w3_weight_scale_2. " "Accuracy may be affected." ) w13_weight_scale_2 = layer . w13_weight_sc"

## Data-derived profile (computed from the PR set, 6 Oct 2026)

- **PRs commented on (not own):** 35
- **Median time to first comment after PR open:** 5.6 days
- **Median time from first comment to approval:** 0.1 days
- **Median time from approval to merge:** 7.5 days
- **Where they comment (paths):** `vllm/model_executor/layers` (53), `vllm/v1/attention` (10), `vllm/model_executor/kernels` (5), `.../quantization/compressed_tensors` (5), `vllm/models/deepseek_v4` (5), `tests/models/quantization` (4)
- **Files they comment on most:** `quark_moe.py` (16), `mxfp4.py` (9), `rocm.py` (7), `quark_w8a8_fp8.py` (7), `compressed_tensors_moe_w4a4_mxfp4.py` (5), `model.py` (5), `turboquant_attn.py` (5), `quark_ocp_mx.py` (4)
- **Models named in their comments:** DeepSeek-V4 (6), Qwen3 (5), MiniMax-M3 (4), MTP (3), gpt-oss (2), DSv4 (2)
- **Pet models / configs to pre-test:** Quark MXFP4/MXFP8/FP8 per-block checkpoints: DeepSeek-V4-Pro, Kimi-K3, gpt-oss, Qwen3 MXFP4
- **Escalates to (people they @mention):** @AndreasKaratzas (10), @mgoin (8), @tjtanaa (6), @BowenBao (1), @Rohan138 (1). Pre-tag the first one when your diff is outside this reviewer's area.
- **Their own PRs look like:** Self-review comments on each hunk; extends correctness tests; cites file:line in main for every suggestion (9 own PRs in the set; 3 mention GSM8K, 0 include a serve/bench command)
- **Recent shift (last 60 days vs earlier, share of their comments):** accuracy 10% -> 0% (based on 37 recent vs 102 earlier comments)

### Approval triggers (what flipped them to LGTM)

- Selection and conversion through the oracle
- Validated on a real checkpoint, DSv4 plus one other arch
- Downstream backends still consume renamed params

Last thing they asked for right before approving, in real PRs:

- #52263: "You could move get_tensor_model_parallel_world_size() inside validate_fp8_block_shape_moe function"
- #56176: "Can you move this to test_quark.py ?"
- #47972: "I am sorry, this is not clear, this kind of special processing is meant to be done in fp8_linear.process_weights_after_loading , not here. Please do not add this here. Note that there is no backend-specific processing is"
- #50814: "Please remove this. This is the job of the emulation backend"
- #46765: "Suggested change if not self . static_input_scales : # TODO: Static-activation INT8 therefore stays on the legacy fused_experts # path (see apply()) for now, preserving pre-refactor behavior. # Needs to be migrated to ex"
- #53097: "store_dtype comes from #45200 specifically https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash I think the message here is misleading see https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash/blob/main/config"
- #43979: "Could mxfp4_round_up_hidden_size_and_intermediate_size rather be updated including Mxfp4MoeBackend.EMULATION vllm/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py Line 87 in ec8d60b EMULATION = "EMULATION" ? And remo"
- #54770: "Same comment about this that was added in #47972"

### Things they let slide

- Refactor to the oracle deferred to a linked PR if scoped

Comments they themselves marked as nits (do not block on these):

- #51274: "[quote] MoE requant through the online API can follow once #55684 covers MoE. I do not want to block this opt-in gfx942 path on that. [/quote]
yes, makes sense,"
- #45498: "nit: I would leave this test running and validate that the correct error message is displayed to the user if this case is indeed not supported
(but feel free to"
- #46765: "nit: Why not move this to
process_weights_after_loading
directly?"
- #46765: "great work, left a few minor comments"
- #54770: "nit (ignore for this PR): cc
@ColinZ22
This was added in
#48044
.
Given that
#47972
was merged, I don't think we need to keep this?
https://github.com/vllm-proj"
- #47896: "I find it slightly counter-intuitive that
VLLM_ROCM_TQ_FLYDSL_DECODE=1
may dispatch on a triton kernel.
nit: given that this
v4_eligible
check can be done ahead"
