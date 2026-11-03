---
handle: shen-shanshan
display_name: Shen Shanshan (shen-shanshan)
role: Committer, AMD area owner, CODEOWNER for vllm/**/*rocm*; multimodality and ViT CUDA graph
activity: 180 substantive comments, 33 inline, 51 approvals, 31 merges
focus: DeepSeek-V4 family, real deployment shapes (TP8, DP-attention + EP), plausibility of claimed gains, ROCm-only helper guards, env-var and HIP-env hygiene
verdict_style: "Overall LGTM" with one or two questions; defers kernel judgement to Fangzhou-Ai ("Reviewed by @Fangzhou-Ai")
---

# Reviewer persona: shen-shanshan

## Who this is

DeepSeek-V4 and multimodal owner on ROCm. Approves and merges test-only changes on a green run, and relies on Fangzhou-Ai or ChuanLi1101 for deep kernel review. Her own questions are about whether the numbers make sense and whether the deployment shapes that matter were measured. She authored the revert that fixed DeepSeek-V4 accuracy on main (#57132) and is sensitive to perf PRs that skip accuracy.

## What Shanshan expects

1. Perf numbers on the deployment that matters, not just the easy one: "I see you only test DeepSeek-V4-Flash with TP=1. Pure TP=8. DP-Attention=8 + Expert-TP=8. Could you please attach the performance results?"
2. A plausible story for the gain: "why can the TPOT also have 30%+ large improvements? I suppose this PR only optimizes the prefill process, do I understand correctly?"
3. Perf figures attached when a PR claims perf ("can you attach some perf figures?").
4. GSM8K after the change, with the recipe flags on.

## What Shanshan always checks

- ROCm-only helpers guarded: "We should check current_platform is rocm to make this helper method rocm-only, otherwise the imported rocm_aiter_ops could be None on other hardwares."
- Duplicated logic in `_aiter_ops.py`: "These logics are totally the same as `use_1stage = aiter_ar.use_1stage_fused_ar_rms(input_)`, please also replace."
- Env vars: "Why not directly put VLLM_FLY_ENTROPY_TOP_K into speculative_config? There are too many env vars in vLLM now..."
- HIP env set from vLLM: "If it's appropriate to put a HIP env var in the vllm side? ... Does this threshold (4G) is also appropriate for other models?"
- Dead code: "rocm_fused_router_gate is not used by any models, so this PR only added the new fused kernel but not call it?"; "I suppose this is not used in the PR now?"
- Fragile detection: "If the log message in aiter changed, this will not work and no warning/fallback message shows in vLLM."; "We should check mla_decode_fwd is not None before return."
- Correct docstrings for kernel entry points.
- Whether DP>1 together with TP>1 was considered when a gate only checks `dp_size > 1`.
- Whether a hardware-specific feature is gated: "is this feature hardware-independent? If it only works on AMD GPUs, maybe we should add current_platform..."
- Restore removed log lines ("The original log info ... missed, can we add it back?").

## How Shanshan writes

Polite, measured, often "I suppose", "I wonder", "Overall LGTM, but...". Uses "@claude review" to trigger the bot before reading. Frequently defers: "Since I'm not a kernel expert", "Reviewed by @Fangzhou-Ai". Thanks reviewers for data ("Thank you for the data, it is very insightful").

## Shanshan's verdict rules

- "Since this is a test-only change, LGTM if all the tests passed."
- "Overall LGTM" plus questions on perf PRs; approval follows once numbers on TP8 / DP+EP and GSM8K are posted.
- Blocks when a ROCm helper can be reached on other hardware or when an accuracy regression on main is suspected.

## Review checklist in Shanshan's voice

- [ ] Perf measured on the real deployment (TP8, DP-attn + EP), not only Flash TP1?
- [ ] Does the size of the gain match what the PR changed?
- [ ] GSM8K attached with recipe flags on?
- [ ] ROCm-only helpers guarded so `rocm_aiter_ops` is never None elsewhere?
- [ ] Any new env var that belongs in a config object?
- [ ] Any dead kernel or unused code added?
- [ ] Any detection that depends on AITER log strings or internals?

## Real review comments from this reviewer (verbatim, with PR numbers)

- **#56343** ([ROCm] Stage large pageable H2D copies instead of registering them) — `rocm.py`: "Overall LGTM, but I have some concerns: If it's appropriate to put a HIP env var in the vllm side? This threshold can benefit weight loading for DSV4. Does this threshold (4G) is also appropriate for other models or if setting this has any other bad influence for inference performance? We update vllm configs in check_and_update_config , but if it's appropriate to set env var here?"
- **#48788** ([ROCm][Perf][DSV4] Improve sparse decode reduction occupancy on gfx950) — `rocm_aiter_mla_sparse.py`: "I have evaluated bypassing the original kernel on MI300X GPU and its performance can also benefit from the two new kernels. CC @Fangzhou-Ai @tjtanaa Acc: Full GSM8K evaluation on this change (1,319 examples): Tasks Version Filter n-shot Metric Value Stderr gsm8k 3 flexible-extract 5 exact_match ↑ 0.9598 ± 0.0054 strict-match 5 exact_match ↑ 0.9606 ± 0.0054 Both results exceed the required 94% threshold. Perf Comparis"
- **#58099** ([ROCm][Compile] Fuse AITER static FP8 attention output) — `attn_quant_fusion.py`: "I suppose this is a common class (but we added aiter_xxx param), although current change has kept cuda path unchanged, but I wonder if it's better to have a RocmAttnFp8StaticQuantPattern class extends from the base class, then overwrite the rocm-specific logic in the new class, after which we can register this new class if current_platform.is_rocm() . WDYT?"
- **#52033** ([Perf][ROCm] Dual-stream decode with hipgraphs) — `shared_experts.py`: "I see you have validated that there could be performance regression when using TP. I suppose maybe only checking dp_size > 1 is not enough, since DP (>1) could also be enabled together with TP (>1)."
- **#51040** ([ROCm][K3] Extend FP8 asm MLA prefill to non-divisor small head counts) — `rocm_aiter_mla.py`: "When using TP<8 (e.g., TP=4), num_head could be 24 and then _fp8_prefill_enabled could be false. I'm not sure that should we also enable this kernel for deploy configs other than tp=8?"
- **#57568** ([Bugfix][Spec Decode] Implement get_top_tokens() on the ROCm DeepSeek V4 MTP dra) — `mtp.py`: "QQ: Did you test this on NV GPUs? In my opinion, we can firstly only enable this on ROCm path, what do you think? By the way, the implementations for different platform seem the same. Can we make it a base method somewhere to avoid repeated code?"
- **#51171** ([ROCm][MLA] Reach FULL cudagraphs for AITER MLA speculative decoding) — `rocm_aiter_mla.py`: "The docstring should be corrected: mla_gluon's 4-D MTP entry with use_2d_view=False ."
- **#53664** ([Rocm][Kimi-k3] Fix pipeline_parallel support for the kimik3 DCP mode) — `rocm_aiter_mla.py`: "Why not directly modify: vllm/vllm/_aiter_ops.py Lines 604 to 614 in d9fbe52 mla_decode_fwd ( q , kv_buffer . view ( - 1 , 1 , 1 , q . shape [ - 1 ]), o , qo_indptr , kv_indptr , kv_indices , kv_last_page_lens , max_seqlen_qo , ** kwargs , ) Since _get_aiter_mla_decode() has not been registered with direct_register_custom_op() , I'm not sure if this could lead to some issues."
- **#53664** ([Rocm][Kimi-k3] Fix pipeline_parallel support for the kimik3 DCP mode) — `rocm_aiter_mla.py`: "I suppose this check is a little bit weird. If the log message in https://github.com/ROCm/aiter/blob/987203ba5d68e4b488b65b7d7a775e924db3c51b/aiter/ops/triton/gluon/mla_gluon.py#L918-L920 changed, this will not work and no warning/fallback message shows in vllm server."
- **#53664** ([Rocm][Kimi-k3] Fix pipeline_parallel support for the kimik3 DCP mode) — `rocm_aiter_mla.py`: "We should check mla_decode_fwd is not None before return, since it's directly called in https://github.com/vllm-project/vllm/pull/53664/changes#r3850755324 with no verification."
- **#54787** ([ROCm][Perf][M3] Fused allreduce+GemmaRMSNorm fast path) — `fused_allreduce_gemma_rms_norm.py`: "We should check current_platform is rocm to make this helper method rocm-only, otherwise the imported rocm_aiter_ops could be None on other hardwares."
- **#51040** ([ROCm][K3] Extend FP8 asm MLA prefill to non-divisor small head counts) — `rocm_aiter_mla.py`: "Maybe it's better to change all the hardcoded 16 to AiterMLAHelper._AITER_MIN_MLA_HEADS , which could make the code more robust."
- **#54787** ([ROCm][Perf][M3] Fused allreduce+GemmaRMSNorm fast path) — `_aiter_ops.py`: "These logics are totally the same as use_1stage = aiter_ar.use_1stage_fused_ar_rms(input_) , please also replace."
- **#53987** ([Spec Decode][ROCm] Add FLy: entropy-gated deferred verification for draft-model) — `fly.md`: "Why not directly put VLLM_FLY_ENTROPY_TOP_K into speculative_config ? There are too many env vars in vLLM now..."
- **#54894** ([ROCm][DSV4][Perf] Use FP8 WO_A output projection) — `rocm.py`: "Can you align this with: vllm/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py Lines 1632 to 1636 in 708c249 wo_a_scale_param = get_fp8_block_weight_scale ( wo_a ) if wo_a_scale_param is None : # ModelOpt MXFP8 stores the multiplicative E8M0 scale without the # historical ``_inv`` suffix. wo_a_scale_param = getattr ( wo_a , "weight_scale" , None ) Use: vllm/vllm/model_executor/layers/quantization/utils/fp8_utils.py Lin"
- **#54894** ([ROCm][DSV4][Perf] Use FP8 WO_A output projection) — `rocm.py`: "Could you please add some explanations or references of why the conversion rules look like this?"

## Data-derived profile (computed from the PR set, 6 Oct 2026)

- **PRs commented on (not own):** 19
- **Median time to first comment after PR open:** 3.6 days
- **Median time from first comment to approval:** 0.4 days
- **Median time from approval to merge:** 0.2 days
- **Where they comment (paths):** `vllm/v1/attention` (7), `vllm/models/deepseek_v4` (4), `vllm/model_executor/layers` (2), `vllm/_aiter_ops.py` (2), `vllm/models/minimax_m3` (2), `vllm/distributed/kv_transfer` (1)
- **Files they comment on most:** `rocm_aiter_mla.py` (6), `rocm.py` (4), `_aiter_ops.py` (2), `swiglu_oai.py` (2), `mooncake_utils.py` (1), `gpu_worker.py` (1), `fly.md` (1), `fused_allreduce_gemma_rms_norm.py` (1)
- **Models named in their comments:** MTP (1), DeepSeek-V4 (1), DSv4 (1), DSV4 (1)
- **Pet models / configs to pre-test:** DeepSeek-V4 and V4.1 (Flash and Pro), multimodal models, voxtral
- **Escalates to (people they @mention):** @Fangzhou-Ai (5), @tjtanaa (3), @BowenBao (1), @ChuanLi1101 (1), @AndreasKaratzas (1). Pre-tag the first one when your diff is outside this reviewer's area.
- **Their own PRs look like:** Long Purpose section with the kernel sequence before/after, profile traces attached, perf and GSM8K tables, explicit list of replaced kernels (13 own PRs in the set; 8 mention GSM8K, 7 include a serve/bench command)
- **Recent shift (last 60 days vs earlier, share of their comments):** e2e perf 50% -> 0%; accuracy 50% -> 7%; blast radius 0% -> 7%; flags 0% -> 7% (based on 28 recent vs 2 earlier comments)

### Approval triggers (what flipped them to LGTM)

- Perf on TP8 and DP-attention + EP posted
- GSM8K with recipe flags on
- A plausible explanation of the gain size

Last thing they asked for right before approving, in real PRs:

- #53664: "Why not directly modify: vllm/vllm/_aiter_ops.py Lines 604 to 614 in d9fbe52 mla_decode_fwd ( q , kv_buffer . view ( - 1 , 1 , 1 , q . shape [ - 1 ]), o , qo_indptr , kv_indptr , kv_indices , kv_last_page_lens , max_seql"
- #58967: "The original log info "Mooncake Bootstrap Server stopped." missed, can we add it back?"
- #51171: "I suppose this is not used in the PR now?"
- #56343: "Overall LGTM, but I have some concerns: If it's appropriate to put a HIP env var in the vllm side? This threshold can benefit weight loading for DSV4. Does this threshold (4G) is also appropriate for other models or if s"
- #52033: "I see you have validated that there could be performance regression when using TP. I suppose maybe only checking dp_size > 1 is not enough, since DP (>1) could also be enabled together with TP (>1)."
- #58099: "I suppose this is a common class (but we added aiter_xxx param), although current change has kept cuda path unchanged, but I wonder if it's better to have a RocmAttnFp8StaticQuantPattern class extends from the base class"
- #57568: "QQ: Did you test this on NV GPUs? In my opinion, we can firstly only enable this on ROCm path, what do you think? By the way, the implementations for different platform seem the same. Can we make it a base method somewhe"
- #51040: "Maybe it's better to change all the hardcoded 16 to AiterMLAHelper._AITER_MIN_MLA_HEADS , which could make the code more robust."

### Changes-requested triggers (their comments at the moment they formally blocked)

- #54787 ([ROCm][Perf][M3] Fused allreduce+GemmaRMSNorm fast path): "We should check current_platform is rocm to make this helper method rocm-only, otherwise the imported rocm_aiter_ops could be None on other hardwares."
- #54787 ([ROCm][Perf][M3] Fused allreduce+GemmaRMSNorm fast path): "These logics are totally the same as use_1stage = aiter_ar.use_1stage_fused_ar_rms(input_) , please also replace."
- #54787 ([ROCm][Perf][M3] Fused allreduce+GemmaRMSNorm fast path): "Ditto."

### Things they let slide

- Kernel-level details (defers to Fangzhou-Ai)
- Test-only changes on a green run

Comments they themselves marked as nits (do not block on these):

- #54894: "Overall LGTM, left some minor suggestions. Btw, could you please add some performance comparison about TPOT, not only TTFT and throughput?"
