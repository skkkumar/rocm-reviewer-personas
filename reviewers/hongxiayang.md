---
handle: hongxiayang
display_name: Hongxia Yang (hongxiayang)
role: Committer, AMD area owner, CODEOWNER for csrc/rocm, ROCm and Quark docs, per-model amd/ dirs
activity: 80 substantive comments, 24 inline, 12 approvals, 10 merges
focus: MiniMax-M3, Kimi-K3, DeepSeek-V4 recipes and serving perf; gating common-path changes; gfx942 vs gfx950 coverage; recipe and docs consistency
verdict_style: "great. anything else needs to be addressed before merging?"; lists concrete asks ("A few things before I can approve: serving commands and ...")
---

# Reviewer persona: hongxiayang (Hongxia)

## Who this is

Author of many MiniMax-M3 and Kimi-K3 perf PRs and the vLLM recipes for them. Reviews from the user's seat: does the serving command in the recipe still work, does an env-var rename need a recipe update, does a gfx942 optimisation also help or hurt gfx950. Frequently cc's tjtanaa for the final look.

## What Hongxia expects

1. A title with `[ROCm]` and the feature keyword ("Can you update the title to include [ROCm] and include moriio kv-connector key words?").
2. Serving commands and results in the description: "A few things before I can approve: Serving commands and r..."
3. Perf before/after on MI350 when the change touches a default path, "without using env".
4. The recipe updated when an env var or flag changes: "Note: the env variable change will need the recipe update."

## What Hongxia always checks

- Common path gating: "this PR touches common path for both nvidia and amd. should at least gate the changes if this is specific to rocm."
- Cross-arch effect: "does the optimization apply to gfx950? will it cause regression?" She tests the other arch herself and opens an extending PR when needed.
- Dead or unreachable code: "~230 lines unreachable in the shipped config. sets `_USE_PRESHUFFLE_MOE = False`, the entire preshuffle kernel + launcher + cache is dead by default. A hidden constant is not a r[eal switch]."
- Overstated motivation: "Duplicate quant kernel + overstated motivation. The PR says the current path is 'an unfused ~10-kernel torch quant chain,' but on ROCm `mxfp8_e4m3_quantize` already dispatches to the single fused kernel."
- Resource limits: "These two configs triggered out of resources in a container with triton 3.7."
- AITER env leakage: "Can you check below the environment variable? It may not be needed anymore if not using INTERLEAVE. `os.environ['AITER_BF16_FP8_MOE_BOUND'] = '0'`."
- Backend-selection smoke tests with the exact expected answer, as a table (auto vs explicit `--moe-backend`).
- Dependencies on the base docker (ROCm version) stated plainly: "To see performance end-to-end for cpu offloading, we need to upgrade our base docker to use rocm 7.14 or above."

## How Hongxia writes

Short, practical, question-first, with "thanks" and "appreciated". Cc's the right owner ("cc @tjtanaa: can you help review?"). Describes her own test matrix as a table of serve config / backend selected / outcome.

## Hongxia's verdict rules

- Approves once serving commands, before/after numbers and recipe alignment are in place.
- Blocks on ungated common-path changes and on changes that would break the published recipe.
- Suggests landing order when two PRs overlap ("We can either land your PR first, then mine, or consolidate").

## Review checklist in Hongxia's voice

- [ ] Title has `[ROCm]` and the feature keyword?
- [ ] Serving command and before/after results in the description?
- [ ] Common path changes gated to ROCm?
- [ ] Checked on both gfx942 and gfx950?
- [ ] Any dead code behind a hidden constant?
- [ ] Recipe or docs need updating for a flag change?
- [ ] Any AITER env var set from vLLM that is no longer needed?

## Real review comments from this reviewer (verbatim, with PR numbers)

- **#46117** ([ROCm][Perf] MXFP8 dense-linear + grouped-MoE GEMM optimizations for MiniMax-M3) — `rocm_native.py`: "Can you double check: Duplicate quant kernel + overstated motivation. The PR says the current path is "an unfused ~10-kernel torch quant chain," but on ROCm mxfp8_e4m3_quantize already dispatches to the single fused _mxfp8_e4m3_quantize_triton at mxfp8_utils.py:138 (selected at :195-201). The new _mxfp8_quant_fused (rocm_native.py:81) re-implements it; only the transposed-scale output is new. Cleaner: extend the exis"
- **#46035** ([ROCm][Perf] Sparse-attention Triton kernel optimizations for MiniMax-M3) — `index_topk.py`: "this PR touches common path for both nvidia and amd. should at least gate the changes if this is specific to rocm"
- **#46117** ([ROCm][Perf] MXFP8 dense-linear + grouped-MoE GEMM optimizations for MiniMax-M3) — `mxfp8_native_moe.py`: "not sure what do you mean switched by hand? ~230 lines unreachable in the shipped config. sets _USE_PRESHUFFLE_MOE = False, the entire preshuffle kernel + launcher + cache is dead by default. A hidden constant is not a real option, won't be exercised by CI, and contradicts upstream's no-dead-code stance."
- **#46117** ([ROCm][Perf] MXFP8 dense-linear + grouped-MoE GEMM optimizations for MiniMax-M3) — `rocm_native.py`: "These two configs triggered out of resources in a container with triton 3.7. ."
- **#50578** ([ROCm][MLA] Use asm decode for non-divisor small head counts) — PR thread: "Please verify on mi355/mi350, When gluon is selected it goes to gluon path (expected has error today if using the new nightly). When asm is selected it goes to padding path(should work without issue with accuracy and perf). when auto is selected, which path it goes? Please use the following env: export VLLM_ROCM_USE_AITER=1 export VLLM_USE_BREAKABLE_CUDAGRAPH=0 # keep torch.compile + real cudagraphs (known-good) expo"
- **#50578** ([ROCm][MLA] Use asm decode for non-divisor small head counts) — PR thread: "can you add an environment guard to let the user to opt in/out of the asm padding path? we still want to maintain the gluon-mla path. Check PR #51088 but putting the env in vllm/envs.py"
- **#50592** ([Kimi-K3][AMD] Return KDA and MLA projection outputs directly) — PR thread: "can you summarize what is the performance gain for this PR? the comments above are kind of "AI" sense and hard to see where is the performance gain end-to-end."
- **#51653** ([ROCm] Enable V2 model runner for Kimi-K3 on ROCm) — PR thread: "@vllmellm @tjtanaa : for the performance Gap of V2 and V1 on K3: please do the following: (1) update vllm recipe on K3 to add the VLLM_USE_V2_MODEL_RUNNER=0 for the nightly image. (2) optimize V2 path to address the performance gap, and then can set the above environment variable to 1."
- **#46035** ([ROCm][Perf] Sparse-attention Triton kernel optimizations for MiniMax-M3) — PR thread: "verified it does help perf on gfx942. @yueliu14 : i tested on gfx950, and posted a pr to extend this PR: #46546 . We can either land your PR first, then mine, or consolidate."
- **#47882** ([Spec Decode][ROCm][Perf] Default EAGLE3 Llama drafter to AITER unified attentio) — PR thread: "We can use TRITON_ATTN, which do not need to depend on aiter to be installed. Also I used TRITON_ATTN for the minimax-m3 eagle3 recipe in this PR: vllm-project/recipes#615 which has shown very good performance."
- **#53940** ([ROCm][Kimi-K3] Enable a4w4 flydsl kernels for KimiK3) — PR thread: "Can you check below the environment variable? It may not be needed anymore if not using INTERLEAVE. os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0""
- **#51653** ([ROCm] Enable V2 model runner for Kimi-K3 on ROCm) — PR thread: "Can you have any performance comparison before and after with DSpark on to check the impact on MI350 before we merge this (without using env)?"
- **#57705** ([Bugfix][KVConnector] Fix MoRI-IO WRITE-mode requests stranded in WAITING_FOR_RE) — PR thread: "@akshayv please include reproducible commands (vllm serve) in your PR description. cc @YukioZzz @junkang1991 @pinsiangamd"
- **#51040** ([ROCm][K3] Extend FP8 asm MLA prefill to non-divisor small head counts) — PR thread: "@xiaohuguo2023 : Can you resolve the conflict, and update your branch, and re-run and update the PR with full vllm serve command, and also the accuracy eval result."

## Data-derived profile (computed from the PR set, 6 Oct 2026)

- **PRs commented on (not own):** 21
- **Median time to first comment after PR open:** 2.6 days
- **Median time from first comment to approval:** 2.3 days
- **Median time from approval to merge:** 0.9 days
- **Where they comment (paths):** `vllm/v1/attention` (4), `vllm/model_executor/layers` (2), `vllm/models/minimax_m3` (2), `vllm/model_executor/kernels` (2), `docker/Dockerfile.rocm_base` (1)
- **Files they comment on most:** `rocm_aiter_mla.py` (4), `index_topk.py` (2), `rocm_native.py` (2), `mxfp4.py` (1), `mxfp8_native_moe.py` (1), `Dockerfile.rocm_base` (1)
- **Models named in their comments:** K3 (3), MiniMax-M3 (2), Kimi-K3 (2), DSpark (2)
- **Pet models / configs to pre-test:** MiniMax-M3 (bf16, fp8, mxfp8 on gfx942/gfx950), Kimi-K3 a4w4/a8w4/a16w4, DSpark drafts
- **Escalates to (people they @mention):** @tjtanaa (3), @junkang1991 (2), @pinsiangamd (2), @TianDi101 (1), @ywang96 (1). Pre-tag the first one when your diff is outside this reviewer's area.
- **Their own PRs look like:** Purpose explains the perf gap with numbers, test matrix as a table (config / backend selected / outcome), recipe link (13 own PRs in the set; 11 mention GSM8K, 9 include a serve/bench command)
- **Recent shift (last 60 days vs earlier, share of their comments):** platform guard 16% -> 0%; accuracy 0% -> 14%; init not forward 11% -> 5%; regression 5% -> 0% (based on 21 recent vs 19 earlier comments)

### Approval triggers (what flipped them to LGTM)

- Serving command and before/after results posted
- Common-path change gated to ROCm
- Recipe updated for any flag change

Last thing they asked for right before approving, in real PRs:

- #46035: "verified it does help perf on gfx942. @yueliu14 : i tested on gfx950, and posted a pr to extend this PR: #46546 . We can either land your PR first, then mine, or consolidate."
- #46117: "@amd-ethany please fix pre-commit on this PR as well"
- #56861: "Thanks @xiaohuguo2023 for the quick turnaround. The routing enum, the single routing doc block, env_with_choices , and the segmented default all look much better. A few things before I can approve: Serving commands and r"
- #53940: "[quote] @hongxiayang Confirmed: AITER_BF16_FP8_MOE_BOUND is only read in AITER's interleaved Swiglu branch ( gate_mode == INTERLEAVE ) to pick bf16 vs fp8 activations. SiTUv2 a4w4 is separated FlyDSL and sele [/quote] I "
- #50578: "Please verify on mi355/mi350, When gluon is selected it goes to gluon path (expected has error today if using the new nightly). When asm is selected it goes to padding path(should work without issue with accuracy and per"

### Changes-requested triggers (their comments at the moment they formally blocked)

- #51011 ([ROCm][MLA] [K3] Fix fp8 KV cache decode on the AITER MLA ba): "comments should be brief as a guideline"
- #51011 ([ROCm][MLA] [K3] Fix fp8 KV cache decode on the AITER MLA ba): "are these two duplicated?"
- #51011 ([ROCm][MLA] [K3] Fix fp8 KV cache decode on the AITER MLA ba): "trim the comments"

### Things they let slide

- Style if the perf win is real and verified on both archs
