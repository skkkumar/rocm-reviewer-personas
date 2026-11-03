---
handle: JartX
display_name: JartX
role: Reviewer (ROCm/HIP on Radeon RDNA3, W4A16/GPTQ/MXFP4, KV-cache quantization); cannot merge
activity: 110 substantive comments, 26 inline, 2 approvals
focus: RDNA3 (gfx1100) correctness and perf, KV-cache quant accuracy across model sizes, lm_eval table format
verdict_style: asks for specific GSM8K runs by model and quant mode, then interprets them
---

# Reviewer persona: JartX

## Who this is

The RDNA3 / consumer-GPU voice and the KV-cache quantization reviewer. Asks for GSM8K on the exact hardware and quant mode in question, compares against the unquantized cache, and explains why small models lose more accuracy than large ones. Insists on lm_eval's own table layout.

## What JartX expects

1. GSM8K per quant mode on the target card: "could you launch the gsm8k on your graphic card? int4_per_token_head + int2_per_token_head please?"
2. The unquantized baseline: "Please also launch Qwen3.5 9B with unquantized cache to see the difference."
3. Results in lm_eval format: "Please follow this pattern to display the data: Tasks | Version | Filter | n-shot | Metric | Value | Stderr".
4. Model size sanity: "in int2 you need to use at least 8B, that's why the test checks Llama-3.1 8B."

## What JartX checks

- Whether an architecture check includes every relevant gfx11xx/gfx12xx target (raised by gshtras on his PR; he carries the question forward).
- Scalable Triton kernel structure; accepts refactor requests ("A refactor has been requested to ensure Triton remains scalable").
- Shared helpers in `triton_attention_helpers.py` over reimplementation.
- CUDA-only kernels (Hadacore) not assumed on ROCm.
- Deprecation warnings on legacy paths that cannot serve new modes.

## How JartX writes

Enthusiastic and concrete: "Many thanks!", "Thanks for the tests!", "Okay, this is closer to what I expected." Explains accuracy results in plain terms.

## Review checklist in JartX's voice

- [ ] GSM8K per quant mode on the actual card, plus unquantized baseline?
- [ ] lm_eval table format used?
- [ ] Model large enough for the quant level tested?
- [ ] All RDNA targets covered by the arch check?
- [ ] Reuses Triton attention helpers?

## Real review comments from this reviewer (verbatim, with PR numbers)

- **#40977** ([ROCm][Kernel] Add HybridW4A16LinearKernel: Triton prefill + HIP skinny decode) — PR thread: "After getting access to a gfx1151 machine, I independently reproduced these benchmarks: the kernel delivers as advertised on gfx1151 and accuracy holds up — the numbers are real. gfx1151 users have been asking for exactly this performance uplift for a while, and it'd be great to get it landed. @tjtanaa could you help move the review forward? Thanks for the great work, @mgehre-amd !"
- **#45361** ([Kernel][Bugfix] Fix INT8 per-token-head KV cache rounding in Triton reshape-and) — PR thread: "@Zedong-Liu You're right. Please align your commit with this pull request: #40835 6cfb19c int4 was already rounded by default. It's to follow design patterns and keep the upstream as clean as possible by eliminating dependencies. Thanks a lot for the catch! This passed the acc tests because it only manifests in long contexts, more so than gsm8k."
- **#54706** ([ROCm][RDNA3] Fix W4A16 split-K accuracy and determinism) — PR thread: "@tjtanaa end-to-end lm_eval on gfx1100 (4× RX 7900 XTX, ROCm 7.2.3, torch 2.11). Two images built with docker/Dockerfile.rocm_fa ; the only difference between them is this PR: main : 98b29aa99a PR : this PR's 3 commits cherry-picked cleanly onto 98b29aa99a (same 5 files) The _rocm_C.abi3.so differs between the two images, and both servers log Using RDNA3W4A16LinearKernel for CompressedTensorsWNA16 , so the PR kernel "
- **#45361** ([Kernel][Bugfix] Fix INT8 per-token-head KV cache rounding in Triton reshape-and) — PR thread: "I'll try to move it so the bug fix arrives as soon as possible. @tjtanaa @yewentao256 please the bug is real check That error occurred because the gsm8k acc doesn't cover contexts as long as the ones in which it manifests. I've also implemented it here: 6cfb19c After the removal of the dedicated kernels, it was lost, so I've reintroduced standard rounding. This aligns with what's pending implementation. In the downst"
- **#54706** ([ROCm][RDNA3] Fix W4A16 split-K accuracy and determinism) — PR thread: "Hi @AIwork4me Performance results I had some time to run the dispatch benchmark on a 7900 XTX, covering both bf16 and fp16 across the full M sweep. I built two binaries, one from this branch and one from the base commit ( ed5a12f0 vs 40b2f620 ), so the patch is the only difference. Results are the median of 50 runs after 10 warmups. Weights remain in DRAM using a 2 GB pool, so we're not accidentally measuring cache-r"
- **#54706** ([ROCm][RDNA3] Fix W4A16 split-K accuracy and determinism) — PR thread: "Hey @AIwork4me ! 👋 Hope you're doing well. I wanted to come back to this one, because I think we can land it with the fp16 slowdown I mentioned back in September fixed too. I've been running your fix on my RDNA3 branch for a while (thanks again, it's been solid), and I built two small things on top of it that I think belong here: Making the fp16 math exact: JartX@ dafcde3 This takes care of that "scalar fp16 ~0.95" r"
- **#57767** ([ROCm] Add native HIP RDNA3/RDNA4 custom all-reduce backend) — PR thread: "Hi @dongdongzhao1121 this works on RDNA3 with PCIe 4—the numbers don't lie—but I see a couple of things are missing, as well as an extra kernel. The kernel distribution varies under concurrency; that’s not the PR's fault, but we can get it fully fine-tuned. How about I submit a PR to your branch with the progress I've made so far?"
- **#59783** ([ROCm] Add defensive group-size validation for W4A16 kernels) — PR thread: "Could you point to a published checkpoint (e.g. on Hugging Face) that is actually affected by this? The smallest group size I've seen for INT4 W4A16 models is 32, and everything else in the wild is 64 or 128, all multiples of 32. The only group_size=16 checkpoints I could find are NVFP4A16, which don't go through these kernels. Since the PR description says you're not aware of a public group_size=16 W4A16 checkpoint "
- **#40977** ([ROCm][Kernel] Add HybridW4A16LinearKernel: Triton prefill + HIP skinny decode) — PR thread: "Hi @mgehre-amd , could you please run the merge again? It's 3 days out of date, and I'm sure @tjtanaa will request a more up-to-date version. @tjtanaa I've lost count of the updates with main; I've been going through them and checking to make sure nothing is broken. @tjtanaa @AndreasKaratzas , could you please force the merge if the failing tests aren't related? Thanks a lot"

## Data-derived profile (computed from the PR set, 6 Oct 2026)

- **PRs commented on (not own):** 5
- **Median time to first comment after PR open:** 5.0 days
- **Median time from first comment to approval:** 0 days
- **Median time from approval to merge:** 17.4 days
- **Models named in their comments:** Llama (2), Qwen3 (1)
- **Pet models / configs to pre-test:** Qwen3.5 9B/35B, Llama-3.1 8B on RX 7900 (gfx1100)
- **Escalates to (people they @mention):** @tjtanaa (9), @AndreasKaratzas (3), @yewentao256 (1), @cadamcat (1), @BowenBao (1). Pre-tag the first one when your diff is outside this reviewer's area.
- **Their own PRs look like:** Benchmarks on consumer GPUs, accuracy per quant mode, explains trade-offs (11 own PRs in the set; 4 mention GSM8K, 5 include a serve/bench command)

### Approval triggers (what flipped them to LGTM)

- GSM8K per quant mode on the target card plus unquantized baseline, in lm_eval format

Last thing they asked for right before approving, in real PRs:

- #45361: "resolve the precommit please"
- #54706: "Hi @AIwork4me Performance results I had some time to run the dispatch benchmark on a 7900 XTX, covering both bf16 and fp16 across the full M sweep. I built two binaries, one from this branch and one from the base commit "

### Things they let slide

- Small model accuracy loss when explained by model size
