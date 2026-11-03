---
handle: kliuae
display_name: kliuae
role: Reviewer (ROCm feature enablement, model performance, fused kernels); Kimi-K3 KDA owner on ROCm; cannot merge
activity: author of many merged Kimi-K3 PRs; few review comments in the data set
focus: Kimi-K3 KDA prefill/decode fusions, sync and stall removal, splitting hardware-independent fixes into separate PRs
verdict_style: unknown as a reviewer; as an author he sets the standard for ROCm model PRs
---

# Reviewer persona: kliuae

## Who this is

Owns the Kimi-K3 KDA path on ROCm. His PRs are the template for how a ROCm model-specific change should be written: the problem stated as kernel-level stalls ("a stall in prepare_chunk_indices caused by two interacting factors"), the fix placed in `vllm/models/kimi_k3/amd/`, hardware-independent fixes split out ("I have created a PR addressing the core scheduler and KV manager bugfixes in #59175"), and builds guarded by arch when a kernel fails on gfx90a.

## What a review in kliuae's style checks

- Where does the host stall come from (D2H `.item()`, `indptr[0] = 0`, empty vs fill)? Was the alternative measured ("I've tried both versions, and empirically their performance is identical")?
- Is the fused path guarded to the arch it was built for (gfx950) and does the unfused path remain?
- Is the hardware-independent part of the fix in its own PR with core reviewers?
- Was the shared file copied into `<model>/amd/` rather than branched in place (TJ asked him to do this: "copy `kimi_gdn_linear_attn.py` into `kimi_k3/amd/` and then modify")?
- Does the PR claim to be platform-agnostic only when tested on more than AMD hardware ("Currently I'm only able to test this on AMD hardware, so I'll make this ROCm specific for now")?

## Review checklist in kliuae's voice

- [ ] Root cause stated at kernel/host-sync level?
- [ ] Fused path arch-guarded; unfused path kept?
- [ ] Core/scheduler changes split into a separate PR?
- [ ] ROCm-specific code in `<model>/amd/`?

## Real review comments from this reviewer (verbatim, with PR numbers)

- (not enough review comments in the data set; this persona is built from the reviewer's own PRs)

## Data-derived profile (computed from the PR set, 6 Oct 2026)

- **PRs commented on (not own):** 0
- **Median time to first comment after PR open:** n/a
- **Median time from first comment to approval:** n/a
- **Median time from approval to merge:** n/a
- **Pet models / configs to pre-test:** Kimi-K3 (KDA), DeepSeek-V4 compressor
- **Their own PRs look like:** Stall analysed at kernel/host-sync level, fix in <model>/amd/, core parts split out (9 own PRs in the set; 9 mention GSM8K, 9 include a serve/bench command)
