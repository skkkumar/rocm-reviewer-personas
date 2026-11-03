#!/usr/bin/env python3
"""Generate the exact validation commands and result tables a vLLM ROCm PR must carry.

Usage:
  python evidence.py --model moonshotai/Kimi-K3 --tp 8 --gpu gfx950 --kind perf [--flags "VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1"] [--port 8005]
  python evidence.py --model deepseek-ai/DeepSeek-V4-Flash --tp 1 --gpu gfx950 --kind bugfix

Reviewer expectations encoded (from the PR set):
  - GSM8K via lm_eval, num_fewshot 30, concurrency 128 (256 for DeepSeek-V4-class), baseline vs PR, lm_eval table layout
  - Serving sweep at 3 ISL/OSL/concurrency points with TTFT and TPOT
  - State the GPU, AITER version, every env var, and whether VLLM_ROCM_USE_AITER=1 was on
  - Second arch tested or gated; TP/EP/DCP/MTP tested or rejected at init
"""
import argparse

DSV4 = ("deepseek-v4", "dsv4", "deepseek-v4.1", "glm-5", "kimi-k3")  # reviewers asked conc 256 on these

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--ep", action="store_true")
    ap.add_argument("--gpu", default="gfx950", help="gfx942 or gfx950")
    ap.add_argument("--kind", choices=["perf", "bugfix"], default="perf")
    ap.add_argument("--flags", default="VLLM_ROCM_USE_AITER=1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--extra", default="", help="extra vllm serve args")
    a = ap.parse_args()
    conc = 256 if any(k in a.model.lower() for k in DSV4) else 128
    other = "gfx942 (MI300X/MI325X)" if a.gpu == "gfx950" else "gfx950 (MI355X)"
    par = f"--tensor-parallel-size {a.tp}" + (f" --data-parallel-size {a.dp}" if a.dp > 1 else "") + (" --enable-expert-parallel" if a.ep else "")
    serve = f"{a.flags} vllm serve {a.model} {par} --max-model-len {a.max_model_len} --trust-remote-code --port {a.port} {a.extra}".strip()

    print("# 1. Launch command (paste verbatim into the PR, once for baseline and once for PR)\n")
    print("```bash\n" + serve + "\n```\n")
    print(f"State: GPU = {a.gpu}, AITER version = $(python -c 'import aiter; print(aiter.__version__)'), vLLM commit = $(git rev-parse --short HEAD), flags above.\n")

    print(f"# 2. Accuracy (reviewers ask: num_fewshot 30, concurrency {conc}, baseline vs PR)\n")
    print("```bash")
    print(f"lm_eval --model local-completions --tasks gsm8k --num_fewshot 30 --batch_size {conc} \\")
    print(f"  --model_args model={a.model},base_url=http://127.0.0.1:{a.port}/v1/completions,num_concurrent={conc},max_retries=3,tokenized_requests=False")
    print("```\n")
    print("Paste the lm_eval table unchanged, twice (baseline, PR):\n")
    print("| Run | Tasks | Version | Filter | n-shot | Metric | Value | Stderr |\n|---|---|---|---|---|---|---|---|\n| baseline <sha> | gsm8k | 3 | flexible-extract | 30 | exact_match | | ± |\n| baseline <sha> | gsm8k | 3 | strict-match | 30 | exact_match | | ± |\n| PR <sha> | gsm8k | 3 | flexible-extract | 30 | exact_match | | ± |\n| PR <sha> | gsm8k | 3 | strict-match | 30 | exact_match | | ± |\n")

    print("# 3. End-to-end serving sweep (TTFT and TPOT; three shapes minimum)\n")
    print("```bash")
    for isl, osl, c in [(1024, 1024, 64), (8192, 1024, 64), (8192, 1024, conc)]:
        print(f"vllm bench serve --model {a.model} --port {a.port} --dataset-name random --random-input-len {isl} --random-output-len {osl} \\\n  --num-prompts {c*4} --max-concurrency {c} --request-rate inf --percentile-metrics ttft,tpot,itl --save-result --result-filename isl{isl}_osl{osl}_c{c}.json")
    print("```\n")
    print("Table to paste (one row per shape, baseline and PR, 3 runs each; state run count):\n")
    print("| ISL/OSL/conc | Build | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | Output tok/s | Runs |\n|---|---|---|---|---|---|---|\n| 1k/1k/64 | baseline | | | | | 3 |\n| 1k/1k/64 | PR | | | | | 3 |\n| 8k/1k/64 | baseline | | | | | 3 |\n| 8k/1k/64 | PR | | | | | 3 |\n| 8k/1k/%d | baseline | | | | | 3 |\n| 8k/1k/%d | PR | | | | | 3 |\n" % (conc, conc))

    if a.kind == "bugfix":
        print("# 4. Bugfix extras\n\n- Reproducer command and the exact error/garbage output before the fix.\n- A unit test that fails without the fix: `pytest -v -s tests/<path>::<test>` output pasted.\n- Perf check: the 8k/1k row above, baseline vs PR, to show no regression (asked on almost every ROCm bugfix).\n")

    print("# 5. Blast radius statement (copy into Details)\n")
    print(f"- Other arch: {other} -> tested / gated off by `<condition>` (pick one).")
    print("- Other models on this path: <list from `grep -rn <function> vllm/`> -> tested / unaffected because <reason>.")
    print("- TP / EP / DCP / DP-attention / MTP: each -> tested / rejected at init with `<error>`.")
    print("- Existing tests that assume the old kernel: <list> -> updated in this PR.")
    print("- Sync check: run the config with `VLLM_GPU_SYNC_CHECK=error` once (pr-checklist 1.5).")

if __name__ == "__main__":
    main()
