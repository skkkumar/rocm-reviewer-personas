#!/usr/bin/env python3
"""Deterministic pre-flight checks for a vLLM ROCm PR. No LLM.

Usage:
  python static_checks.py --diff pr.diff --body pr_body.md [--title "..."] [--repo /path/to/vllm] [--json]
  gh pr diff 59280 --repo vllm-project/vllm | python static_checks.py --diff - --body <(gh pr view 59280 --repo vllm-project/vllm --json body -q .body)

Each rule maps to a comment that a ROCm reviewer left at least 5 times in the PR set this tool was built from.
Severity: BLOCK (a committer will not merge), FIX (will be asked before approval), NOTE (nit or heads-up).
"""
import argparse, json, re, subprocess, sys, os
from collections import defaultdict

SHARED_PATHS = [
    "vllm/model_executor/layers/fused_moe/", "vllm/model_executor/layers/quantization/",
    "vllm/compilation/", "vllm/v1/worker/", "vllm/v1/core/", "vllm/v1/attention/backends/mla/common.py",
    "vllm/v1/attention/backends/utils.py", "vllm/model_executor/layers/layernorm.py", "vllm/envs.py",
    "vllm/platforms/interface.py", "csrc/", "CMakeLists.txt", "setup.py",
]
ROCM_ONLY_HINT = re.compile(r"rocm|aiter|amd/|hip|gfx|quark|flydsl|mori", re.I)

def parse_diff(text):
    files = {}
    cur = None; new_line = 0
    for line in text.splitlines():
        if line.startswith("diff --git"):
            m = re.search(r" b/(\S+)$", line); cur = m.group(1) if m else line; files[cur] = {"added": [], "removed": [], "is_new": False, "hunks": []}
        elif cur and line.startswith("new file mode"): files[cur]["is_new"] = True
        elif cur and line.startswith("@@"):
            m = re.search(r"\+(\d+)", line); new_line = int(m.group(1)) - 1 if m else 0
        elif cur and line.startswith("+") and not line.startswith("+++"):
            new_line += 1; files[cur]["added"].append((new_line, line[1:]))
        elif cur and line.startswith("-") and not line.startswith("---"):
            files[cur]["removed"].append(line[1:])
        elif cur and not line.startswith("\\"):
            new_line += 1
    return files

def in_forward_context(added, idx):
    """Heuristic: is added line idx inside a def whose name looks like a per-step path?"""
    for j in range(idx, -1, -1):
        ln = added[j][1]
        m = re.match(r"\s*def\s+(\w+)", ln)
        if m:
            return bool(re.search(r"forward|_forward|apply|build|__call__|compute|decode|prefill|schedule|_run", m.group(1))) and not m.group(1).startswith("__init__")
    return False

def run(diff_text, body, title, repo):
    files = parse_diff(diff_text)
    findings = []
    def add(sev, rule, where, msg, who):
        findings.append({"severity": sev, "rule": rule, "where": where, "message": msg, "asked_by": who})

    added_py = {f: v for f, v in files.items() if f.endswith(".py")}
    test_files = [f for f in files if "/tests/" in "/" + f or f.startswith("tests/")]
    changed_loc = sum(len(v["added"]) for f, v in files.items() if not (f.startswith("tests/") or f.startswith("docs/")))

    # --- Description checks ---
    b = body or ""
    if len(b.strip()) < 200 or "Essential Elements of an Effective PR Description Checklist" in b and len(b) < 900:
        add("BLOCK", "desc-empty", "PR body", "Description is empty or the untouched template. Fill Overview / Claims / Validation / Details.", "tjtanaa, AndreasKaratzas")
    if not re.search(r"vllm serve|vllm bench|python -m vllm", b):
        add("FIX", "desc-no-command", "PR body", "No `vllm serve` / `vllm bench` command. tjtanaa's first ask on every ROCm PR: 'Please provide the vllm launch command'.", "tjtanaa, Rohan138")
    if not re.search(r"gsm8k|lm[ _-]?eval", b, re.I):
        add("FIX", "desc-no-accuracy", "PR body", "No GSM8K / lm_eval result. Expected: num_fewshot 30, concurrency 128 (256 for DSv4-class), before and after.", "tjtanaa, simondanielsson, Fangzhou-Ai")
    elif re.search(r"gsm8k", b, re.I) and not re.search(r"(num.?few.?shot|n-shot|numshot)\D{0,10}30|\b30\b.{0,20}(shot)", b, re.I):
        add("NOTE", "desc-accuracy-shots", "PR body", "GSM8K present but num_fewshot 30 not stated. tjtanaa rejects numshot 5/8 when kernels change numerics.", "tjtanaa")
    if re.search(r"\[Perf\]|perf", title or "", re.I) and not re.search(r"ttft|tpot|throughput|tok/s|tokens/s|itl|latency", b, re.I):
        add("FIX", "desc-no-e2e", "PR body", "Perf PR without end-to-end serving numbers (TTFT/TPOT/throughput). Microbenchmarks alone are asked to be supplemented every time.", "tjtanaa, dllehr-amd, simondanielsson")
    if re.search(r"\[Perf\]", title or "", re.I) and not re.search(r"isl|osl|conc|concurrency|input.?len|output.?len", b, re.I):
        add("NOTE", "desc-no-sweep", "PR body", "No ISL/OSL/concurrency shapes stated. dllehr-amd asks for 'a few more ISL/OSL/Conc sweeps'.", "dllehr-amd, Fangzhou-Ai")
    if not re.search(r"MI3\d\d|gfx9\d\d|gfx1\d\d\d|MI355|MI300|MI325|MI350", b):
        add("NOTE", "desc-no-gpu", "PR body", "GPU/arch not stated. Say gfx942 or gfx950 and whether the other was tested or gated off.", "dllehr-amd, hongxiayang")
    if re.search(r"aiter", diff_text, re.I) and not re.search(r"aiter[ _-]?(v?0\.\d|version|bump|pin)", b, re.I):
        add("NOTE", "desc-no-aiter-version", "PR body", "Diff touches AITER but the required AITER version is not stated. tjtanaa parks PRs whose kernel is not in the pinned version.", "tjtanaa, Rohan138")
    if re.search(r"VLLM_ROCM_USE_AITER", diff_text) and not re.search(r"VLLM_ROCM_USE_AITER", b):
        add("NOTE", "desc-aiter-flag", "PR body", "Say whether VLLM_ROCM_USE_AITER=1 was on during the runs (tjtanaa asked on #52737).", "tjtanaa")
    if re.search(r"Bugfix|Fix", title or "", re.I) and not re.search(r"root cause|caused by|because|the bug|regress", b, re.I):
        add("FIX", "desc-no-rootcause", "PR body", "Bugfix without a root-cause sentence. The pr-checklist and every ROCm committer ask for it.", "AndreasKaratzas, dllehr-amd")
    if re.search(r"Bugfix|Fix", title or "", re.I) and not re.search(r"perf|tpot|regress", b, re.I):
        add("NOTE", "desc-bugfix-perf", "PR body", "Bugfix without a perf check. 'Did you do any perf tests after this?' is asked on almost every ROCm bugfix.", "AndreasKaratzas")
    if not re.search(r"^\[[^\]]+\]", (title or "").strip()):
        add("FIX", "title-tags", "title", "Title must start with bracket tags, e.g. [ROCm][Perf] or [Bugfix][ROCm].", "contributing guide, hongxiayang")
    elif not re.search(r"ROCm|AMD", title or "", re.I) and ROCM_ONLY_HINT.search(diff_text):
        add("NOTE", "title-rocm", "title", "ROCm-specific diff without [ROCm] in the title; hongxiayang asks for it.", "hongxiayang")

    # --- Size ---
    if changed_loc > 500:
        add("FIX", "size-rfc", "diff", f"{changed_loc} non-test lines added. >500 LOC needs an RFC issue or gets 'rfc-required'. Split it.", "contributing guide")
    elif changed_loc > 250 and re.search(r"Bugfix|Fix", title or "", re.I):
        add("NOTE", "size-bugfix", "diff", f"{changed_loc} lines for a bugfix is above the pr-checklist 'high complexity' line (~250). Expect a split request.", "pr-checklist, kliuae precedent")

    # --- Per-file code rules ---
    for f, v in files.items():
        added = v["added"]
        is_test = f.startswith("tests/") or "/tests/" in f
        base = os.path.basename(f)
        if base.lower().startswith("readme") or f.startswith("benchmarks/") and ROCM_ONLY_HINT.search(f):
            add("FIX", "stray-files", f, "Benchmark scripts and READMEs for AITER kernels should live in AITER, not vLLM ('remove this file before final merge').", "tjtanaa")
        if not f.endswith(".py"):
            if f.endswith((".cu", ".hip", ".cuh", ".h")):
                for n, ln in added:
                    if re.search(r"!\s*defined\s*\(\s*__gfx950__\s*\)", ln):
                        add("FIX", "hip-fnuz-gate", f"{f}:{n}", "Use `#if defined(__gfx942__)` for fnuz, not `!defined(__gfx950__)` (only one AMD arch uses fnuz).", "tjtanaa")
                if re.search(r"jit|hipModuleLoad|hiprtc", "\n".join(l for _, l in added), re.I):
                    add("BLOCK", "jit-kernel", f, "Custom kernels must compile at build time through CMake into _rocm_C, not JIT.", "tjtanaa")
            continue
        text_added = "\n".join(l for _, l in added)
        for i, (n, ln) in enumerate(added):
            s = ln.strip()
            if s.startswith("#"): continue
            # top-level rocm import
            if re.match(r"from vllm\.platforms\.rocm import|import vllm\.platforms\.rocm", s) and not ln.startswith((" ", "\t")):
                add("BLOCK", "rocm-import-toplevel", f"{f}:{n}", "Top-level `vllm.platforms.rocm` import. Guard with `if current_platform.is_rocm():` or import inside the function.", "tjtanaa, AndreasKaratzas")
            if re.search(r"\benvs\.VLLM_", s) and in_forward_context(added, i):
                add("BLOCK", "envs-in-forward", f"{f}:{n}", "`envs.*` read inside a per-step path. Cache in __init__ or as a rocm_aiter_ops classmethod (issue #17067).", "tjtanaa")
            if re.search(r"os\.environ\[.*\]\s*=|os\.environ\.setdefault|os\.putenv", s):
                add("FIX", "sets-env", f"{f}:{n}", "Sets an environment variable at runtime. shen-shanshan and hongxiayang question HIP/AITER env vars set from vLLM; justify or move to init with a TODO.", "shen-shanshan, hongxiayang")
            if re.search(r"torch\.finfo\(", s) and in_forward_context(added, i):
                add("FIX", "finfo-forward", f"{f}:{n}", "`torch.finfo(...)` in a per-step path is costly; precompute as a constant.", "tjtanaa")
            if re.search(r"\.item\(\)", s) and in_forward_context(added, i) and not is_test:
                add("FIX", "item-sync", f"{f}:{n}", "`.item()` forces a device sync in a per-step path. kliuae/tjtanaa remove these; pr-checklist says run with VLLM_GPU_SYNC_CHECK=error.", "tjtanaa, kliuae")
            if re.search(r"current_platform\.fp8_dtype\(\)", s) and in_forward_context(added, i):
                add("FIX", "fp8-dtype-forward", f"{f}:{n}", "`current_platform.fp8_dtype()` invokes host ops; cache it (tjtanaa, #57435).", "tjtanaa")
            if re.search(r"gfx9\d\d|gfx1\d\d\d", s) and not is_test and "platforms/rocm.py" not in f and not re.search(r"on_gfx|is_gfx|__gfx", s):
                add("FIX", "gfx-literal", f"{f}:{n}", "Hard-coded gfx string outside platforms/rocm.py. Use on_gfx950() / CDNA family helpers so MI450 inherits it.", "AndreasKaratzas, tjtanaa")
            if re.search(r"^\s*try:\s*$", ln):
                # look ahead for aiter import or kernel dispatch
                window = "\n".join(l for _, l in added[i:i+6])
                if re.search(r"import aiter|from aiter|rocm_aiter_ops\.|_aiter_ops|flydsl|getattr\(aiter", window):
                    add("BLOCK", "try-except-dispatch", f"{f}:{n}", "try/except around AITER import or kernel selection. vLLM pins one AITER version; select in __init__ with no fallback probe.", "tjtanaa, simondanielsson")
            if re.search(r"hasattr\(\s*aiter|getattr\(\s*aiter|importlib\.util\.find_spec\([\"']aiter", s):
                add("BLOCK", "aiter-probe", f"{f}:{n}", "Probing AITER internals for feature detection. Assume the pinned version ships it (simondanielsson, tjtanaa, Rohan138 #57426).", "simondanielsson, tjtanaa")
            if re.search(r"except (Exception|RuntimeError|BaseException)\b", s) and not is_test:
                add("FIX", "wide-except", f"{f}:{n}", "Broad except. dllehr-amd: 'now you'll suppress all RuntimeErrors ... more dangerous than before'.", "dllehr-amd")
            if re.search(r"^\s*#.*(this PR|previously|used to|old (code|behaviou?r)|before this change|see PR|PR #\d+)", ln, re.I):
                add("NOTE", "history-comment", f"{f}:{n}", "Comment explains history / the old behaviour. Reviewers ask to delete these; put it in the PR description.", "tjtanaa, simondanielsson, Rohan138")
            if re.search(r"(print\(|breakpoint\(\)|pdb\.set_trace)", s) and not is_test:
                add("FIX", "debug-left", f"{f}:{n}", "Debug print/breakpoint left in the diff.", "Rohan138")
            if re.search(r"logger\.(info|warning)\(", s) and in_forward_context(added, i):
                add("NOTE", "log-in-forward", f"{f}:{n}", "Logging on a per-step path. simondanielsson: 'do we need this logged on every request? I'd suggest .debug'.", "simondanielsson, tjtanaa")
        # file-level rules
        if f == "vllm/envs.py" and re.search(r'^\s*"VLLM_ROCM_\w+":', text_added, re.M):
            add("FIX", "new-env-flag", f, "New VLLM_ROCM_* env flag. tjtanaa: 'Let's not add new flags'. Prefer --moe-backend / --attention-backend, config inference, or an existing VLLM_ROCM_USE_AITER_* flag.", "tjtanaa, shen-shanshan, hongxiayang")
        if re.search(r"direct_register_custom_op|_fake\s*\(|\.register_fake", text_added) and "_aiter_ops.py" not in f and re.search(r"aiter", text_added, re.I) and not is_test:
            add("FIX", "op-registration-place", f, "AITER op registration (_impl/_fake/register) outside vllm/_aiter_ops.py. tjtanaa centralises all of these there.", "tjtanaa")
        if re.search(r"is_rocm\(\)", text_added) and re.search(r"vllm/models?/", f) and "/amd/" not in f and not is_test:
            add("FIX", "rocm-branch-in-shared-model", f, "ROCm branch added to a shared model file. Per RFC #42770, copy into vllm/models/<model>/amd/ and change there.", "tjtanaa, simondanielsson, kliuae")
        if any(f.startswith(p) for p in SHARED_PATHS) and not ROCM_ONLY_HINT.search(base) and not is_test:
            add("FIX", "shared-path", f, "Shared CUDA/ROCm path changed. Keep CUDA behaviour byte-identical (gate with is_rocm()) and tag a core reviewer (mgoin / ProExpertProg / njhill). tjtanaa: 'we need to keep the original config for CUDA platform'.", "tjtanaa, AndreasKaratzas, hongxiayang")
        if is_test:
            whole = text_added if v["is_new"] else text_added
            if ROCM_ONLY_HINT.search(f) or re.search(r"aiter|rocm", text_added, re.I):
                if not re.search(r"skipif\([^)]*is_rocm|pytestmark.*rocm|skip_if_not_rocm", text_added, re.I) and v["is_new"]:
                    add("BLOCK", "test-no-rocm-skip", f, "New ROCm test without a file-level `@pytest.mark.skipif(not current_platform.is_rocm(), ...)`.", "AndreasKaratzas, tjtanaa")
                if re.search(r"aiter", text_added, re.I) and not re.search(r"aiter.*(available|found|installed)|skipif\([^)]*aiter", text_added, re.I) and v["is_new"]:
                    add("FIX", "test-no-aiter-skip", f, "Test needs AITER but has no AITER-availability skip ('there should be a skip if aiter not found check too').", "AndreasKaratzas")
            if re.search(r"torch\.manual_seed\(", text_added):
                add("NOTE", "test-torch-seed", f, "Use vLLM's set_manual_seed util instead of torch.manual_seed.", "AndreasKaratzas, simondanielsson")
            if re.search(r"(atol|rtol)\s*=\s*[0-9.e-]+", text_added) and not re.search(r"#.*(tolerance|atol|rtol|because|dtype)", text_added, re.I):
                add("NOTE", "test-tolerance", f, "Numeric tolerance without a comment deriving it. AndreasKaratzas: 'are these thresholds numerically backed?'", "AndreasKaratzas")
            if re.search(r"tensor_parallel_size\s*=\s*([4-9]|\d\d)|tp_size\s*=\s*([4-9]|\d\d)", text_added):
                add("FIX", "test-tp-size", f, "Test TP size above 2. tjtanaa: 'For this test, we either go for TP1 or TP2.'", "tjtanaa")
            if re.search(r"gfx950|gfx942", text_added) and not re.search(r"on_gfx|cdna", text_added, re.I):
                add("NOTE", "test-gfx-literal", f, "Hard-coded gfx string in a test; use on_gfx950()/CDNA helpers.", "AndreasKaratzas")
            if v["is_new"] and repo:
                tay = os.path.join(repo, ".buildkite", "test-amd.yaml")
                if os.path.exists(tay):
                    txt = open(tay, encoding="utf-8", errors="replace").read()
                    d = os.path.dirname(f)
                    if f not in txt and (d + "/") not in txt and not any(seg and seg in txt for seg in [d, os.path.dirname(d)]):
                        add("BLOCK", "test-not-in-ci", f, "New test path does not appear in .buildkite/test-amd.yaml. AndreasKaratzas treats this as untested: 'point me to the test group'.", "AndreasKaratzas")
    # test presence
    if not test_files and not re.search(r"\[CI\]|\[Doc", title or "", re.I):
        add("FIX", "no-tests", "diff", "No test changes. Bugfixes need a test that fails without the fix; perf PRs need at least a dispatch/correctness test on the new path.", "AndreasKaratzas, tjtanaa")

    # AITER pin check
    if repo:
        df = os.path.join(repo, "docker", "Dockerfile.rocm_base")
        if os.path.exists(df):
            m = re.search(r'AITER_BRANCH="?([^"\s]+)', open(df, encoding="utf-8", errors="replace").read())
            if m: add("NOTE", "aiter-pin", "docker/Dockerfile.rocm_base", f"Pinned AITER is `{m.group(1)}`. Confirm every aiter.* symbol you call exists at that tag.", "tjtanaa, Rohan138")

    # DCO / branch checks via git (if repo given and we are on the branch)
    if repo:
        try:
            log = subprocess.run(["git", "-C", repo, "log", "--format=%H%n%b%n==END==", "origin/main..HEAD"], capture_output=True, text=True, timeout=20).stdout
            commits = [c for c in log.split("==END==") if c.strip()]
            missing = [c.splitlines()[0][:10] for c in commits if "Signed-off-by:" not in c]
            if missing: add("BLOCK", "dco", "commits", f"Commits without Signed-off-by: {', '.join(missing)}. Rebase with --signoff.", "DCO bot, dllehr-amd")
            behind = subprocess.run(["git", "-C", repo, "rev-list", "--count", "HEAD..origin/main"], capture_output=True, text=True, timeout=20).stdout.strip()
            if behind and behind != "0": add("FIX", "behind-main", "branch", f"Branch is {behind} commits behind origin/main. `/ci run` refuses stale branches; rebase first.", "CI bot, AndreasKaratzas")
        except Exception: pass

    order = {"BLOCK": 0, "FIX": 1, "NOTE": 2}
    findings.sort(key=lambda x: (order[x["severity"]], x["rule"]))
    # predicted reviewers
    who = defaultdict(int)
    for x in findings:
        for w in re.split(r",\s*", x["asked_by"]): who[w] += 1
    return {"findings": findings, "changed_non_test_loc": changed_loc, "files": list(files), "likely_commenters": sorted(who.items(), key=lambda kv: -kv[1])[:5]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", required=True, help="path to diff or - for stdin")
    ap.add_argument("--body", default="", help="path to PR body markdown")
    ap.add_argument("--title", default="")
    ap.add_argument("--repo", default="", help="path to vllm checkout (enables CI group, AITER pin, DCO and rebase checks)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    diff = sys.stdin.read() if a.diff == "-" else open(a.diff, encoding="utf-8", errors="replace").read()
    body = open(a.body, encoding="utf-8", errors="replace").read() if a.body and os.path.exists(a.body) else a.body
    res = run(diff, body, a.title, a.repo)
    if a.json: print(json.dumps(res, indent=1)); return
    n = {"BLOCK": 0, "FIX": 0, "NOTE": 0}
    for x in res["findings"]: n[x["severity"]] += 1
    print(f"Pre-flight: {n['BLOCK']} blocking, {n['FIX']} fix-before-approval, {n['NOTE']} notes  ({res['changed_non_test_loc']} non-test lines added)\n")
    for x in res["findings"]:
        print(f"[{x['severity']}] {x['where']}\n    {x['message']}\n    asked by: {x['asked_by']}\n")
    if res["likely_commenters"]:
        print("Most likely to comment first:", ", ".join(f"{w} ({c})" for w, c in res["likely_commenters"]))
    sys.exit(1 if n["BLOCK"] else 0)

if __name__ == "__main__":
    main()
