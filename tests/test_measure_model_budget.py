#!/usr/bin/env python3
"""test_measure_model_budget.py — test the measurement tool's parsing,
refusal behavior, and TOML emission.

Run directly: python3 tests/test_measure_model_budget.py
Covers: dense single-pool, hybrid/Mamba, multiple KV pools (refuse),
missing markers (refuse), malformed KV (refuse), invalid mem-fraction.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "memory_planner" / "measure_model_budget.py"

# ---- Inline log fixtures (sanitized; no real model data) ----

DENSE_LOG = """\
[2026-01-01 00:00:00] Load weight begin. avail mem=100.00 GB
Multi-thread loading shards: 100% Completed
[2026-01-01 00:01:00] Load weight end. elapsed=60s, type=DenseModel, quant=fp8, avail mem=80.00 GB, mem usage=20.00 GB.
[2026-01-01 00:01:01] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 500000, K size: 10.00 GB, V size: 10.00 GB
[2026-01-01 00:01:02] Capture decode CUDA graph end. mem usage=1.50 GB, avail mem=58.50 GB.
[2026-01-01 00:01:03] max_total_num_tokens=500000, context_len=131072, available_gpu_mem=58.50 GB
"""

HYBRID_LOG = """\
[2026-01-01 00:00:00] Load weight begin. avail mem=80.00 GB
Multi-thread loading shards: 100% Completed
[2026-01-01 00:01:00] Load weight end. elapsed=60s, type=HybridModel, quant=fp8, avail mem=60.00 GB, mem usage=20.00 GB.
[2026-01-01 00:01:01] Mamba Cache is allocated. max_mamba_cache_size: 22, conv_state size: 0.06GB, ssm_state size: 3.23GB intermediate_ssm_state_cache size: 2.25GB intermediate_conv_window_cache size: 0.02GB
[2026-01-01 00:01:02] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 300000, K size: 3.00 GB, V size: 3.00 GB
[2026-01-01 00:01:03] max_total_num_tokens=300000, context_len=262144, available_gpu_mem=50.00 GB
"""

MULTI_POOL_LOG = """\
[2026-01-01 00:00:00] Load weight begin. avail mem=80.00 GB
[2026-01-01 00:01:00] Load weight end. elapsed=60s, type=MTPModel, quant=fp8, avail mem=60.00 GB, mem usage=20.00 GB.
[2026-01-01 00:01:01] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 300000, K size: 3.00 GB, V size: 3.00 GB
[2026-01-01 00:01:02] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 50000, K size: 0.10 GB, V size: 0.10 GB
[2026-01-01 00:01:03] max_total_num_tokens=300000, context_len=262144, available_gpu_mem=50.00 GB
"""

MISSING_MARKERS_LOG = """\
[2026-01-01 00:00:00] Some random sglang startup line
[2026-01-01 00:00:01] Another line with no memory data
"""

MALFORMED_KV_LOG = """\
[2026-01-01 00:00:00] Load weight begin. avail mem=80.00 GB
[2026-01-01 00:01:00] Load weight end. elapsed=60s, type=BadModel, avail mem=60.00 GB, mem usage=20.00 GB.
[2026-01-01 00:01:01] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 0, K size: 0.00 GB, V size: 0.00 GB
"""


def run_tool(log_text: str, *args: str) -> tuple[int, str, str]:
    """Run the measurement tool with the given log on stdin. Returns (rc, stdout, stderr)."""
    p = subprocess.run(
        [sys.executable, str(TOOL), *args],
        input=log_text, capture_output=True, text=True, timeout=10,
    )
    return p.returncode, p.stdout, p.stderr


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {name}" + (f"  {detail}" if detail else ""))
    return ok


def main() -> int:
    fail = 0
    say = lambda s: print(f"\n=== {s} ===")

    say("Dense single-pool: exit 0, parseable TOML")
    rc, out, err = run_tool(DENSE_LOG, "--model-id", "dense-test", "--mem-fraction", "0.60")
    if not check("exit 0", rc == 0, f"rc={rc}"):
        fail += 1
    # stdout should be parseable TOML with the right model_id
    try:
        # The tool emits a [[profiles]] block; parse it as a ledger and extract
        entry = tomllib.loads(out.strip())
        prof = entry["profiles"][0]
        ok = prof["model_id"] == "dense-test"
        ok = ok and prof["budget"]["weights_gib"] == 20.0
        ok = ok and prof["budget"]["target_kv_tokens"] == 500000
        if not check("TOML parses + correct values", ok):
            fail += 1
    except Exception as e:
        check("TOML parses", False, str(e))
        fail += 1

    say("Hybrid/Mamba: exit 0, mamba reflected in overhead")
    rc, out, err = run_tool(HYBRID_LOG, "--model-id", "hybrid-test", "--mem-fraction", "0.50")
    if not check("exit 0", rc == 0, f"rc={rc}"):
        fail += 1
    try:
        entry = tomllib.loads(out.strip())
        prof = entry["profiles"][0]
        overhead = prof["budget"].get("static_overhead_gib", 0)
        mamba_in_trace = "5.56" in err or "5.56" in out
        if not check("overhead > 0 (includes Mamba)", overhead > 0, f"overhead={overhead}"):
            fail += 1
        if not check("Mamba state in trace", mamba_in_trace):
            fail += 1
    except Exception as e:
        check("TOML parses", False, str(e))
        fail += 1

    say("Multiple KV pools: nonzero, empty stdout, stderr explains")
    rc, out, err = run_tool(MULTI_POOL_LOG, "--model-id", "mtp-test", "--mem-fraction", "0.60")
    if not check("exit nonzero", rc != 0, f"rc={rc}"):
        fail += 1
    if not check("empty stdout", out.strip() == "", repr(out[:80])):
        fail += 1
    if not check("stderr mentions composite", "composite" in err.lower() or "multi-pool" in err.lower()):
        fail += 1

    say("Missing required markers: nonzero, empty stdout")
    rc, out, err = run_tool(MISSING_MARKERS_LOG, "--model-id", "bad-test", "--mem-fraction", "0.60")
    if not check("exit nonzero", rc != 0, f"rc={rc}"):
        fail += 1
    if not check("empty stdout", out.strip() == "", repr(out[:80])):
        fail += 1

    say("Malformed KV (zero tokens): nonzero, empty stdout")
    rc, out, err = run_tool(MALFORMED_KV_LOG, "--model-id", "bad-kv", "--mem-fraction", "0.60")
    if not check("exit nonzero", rc != 0, f"rc={rc}"):
        fail += 1
    if not check("empty stdout", out.strip() == "", repr(out[:80])):
        fail += 1

    say("Invalid mem-fraction (0): nonzero")
    rc, out, err = run_tool("test", "--model-id", "x", "--mem-fraction", "0")
    if not check("exit nonzero", rc != 0, f"rc={rc}"):
        fail += 1

    say("Invalid mem-fraction (1.5): nonzero")
    rc, out, err = run_tool("test", "--model-id", "x", "--mem-fraction", "1.5")
    if not check("exit nonzero", rc != 0, f"rc={rc}"):
        fail += 1

    say("Invalid mem-fraction (-1): nonzero")
    rc, out, err = run_tool("test", "--model-id", "x", "--mem-fraction", "-1")
    if not check("exit nonzero", rc != 0, f"rc={rc}"):
        fail += 1

    print()
    if fail:
        print(f"FAIL: {fail} check(s) failed")
        return 1
    print("PASS: all measurement-tool checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
