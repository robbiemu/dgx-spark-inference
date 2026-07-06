#!/usr/bin/env python3
"""test_fraction_base_validation.py — test the resolver's fraction_base
enum validation and derivation behavior for both valid bases + invalid rejection.

Run directly: python3 tests/test_fraction_base_validation.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESOLVER = ROOT / "tools" / "memory_planner" / "resolve_memory_plan.py"

LEDGER_TEMPLATE = """\
[[profiles]]
model_id = "{mid}"
[profiles.budget]
weights_gib = 20.0
target_kv_tokens = 300000
kv_bytes_per_token = 32768.0
static_pad_gib = 0.5
static_overhead_gib = 10.0
cuda_graph_peak_gib = 1.0
request_workspace_gib = 2.0
gpu_headroom_gib = 1.0
{fraction_base_line}
"""

PLAN = """\
device.total_gib = 121.7
[observed]
gpu_free_now_gib = 80.0
memavailable_now_gib = 80.0
[[admit]]
role = "test"
model_id = "{mid}"
"""


def run_resolver(ledger_text: str, plan_text: str) -> tuple[int, str, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as lf:
        lf.write(ledger_text); lf.flush(); ledger = lf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as pf:
        pf.write(plan_text); pf.flush(); plan = pf.name
    p = subprocess.run(
        [sys.executable, str(RESOLVER), ledger, plan],
        capture_output=True, text=True, timeout=10,
    )
    return p.returncode, p.stdout, p.stderr


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {name}" + (f"  {detail}" if detail else ""))
    return ok


def main() -> int:
    fail = 0
    mid = "test-model"

    print("\n=== fraction_base = a_preload (default) ===")
    ledger = LEDGER_TEMPLATE.format(mid=mid, fraction_base_line="")
    plan = PLAN.format(mid=mid)
    rc, out, err = run_resolver(ledger, plan)
    if not check("ADMIT", rc == 0, f"rc={rc}"):
        fail += 1
    # fraction should be static_required / A_preload = (20+9.16+10+0.5) / 80 = 0.496
    if not check("fraction derived against A_preload", "0.496" in out or "0.495" in out, out[:200]):
        fail += 1

    print("\n=== fraction_base = device_total ===")
    ledger = LEDGER_TEMPLATE.format(mid=mid, fraction_base_line='fraction_base = "device_total"')
    rc, out, err = run_resolver(ledger, plan)
    if not check("ADMIT", rc == 0, f"rc={rc}"):
        fail += 1
    # fraction should be static_required / device_total = 39.66 / 121.7 = 0.326
    if not check("fraction derived against device_total", "0.326" in out or "0.325" in out, out[:200]):
        fail += 1

    print("\n=== fraction_base = invalid value ===")
    ledger = LEDGER_TEMPLATE.format(mid=mid, fraction_base_line='fraction_base = "total_device"')
    rc, out, err = run_resolver(ledger, plan)
    if not check("exit nonzero (controlled refusal)", rc != 0, f"rc={rc}"):
        fail += 1
    if not check("REFUSING in stderr", "REFUSING" in err, err[:100]):
        fail += 1
    if not check("no Python traceback", "Traceback" not in err):
        fail += 1
    if not check("mentions valid options", "a_preload" in err and "device_total" in err):
        fail += 1

    print()
    if fail:
        print(f"FAIL: {fail} check(s) failed")
        return 1
    print("PASS: all fraction_base checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
