#!/usr/bin/env python3
"""test_adapter_precedence.py — integration test for the three mem_fraction_static
launch states:

1. Planner active (DGX_MEM_FRACTION_STATIC exported) → overrides the profile value.
2. No planner pair, ordinary mode → profile fallback passes through.
3. No planner pair, DGX_MEMORY_PREFLIGHT=required → launch refuses before SGLang.

States 1 and 2 are verified via the adapter's emit-yaml mode (no docker/GPU).
State 3 is verified via dispatch.sh with stubs (no docker/GPU).

Run directly: python3 tests/test_adapter_precedence.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADAPTER = ROOT / "runtime" / "sglang" / "adapters" / "sglang.sh"
DISPATCH = ROOT / "src" / "inferencectl" / "dispatch.sh"
RUNTIME_ROOT = ROOT / "runtime" / "sglang"
SPEC_REL = "profiles/qwen36-27b-fp8/sglang.toml"  # pins mem_fraction_static = 0.60


def _render(env_overrides: dict[str, str]) -> str:
    """Invoke the REAL adapter in emit-yaml mode and return the rendered YAML."""
    env = dict(os.environ)
    td = tempfile.mkdtemp()
    (Path(td) / "inference.env").write_text(
        'MODEL_CACHE_ROOT="/tmp/nonexistent"\n'
        'PORT="30000"\n'
        'CONTAINER_NAME="inference-agentic"\n'
        f'PROJECT_ROOT="{ROOT}"\n'
    )
    env["CONFIG_ROOT"] = td
    env.update(env_overrides)
    proc = subprocess.run(
        ["bash", str(ADAPTER), "agentic", str(RUNTIME_ROOT), str(ROOT),
         "qwen36-27b-fp8", "model", SPEC_REL, "qwen3.6-27b-agentic", "emit-yaml"],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, f"adapter emit-yaml failed:\n{proc.stderr}"
    return proc.stdout


def _mfs_value(out: str) -> float | None:
    for line in out.splitlines():
        if line.startswith("mem-fraction-static:"):
            return float(line.split(":", 1)[1].strip())
    return None


def _run_dispatch_required_mode() -> tuple[int, str]:
    """Run dispatch.sh with DGX_MEMORY_PREFLIGHT=required but no planner pair.
    Should refuse (exit nonzero) before reaching the adapter."""
    td = Path(tempfile.mkdtemp())
    env = dict(os.environ)
    env["CONFIG_ROOT"] = str(td)
    env["DGX_MEMORY_PREFLIGHT"] = "required"
    # No memory_ledger.toml or memory_plan.toml → no planner pair
    (td / "inference.env").write_text(
        f'MODEL_CACHE_ROOT="{td}"\nPORT="30000"\nCONTAINER_NAME="x"\nPROJECT_ROOT="{ROOT}"\n'
    )
    (td / "active-models.toml").write_text(
        '[active.agentic]\nmodel_id="qwen36-27b-fp8"\nruntime_id="sglang"\n'
    )
    (td / "runtimes.toml").write_text(
        f'[runtimes.sglang]\nproject_root="{RUNTIME_ROOT}"\n'
    )
    proc = subprocess.run(
        ["bash", str(DISPATCH), "agentic"],
        capture_output=True, text=True, env=env, timeout=10,
    )
    return proc.returncode, proc.stderr


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {name}" + (f"  {detail}" if detail else ""))
    return ok


def main() -> int:
    fail = 0

    print("\n=== State 1: planner active (env override wins over profile) ===")
    out = _render({"DGX_MEM_FRACTION_STATIC": "0.42"})
    mfs = _mfs_value(out)
    if not check("override 0.42 wins over profile 0.60", mfs == 0.42, f"got {mfs}"):
        fail += 1

    print("\n=== State 2: no planner pair, ordinary mode (profile fallback) ===")
    out2 = _render({})
    mfs2 = _mfs_value(out2)
    if not check("profile 0.60 passes through as fallback", mfs2 == 0.6, f"got {mfs2}"):
        fail += 1

    print("\n=== State 3: no planner pair, DGX_MEMORY_PREFLIGHT=required (refuses) ===")
    rc, err = _run_dispatch_required_mode()
    if not check("launch refuses (nonzero exit)", rc != 0, f"rc={rc}"):
        fail += 1
    if not check("admission refuses (not legacy launch)", "REFUSING" in err.upper() or "REFUSE" in err.upper()):
        fail += 1

    print()
    if fail:
        print(f"FAIL: {fail} check(s) failed")
        return 1
    print("PASS: adapter precedence — all three launch states verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
