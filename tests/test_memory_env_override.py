#!/usr/bin/env python3
"""Test 7 — memory env-var override tier (DGX_MEM_FRACTION_STATIC / DGX_MAX_TOTAL_TOKENS).

Mirrors Test 4 (invokes the REAL adapter in emit-yaml mode) but exercises the
env-var override precedence added for the v0.2 memory preflight:

  - When DGX_MEM_FRACTION_STATIC + DGX_MAX_TOTAL_TOKENS are exported, the rendered
    YAML uses THEM (override wins over the spec's pinned 0.60).
  - max-total-tokens is emitted ONLY when DGX_MAX_TOTAL_TOKENS is set (preserves
    "empty = omit key"; the v0.1 baseline must NOT gain a max-total-tokens line).
  - When the env vars are UNSET, behavior is unchanged: spec's 0.60 reaches the
    YAML and no max-total-tokens line appears (regression guard for the override tier).

No GPU, no docker: emit-yaml mode performs NO safety rails and launches nothing.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "runtime" / "sglang" / "adapter.sh"
RUNTIME_ROOT = ROOT / "runtime" / "sglang"
SPEC_REL = "profiles/qwen36-27b-fp8/sglang.toml"

# The spec pins mem_fraction_static = 0.60; the override must win over it.
OVERRIDE_FRAC = "0.42"
OVERRIDE_MTT = "200000"


def _render(env_overrides: dict[str, str]) -> str:
    """Invoke the REAL adapter in emit-yaml mode and return the rendered YAML."""
    env = dict(os.environ)
    td = tempfile.mkdtemp()
    (Path(td) / "inference.env").write_text(
        'MODEL_CACHE_ROOT="/tmp/nonexistent-model-cache"\n'
        'PORT="30000"\n'
        'CONTAINER_NAME="inference-agentic"\n'
        'PROJECT_ROOT="%s"\n' % ROOT
    )
    env["CONFIG_ROOT"] = td
    env.update(env_overrides)
    proc = subprocess.run(
        ["bash", str(ADAPTER), "agentic", str(RUNTIME_ROOT), str(ROOT),
         "qwen36-27b-fp8", "model", SPEC_REL, "qwen3.6-27b-agentic", "emit-yaml"],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, f"adapter emit-yaml failed (rc={proc.returncode}):\n{proc.stderr}"
    return proc.stdout


def _mfs_value(out: str):
    for line in out.splitlines():
        if line.startswith("mem-fraction-static:"):
            return float(line.split(":", 1)[1].strip())
    return None


def _has_line(out: str, prefix: str) -> str | None:
    for line in out.splitlines():
        if line.startswith(prefix):
            return line
    return None


def main() -> int:
    failures = []

    # --- Case 1: env override ACTIVE -> both knobs applied ---
    out = _render({"DGX_MEM_FRACTION_STATIC": OVERRIDE_FRAC,
                   "DGX_MAX_TOTAL_TOKENS": OVERRIDE_MTT})
    mfs = _mfs_value(out)
    if mfs != float(OVERRIDE_FRAC):
        failures.append(f"override ACTIVE: mem-fraction-static == {OVERRIDE_FRAC} (got {mfs})")
    mtt_line = _has_line(out, "max-total-tokens:")
    if mtt_line is None or mtt_line.split(":", 1)[1].strip() != OVERRIDE_MTT:
        failures.append(f"override ACTIVE: max-total-tokens: {OVERRIDE_MTT} (got {mtt_line})")

    # --- Case 2: env UNSET -> spec value (0.6) + NO max-total-tokens line ---
    out2 = _render({})
    mfs2 = _mfs_value(out2)
    if mfs2 != 0.6:
        failures.append(f"env UNSET: mem-fraction-static == 0.6 (got {mfs2}) — override tier broke the fallback")
    if _has_line(out2, "max-total-tokens:") is not None:
        failures.append("env UNSET: max-total-tokens line SHOULD be absent (empty = omit key)")

    # --- Case 3: only FRAC set -> frac overridden, max-total-tokens still absent ---
    out3 = _render({"DGX_MEM_FRACTION_STATIC": OVERRIDE_FRAC})
    if _mfs_value(out3) != float(OVERRIDE_FRAC):
        failures.append("partial: DGX_MEM_FRACTION_STATIC alone should still override frac")
    if _has_line(out3, "max-total-tokens:") is not None:
        failures.append("partial: max-total-tokens should be absent when DGX_MAX_TOTAL_TOKENS unset")

    if failures:
        print("FAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("PASS: env-var override tier (DGX_MEM_FRACTION_STATIC > spec; DGX_MAX_TOTAL_TOKENS emits when set, omits when empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
