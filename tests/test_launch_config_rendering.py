#!/usr/bin/env python3
"""Test 4 — Launch-config rendering via the REAL adapter.

Invokes the REAL adapter in its `emit-yaml` mode (the adapter's own render path,
not a reimplementation) with the baseline sglang.toml + the runtime manifest ->
asserts the rendered YAML contains the four pinned values. Guards the regression
that motivated the prep work (mem-fraction-static must reach the YAML).

No GPU, no docker: emit-yaml mode performs NO safety rails and launches nothing;
it merges manifest defaults + spec overrides and prints the YAML using a
placeholder key. We also stub MODEL_CACHE_ROOT so the adapter's config-discovery
succeeds without a real host path."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "runtime" / "sglang" / "adapters" / "sglang.sh"
RUNTIME_ROOT = ROOT / "runtime" / "sglang"
SPEC_REL = "profiles/qwen36-27b-fp8/sglang.toml"

EXPECTED = {
    "context-length: 262144",
    "max-running-requests: 1",
    "max-queued-requests: 1",
}


def main() -> int:
    assert ADAPTER.is_file(), f"missing {ADAPTER}"
    env = dict(os.environ)
    # Config discovery: point CONFIG_ROOT at a temp dir carrying inference.env so
    # the adapter does not touch /etc. No real model cache is needed for emit-yaml.
    import tempfile
    td = tempfile.mkdtemp()
    (Path(td) / "inference.env").write_text(
        'MODEL_CACHE_ROOT="/tmp/nonexistent-model-cache"\n'
        'PORT="30000"\n'
        'CONTAINER_NAME="inference-agentic"\n'
        'PROJECT_ROOT="%s"\n' % ROOT
    )
    env["CONFIG_ROOT"] = td

    proc = subprocess.run(
        ["bash", str(ADAPTER), "agentic", str(RUNTIME_ROOT), str(ROOT),
         "qwen36-27b-fp8", "model", SPEC_REL, "qwen3.6-27b-agentic", "emit-yaml"],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        print(f"adapter emit-yaml failed (rc={proc.returncode}):\n{proc.stderr}", file=sys.stderr)
        return 1
    out = proc.stdout
    missing = [e for e in EXPECTED if e not in out]
    # mem-fraction-static must be PRESENT and equal 0.6 (0.60 and 0.6 are the same
    # value; TOML parses 0.60 -> float 0.6, which the adapter prints as "0.6". We
    # assert on the value semantically so the regression guard catches an omitted
    # or wrong value without being brittle to float formatting).
    has_mfs = False
    for line in out.splitlines():
        if line.startswith("mem-fraction-static:"):
            try:
                if float(line.split(":", 1)[1].strip()) == 0.6:
                    has_mfs = True
            except ValueError:
                pass
    if not has_mfs:
        missing.append("mem-fraction-static: <present and == 0.6>")
    if missing:
        print("FAIL: rendered YAML missing expected lines:", file=sys.stderr)
        for m in missing:
            print(f"  missing: {m}", file=sys.stderr)
        print("--- rendered YAML ---", file=sys.stderr)
        print(out, file=sys.stderr)
        return 1
    # also assert the placeholder (never a real key) is what appears
    if "api-key: \"REDACTED-PLACEHOLDER\"" not in out:
        print("FAIL: rendered YAML did not use the placeholder key", file=sys.stderr)
        return 1
    print("PASS: real adapter render path emits the pinned launch values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
