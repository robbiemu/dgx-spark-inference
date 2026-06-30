#!/usr/bin/env python3
"""Test 3 — Available-catalog compatibility.

Every candidate listed in the PRODUCTION runtime catalog
(runtime/sglang/available.toml[agentic].models) must satisfy the agentic role
according to the REAL resolver, using the REAL capability records in this repo.
Guards the live artifact against a future edit reintroducing DFlash (or any other
incompatible candidate) under agentic."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOLVER = ROOT / "tools" / "resolve_service_plan.py"

AVAILABLE = ROOT / "runtime" / "sglang" / "available.toml"
RUNTIME_CAP = ROOT / "runtime" / "sglang" / "capability.toml"
ROLES = ROOT / "config" / "examples" / "roles.toml"

# Where capability records live in this repo, keyed by model_id.
CAP_PATHS = {
    "qwen36-27b-fp8": ROOT / "profiles" / "qwen36-27b-fp8" / "capability.toml",
}


def main() -> int:
    for p in (AVAILABLE, RUNTIME_CAP, ROLES):
        assert p.is_file(), f"missing {p}"
    avail = tomllib.loads(AVAILABLE.read_text())
    role_block = avail.get("roles", {}).get("agentic", {})
    candidates = role_block.get("models", [])
    assert candidates, "available.toml[agentic] lists no candidates"

    required = tomllib.loads(ROLES.read_text())["roles"]["agentic"]["required_model_capabilities"]
    request_toml = (
        'requested_roles = ["agentic"]\n'
        "[roles.agentic]\n"
        f"required_model_capabilities = {json.dumps(required)}\n"
    )

    failures = []
    for cand in candidates:
        mid = cand["id"]
        cap_path = CAP_PATHS.get(mid)
        if not cap_path or not cap_path.is_file():
            failures.append(f"{mid}: no capability record found for this catalog candidate")
            continue
        with tempfile.TemporaryDirectory() as td:
            tdpath = Path(td)
            (tdpath / "request.toml").write_text(request_toml)
            models_dir = tdpath / "models"; models_dir.mkdir()
            runtimes_dir = tdpath / "runtimes"; runtimes_dir.mkdir()
            (models_dir / "m.toml").write_text(cap_path.read_text())
            (runtimes_dir / "r.toml").write_text(RUNTIME_CAP.read_text())
            proc = subprocess.run(
                [sys.executable, str(RESOLVER), "--request", str(tdpath / "request.toml"),
                 "--models-dir", str(models_dir), "--runtimes-dir", str(runtimes_dir),
                 "--allow-unresolved"],
                capture_output=True, text=True,
            )
            if proc.returncode not in (0, 2):
                failures.append(f"{mid}: resolver crashed: {proc.stderr.strip()}")
                continue
            res = json.loads(proc.stdout)
            ag = next((r for r in res.get("roles", []) if r.get("role") == "agentic"), None)
            if not ag or ag.get("status") != "resolved" or ag.get("model_id") != mid:
                failures.append(f"{mid}: did NOT resolve as required ({ag})")
            else:
                print(f"PASS: catalog candidate '{mid}' resolves for agentic")

    if failures:
        print("FAIL: catalog compatibility:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("PASS: all production available.toml[agentic] candidates are capability-compatible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
