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

RUNTIME_DIRS = [
    ROOT / "runtime" / "sglang",
]
ROLES = ROOT / "config" / "examples" / "roles.toml"

# Where capability records live in this repo, keyed by model_id.
CAP_PATHS = {
    "qwen36-27b-fp8": ROOT / "profiles" / "qwen36-27b-fp8" / "capability.toml",
}


def main() -> int:
    assert ROLES.is_file(), f"missing {ROLES}"
    for runtime_dir in RUNTIME_DIRS:
        assert runtime_dir.is_dir(), f"missing {runtime_dir}"

    role_policy = tomllib.loads(ROLES.read_text())["roles"]

    failures = []
    for runtime_dir in RUNTIME_DIRS:
        available = runtime_dir / "available.toml"
        runtime_cap = runtime_dir / "capability.toml"
        for path in (available, runtime_cap):
            assert path.is_file(), f"missing {path}"
        avail = tomllib.loads(available.read_text())
        for role, role_catalog in avail.get("roles", {}).items():
            candidates = role_catalog.get("models", [])
            assert candidates, f"{available}[{role}] lists no candidates"
            if role not in role_policy:
                # Role present in a runtime catalog but not in the policy example.
                # Skip — this lets downstream/installed catalogs carry roles the
                # reference policy does not define without failing the gate.
                continue
            required = role_policy[role]["required_model_capabilities"]
            request_toml = (
                f'requested_roles = ["{role}"]\n'
                f"[roles.{role}]\n"
                f"required_model_capabilities = {json.dumps(required)}\n"
            )
            for cand in candidates:
                check_candidate(
                    runtime_dir, runtime_cap, role, cand, request_toml, failures
                )

    if failures:
        print("FAIL: catalog compatibility:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("PASS: all production catalog candidates are capability-compatible")
    return 0


def check_candidate(runtime_dir, runtime_cap, role, cand, request_toml, failures):
            mid = cand["id"]
            cap_path = CAP_PATHS.get(mid)
            if not cap_path or not cap_path.is_file():
                failures.append(
                    f"{runtime_dir.name}/{role}/{mid}: no capability record found"
                )
                return
            with tempfile.TemporaryDirectory() as td:
                tdpath = Path(td)
                (tdpath / "request.toml").write_text(request_toml)
                models_dir = tdpath / "models"
                models_dir.mkdir()
                runtimes_dir = tdpath / "runtimes"
                runtimes_dir.mkdir()
                (models_dir / "m.toml").write_text(cap_path.read_text())
                (runtimes_dir / "r.toml").write_text(runtime_cap.read_text())
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(RESOLVER),
                        "--request",
                        str(tdpath / "request.toml"),
                        "--models-dir",
                        str(models_dir),
                        "--runtimes-dir",
                        str(runtimes_dir),
                        "--allow-unresolved",
                    ],
                    capture_output=True,
                    text=True,
                )
                if proc.returncode not in (0, 2):
                    failures.append(
                        f"{runtime_dir.name}/{role}/{mid}: resolver crashed: "
                        f"{proc.stderr.strip()}"
                    )
                    return
                res = json.loads(proc.stdout)
                ag = next(
                    (
                        resolved
                        for resolved in res.get("roles", [])
                        if resolved.get("role") == role
                    ),
                    None,
                )
                if (
                    not ag
                    or ag.get("status") != "resolved"
                    or ag.get("model_id") != mid
                ):
                    failures.append(
                        f"{runtime_dir.name}/{role}/{mid}: did NOT resolve ({ag})"
                    )
                else:
                    print(
                        f"PASS: {runtime_dir.name} candidate '{mid}' "
                        f"resolves for {role}"
                    )


if __name__ == "__main__":
    raise SystemExit(main())
