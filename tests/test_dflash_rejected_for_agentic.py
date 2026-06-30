#!/usr/bin/env python3
"""Test 2 — DFlash bundle is REJECTED for the `agentic` role.

temp models-dir containing ONLY DFlash's capability.toml (which OMITS
structured_output — there is deliberately no structured_output_prompt_only
identifier) + the REAL runtime record -> run the REAL resolver -> assert
status=unresolved. This is the test that makes the capability split real.

Why unresolved: agentic requires structured_output; DFlash's capabilities list
omits it (DFlash can only do prompt-only JSON, which is not grammar-constrained).
The role vocabulary is "things a role can require"; DFlash just doesn't provide
the one agentic requires."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOLVER = ROOT / "tools" / "resolve_service_plan.py"

DFLASH_CAP = ROOT / "bundles" / "experimental" / "qwen36-27b-fp8-dflash" / "capability.toml"
RUNTIME_CAP = ROOT / "runtime" / "sglang" / "capability.toml"

REQUEST_TOML = """\
requested_roles = ["agentic"]
[roles.agentic]
required_model_capabilities = [
  "chat_completion",
  "tool_calling",
  "structured_output",
]
"""


def main() -> int:
    assert DFLASH_CAP.is_file(), f"missing {DFLASH_CAP}"
    assert RUNTIME_CAP.is_file(), f"missing {RUNTIME_CAP}"
    # Sanity: the DFlash record must NOT claim structured_output (this is what we
    # are testing the resolver enforces; if someone "fixes" it, fail loudly here).
    import tomllib
    caps = tomllib.loads(DFLASH_CAP.read_text()).get("capabilities", [])
    assert "structured_output" not in caps, (
        "DFlash capability.toml must not claim structured_output; got: " + repr(caps)
    )

    with tempfile.TemporaryDirectory() as td:
        tdpath = Path(td)
        (tdpath / "request.toml").write_text(REQUEST_TOML)
        models_dir = tdpath / "models"; models_dir.mkdir()
        runtimes_dir = tdpath / "runtimes"; runtimes_dir.mkdir()
        (models_dir / "dflash.toml").write_text(DFLASH_CAP.read_text())
        (runtimes_dir / "sglang.toml").write_text(RUNTIME_CAP.read_text())

        proc = subprocess.run(
            [
                sys.executable, str(RESOLVER),
                "--request", str(tdpath / "request.toml"),
                "--models-dir", str(models_dir),
                "--runtimes-dir", str(runtimes_dir),
                "--allow-unresolved",
            ],
            capture_output=True, text=True,
        )
        if proc.returncode not in (0, 2):
            print(f"resolver crashed (rc={proc.returncode}):\n{proc.stderr}", file=sys.stderr)
            return 1
        result = json.loads(proc.stdout)
        agentic = next((r for r in result.get("roles", []) if r.get("role") == "agentic"), None)
        if agentic and agentic.get("status") == "resolved":
            print(f"FAIL: DFlash was WRONGLY resolved for agentic: {agentic}", file=sys.stderr)
            return 1
        print("PASS: DFlash rejected for agentic (lacks grammar-constrained structured_output)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
