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

DFLASH_CAPS = [
    ROOT / "bundles" / "experimental" / "qwen36-27b-fp8-dflash" / "capability.toml",
    ROOT / "bundles" / "experimental" / "laguna-s-2.1-nvfp4-dflash" / "capability.toml",
]
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
    assert RUNTIME_CAP.is_file(), f"missing {RUNTIME_CAP}"
    import tomllib
    for dflash_cap in DFLASH_CAPS:
        assert dflash_cap.is_file(), f"missing {dflash_cap}"
        caps = tomllib.loads(dflash_cap.read_text()).get("capabilities", [])
        assert "structured_output" not in caps, (
            f"{dflash_cap} must not claim structured_output; got: {caps!r}"
        )
        assert "logprobs" not in caps and "return_logprob" not in caps, (
            f"{dflash_cap} must not claim logprobs; got: {caps!r}"
        )
        if dflash_cap.parent.name == "laguna-s-2.1-nvfp4-dflash":
            roles = tomllib.loads(dflash_cap.read_text()).get("roles", [])
            assert roles == ["agentic-experimental"], (
                f"Laguna DFlash must be isolated to agentic-experimental; "
                f"got: {roles!r}"
            )

        with tempfile.TemporaryDirectory() as td:
            tdpath = Path(td)
            (tdpath / "request.toml").write_text(REQUEST_TOML)
            models_dir = tdpath / "models"; models_dir.mkdir()
            runtimes_dir = tdpath / "runtimes"; runtimes_dir.mkdir()
            (models_dir / "dflash.toml").write_text(dflash_cap.read_text())
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
                print(f"FAIL: {dflash_cap.parent.name} was WRONGLY resolved for agentic: {agentic}", file=sys.stderr)
                return 1
            print(f"PASS: {dflash_cap.parent.name} rejected for production agentic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
