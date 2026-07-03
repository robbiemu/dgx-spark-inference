#!/usr/bin/env python3
"""Cache dtype is explicit for NVFP4 candidates and omitted for FP8 baseline."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "runtime" / "sglang" / "adapters" / "sglang.sh"
RUNTIME = ROOT / "runtime" / "sglang"

CASES = {
    "profiles/qwen36-27b-fp8/sglang.toml": None,
    "profiles/qwen36-27b-nvfp4-kv-fp8-e4m3/sglang.toml": "fp8_e4m3",
}


def render(spec: str) -> str:
    env = os.environ.copy()
    env["MODEL_CACHE_ROOT"] = "/nonexistent-test-cache"
    proc = subprocess.run(
        [
            str(ADAPTER),
            "agentic",
            str(RUNTIME),
            str(ROOT),
            Path(spec).parent.name,
            "model",
            spec,
            "qwen3.6-27b-agentic",
            "emit-yaml",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def main() -> int:
    for spec, expected in CASES.items():
        output = render(spec)
        lines = [
            line
            for line in output.splitlines()
            if line.startswith("kv-cache-dtype:")
        ]
        if expected is None:
            assert not lines, f"{spec}: cache dtype should be omitted: {lines}"
        else:
            assert lines == [f'kv-cache-dtype: "{expected}"'], (
                f"{spec}: unexpected cache dtype lines: {lines}"
            )
        print(f"PASS: {spec} -> {expected or 'omitted'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
