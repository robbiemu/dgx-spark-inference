#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "runtime" / "sglang" / "adapters" / "sglang.sh"

with tempfile.TemporaryDirectory() as td:
    config = Path(td)
    (config / "inference.env").write_text(
        f'MODEL_CACHE_ROOT="/tmp/models"\n'
        f'PROJECT_ROOT="{ROOT}"\n'
        'ROLE="agentic"\n'
        'PORT="30000"\n'
        'CONTAINER_NAME="inference-agentic"\n'
    )
    env = dict(os.environ)
    env.update(
        CONFIG_ROOT=td,
        PORT="30001",
        CONTAINER_NAME="inference-agentic-helper",
    )
    proc = subprocess.run(
        [
            "bash", str(ADAPTER), "agentic-helper",
            str(ROOT / "runtime" / "sglang-nvfp4fix"), str(ROOT),
            "ornith-1.0-9b-fp8", "model",
            "profiles/ornith-1.0-9b-fp8/sglang.toml",
            "agentic-helper", "emit-yaml",
        ],
        text=True, capture_output=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "port: 30001" in proc.stdout, proc.stdout

print("PASS: helper unit port survives shared primary inference.env")
