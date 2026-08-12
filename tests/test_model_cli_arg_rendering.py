#!/usr/bin/env python3
"""Model-specific CLI vocabulary is rendered by the real adapter path."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "runtime" / "sglang" / "adapters" / "sglang.sh"
RUNTIME_ROOT = ROOT / "runtime" / "sglang"


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        temporary = Path(td)
        config_root = temporary / "config"
        config_root.mkdir()
        (config_root / "inference.env").write_text(
            'MODEL_CACHE_ROOT="/tmp/nonexistent-model-cache"\n'
            'PORT="30135"\n'
            'CONTAINER_NAME="inference-test"\n'
            f'PROJECT_ROOT="{ROOT}"\n'
        )
        spec = temporary / "hybrid.toml"
        spec.write_text(
            'kind = "model-runtime-config"\n'
            'model_id = "hybrid-test"\n'
            'runtime_id = "sglang-test"\n'
            '[identity]\n'
            'source_repository = "nvidia/test"\n'
            'source_revision = "0123456789abcdef"\n'
            'quantization = "nvfp4"\n'
            '[launch]\n'
            'needs_trust_remote_code = true\n'
            'fp4_gemm_backend = "marlin"\n'
            'moe_runner_backend = "marlin"\n'
            'mamba_backend = "flashinfer"\n'
            'mamba_ssm_dtype = "float16"\n'
            'max_mamba_cache_size = 20\n'
            'mamba_radix_cache_strategy = "extra_buffer"\n'
            'chunked_prefill_size = 32768\n'
            'cuda_graph_max_bs_decode = 16\n'
        )
        env = dict(os.environ)
        env["CONFIG_ROOT"] = str(config_root)
        proc = subprocess.run(
            [
                "bash",
                str(ADAPTER),
                "test-role",
                str(RUNTIME_ROOT),
                str(ROOT),
                "hybrid-test",
                "model",
                str(spec),
                "hybrid-test",
                "emit-model-args",
            ],
            capture_output=True,
            text=True,
            env=env,
        )

    assert proc.returncode == 0, proc.stderr
    expected = {
        "--trust-remote-code",
        "--fp4-gemm-backend marlin",
        "--moe-runner-backend marlin",
        "--mamba-backend flashinfer",
        "--mamba-ssm-dtype float16",
        "--max-mamba-cache-size 20",
        "--mamba-radix-cache-strategy extra_buffer",
        "--chunked-prefill-size 32768",
        "--cuda-graph-max-bs-decode 16",
    }
    missing = sorted(item for item in expected if item not in proc.stdout)
    if missing:
        raise AssertionError(f"missing CLI args: {missing}; rendered={proc.stdout!r}")
    print("PASS: real adapter renders hybrid-model CLI vocabulary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
