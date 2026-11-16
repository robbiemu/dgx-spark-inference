#!/usr/bin/env python3
"""vLLM implementation of dispatch's seven-argument adapter contract."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib


def positive_integer(value, label: str) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"[1-9][0-9]*", str(value)):
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def build_command(runtime: Path, project: Path, model: str, kind: str,
                  spec: str, served: str, env: dict) -> tuple[list[str], dict, dict]:
    manifest = tomllib.loads((runtime / "runtime-manifest.toml").read_text())
    profile = tomllib.loads((project / spec).read_text())
    if (kind != "model" or profile.get("kind") != "model-runtime-config"
            or manifest.get("runtime_kind") != "vllm"
            or profile.get("model_id") != model
            or profile.get("runtime_id") != manifest.get("runtime_id")):
        raise ValueError("model/runtime identity or kind mismatch")
    if manifest.get("allocation_probe") != "capacity_probe.py":
        raise ValueError("vLLM requires its allocation probe")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", manifest["image_id"]):
        raise ValueError("a pinned image ID is required")
    if not manifest["image"] or manifest["image"].startswith("-"):
        raise ValueError("invalid image reference")
    if not served or any(c.isspace() for c in served):
        raise ValueError("a served model name is required")
    name = env["CONTAINER_NAME"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("invalid container name")
    port = positive_integer(env["PORT"], "PORT")
    if port > 65535:
        raise ValueError("PORT is out of range")
    identity = profile["identity"]
    repo, revision = identity["source_repository"], identity["source_revision"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("invalid model repository")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("a pinned model revision is required")
    cache = Path(env["MODEL_CACHE_ROOT"]).resolve()
    relative = Path("models--" + repo.replace("/", "--")) / "snapshots" / revision
    snapshot = (cache / relative).resolve()
    if not snapshot.is_relative_to(cache) or not (snapshot / "config.json").is_file():
        raise ValueError("pinned model snapshot is missing or outside the cache")
    allocation = profile["allocation"]
    tokens = positive_integer(allocation["pool_tokens"], "allocation.pool_tokens")
    cache_bytes = positive_integer(allocation["cache_bytes"], "allocation.cache_bytes")
    if positive_integer(env["DGX_MAX_TOTAL_TOKENS"], "admitted token cap") != tokens:
        raise ValueError("admitted token cap differs from the qualified fixed allocation")
    config = dict(profile.get("launch", {}))
    reserved = {"api-key", "model", "config", "host", "port", "served-model-name",
                "kv-cache-memory-bytes", "gpu-memory-utilization", "device-memory-utilization"}
    if any(k.replace("_", "-") in reserved for k in config):
        raise ValueError("launch config overrides a managed identity, auth, or allocation field")
    config.update({"host": "0.0.0.0", "port": port, "served-model-name": served,
                   "kv-cache-memory-bytes": cache_bytes})
    secret = env.get("SGLANG_API_KEY", "")
    if not re.fullmatch(r"[0-9a-f]{64}", secret):
        raise ValueError("the operator API key must have the existing 64-hex shape")
    configdir = Path(env["DGX_RUNTIME_CONFIG_DIR"]).resolve()
    configdir.mkdir(mode=0o700, parents=True, exist_ok=True)
    configdir.chmod(0o700)
    fd, path = tempfile.mkstemp(prefix="vllm-", suffix=".json", dir=configdir)
    with os.fdopen(fd, "w") as output:
        json.dump(config, output)
    # mkstemp creates mode 0600. JSON is valid YAML for vLLM's config loader.
    ledger = env.get("DGX_MEMORY_LEDGER")
    ledger_revision = hashlib.sha256(Path(ledger).read_bytes()).hexdigest()[:16] if ledger else ""
    args = ["docker", "run", "--rm", "--name", name, "--gpus", "all",
            "--publish", f"{port}:{port}", "--env", "VLLM_API_KEY",
            "--label", "io.inferencectl.managed=true",
            "--label", f"io.inferencectl.role={env['ROLE']}",
            "--label", f"io.inferencectl.memory_profile={model}",
            "--label", f"io.inferencectl.ledger_revision={ledger_revision}",
            "--volume", f"{cache}:/model-cache:ro",
            "--volume", f"{path}:/runtime-config.json:ro",
            "--entrypoint", "vllm", manifest["image"],
            "serve", str(Path("/model-cache") / relative), "--config", "/runtime-config.json"]
    child_env = dict(env)
    child_env["VLLM_API_KEY"] = secret  # Docker receives the name, never the value in argv.
    environment = manifest.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("runtime environment must be a table")
    extra_env = []
    for key, value in environment.items():
        if (not re.fullmatch(r"[A-Z][A-Z0-9_]*", key)
                or key in {"VLLM_API_KEY", "SGLANG_API_KEY"} or not isinstance(value, str)):
            raise ValueError("invalid runtime environment entry")
        extra_env.extend(["--env", key])
        child_env[key] = value
    # Extra runtime settings precede the image and are passed by name only.
    image_index = args.index("--entrypoint")
    args[image_index:image_index] = extra_env
    return args, manifest, child_env


def main() -> int:
    try:
        role, runtime, project, model, kind, spec, served = sys.argv[1:]
        env = dict(os.environ, ROLE=role)
        args, manifest, child_env = build_command(
            Path(runtime), Path(project), model, kind, spec, served, env
        )
        actual = subprocess.run(
            ["docker", "image", "inspect", manifest["image"], "--format", "{{.Id}}"],
            text=True, capture_output=True, check=True, timeout=10,
        ).stdout.strip()
        if actual != manifest["image_id"]:
            raise ValueError("runtime image ID drifted")
        os.execvpe("docker", args, child_env)
    except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError):
        # Do not echo config, environment, command arguments, or Docker stderr.
        print("REFUSING: invalid vLLM runtime/profile, allocation, or image identity", file=sys.stderr)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
