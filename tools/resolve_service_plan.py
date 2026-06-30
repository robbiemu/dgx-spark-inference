#!/usr/bin/env python3
"""Resolve requested roles using approved model and runtime capability records."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def require_string(record: dict[str, Any], key: str, source: Path) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: {key} must be a non-empty string")
    return value


def string_list(record: dict[str, Any], key: str, source: Path) -> list[str]:
    value = record.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{source}: {key} must be a list of strings")
    return value


def load_catalog(directory: Path, expected_kind: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for path in sorted(directory.glob("*.toml")):
        record = load_toml(path)

        if record.get("kind") != expected_kind:
            raise ValueError(
                f"{path}: expected kind {expected_kind!r}, got {record.get('kind')!r}"
            )

        record["_source"] = str(path)
        records.append(record)

    return records


def resolve(
    request: dict[str, Any],
    models: list[dict[str, Any]],
    runtimes: list[dict[str, Any]],
) -> dict[str, Any]:
    requested_roles = string_list(request, "requested_roles", Path("request"))
    role_definitions = request.get("roles")

    if not isinstance(role_definitions, dict):
        raise ValueError("request: [roles] table is required")

    approved_runtimes: dict[str, dict[str, Any]] = {}

    for runtime in runtimes:
        source = Path(runtime["_source"])

        if runtime.get("promotion_state") != "approved":
            continue

        runtime_id = require_string(runtime, "runtime_id", source)
        approved_runtimes[runtime_id] = runtime

    resolved_roles: list[dict[str, Any]] = []

    for role_name in requested_roles:
        requirements = role_definitions.get(role_name)

        if not isinstance(requirements, dict):
            raise ValueError(f"request: missing [roles.{role_name}] table")

        required_capabilities = set(
            string_list(requirements, "required_model_capabilities", Path("request"))
        )

        candidates: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []

        for model in models:
            source = Path(model["_source"])

            if model.get("promotion_state") != "approved":
                continue

            model_id = require_string(model, "model_id", source)
            model_roles = set(string_list(model, "roles", source))
            model_capabilities = set(string_list(model, "capabilities", source))
            runtime_ids = string_list(model, "compatible_runtime_ids", source)

            if role_name not in model_roles:
                continue

            if not required_capabilities.issubset(model_capabilities):
                continue

            for runtime_id in runtime_ids:
                runtime = approved_runtimes.get(runtime_id)

                if runtime is not None:
                    candidates.append((model_id, runtime_id, model, runtime))

        if not candidates:
            resolved_roles.append(
                {
                    "role": role_name,
                    "status": "unresolved",
                    "reason": "no approved compatible model/runtime capability pair",
                }
            )
            continue

        model_id, runtime_id, model, runtime = min(
            candidates,
            key=lambda candidate: (candidate[0], candidate[1]),
        )

        resolved_roles.append(
            {
                "role": role_name,
                "status": "resolved",
                "model_id": model_id,
                "runtime_id": runtime_id,
                "model_evidence": model["_source"],
                "runtime_evidence": runtime["_source"],
            }
        )

    is_resolved = all(item["status"] == "resolved" for item in resolved_roles)

    return {
        "status": "resolved" if is_resolved else "unresolved",
        "roles": resolved_roles,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--runtimes-dir", required=True, type=Path)
    parser.add_argument("--allow-unresolved", action="store_true")
    args = parser.parse_args()

    try:
        result = resolve(
            load_toml(args.request),
            load_catalog(args.models_dir, "model-capability"),
            load_catalog(args.runtimes_dir, "runtime-capability"),
        )
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"resolver error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))

    if result["status"] == "unresolved" and not args.allow_unresolved:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
