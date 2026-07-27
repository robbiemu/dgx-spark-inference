#!/usr/bin/env python3
"""Managed model profiles must use the standard Hugging Face cache layout."""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "runtime" / "sglang" / "available.toml"


def main() -> int:
    catalog = tomllib.loads(CATALOG.read_text())
    failures: list[str] = []
    checked = 0

    for role, role_config in catalog["roles"].items():
        for candidate in role_config.get("models", []):
            if candidate["kind"] != "model":
                continue

            spec_path = ROOT / candidate["spec"]
            spec = tomllib.loads(spec_path.read_text())
            identity = spec.get("identity", {})
            launch = spec.get("launch", {})
            checked += 1

            if identity.get("model_dir"):
                failures.append(
                    f"{role}/{candidate['id']}: identity.model_dir bypasses "
                    "the standard HF snapshot resolver"
                )
            if not identity.get("source_repository"):
                failures.append(
                    f"{role}/{candidate['id']}: missing identity.source_repository"
                )
            if not identity.get("source_revision"):
                failures.append(
                    f"{role}/{candidate['id']}: missing identity.source_revision"
                )
            if launch.get("attention_backend") == "auto":
                failures.append(
                    f"{role}/{candidate['id']}: attention_backend='auto' is not "
                    "a valid SGLang CLI value; omit it to use SGLang's default"
                )

    if failures:
        print("Managed profile cache-contract failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        f"PASS: {checked} managed model profiles use pinned standard-cache snapshots"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
