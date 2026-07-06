#!/usr/bin/env python3
"""Atomically bind a role to a candidate in a registered runtime catalog."""

from __future__ import annotations

import argparse
import datetime
import os
import re
import shutil
import tempfile
import tomllib
from pathlib import Path


def atomic_write(path: Path, content: str) -> None:
    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def nested_runtime(registry: dict, runtime_id: str) -> dict | None:
    current = registry.get("runtimes", {})
    try:
        for part in runtime_id.split("."):
            current = current[part]
        return current
    except (KeyError, TypeError):
        return registry.get("runtimes", {}).get(runtime_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-root", type=Path, default=Path("/etc/dgx-spark-inference"))
    parser.add_argument("--role", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()

    active_path = args.config_root / "active-models.toml"
    runtimes_path = args.config_root / "runtimes.toml"
    available_path = args.runtime_root / "available.toml"
    for path in (active_path, runtimes_path, available_path):
        if not path.is_file():
            raise SystemExit(f"REFUSE: missing {path}")

    available = tomllib.loads(available_path.read_text())
    role_catalog = available.get("roles", {}).get(args.role, {})
    offered = {item["id"] for item in role_catalog.get("models", [])}
    if args.model_id not in offered:
        raise SystemExit(
            f"REFUSE: {args.model_id!r} is not offered for {args.role!r} "
            f"by {available_path}"
        )

    runtimes_text = runtimes_path.read_text()
    runtimes = tomllib.loads(runtimes_text)
    existing = nested_runtime(runtimes, args.runtime_id)
    expected_root = str(args.runtime_root)
    if existing:
        if existing.get("project_root") != expected_root:
            raise SystemExit(
                f"REFUSE: runtime {args.runtime_id!r} already maps to "
                f"{existing.get('project_root')!r}"
            )
        new_runtimes = runtimes_text
    else:
        new_runtimes = (
            runtimes_text.rstrip()
            + "\n\n"
            + f'[runtimes."{args.runtime_id}"]\n'
            + f'project_root = "{expected_root}"\n'
        )
        tomllib.loads(new_runtimes)

    active_text = active_path.read_text()
    section = re.compile(
        r"(\[active\."
        + re.escape(args.role)
        + r"\])(?P<body>.*?)(?=\n\[|\Z)",
        re.S,
    )
    match = section.search(active_text)
    if match:
        body = match.group("body")
        for key, value in (
            ("model_id", args.model_id),
            ("runtime_id", args.runtime_id),
        ):
            body, count = re.subn(
                rf'({key}\s*=\s*")[^"]*(")',
                rf"\g<1>{value}\g<2>",
                body,
                count=1,
            )
            if count != 1:
                raise SystemExit(f"REFUSE: could not update {key} for {args.role!r}")
        new_active = (
            active_text[: match.start("body")]
            + body
            + active_text[match.end("body") :]
        )
    else:
        new_active = (
            active_text.rstrip()
            + "\n\n"
            + f"[active.{args.role}]\n"
            + f'model_id = "{args.model_id}"\n'
            + f'runtime_id = "{args.runtime_id}"\n'
        )
    tomllib.loads(new_active)

    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    shutil.copy2(runtimes_path, f"{runtimes_path}.pre-bind.{stamp}")
    shutil.copy2(active_path, f"{active_path}.pre-bind.{stamp}")
    atomic_write(runtimes_path, new_runtimes)
    atomic_write(active_path, new_active)
    print(
        f"bound {args.role}: model={args.model_id} runtime={args.runtime_id} "
        f"root={args.runtime_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
