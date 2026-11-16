#!/usr/bin/env python3
"""Invoke a runtime's declared allocation probe without exposing its output."""
from pathlib import Path
import re
import subprocess
import sys
import tomllib


def probe_path(runtime: Path) -> Path | None:
    manifest = tomllib.loads((runtime / "runtime-manifest.toml").read_text())
    name = manifest.get("allocation_probe")
    if name is None:
        return None
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("invalid allocation probe")
    root = runtime.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("allocation probe must be an installed runtime file")
    return path


def main() -> int:
    try:
        mode, root, container = sys.argv[1:]
        path = probe_path(Path(root))
        if mode == "select":
            print("custom" if path else "sglang")
            return 0
        if mode != "probe" or path is None:
            raise ValueError("no custom probe")
        result = subprocess.run([sys.executable, str(path), container],
                                capture_output=True, text=True, check=True, timeout=25)
        value = result.stdout.strip()
        if not re.fullmatch(r"[1-9][0-9]*", value):
            raise ValueError("invalid capacity")
        print(value)
        return 0
    except (ValueError, TypeError, OSError, subprocess.SubprocessError):
        return 1  # Fail closed, without forwarding probe diagnostics/logs.


if __name__ == "__main__":
    raise SystemExit(main())
