#!/usr/bin/env python3
"""Report realized vLLM cache capacity, never raw container logs."""
import re
import subprocess
import sys


def capacity(log: str) -> int:
    reports = re.findall(r"GPU KV cache size:\s*([^\s]+)\s*tokens", log)
    if not reports or not re.fullmatch(r"(?:[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)", reports[-1]):
        raise ValueError("no valid realized KV capacity")
    return int(reports[-1].replace(",", ""))


def probe(container: str) -> int:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", container):
        raise ValueError("invalid container name")
    started = subprocess.run(
        ["docker", "inspect", container, "--format", "{{.State.StartedAt}}"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9TZ:.+-]+", started):
        raise ValueError("missing container start time")
    result = subprocess.run(
        ["docker", "logs", "--since", started, "--tail", "10000", container],
        capture_output=True, text=True, check=True, timeout=10,
    )
    return capacity(result.stdout + result.stderr)


def main() -> int:
    try:
        if len(sys.argv) != 2:
            raise ValueError("expected one container")
        print(probe(sys.argv[1]))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError):
        print("capacity unavailable", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
