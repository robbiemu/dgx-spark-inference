#!/usr/bin/env python3
"""Optional host-memory guard for one explicitly selected inference service."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import subprocess
import sys
import time


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def available_gib(text: str) -> float:
    fields = re.findall(r"^MemAvailable:\s*([0-9]+)\s+kB\s*$", text, re.MULTILINE)
    if len(fields) != 1:
        raise ValueError("missing or ambiguous MemAvailable")
    return int(fields[0]) / (1 << 20)


def check_once(service: str, container: str, memory_profile: str, floor: float,
               timeout: float, read_memory=None, run=None) -> str:
    """Return inactive/waiting/healthy/stopped. Never stop an unverified target."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*\.service", service):
        raise ValueError("invalid target service")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", container):
        raise ValueError("invalid target container")
    if not memory_profile or not math.isfinite(floor) or floor <= 0:
        raise ValueError("explicit memory profile and positive floor are required")
    run = run or subprocess.run
    read_memory = read_memory or (lambda: Path('/proc/meminfo').read_text())
    state = run(['systemctl', 'is-active', service], capture_output=True,
                text=True, timeout=timeout)
    if state.stdout.strip() not in {'active', 'activating', 'reloading'}:
        if state.stdout.strip() in {'inactive', 'failed', 'deactivating', 'unknown'}:
            return 'inactive'
        raise ValueError('service state unavailable')
    inspected = run(['docker', 'inspect', '--format',
                     '{{index .Config.Labels "io.inferencectl.memory_profile"}}', container],
                    capture_output=True, text=True, timeout=timeout)
    if inspected.returncode != 0 or inspected.stdout.strip() != memory_profile:
        return 'waiting'
    if available_gib(read_memory()) >= floor:
        return 'healthy'
    # Stopping the unit prevents Restart= from immediately relaunching it.
    # Request only this unit's stop; do not kill unrelated containers or services.
    run(['systemctl', '--no-block', 'stop', service], capture_output=True,
        text=True, check=True, timeout=timeout)
    return 'stopped'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service', required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--memory-profile', required=True)
    parser.add_argument('--minimum-available-gib', type=positive, required=True)
    parser.add_argument('--poll-interval', type=positive, default=1.0)
    parser.add_argument('--command-timeout', type=positive, default=5.0)
    args = parser.parse_args(argv)
    try:
        while True:
            state = check_once(args.service, args.container, args.memory_profile,
                               args.minimum_available_gib, args.command_timeout)
            if state in {'inactive', 'stopped'}:
                if state == 'stopped':
                    print('memory watchdog requested the configured service stop', flush=True)
                return 0
            time.sleep(args.poll_interval)
    except (ValueError, OSError, subprocess.SubprocessError):
        print('memory watchdog could not verify state or request the configured stop', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
