#!/usr/bin/env bash
# run_all.sh — the v0.1 no-GPU test gate.
# Runs the six load-bearing tests + TOML schema validation + the contextual
# secret scan over the whole repo. No GPU, no docker, no network, no package
# install (python3 3.12 stdlib only). Exits nonzero on any failure.
#
# ShellCheck and the external secret scanners (gitleaks, trufflehog) are run
# separately by the owner after tool install — they are NOT part of this gate.
set -u
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

fail=0
say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

say "TOML schema validation (parse every .toml)"
python3 - "$ROOT" <<'PY' || fail=1
import sys, tomllib
from pathlib import Path
root = Path(sys.argv[1])
tomls = sorted(p for p in root.rglob("*.toml") if ".git" not in p.parts)
assert tomls, "no toml files found"
bad = []
for p in tomls:
    try:
        tomllib.loads(p.read_text())
    except Exception as e:
        bad.append(f"{p.relative_to(root)}: {e}")
if bad:
    print("TOML parse failures:", file=sys.stderr)
    for b in bad: print("  " + b, file=sys.stderr)
    sys.exit(1)
print(f"OK: {len(tomls)} TOML files parse")
PY

say "Test 1 — baseline satisfies agentic"
python3 tests/test_baseline_satisfies_agentic.py || fail=1

say "Test 2 — DFlash rejected for agentic"
python3 tests/test_dflash_rejected_for_agentic.py || fail=1

say "Test 3 — available catalog compatibility"
python3 tests/test_available_catalog_compatible.py || fail=1

say "Test 4 — launch-config rendering (real adapter)"
python3 tests/test_launch_config_rendering.py || fail=1

say "Test 5 — dry-run has no secrets"
python3 tests/test_dry_run_no_secrets.py || fail=1

say "Test 6 — experimental isolation (adapter honors caller port, not prod)"
python3 tests/test_experimental_isolation.py || fail=1

say "Test 7 — memory env-var override tier (DGX_MEM_FRACTION_STATIC / DGX_MAX_TOTAL_TOKENS)"
python3 tests/test_memory_env_override.py || fail=1

say "Test 8 — admission serialized (race closed; knobs exported; refuse safe; legacy preserved)"
python3 tests/test_admission_serialized.py || fail=1

say "Test 9 — admission matched-pair atomicity + GPU-probe fail-closed"
python3 tests/test_admission_pair_atomicity.py || fail=1

say "Test 10 — admission live-path invariants (review blockers: dispatch delegation, measured A_preload, plan floor, probe-fail-auto)"
python3 tests/test_admission_live_path.py || fail=1

say "Test 11 — explicit KV-cache dtype profiles preserve baseline auto behavior"

say "Test 12 — health-wait gate succeeds and fails closed"
python3 tests/test_wait_for_health.py || fail=1
# Planner: measurement tool, fraction_base validation, adapter precedence.
python3 tests/test_measure_model_budget.py || fail=1
python3 tests/test_fraction_base_validation.py || fail=1
python3 tests/test_adapter_precedence.py || fail=1

say "Test 14 — authenticated endpoint validator parses"
python3 -m py_compile tools/validate_agentic_endpoint.py || fail=1

say "Repository-wide contextual secret scan"
python3 tests/scan_secrets.py "$ROOT" || fail=1

echo
if [ "$fail" = 0 ]; then
  echo "ALL TESTS PASSED"
  exit 0
else
  echo "ONE OR MORE TESTS FAILED" >&2
  exit 1
fi
