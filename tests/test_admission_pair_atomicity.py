#!/usr/bin/env python3
"""Test 9 — admission matched-pair atomicity + GPU-probe fail-closed.

Phase C: the ledger + plan are a MATCHED PAIR. Exactly-one present must REFUSE
in both auto and required modes (a lone file signals a half-edited deployment;
never silently pair it with a repo copy of the other). Both-absent legacy-
launches in auto, refuses in required. GPU-probe failure refuses in required
(fail-closed — /proc/meminfo alone cannot derive the fraction SGLang will use).

No GPU/docker: PATH-stubbed. Exercises the REAL admission.sh enrollment + pair
logic (the lock/resolver path is covered by T8).
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADMISSION = ROOT / "src" / "inferencectl" / "admission.sh"

failures = []
def check(name, cond, detail=""):
    if cond: print(f"  PASS  {name}")
    else: print(f"  FAIL  {name}  {detail}"); failures.append(name)


def _stub(dirpath: Path, name: str, body: str) -> Path:
    p = dirpath / name
    p.write_text("#!/usr/bin/env bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _run(stub_dir: Path, *, preflight: str, ledger_exists: bool, plan_exists: bool,
         docker_body: str, env_extra: dict | None = None) -> tuple[int, str, str, bool]:
    """Run admission.sh with the given pair state + docker stub. Returns
    (rc, stdout, stderr, adapter_ran). adapter writes a marker if it ran."""
    td = stub_dir
    ledger = td / "memory_ledger.toml"
    plan = td / "memory_plan.toml"
    ledger.unlink(missing_ok=True); plan.unlink(missing_ok=True)
    if ledger_exists: ledger.touch()
    if plan_exists: plan.touch()
    marker = td / "ran.txt"; marker.unlink(missing_ok=True)
    _stub(td, "docker", docker_body)
    _stub(td, "curl", "echo {}; exit 0")  # no realized -> would loop, but we exit before that
    _stub(td, "resolve_memory_plan.py",
          'import json\nprint(json.dumps({"result":"ADMIT","exit_code":0,"models":[]}))')
    _stub(td, "fake-adapter", f"echo ran > {marker}; exec sleep 5")
    env = dict(os.environ)
    env["PATH"] = f"{td}:{env['PATH']}"
    env["CONFIG_ROOT"] = str(td)
    env["PROJECT_ROOT"] = str(ROOT)
    env["DGX_MEMORY_PREFLIGHT"] = preflight
    env["DGX_MEMORY_LEDGER"] = str(ledger)
    env["DGX_MEMORY_PLAN"] = str(plan)
    env["DGX_MEMORY_PLANNER"] = str(td / "resolve_memory_plan.py")
    env["DGX_ADMISSION_LOCK"] = str(td / "lock")
    env["DGX_ADMISSION_READY_TIMEOUT"] = "3"
    env["PORT"] = "30199"
    env["SGLANG_API_KEY"] = "0" * 64
    env["DGX_INFERENCE_EXPERIMENTAL"] = "1"
    if env_extra: env.update(env_extra)
    p = subprocess.Popen(
        ["bash", str(ADMISSION), "r", str(ROOT / "runtime/sglang"), str(ROOT),
         "ornith-1.0-9b-fp8", "model", "s", "n", str(td / "fake-adapter")],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        p.kill(); out, err = p.communicate()
    return p.returncode, out, err, marker.exists()


DOCKER_OK = '''[ "$1" = "run" ] && echo "FREE_GIB 40.00 TOTAL_GIB 121.70" && exit 0
[ "$1" = "ps" ] && exit 0; exit 0'''
DOCKER_BROKEN = "exit 1"  # GPU probe fails


def main() -> int:
    assert ADMISSION.is_file(), f"missing {ADMISSION}"
    td = Path(tempfile.mkdtemp())

    print("=== Case 1: required + exactly-one present (ledger only) -> REFUSE 75 ===")
    rc, out, err, ran = _run(td, preflight="required", ledger_exists=True, plan_exists=False,
                             docker_body=DOCKER_OK)
    check("required+ledger-only refuses", rc == 75 and not ran, f"rc={rc} ran={ran}")
    check("required+ledger-only message names the lone file", "ledger present but plan missing" in err, err[-150:])

    print("=== Case 2: required + exactly-one present (plan only) -> REFUSE 75 ===")
    rc, out, err, ran = _run(td, preflight="required", ledger_exists=False, plan_exists=True,
                             docker_body=DOCKER_OK)
    check("required+plan-only refuses", rc == 75 and not ran, f"rc={rc} ran={ran}")
    check("required+plan-only message names the lone file", "plan present but ledger missing" in err, err[-150:])

    print("=== Case 3: auto + exactly-one present -> REFUSE (atomic even in auto) ===")
    rc, out, err, ran = _run(td, preflight="auto", ledger_exists=True, plan_exists=False,
                             docker_body=DOCKER_OK)
    check("auto+ledger-only refuses (no fail-open)", rc == 75 and not ran, f"rc={rc} ran={ran}")

    print("=== Case 4: auto + both absent -> legacy launch (adapter runs) ===")
    rc, out, err, ran = _run(td, preflight="auto", ledger_exists=False, plan_exists=False,
                             docker_body=DOCKER_OK)
    check("auto+both-absent legacy launches adapter", ran, f"rc={rc} ran={ran}")

    print("=== Case 5: required + both absent -> REFUSE 75 ===")
    rc, out, err, ran = _run(td, preflight="required", ledger_exists=False, plan_exists=False,
                             docker_body=DOCKER_OK)
    check("required+both-absent refuses", rc == 75 and not ran, f"rc={rc} ran={ran}")

    print("=== Case 6: required + GPU probe FAILS -> REFUSE (fail-closed) ===")
    # both present (so we get past the pair check), but docker probe fails.
    rc, out, err, ran = _run(td, preflight="required", ledger_exists=True, plan_exists=True,
                             docker_body=DOCKER_BROKEN)
    check("required+GPU-probe-fail refuses (fail-closed)", rc == 75 and not ran,
          f"rc={rc} ran={ran}")
    check("GPU-probe-fail message names the probe", "GPU free-memory probe failed" in err, err[-200:])

    print("=== Case 7: off mode -> legacy launch regardless of pair ===")
    rc, out, err, ran = _run(td, preflight="off", ledger_exists=False, plan_exists=False,
                             docker_body=DOCKER_OK)
    check("off mode legacy launches adapter", ran, f"rc={rc} ran={ran}")

    if failures:
        print(f"\nFAIL: {len(failures)} check(s): {failures}", file=sys.stderr)
        return 1
    print("\nPASS: matched-pair atomic (exactly-one refuses in both modes; both-absent legacy/refuse by mode); GPU-probe fail-closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
