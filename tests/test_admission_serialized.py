#!/usr/bin/env python3
"""Test 8 — admission.sh serialized admission (the concurrency correctness fix).

Exercises the REAL admission.sh with PATH-stubbed docker / resolver / adapter so
NO GPU, NO real docker, NO network is used. Proves the race the wrapper exists to
close: two concurrent dispatchers cannot both pass preflight while the first
candidate is between preflight and allocation commitment.

Cases:
  A) ADMIT exports DGX_MEM_FRACTION_STATIC + DGX_MAX_TOTAL_TOKENS to the captured
     adapter env, and clears any inherited DGX_MEM_* first.
  B) REFUSE (resolver returns non-ADMIT / missing pair / probe fail) never
     launches the adapter; exit code 75 (deliberate refusal).
  C) SERIALIZATION: a fake adapter that BLOCKS (holds the allocation window open)
     keeps the admission lock held; a second concurrent admission for a different
     role cannot complete admission until the first releases.
  D) legacy/auto mode with no planner pair -> adapter runs directly (no lock).
  E) timeout cleanup removes the candidate container before waiting for a
     foreground Docker client that does not exit on TERM.

No GPU/docker: the stubs return canned docker/resolver output. The real
admission.sh logic (lock acquire, enrollment, env clearing, verify loop) runs.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADMISSION = ROOT / "src" / "inferencectl" / "admission.sh"
PYTHON_BIN = Path(
    shutil.which(f"python{sys.version_info.major}.{sys.version_info.minor}")
    or sys.executable
).parent

failures = []
def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}"); failures.append(name)


def _write_stub(dirpath: Path, name: str, body: str) -> Path:
    p = dirpath / name
    p.write_text("#!/usr/bin/env bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _fake_adapter_capture(dirpath: Path, marker: Path) -> Path:
    """Adapter stub: records the DGX_MEM_* env it was called with, then stays alive
    so admission's supervise loop (wait) is plausible; caller kills us."""
    body = (
        'echo "DGX_MEM_FRACTION_STATIC=${DGX_MEM_FRACTION_STATIC:-<unset>}" > ' + str(marker) + '\n'
        'echo "DGX_MAX_TOTAL_TOKENS=${DGX_MAX_TOTAL_TOKENS:-<unset>}" >> ' + str(marker) + '\n'
        'exec sleep 300\n'
    )
    return _write_stub(dirpath, "fake-adapter", body)


def _fake_docker_probe(dirpath: Path) -> Path:
    """docker stub: returns a fixed GPU mem probe line + empty resident list."""
    return _write_stub(dirpath, "docker", '''
if [ "$1" = "run" ]; then echo "FREE_GIB 40.00 TOTAL_GIB 121.70"; exit 0; fi
if [ "$1" = "ps" ]; then exit 0; fi   # no labeled residents
exit 0
''')


def _fake_flock(dirpath: Path) -> Path:
    """Portable flock(1) stand-in for macOS; locks the inherited numeric fd."""
    return _write_stub(dirpath, "flock", '''
python3 - "$@" <<'PY'
import fcntl, sys
fcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX)
PY
''')


def _fake_awk(dirpath: Path) -> Path:
    """Supply a deterministic MemAvailable value on hosts without /proc."""
    return _write_stub(dirpath, "awk", '''
if [ "${!#}" = "/proc/meminfo" ]; then echo 104857600; exit 0; fi
exec /usr/bin/awk "$@"
''')


def _fake_sha256sum(dirpath: Path) -> Path:
    return _write_stub(
        dirpath,
        "sha256sum",
        'echo "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  $1"\n',
    )


def _fake_resolver_admit(dirpath: Path, frac="0.80", mtt="376048", mintok="32768") -> Path:
    return _write_stub(dirpath, "resolve_memory_plan.py", f'''
import json, sys, tomllib
plan = tomllib.load(open(sys.argv[2], "rb"))
models = []
for slot in plan.get("admit", []):
    models.append({{
        "role": slot["role"],
        "model_id": slot["model_id"],
        "mem_fraction_static": {frac},
        "max_total_tokens": int(slot.get("allocated_kv_tokens", {mtt})),
        "minimum_admissible_pool_tokens": {mintok},
        "overall_pass": True,
    }})
doc = {{"result":"ADMIT","exit_code":0,"models":models}}
print(json.dumps(doc))
''')


def _run_admission(stub_dir: Path, env_overrides: dict, preflight="required",
                   model_id="ornith-1.0-9b-fp8", marker=None) -> subprocess.Popen:
    """Launch admission.sh with stubs on PATH; returns the Popen."""
    stubs = stub_dir
    _fake_flock(stubs)
    _fake_awk(stubs)
    _fake_sha256sum(stubs)
    ledger = stubs / "memory_ledger.toml"; ledger.touch()
    plan = stubs / "memory_plan.toml"; plan.touch()
    (stubs / "active-models.toml").write_text(
        f'[active.agentic-helper]\nmodel_id="{model_id}"\nruntime_id="test"\n'
    )
    env = dict(os.environ)
    env["PATH"] = f"{stubs}:{PYTHON_BIN}:{env['PATH']}"
    env["CONFIG_ROOT"] = str(stubs)
    env["PROJECT_ROOT"] = str(ROOT)
    env["DGX_MEMORY_PREFLIGHT"] = preflight
    env["DGX_MEMORY_LEDGER"] = str(ledger)
    env["DGX_MEMORY_PLAN"] = str(plan)
    env["DGX_MEMORY_PLANNER"] = str(stubs / "resolve_memory_plan.py")
    env["ACTIVE_MODELS"] = str(stubs / "active-models.toml")
    env["DGX_ADMISSION_LOCK"] = str(stubs / "test.lock")
    env["DGX_ADMISSION_READY_TIMEOUT"] = "20"
    env["PORT"] = "30199"
    env["SGLANG_API_KEY"] = "0" * 64
    # each admission gets its OWN lock file (cases share td; orphaned fake-adapter
    # children from earlier cases must not hold a later case's lock).
    env["DGX_ADMISSION_LOCK"] = str(stubs / f"test.{marker.name if marker else 'default'}.lock")
    env["DGX_MEMORY_PLAN_STATE"] = str(
        stubs / f"state.{marker.name if marker else 'default'}.json"
    )
    # point curl at nothing real; verify_ready will fail-timeout OR be satisfied by stub
    env["DGX_INFERENCE_EXPERIMENTAL"] = "1"  # avoid sourcing real inference.env
    env.update(env_overrides)
    adapter = _fake_adapter_capture(stubs, marker or stubs / "adapter_env.txt")
    # a curl stub that reports a realized pool in-band so verify_ready succeeds quickly.
    # DON'T clobber a curl stub the caller already wrote (Case C sets one returning {}
    # to force verify_ready to loop, holding the lock for the serialization test).
    if not (stubs / "curl").exists():
        _write_stub(stubs, "curl", f'''
echo '{{"max_total_num_tokens":{env_overrides.get("STUB_REALIZED","376048")}}}'
exit 0
''')
    return subprocess.Popen(
        ["bash", str(ADMISSION), "agentic-helper", str(ROOT/"runtime/sglang"),
         str(ROOT), model_id, "model", "profiles/qwen36-27b-fp8/sglang.toml",
         "agentic-helper", str(adapter)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main() -> int:
    assert ADMISSION.is_file(), f"missing {ADMISSION}"
    td = Path(tempfile.mkdtemp())

    print("=== Case A: ADMIT exports derived knobs to adapter, clears inherited ===")
    marker = td / "A_env.txt"
    # pre-pollute the env to prove admission clears inherited overrides
    env_a = {"DGX_MEM_FRACTION_STATIC": "0.99", "DGX_MAX_TOTAL_TOKENS": "999",
             "STUB_REALIZED": "376048"}
    _fake_resolver_admit(td, "0.80", "376048", "32768")
    _fake_docker_probe(td)
    p = _run_admission(td, env_a, marker=marker)
    time.sleep(6)  # let it admit + verify
    # admission should still be supervising (sleep 300). Kill it now; we have the marker.
    p.terminate()
    try: p.wait(timeout=5)
    except Exception: p.kill()
    cap = marker.read_text() if marker.exists() else ""
    check("ADMIT launched the adapter", "DGX_MEM_FRACTION_STATIC=" in cap, f"cap={cap!r}")
    check("ADMIT exported derived fraction (0.80, not inherited 0.99)",
          "DGX_MEM_FRACTION_STATIC=0.8" in cap, f"cap={cap!r}")
    check("ADMIT exported derived max_total_tokens (376048, not inherited 999)",
          "DGX_MAX_TOTAL_TOKENS=376048" in cap, f"cap={cap!r}")

    print("=== Case B: REFUSE never launches adapter; exit 75 ===")
    # resolver that returns REFUSE
    _write_stub(td, "resolve_memory_plan.py", '''
import json; print(json.dumps({"result":"REFUSE","exit_code":1,"models":[]})); exit(1)
''')
    marker_b = td / "B_env.txt"
    pb = _run_admission(td, {}, marker=marker_b)
    try:
        out_b, err_b = pb.communicate(timeout=15); rc = pb.returncode
    except subprocess.TimeoutExpired:
        pb.kill(); out_b, err_b = pb.communicate(); rc = -1
    check("REFUSE did not launch adapter", not marker_b.exists(),
          f"adapter ran: {marker_b.read_text() if marker_b.exists() else ''!r}")
    check("REFUSE exit code 75 (deliberate refusal)", rc == 75,
          f"rc={rc} out={out_b[-200:]!r} err={err_b[-300:]!r}")

    print("=== Case C: SERIALIZATION — second admission blocks while first holds lock ===")
    # reset to an admitting resolver
    _fake_resolver_admit(td, "0.80", "376048", "32768")
    # The fake adapter sleeps 300 (holds the allocation window). Launch role 1;
    # while it's verified+holding (after verify_ready it releases the lock and
    # supervises), launch role 2. The CORRECTNESS claim: the lock is RELEASED
    # after verify, so a second role CAN admit after the first commits — but two
    # roles cannot BOTH be in the preflight->verify window. We test the window:
    # a fake adapter that BLOCKS verify_ready (curl returns no realized pool)
    # keeps the first admission stuck holding the lock; a second concurrent one
    # must not have admitted by then.
    marker_c1 = td / "C1_env.txt"
    # curl stub that NEVER reports a valid realized pool -> verify_ready loops.
    # Remove any prior curl stub (Case A wrote one returning a valid pool) first.
    (td / "curl").unlink(missing_ok=True)
    _write_stub(td, "curl", 'echo "{}"; exit 0')
    # BOTH Case C admissions share ONE lock (the serialization claim under test).
    shared_lock = str(td / "caseC.shared.lock")
    p1 = _run_admission(td, {"DGX_ADMISSION_READY_TIMEOUT": "8",
                             "DGX_ADMISSION_LOCK": shared_lock}, marker=marker_c1)
    time.sleep(4)  # role 1 is now in the verify loop, holding the lock
    # role 2 starts; with role 1 holding the lock, role 2 blocks on flock
    marker_c2 = td / "C2_env.txt"
    p2 = _run_admission(td, {"DGX_ADMISSION_READY_TIMEOUT": "8",
                             "DGX_ADMISSION_LOCK": shared_lock}, marker=marker_c2)
    time.sleep(3)
    # role 2 must NOT have launched its adapter yet (lock held by role 1)
    c2_ran = marker_c2.exists()
    # cleanup
    for px in (p1, p2):
        px.terminate()
        try: px.wait(timeout=5)
        except Exception: px.kill()
    check("SERIALIZATION: second admission did not launch while first held lock",
          not c2_ran, "second adapter launched during first's verify window -> race NOT closed")

    print("=== Case D: auto mode, no planner pair -> legacy direct launch ===")
    td2 = Path(tempfile.mkdtemp())
    # no memory_ledger.toml / memory_plan.toml in CONFIG_ROOT -> legacy
    adapter_d = _fake_adapter_capture(td2, td2 / "D_env.txt")
    env_d = dict(os.environ)
    env_d["PATH"] = f"{td2}:{PYTHON_BIN}:{env_d['PATH']}"
    env_d["CONFIG_ROOT"] = str(td2)
    env_d["PROJECT_ROOT"] = str(ROOT)
    env_d["DGX_MEMORY_PREFLIGHT"] = "auto"
    env_d["DGX_INFERENCE_EXPERIMENTAL"] = "1"
    env_d["SGLANG_API_KEY"] = "0" * 64
    pd = subprocess.Popen(
        ["bash", str(ADMISSION), "agentic-helper", str(ROOT/"runtime/sglang"),
         str(ROOT), "qwen36-27b-fp8", "model", "profiles/qwen36-27b-fp8/sglang.toml",
         "qwen3.6-27b-agentic", str(adapter_d)],
        env=env_d, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(2)
    pd.terminate()
    try: pd.wait(timeout=5)
    except Exception: pd.kill()
    check("auto/no-pair legacy: adapter launched directly",
          (td2 / "D_env.txt").exists(), "adapter did not run in legacy mode")

    print("=== Case E: timeout removes candidate before waiting for Docker client ===")
    td3 = Path(tempfile.mkdtemp())
    _fake_flock(td3)
    _fake_awk(td3)
    _fake_sha256sum(td3)
    _fake_resolver_admit(td3, "0.80", "376048", "32768")
    (td3 / "memory_ledger.toml").touch()
    (td3 / "memory_plan.toml").touch()
    (td3 / "active-models.toml").write_text(
        '[active.agentic-helper]\nmodel_id="ornith-1.0-9b-fp8"\nruntime_id="test"\n'
    )
    cleanup_marker = td3 / "docker-rm.txt"
    _write_stub(td3, "docker", f'''case "$1" in
  run) echo "FREE_GIB 40.00 TOTAL_GIB 121.70" ;;
  ps) : ;;
  inspect) echo "null" ;;
  rm) touch "{cleanup_marker}" ;;
esac
exit 0
''')
    _write_stub(td3, "curl", 'echo "{}"; exit 0')
    adapter_e = _write_stub(td3, "stubborn-adapter", f'''
trap '' TERM
while [ ! -f "{cleanup_marker}" ]; do sleep 0.1; done
exit 0
''')
    env_e = dict(os.environ)
    env_e.update({
        "PATH": f"{td3}:{PYTHON_BIN}:{env_e['PATH']}",
        "CONFIG_ROOT": str(td3),
        "PROJECT_ROOT": str(ROOT),
        "DGX_MEMORY_PREFLIGHT": "required",
        "DGX_MEMORY_LEDGER": str(td3 / "memory_ledger.toml"),
        "DGX_MEMORY_PLAN": str(td3 / "memory_plan.toml"),
        "DGX_MEMORY_PLANNER": str(td3 / "resolve_memory_plan.py"),
        "ACTIVE_MODELS": str(td3 / "active-models.toml"),
        "DGX_ADMISSION_LOCK": str(td3 / "test.lock"),
        "DGX_MEMORY_PLAN_STATE": str(td3 / "state.json"),
        "DGX_ADMISSION_READY_TIMEOUT": "1",
        "DGX_INFERENCE_EXPERIMENTAL": "1",
        "PORT": "30199",
        "CONTAINER_NAME": "candidate-helper",
        "SGLANG_API_KEY": "0" * 64,
    })
    pe = subprocess.run(
        ["bash", str(ADMISSION), "agentic-helper", str(ROOT / "runtime/sglang"),
         str(ROOT), "ornith-1.0-9b-fp8", "model", "unused.toml",
         "agentic-helper", str(adapter_e)],
        env=env_e, capture_output=True, text=True, timeout=8,
    )
    check("timeout forcibly removed the bounded candidate container",
          pe.returncode == 75 and cleanup_marker.exists(),
          f"rc={pe.returncode} out={pe.stdout[-200:]!r} err={pe.stderr[-200:]!r}")

    print(f"\n=== {['Case A','Case B','Case C','Case D','Case E']} ===" )
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed: {failures}", file=sys.stderr)
        return 1
    print("PASS: admission.sh serialized admission — race closed, knobs exported, refuse safe, legacy preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
