#!/usr/bin/env python3
"""The live wrapper plans the whole configured topology once, then reuses it.

No GPU, Docker daemon, or network is used.  Stubs exercise the real
``admission.sh`` twice: first as a cold ``alpha`` launch, then as a serialized
``beta`` launch with alpha reported resident.  The resolver call trace proves:

* cold planning receives every configured active-model slot;
* target/floor allocation is persisted as one state document;
* the second launch reuses that allocation rather than replanning only beta;
* fractions are still derived separately from each launch's live snapshot.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADMISSION = ROOT / "src" / "inferencectl" / "admission.sh"
PYTHON_BIN = Path(
    shutil.which(f"python{sys.version_info.major}.{sys.version_info.minor}")
    or sys.executable
).parent


def stub(root: Path, name: str, body: str) -> Path:
    path = root / name
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def run_slot(
    root: Path,
    runtime: Path,
    role: str,
    model: str,
    realized: int,
    resident: str = "",
) -> tuple[int, str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{root}:{PYTHON_BIN}:{env['PATH']}",
            "CONFIG_ROOT": str(root),
            "PROJECT_ROOT": str(ROOT),
            "DGX_MEMORY_PREFLIGHT": "auto",
            "DGX_MEMORY_LEDGER": str(root / "memory_ledger.toml"),
            "DGX_MEMORY_PLAN": str(root / "memory_plan.toml"),
            "DGX_MEMORY_PLANNER": str(root / "resolver.py"),
            "DGX_ADMISSION_LOCK": str(root / "admission.lock"),
            "DGX_MEMORY_PLAN_STATE": str(root / "joint-state.json"),
            "DGX_DROP_CACHES_PATH": str(root / "drop-caches"),
            "ACTIVE_MODELS": str(root / "active-models.toml"),
            "DGX_ADMISSION_READY_TIMEOUT": "10",
            "DGX_INFERENCE_EXPERIMENTAL": "1",
            "SGLANG_API_KEY": "0" * 64,
            "PORT": "30199",
            "STUB_REALIZED": str(realized),
            "STUB_RESIDENT": resident,
            "STUB_CALL_LOG": str(root / "resolver-calls.jsonl"),
            "STUB_ADAPTER_LOG": str(root / "adapter-calls.txt"),
            "STUB_RECLAIM_LOG": str(root / "reclaim-calls.txt"),
        }
    )
    proc = subprocess.run(
        [
            "bash",
            str(ADMISSION),
            role,
            str(runtime),
            str(ROOT),
            model,
            "model",
            "unused.toml",
            role,
            str(root / "adapter"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return proc.returncode, proc.stdout, proc.stderr


def main() -> int:
    root = Path(tempfile.mkdtemp())
    runtime = root / "runtime"
    runtime.mkdir()
    (runtime / "runtime-manifest.toml").write_text('image="test/image"\n')
    (root / "memory_ledger.toml").write_text("# resolver stub owns budgets\n")
    (root / "memory_plan.toml").write_text(
        "[policy]\nmemavailable_floor_gib=8.0\n"
        "reclaim_page_cache_before_probe=true\n"
    )
    (root / "active-models.toml").write_text(
        '[active.alpha]\nmodel_id="model-one"\nruntime_id="test"\n'
        '[active.beta]\nmodel_id="model-two"\nruntime_id="test"\n'
    )

    stub(
        root,
        "flock",
        '''python3 - "$@" <<'PY'
import fcntl, sys
fcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX)
PY
''',
    )
    stub(
        root,
        "awk",
        '''if [ "${!#}" = "/proc/meminfo" ]; then echo 104857600; exit 0; fi
exec /usr/bin/awk "$@"
''',
    )
    stub(
        root,
        "sha256sum",
        'echo "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  $1"\n',
    )
    stub(root, "sync", 'echo sync >> "$STUB_RECLAIM_LOG"\n')
    stub(
        root,
        "docker",
        '''if [ "$1" = "run" ]; then
  echo "FREE_GIB 100.00 TOTAL_GIB 121.00"; exit 0
fi
if [ "$1" = "ps" ]; then
  case "$*" in
    *io.inferencectl.role*) [ -n "${STUB_RESIDENT:-}" ] && echo "$STUB_RESIDENT" ;;
    *) : ;;
  esac
  exit 0
fi
if [ "$1" = "inspect" ]; then echo "null"; exit 0; fi
exit 0
''',
    )
    stub(root, "curl", 'echo "{\\"max_total_num_tokens\\":${STUB_REALIZED}}"\n')
    stub(
        root,
        "adapter",
        '''echo "$1 ${DGX_MEM_FRACTION_STATIC} ${DGX_MAX_TOTAL_TOKENS}" >> "$STUB_ADAPTER_LOG"
sleep 1
''',
    )
    stub(
        root,
        "resolver.py",
        '''import json, os, sys, tomllib
plan = tomllib.load(open(sys.argv[2], "rb"))
mode = plan.get("policy", {}).get("allocation_mode", "fixed_targets")
slots = plan.get("admit", [])
with open(os.environ["STUB_CALL_LOG"], "a") as handle:
    handle.write(json.dumps({"mode": mode, "slots": slots}) + "\\n")
models = []
for slot in slots:
    role = slot["role"]
    cold_tokens = 300 if role == "alpha" else 500
    tokens = int(slot.get("allocated_kv_tokens", cold_tokens))
    models.append({
        "role": role,
        "model_id": slot["model_id"],
        "mem_fraction_static": 0.4 if role == "alpha" else 0.5,
        "max_total_tokens": tokens,
        "minimum_admissible_pool_tokens": 256,
        "overall_pass": True,
    })
print(json.dumps({"result": "ADMIT", "exit_code": 0, "models": models}))
''',
    )

    rc1, out1, err1 = run_slot(root, runtime, "alpha", "model-one", 300)
    assert rc1 == 0, (rc1, out1, err1)
    state = json.loads((root / "joint-state.json").read_text())
    assert state["allocation"]["result"] == "ADMIT"

    resident = "alpha|model-one|0123456789abcdef"
    rc2, out2, err2 = run_slot(
        root, runtime, "beta", "model-two", 500, resident=resident
    )
    assert rc2 == 0, (rc2, out2, err2)

    calls = [
        json.loads(line)
        for line in (root / "resolver-calls.jsonl").read_text().splitlines()
    ]
    assert [call["mode"] for call in calls] == [
        "floor_weighted",
        "fixed_targets",
        "fixed_targets",
    ], calls
    cold_slots = {(s["role"], s["model_id"]) for s in calls[0]["slots"]}
    assert cold_slots == {("alpha", "model-one"), ("beta", "model-two")}
    assert calls[1]["slots"][0]["allocated_kv_tokens"] == 300
    assert calls[2]["slots"][0]["allocated_kv_tokens"] == 500
    adapter_calls = (root / "adapter-calls.txt").read_text().splitlines()
    assert adapter_calls == ["alpha 0.4 300", "beta 0.5 500"], adapter_calls
    reclaim_calls = (root / "reclaim-calls.txt").read_text().splitlines()
    assert reclaim_calls == ["sync", "sync"], reclaim_calls
    assert (root / "drop-caches").read_text() == "3\n"

    print("PASS: admission plans all configured models once and reuses joint allocations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
