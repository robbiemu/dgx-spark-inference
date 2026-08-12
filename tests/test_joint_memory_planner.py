#!/usr/bin/env python3
"""Regression tests for generic floor-first, configuration-weighted planning."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLANNER = ROOT / "tools" / "memory_planner" / "resolve_memory_plan.py"


def run_plan(ledger: str, plan: str) -> tuple[int, dict]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ledger_path = root / "ledger.toml"
        plan_path = root / "plan.toml"
        ledger_path.write_text(ledger)
        plan_path.write_text(plan)
        proc = subprocess.run(
            [
                sys.executable,
                "-B",
                str(PLANNER),
                str(ledger_path),
                str(plan_path),
                "--format",
                "json",
            ],
            text=True,
            capture_output=True,
        )
    if not proc.stdout:
        raise AssertionError(f"planner produced no JSON; stderr={proc.stderr!r}")
    return proc.returncode, json.loads(proc.stdout)


def record(model: str, floor: int, target: int, bytes_per_token: int = 1 << 30) -> str:
    return f'''\
[[profiles]]
model_id = "{model}"
[profiles.budget]
weights_gib = 1.0
minimum_admissible_pool_tokens = {floor}
maximum_useful_pool_tokens = {target * 10}
target_kv_tokens = {target}
kv_bytes_per_token = {bytes_per_token}
static_pad_gib = 0.0
static_overhead_gib = 0.0
cuda_graph_peak_gib = 0.0
request_workspace_gib = 0.0
gpu_headroom_gib = 0.0
'''


def joint_plan(models: list[tuple[str, str]], gpu: float = 100, host: float = 100) -> str:
    slots = "".join(
        f'[[admit]]\nrole = "{role}"\nmodel_id = "{model}"\n'
        for role, model in models
    )
    return f'''\
[device]
total_gib = {gpu}
[policy]
allocation_mode = "floor_weighted"
memavailable_floor_gib = 10.0
[observed]
gpu_free_now_gib = {gpu}
memavailable_now_gib = {host}
{slots}'''


def by_role(document: dict) -> dict[str, dict]:
    return {item["role"]: item for item in document["models"]}


def main() -> int:
    # The requested relationship comes from target/floor in configuration.
    ledger = record("one", 10, 20) + record("two", 10, 40)
    rc, document = run_plan(
        ledger, joint_plan([("alpha", "one"), ("beta", "two")])
    )
    assert rc == 0, document
    models = by_role(document)
    assert models["alpha"]["total_pool_weight"] == 1.0
    assert models["beta"]["total_pool_weight"] == 2.0
    assert models["beta"]["max_total_tokens"] in (
        2 * models["alpha"]["max_total_tokens"],
        2 * models["alpha"]["max_total_tokens"] + 1,
    )

    # Changing configuration changes the relationship; no role-specific ratio
    # may be embedded in the planner.
    changed = record("one", 10, 20) + record("two", 10, 30)
    rc, document = run_plan(
        changed, joint_plan([("alpha", "one"), ("beta", "two")])
    )
    assert rc == 0, document
    models = by_role(document)
    assert models["beta"]["total_pool_weight"] == 1.5
    assert abs(
        models["beta"]["max_total_tokens"]
        - models["alpha"]["max_total_tokens"] * 1.5
    ) <= 1

    # Total pools remain proportional until one slot reaches its configured
    # useful ceiling; surplus is then redistributed to an uncapped slot.
    capped = (
        record("one", 10, 20).replace(
            "maximum_useful_pool_tokens = 200",
            "maximum_useful_pool_tokens = 30",
        )
        + record("two", 10, 40).replace(
            "maximum_useful_pool_tokens = 400",
            "maximum_useful_pool_tokens = 100",
        )
    )
    rc, document = run_plan(
        capped,
        joint_plan(
            [("alpha", "one"), ("beta", "two")], gpu=120, host=120
        ),
    )
    assert rc == 0, document
    models = by_role(document)
    assert models["alpha"]["max_total_tokens"] == 30
    assert 60 < models["beta"]["max_total_tokens"] <= 100

    # One configured model and an arbitrary ten-model topology both work.
    rc, document = run_plan(record("solo", 10, 20), joint_plan([("r0", "solo")]))
    assert rc == 0 and len(document["models"]) == 1

    ten_ledger = "".join(record(f"m{i}", 1, i + 2, 1 << 20) for i in range(10))
    ten_slots = [(f"r{i}", f"m{i}") for i in range(10)]
    rc, document = run_plan(ten_ledger, joint_plan(ten_slots, gpu=50, host=50))
    assert rc == 0 and len(document["models"]) == 10

    # All floors are a joint hard gate. Nothing is partially admitted.
    impossible = record("one", 40, 80) + record("two", 40, 160)
    rc, document = run_plan(
        impossible,
        joint_plan([("alpha", "one"), ("beta", "two")], gpu=50, host=50),
    )
    assert rc == 2
    assert document["result"] == "REFUSE"
    assert "joint minimums do not fit" in document["error"]
    assert document["models"] == []

    missing_ceiling = record("one", 10, 20).replace(
        "maximum_useful_pool_tokens = 200\n", ""
    )
    rc, document = run_plan(
        missing_ceiling, joint_plan([("alpha", "one")])
    )
    assert rc == 2
    assert "requires a positive maximum_useful_pool_tokens" in document["error"]

    # Explicit Mamba ratio is part of static_required, while fixed overhead is
    # independently visible and does not silently scale with the pool.
    spec = importlib.util.spec_from_file_location("memory_planner", PLANNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    budget = module.Budget(
        model_id="hybrid",
        weights_gib=2.0,
        target_kv_tokens=10,
        minimum_admissible_pool_tokens=5,
        kv_bytes_per_token=(1 << 30) / 10,
        mamba_kv_memory_ratio=0.9,
        static_overhead_gib=0.25,
        static_pad_gib=0.5,
    )
    assert abs(budget.kv_gib(10) - 1.0) < 1e-9
    assert abs(budget.mamba_gib(10) - 0.9) < 1e-9
    assert abs(budget.static_required_for_tokens(10) - 4.65) < 1e-9
    fixed_budget = module.Budget(
        model_id="fixed-hybrid",
        weights_gib=2.0,
        target_kv_tokens=10,
        minimum_admissible_pool_tokens=5,
        kv_bytes_per_token=(1 << 30) / 10,
        fixed_mamba_cache_gib=0.49,
        static_overhead_gib=0.15,
        static_pad_gib=0.5,
    )
    assert abs(fixed_budget.static_required_for_tokens(10) - 4.14) < 1e-9

    print("PASS: joint planner derives generic floor-first growth from configuration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
