#!/usr/bin/env python3
"""Synthetic tied-component measurement and planner contract regressions.

No backend is launched. These fixtures describe independent memory allocations,
not copies of deployment logs or evidence that any backend supports N pools.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEASURE = ROOT / "tools/memory_planner/measure_model_budget.py"
PLANNER = ROOT / "tools/memory_planner/resolve_memory_plan.py"
TOKENS = 123457  # Deliberately exercises fractional bytes-per-token round trips.
GIB = 1 << 30


def component(index: int, tokens: int = TOKENS) -> str:
    return (
        f"KV Cache is allocated. #tokens: {tokens}, "
        f"K size: {index * 0.01:.2f} GB, V size: {index * 0.02:.2f} GB"
    )


def startup(reports: list[str], extra: str = "") -> str:
    return "\n".join([
        "Load weight begin. avail mem=100.00 GB",
        "Load weight end. type=SyntheticModel, mem usage=2.00 GB.",
        extra,
        *reports,
    ]) + "\n"


def measure(log: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MEASURE), "--model-id", "synthetic-model",
         "--mem-fraction", "0.80", *args],
        input=log, text=True, capture_output=True, timeout=10,
    )


def resolve(ledger: str, role_count: int = 1, mode: str = "fixed_targets") -> tuple[int, dict]:
    slots = "".join(
        f'[[admit]]\nrole = "slot-{i}"\nmodel_id = "synthetic-model"\n'
        for i in range(role_count)
    )
    plan = (
        '[device]\ntotal_gib = 100.0\n'
        f'[policy]\nallocation_mode = "{mode}"\n'
        '[observed]\ngpu_free_now_gib = 100.0\nmemavailable_now_gib = 100.0\n'
        + slots
    )
    with tempfile.TemporaryDirectory() as directory:
        ledger_path = Path(directory) / "ledger.toml"
        plan_path = Path(directory) / "plan.toml"
        ledger_path.write_text(ledger)
        plan_path.write_text(plan)
        proc = subprocess.run(
            [sys.executable, str(PLANNER), str(ledger_path), str(plan_path), "--format", "json"],
            text=True, capture_output=True, timeout=10,
        )
    return proc.returncode, json.loads(proc.stdout)


def ledger(layout: str = "tied", components: str = "[1.0, 2.0, 3.0]", total: str = "6.0") -> str:
    return f'''
[[profiles]]
model_id = "synthetic-model"
[profiles.budget]
weights_gib = 1.0
target_kv_tokens = 200
minimum_admissible_pool_tokens = 100
maximum_useful_pool_tokens = 1000
static_pad_gib = 0.0
kv_bytes_per_token = {total}
kv_pool_layout = "{layout}"
kv_pool_bytes_per_token = {components}
'''


class TiedPools(unittest.TestCase):
    def refused(self, log: str, *args: str) -> None:
        proc = measure(log, *args)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertEqual(proc.stdout, "")
        self.assertNotIn("Traceback", proc.stderr)

    def test_single_pool_default_and_explicit_one_are_compatible(self):
        log = startup([component(1)])
        original = measure(log)
        explicit = measure(log, "--tied-kv-pools", "1")
        self.assertEqual(original.returncode, 0, original.stderr)
        self.assertEqual(original.stdout, explicit.stdout)
        budget = tomllib.loads(original.stdout)["profiles"][0]["budget"]
        self.assertNotIn("kv_pool_layout", budget)
        self.assertNotIn("kv_pool_bytes_per_token", budget)
        self.assertEqual(resolve(original.stdout)[0], 0)

    def test_two_three_five_and_ten_components_round_trip(self):
        for count in (2, 3, 5, 10):
            with self.subTest(count=count):
                proc = measure(startup([component(i) for i in range(1, count + 1)]),
                               "--tied-kv-pools", str(count))
                self.assertEqual(proc.returncode, 0, proc.stderr)
                budget = tomllib.loads(proc.stdout)["profiles"][0]["budget"]
                self.assertEqual(budget["kv_pool_layout"], "tied")
                parts = budget["kv_pool_bytes_per_token"]
                self.assertEqual(len(parts), count)
                for i, part in enumerate(parts, 1):
                    self.assertTrue(math.isclose(part, i * 0.03 * GIB / TOKENS))
                self.assertEqual(budget["kv_bytes_per_token"], sum(parts))
                self.assertEqual(budget["target_kv_tokens"], TOKENS)
                rc, result = resolve(proc.stdout)
                self.assertEqual(rc, 0, result)
                self.assertEqual(len(result["models"]), 1)
                row = result["models"][0]
                self.assertEqual(row["max_total_tokens"], TOKENS)
                self.assertAlmostEqual(row["mem_fraction_static"], 0.8, places=4)

    def test_component_count_is_independent_of_serving_role_count(self):
        tied = ledger(components="[2.0, 4.0]")
        single = tied.replace('kv_pool_layout = "tied"', 'kv_pool_layout = "single"').replace(
            "kv_pool_bytes_per_token = [2.0, 4.0]", "kv_pool_bytes_per_token = []"
        )
        for role_count in (1, 4):
            for mode in ("fixed_targets", "floor_weighted"):
                with self.subTest(roles=role_count, mode=mode):
                    rc, result = resolve(tied, role_count, mode)
                    self.assertEqual(rc, 0, result)
                    self.assertEqual(len(result["models"]), role_count)
                    self.assertEqual((rc, result), resolve(single, role_count, mode))

    def test_multi_pool_requires_explicit_exact_count(self):
        reports = [component(1), component(2), component(3)]
        self.refused(startup(reports))
        for expected in (1, 2, 4):
            self.refused(startup(reports), "--tied-kv-pools", str(expected))
        self.refused(startup([]), "--tied-kv-pools", "2")
        self.refused(startup([component(1)]), "--tied-kv-pools", "2")

    def test_count_option_rejects_nonpositive_and_noninteger_values(self):
        for value in ("0", "-1", "2.5", "many"):
            with self.subTest(value=value):
                self.refused(startup([component(1)]), "--tied-kv-pools", value)

    def test_unequal_zero_negative_and_malformed_capacities_refuse(self):
        for value in (str(TOKENS - 1), "0", "-1", "1.5", "123x", "nan"):
            with self.subTest(value=value):
                bad = component(2).replace(str(TOKENS), value)
                self.refused(startup([component(1), bad]), "--tied-kv-pools", "2")

    def test_missing_or_repeated_fields_refuse(self):
        bad_reports = [
            component(2).replace(f"#tokens: {TOKENS}, ", ""),
            component(2) + f", tokens: {TOKENS}",
            component(2).replace("K size: 0.02 GB, ", ""),
            component(2).replace(", V size: 0.04 GB", ""),
            component(2) + ", K size: 0.02 GB",
            component(2) + ", K size: invalid",
            component(2).replace("0.02 GB", "0.02 MB"),
            component(2).replace("0.02 GB", "0.02"),
        ]
        for bad in bad_reports:
            with self.subTest(report=bad):
                self.refused(startup([component(1), bad]), "--tied-kv-pools", "2")

    def test_invalid_sizes_and_zero_total_refuse(self):
        for value in ("-0.02", "nan", "inf", "1e999", "bad", "."):
            with self.subTest(value=value):
                bad = component(2).replace("0.02 GB", value + " GB")
                self.refused(startup([component(1), bad]), "--tied-kv-pools", "2")
        zero = component(2).replace("0.02 GB", "0.00 GB").replace("0.04 GB", "0.00 GB")
        self.refused(startup([component(1), zero]), "--tied-kv-pools", "2")

    def test_duplicate_anonymous_reports_are_ambiguous_not_deduplicated(self):
        reports = ["[worker-a] " + component(1), "[worker-b] " + component(1)]
        self.refused(startup(reports), "--tied-kv-pools", "2")

    def test_zero_v_and_gib_units_have_positive_combined_cost(self):
        second = component(2).replace("0.04 GB", "0.00 GB").replace(" GB", " GiB")
        proc = measure(startup([component(1), second]), "--tied-kv-pools", "2")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(resolve(proc.stdout)[0], 0)

    def test_fixed_mamba_is_supported_but_proportional_basis_is_not_guessed(self):
        reports = [component(1), component(2), component(3)]
        extra = "Mamba Cache is allocated. state size: 1.00GB"
        fixed = measure(startup(reports, extra), "--tied-kv-pools", "3",
                        "--mamba-cache-allocation", "fixed")
        self.assertEqual(fixed.returncode, 0, fixed.stderr)
        budget = tomllib.loads(fixed.stdout)["profiles"][0]["budget"]
        self.assertEqual(budget["fixed_mamba_cache_gib"], 1.0)
        self.assertEqual(resolve(fixed.stdout)[0], 0)
        self.refused(startup(reports, extra), "--tied-kv-pools", "3")
        self.refused(startup(reports, extra), "--tied-kv-pools", "3",
                     "--mamba-cache-allocation", "proportional")
        rc, result = resolve(ledger() + "mamba_kv_memory_ratio = 0.5\n")
        self.assertEqual(rc, 2, result)

    def test_planner_rejects_invalid_component_contracts(self):
        cases = [
            ledger(layout="unknown"), ledger(layout="single"),
            ledger(components="[]"), ledger(components="[6.0]"),
            ledger(components="[0.0, 6.0]"), ledger(components="[-1.0, 7.0]"),
            ledger(components="[nan, 6.0]"), ledger(components="[inf, 6.0]"),
            ledger(components="[1e308, 1e308]", total="inf"),
            ledger(total="7.0"), ledger(components='["bad", 6.0]'),
            ledger(components="6.0"), ledger(components='"24"'),
            ledger(components="[true, 5.0]"), ledger(components='["2", "4"]'),
        ]
        for entry in cases:
            with self.subTest(entry=entry):
                rc, result = resolve(entry)
                self.assertEqual(rc, 2, result)
                self.assertEqual(result["result"], "REFUSE")
                self.assertEqual(result["models"], [])

    def test_two_component_compatibility_alias(self):
        rc, result = resolve(ledger(layout="tied_target_draft", components="[2.0, 4.0]"))
        self.assertEqual(rc, 0, result)
        rc, result = resolve(ledger(layout="tied_target_draft"))
        self.assertEqual(rc, 2, result)


if __name__ == "__main__":
    unittest.main()
