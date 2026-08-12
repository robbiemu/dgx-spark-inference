#!/usr/bin/env python3
"""Resolve fixed-target or joint floor-weighted SGLang memory plans.

The joint allocator is deliberately model- and role-agnostic.  Every configured
model contributes a measured budget, a required token floor, and an expected
token target.  The allocator first proves that all floors fit together, then
uses target/floor as the configured total-pool weight and spends the remaining
usable unified memory with bounded weighted water-filling.  A configured useful
ceiling prevents assigning more tokens than a slot's context/concurrency can use.

`mamba_kv_memory_ratio` models hybrid runtimes whose automatic state-cache
allocator reserves Mamba/linear-attention memory in proportion to ordinary KV.
It must not be hidden in a fixed `static_overhead_gib`, because that produces a
wrong fraction whenever the planned KV pool changes.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import tomllib
from dataclasses import dataclass, field
from typing import Any

GIB = 1 << 30
ALLOCATION_MODES = frozenset({"fixed_targets", "floor_weighted"})


@dataclass
class Budget:
    model_id: str
    weights_gib: float
    target_kv_tokens: int
    kv_bytes_per_token: float
    minimum_admissible_pool_tokens: int = 0
    maximum_useful_pool_tokens: int = 0
    static_pad_gib: float = 0.5
    # Fixed non-KV, non-Mamba static memory.  Target-dependent Mamba state is
    # represented separately by mamba_kv_memory_ratio.
    static_overhead_gib: float = 0.0
    fixed_mamba_cache_gib: float = 0.0
    mamba_kv_memory_ratio: float = 0.0
    cuda_graph_peak_gib: float = 0.0
    request_workspace_gib: float = 0.0
    memavailable_floor_gib: float = 8.0
    gpu_headroom_gib: float = 1.0
    fraction_base: str = "a_preload"

    VALID_FRACTION_BASES = frozenset({"a_preload", "device_total"})

    def __post_init__(self) -> None:
        if self.fraction_base not in self.VALID_FRACTION_BASES:
            choices = ", ".join(sorted(self.VALID_FRACTION_BASES))
            raise ValueError(
                f"invalid fraction_base {self.fraction_base!r} for "
                f"{self.model_id} (must be one of: {choices})"
            )
        numeric_nonnegative = {
            "weights_gib": self.weights_gib,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "static_pad_gib": self.static_pad_gib,
            "static_overhead_gib": self.static_overhead_gib,
            "fixed_mamba_cache_gib": self.fixed_mamba_cache_gib,
            "mamba_kv_memory_ratio": self.mamba_kv_memory_ratio,
            "cuda_graph_peak_gib": self.cuda_graph_peak_gib,
            "request_workspace_gib": self.request_workspace_gib,
            "gpu_headroom_gib": self.gpu_headroom_gib,
        }
        for name, value in numeric_nonnegative.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{self.model_id}: {name} must be finite and >= 0")
        if self.weights_gib <= 0 or self.kv_bytes_per_token <= 0:
            raise ValueError(
                f"{self.model_id}: weights_gib and kv_bytes_per_token must be > 0"
            )
        if (
            self.target_kv_tokens <= 0
            or self.minimum_admissible_pool_tokens < 0
            or self.maximum_useful_pool_tokens < 0
        ):
            raise ValueError(
                f"{self.model_id}: token target must be > 0 and token bounds "
                "must be >= 0"
            )

    def kv_gib(self, tokens: int) -> float:
        return tokens * self.kv_bytes_per_token / GIB

    def mamba_gib(self, tokens: int) -> float:
        return self.kv_gib(tokens) * self.mamba_kv_memory_ratio

    def cache_gib(self, tokens: int) -> float:
        return self.kv_gib(tokens) + self.mamba_gib(tokens)

    @property
    def effective_cache_bytes_per_token(self) -> float:
        return self.kv_bytes_per_token * (1.0 + self.mamba_kv_memory_ratio)

    def static_required_for_tokens(self, tokens: int) -> float:
        return (
            self.weights_gib
            + self.cache_gib(tokens)
            + self.fixed_mamba_cache_gib
            + self.static_overhead_gib
            + self.static_pad_gib
        )

    def peak_required_for_tokens(self, tokens: int) -> float:
        return (
            self.static_required_for_tokens(tokens)
            + self.cuda_graph_peak_gib
            + self.request_workspace_gib
        )

    @property
    def static_required_gib(self) -> float:
        return self.static_required_for_tokens(self.target_kv_tokens)

    @property
    def peak_required_gib(self) -> float:
        return self.peak_required_for_tokens(self.target_kv_tokens)

    def derive_fraction_for_tokens(
        self,
        tokens: int,
        a_preload_gib: float,
        device_total_gib: float = 0.0,
    ) -> float:
        static_required = self.static_required_for_tokens(tokens)
        if self.fraction_base == "device_total":
            if device_total_gib <= 0:
                raise ValueError(
                    f"{self.model_id}: device_total required for "
                    "fraction_base=device_total"
                )
            return static_required / device_total_gib
        if a_preload_gib <= 0:
            raise ValueError(
                f"{self.model_id}: A_preload must be > 0 (got {a_preload_gib})"
            )
        return static_required / a_preload_gib


@dataclass
class AdmissionResult:
    role: str
    model_id: str
    fraction: float
    max_total_tokens: int
    configured_target_tokens: int
    total_pool_weight: float
    static_required_gib: float
    peak_required_gib: float
    a_preload_gib: float
    gpu_headroom_gib: float
    gpu_gate_pass: bool
    gpu_gate_detail: str
    linux_gate_pass: bool
    linux_gate_detail: str
    overall_pass: bool
    notes: list[str] = field(default_factory=list)
    minimum_admissible_pool_tokens: int = 0
    maximum_useful_pool_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "model_id": self.model_id,
            "mem_fraction_static": round(self.fraction, 6),
            "max_total_tokens": int(self.max_total_tokens),
            "configured_target_tokens": int(self.configured_target_tokens),
            "total_pool_weight": round(self.total_pool_weight, 8),
            "minimum_admissible_pool_tokens": int(
                self.minimum_admissible_pool_tokens
            ),
            "maximum_useful_pool_tokens": int(self.maximum_useful_pool_tokens),
            "overall_pass": bool(self.overall_pass),
            "gpu_gate_pass": bool(self.gpu_gate_pass),
            "linux_gate_pass": bool(self.linux_gate_pass),
            "static_required_gib": round(self.static_required_gib, 4),
            "peak_required_gib": round(self.peak_required_gib, 4),
            "a_preload_gib": round(self.a_preload_gib, 4),
            "gpu_headroom_gib": round(self.gpu_headroom_gib, 4),
        }


def admit(
    budget: Budget,
    role: str,
    allocated_tokens: int,
    total_pool_weight: float,
    a_preload_gib: float,
    device_total_gib: float,
    memavailable_before_gib: float,
) -> AdmissionResult:
    static_required = budget.static_required_for_tokens(allocated_tokens)
    peak_required = budget.peak_required_for_tokens(allocated_tokens)
    fraction = budget.derive_fraction_for_tokens(
        allocated_tokens, a_preload_gib, device_total_gib
    )

    post_static = a_preload_gib - static_required
    transient_and_headroom = (
        budget.cuda_graph_peak_gib
        + budget.request_workspace_gib
        + budget.gpu_headroom_gib
    )
    gpu_pass = 0 < fraction < 1 and post_static >= transient_and_headroom
    gpu_detail = (
        f"A_preload({a_preload_gib:.2f}) - static({static_required:.2f}) "
        f"= {post_static:.2f} >= graph+ws+headroom({transient_and_headroom:.2f})"
    )

    memavailable_after = memavailable_before_gib - peak_required
    linux_pass = memavailable_after >= budget.memavailable_floor_gib
    linux_detail = (
        f"MemAvailable after load = {memavailable_before_gib:.2f} - "
        f"peak({peak_required:.2f}) = {memavailable_after:.2f} "
        f">= floor({budget.memavailable_floor_gib:.2f})"
    )

    notes: list[str] = []
    if allocated_tokens < budget.target_kv_tokens:
        notes.append(
            f"allocated pool {allocated_tokens} is below configured expected "
            f"target {budget.target_kv_tokens}"
        )
    elif allocated_tokens > budget.target_kv_tokens:
        notes.append(
            f"allocated pool {allocated_tokens} exceeds configured expected "
            f"target {budget.target_kv_tokens}; surplus remains useful to radix cache"
        )

    return AdmissionResult(
        role=role,
        model_id=budget.model_id,
        fraction=fraction,
        max_total_tokens=allocated_tokens,
        configured_target_tokens=budget.target_kv_tokens,
        total_pool_weight=total_pool_weight,
        static_required_gib=static_required,
        peak_required_gib=peak_required,
        a_preload_gib=a_preload_gib,
        gpu_headroom_gib=budget.gpu_headroom_gib,
        gpu_gate_pass=gpu_pass,
        gpu_gate_detail=gpu_detail,
        linux_gate_pass=linux_pass,
        linux_gate_detail=linux_detail,
        overall_pass=gpu_pass and linux_pass,
        notes=notes,
        minimum_admissible_pool_tokens=budget.minimum_admissible_pool_tokens,
        maximum_useful_pool_tokens=budget.maximum_useful_pool_tokens,
    )


def load_budgets(ledger_path: str) -> dict[str, Budget]:
    with open(ledger_path, "rb") as handle:
        document = tomllib.load(handle)
    budgets: dict[str, Budget] = {}
    for record in document.get("profiles", []):
        values = record.get("budget")
        if not values:
            continue
        model_id = record["model_id"]
        budgets[model_id] = Budget(
            model_id=model_id,
            weights_gib=float(values["weights_gib"]),
            target_kv_tokens=int(values["target_kv_tokens"]),
            minimum_admissible_pool_tokens=int(
                values.get("minimum_admissible_pool_tokens", 0)
            ),
            maximum_useful_pool_tokens=int(
                values.get("maximum_useful_pool_tokens", 0)
            ),
            kv_bytes_per_token=float(values["kv_bytes_per_token"]),
            static_pad_gib=float(values.get("static_pad_gib", 0.5)),
            static_overhead_gib=float(values.get("static_overhead_gib", 0.0)),
            fixed_mamba_cache_gib=float(
                values.get("fixed_mamba_cache_gib", 0.0)
            ),
            mamba_kv_memory_ratio=float(
                values.get("mamba_kv_memory_ratio", 0.0)
            ),
            cuda_graph_peak_gib=float(values.get("cuda_graph_peak_gib", 0.0)),
            request_workspace_gib=float(
                values.get("request_workspace_gib", 0.0)
            ),
            memavailable_floor_gib=float(
                values.get("memavailable_floor_gib", 8.0)
            ),
            gpu_headroom_gib=float(values.get("gpu_headroom_gib", 1.0)),
            fraction_base=str(values.get("fraction_base", "a_preload")),
        )
    return budgets


def configured_slots(
    plan: dict[str, Any], budgets: dict[str, Budget]
) -> list[tuple[str, Budget, dict[str, Any]]]:
    slots: list[tuple[str, Budget, dict[str, Any]]] = []
    seen_roles: set[str] = set()
    for slot in plan.get("admit", plan.get("slots", [])):
        role = str(slot["role"])
        model_id = str(slot["model_id"])
        if role in seen_roles:
            raise ValueError(f"duplicate configured role {role!r}")
        if model_id not in budgets:
            raise ValueError(f"slot {role!r} references unknown model {model_id!r}")
        seen_roles.add(role)
        slots.append((role, budgets[model_id], slot))
    if not slots:
        raise ValueError("plan contains no configured [[admit]] models")
    return slots


def floor_weighted_token_allocations(
    slots: list[tuple[str, Budget, dict[str, Any]]],
    gpu_free_gib: float,
    memavailable_gib: float,
    memavailable_floor_gib: float,
) -> tuple[dict[str, int], dict[str, float], float, float]:
    """Return bounded, total-proportional pools after satisfying all floors.

    ``target/floor`` supplies each slot's configured relative total-token
    weight. Floors are lower bounds, so a slot below its proportional share is
    caught up first. Ceilings stop allocation at the largest pool that the
    slot's configured context/concurrency can use; surplus is redistributed
    among slots that have not reached their ceiling.
    """
    floor_static = 0.0
    transients = 0.0
    headroom = 0.0
    raw_weights: dict[str, float] = {}
    ceilings: dict[str, int] = {}

    for role, budget, _slot in slots:
        floor_tokens = budget.minimum_admissible_pool_tokens
        if floor_tokens <= 0:
            raise ValueError(
                f"{role}/{budget.model_id}: floor_weighted mode requires a "
                "positive minimum_admissible_pool_tokens"
            )
        if budget.target_kv_tokens < floor_tokens:
            raise ValueError(
                f"{role}/{budget.model_id}: target_kv_tokens "
                f"{budget.target_kv_tokens} is below floor {floor_tokens}"
            )
        ceiling = budget.maximum_useful_pool_tokens
        if ceiling <= 0:
            raise ValueError(
                f"{role}/{budget.model_id}: floor_weighted mode requires a "
                "positive maximum_useful_pool_tokens"
            )
        if ceiling < floor_tokens:
            raise ValueError(
                f"{role}/{budget.model_id}: maximum useful pool {ceiling} is "
                f"below floor {floor_tokens}"
            )
        floor_static += budget.static_required_for_tokens(floor_tokens)
        transients += budget.cuda_graph_peak_gib + budget.request_workspace_gib
        headroom += budget.gpu_headroom_gib
        # The relationship is entirely configuration-derived.  With current
        # values 512/256 and 1024/256 become 2 and 4, normalized below to 1:2.
        raw_weights[role] = budget.target_kv_tokens / floor_tokens
        ceilings[role] = ceiling

    gpu_growth_gib = gpu_free_gib - floor_static - transients - headroom
    host_growth_gib = (
        memavailable_gib
        - memavailable_floor_gib
        - floor_static
        - transients
    )
    growth_gib = min(gpu_growth_gib, host_growth_gib)
    if growth_gib < 0:
        raise ValueError(
            "joint minimums do not fit: "
            f"floor_static={floor_static:.2f}GiB transients={transients:.2f}GiB "
            f"gpu_headroom={headroom:.2f}GiB system_floor={memavailable_floor_gib:.2f}GiB "
            f"gpu_free={gpu_free_gib:.2f}GiB memavailable={memavailable_gib:.2f}GiB"
        )

    weight_base = min(raw_weights.values())
    normalized = {role: weight / weight_base for role, weight in raw_weights.items()}
    growth_bytes = growth_gib * GIB

    def allocations_at_scale(scale: float) -> dict[str, int]:
        return {
            role: min(
                ceilings[role],
                max(
                    budget.minimum_admissible_pool_tokens,
                    math.floor(scale * normalized[role] + 1e-9),
                ),
            )
            for role, budget, _slot in slots
        }

    def incremental_bytes(allocations: dict[str, int]) -> float:
        return sum(
            (allocations[role] - budget.minimum_admissible_pool_tokens)
            * budget.effective_cache_bytes_per_token
            for role, budget, _slot in slots
        )

    maximum_scale = max(
        ceilings[role] / normalized[role]
        for role, _budget, _slot in slots
    )
    ceiling_allocations = allocations_at_scale(maximum_scale + 1.0)
    ceiling_cost = incremental_bytes(ceiling_allocations)
    if ceiling_cost <= growth_bytes:
        allocations = ceiling_allocations
        used_growth_bytes = ceiling_cost
    else:
        low = 0.0
        high = maximum_scale + 1.0
        # Monotone weighted water-fill. Eighty iterations is far beyond the
        # precision needed to identify integer token boundaries here.
        for _ in range(80):
            middle = (low + high) / 2.0
            candidate = allocations_at_scale(middle)
            if incremental_bytes(candidate) <= growth_bytes:
                low = middle
            else:
                high = middle
        allocations = allocations_at_scale(low)
        used_growth_bytes = incremental_bytes(allocations)

    unused_growth_gib = max(0.0, (growth_bytes - used_growth_bytes) / GIB)
    return allocations, normalized, growth_gib, unused_growth_gib


def build_results(
    slots: list[tuple[str, Budget, dict[str, Any]]],
    allocations: dict[str, int],
    total_pool_weights: dict[str, float],
    gpu_free_gib: float,
    device_total_gib: float,
    memavailable_gib: float,
) -> list[AdmissionResult]:
    results: list[AdmissionResult] = []
    current_gpu_free = gpu_free_gib
    current_memavailable = memavailable_gib
    for role, budget, _slot in slots:
        result = admit(
            budget=budget,
            role=role,
            allocated_tokens=allocations[role],
            total_pool_weight=total_pool_weights[role],
            a_preload_gib=current_gpu_free,
            device_total_gib=device_total_gib,
            memavailable_before_gib=current_memavailable,
        )
        results.append(result)
        if result.overall_pass:
            current_gpu_free -= result.peak_required_gib
            current_memavailable -= result.peak_required_gib
    return results


def resolve(ledger_path: str, plan_path: str, dry_run: bool = False):
    del dry_run  # The resolver is always read-only.
    budgets = load_budgets(ledger_path)
    with open(plan_path, "rb") as handle:
        plan = tomllib.load(handle)

    device_total_gib = float(plan["device"]["total_gib"])
    memavailable_start_gib = float(plan["observed"]["memavailable_now_gib"])
    gpu_free_now = plan.get("observed", {}).get("gpu_free_now_gib")
    has_live_free = gpu_free_now is not None
    gpu_free_gib = float(gpu_free_now) if has_live_free else device_total_gib

    policy = plan.get("policy", {})
    floor_gib = float(
        policy.get(
            "memavailable_floor_gib",
            next(iter(budgets.values())).memavailable_floor_gib if budgets else 8.0,
        )
    )
    if not math.isfinite(floor_gib) or floor_gib <= 0:
        raise ValueError("memavailable_floor_gib must be finite and > 0")
    allocation_mode = str(policy.get("allocation_mode", "fixed_targets"))
    if allocation_mode not in ALLOCATION_MODES:
        raise ValueError(
            f"unknown allocation_mode {allocation_mode!r}; "
            f"expected one of {sorted(ALLOCATION_MODES)}"
        )
    for budget in budgets.values():
        budget.memavailable_floor_gib = floor_gib

    residents = plan.get("resident", [])
    if allocation_mode == "floor_weighted" and residents:
        raise ValueError(
            "floor_weighted mode requires one cold joint plan; configured models "
            "must be [[admit]] entries, not already-fixed [[resident]] entries"
        )

    if allocation_mode == "fixed_targets" and residents and not has_live_free:
        for resident in residents:
            model_id = resident["model_id"]
            if model_id not in budgets:
                raise ValueError(f"resident references unknown model {model_id!r}")
            gpu_free_gib -= budgets[model_id].peak_required_gib

    slots = configured_slots(plan, budgets)
    print(
        f"# policy: allocation_mode={allocation_mode}, "
        f"memavailable_floor={floor_gib:.1f} GiB\n"
    )
    print(
        f"# memory plan — device {device_total_gib:.1f} GiB, "
        f"GPU free {gpu_free_gib:.1f} GiB, "
        f"MemAvailable {memavailable_start_gib:.1f} GiB\n"
    )

    if allocation_mode == "floor_weighted":
        (
            allocations,
            total_pool_weights,
            growth_gib,
            unused_growth_gib,
        ) = floor_weighted_token_allocations(
            slots, gpu_free_gib, memavailable_start_gib, floor_gib
        )
        print(f"# usable growth memory after all configured floors = {growth_gib:.2f} GiB\n")
        if unused_growth_gib > 0.001:
            print(
                "# growth memory left unassigned after useful pool ceilings = "
                f"{unused_growth_gib:.2f} GiB\n"
            )
    else:
        allocations = {}
        total_pool_weights = {}
        for role, budget, slot in slots:
            allocations[role] = int(
                slot.get("allocated_kv_tokens", budget.target_kv_tokens)
            )
            total_pool_weights[role] = 1.0

    results = build_results(
        slots,
        allocations,
        total_pool_weights,
        gpu_free_gib,
        device_total_gib,
        memavailable_start_gib,
    )
    any_fail = any(not result.overall_pass for result in results)

    for result in results:
        budget = budgets[result.model_id]
        print(f"## role={result.role}  model={result.model_id}")
        print(f"  configured floor     = {result.minimum_admissible_pool_tokens} tokens")
        print(f"  configured target    = {result.configured_target_tokens} tokens")
        print(f"  configured ceiling   = {result.maximum_useful_pool_tokens} tokens")
        print(f"  derived total weight = {result.total_pool_weight:.4f}")
        print(f"  allocated pool       = {result.max_total_tokens} tokens")
        print(
            f"  static_required      = {result.static_required_gib:.2f} GiB "
            f"(weights {budget.weights_gib:.2f} + KV "
            f"{budget.kv_gib(result.max_total_tokens):.2f} + Mamba "
            f"{budget.mamba_gib(result.max_total_tokens):.2f} proportional + "
            f"{budget.fixed_mamba_cache_gib:.2f} fixed-Mamba + fixed overhead "
            f"{budget.static_overhead_gib:.2f} + pad {budget.static_pad_gib:.2f})"
        )
        print(f"  A_preload            = {result.a_preload_gib:.2f} GiB")
        print(f"  -> mem_fraction_static = {result.fraction:.6f}")
        print(f"  -> max_total_tokens    = {result.max_total_tokens}")
        print(
            f"  GPU gate:   [{'PASS' if result.gpu_gate_pass else 'FAIL'}] "
            f"{result.gpu_gate_detail}"
        )
        print(
            f"  Linux gate: [{'PASS' if result.linux_gate_pass else 'FAIL'}] "
            f"{result.linux_gate_detail}"
        )
        for note in result.notes:
            print(f"  NOTE: {note}")
        print()

    if any_fail:
        print("RESULT: REFUSE — at least one configured model failed admission.")
        return results, 1
    print(f"RESULT: ADMIT — all {len(results)} configured models pass.")
    print("\n# emit (for serialized admission):")
    for result in results:
        print(
            f"  {result.role}/{result.model_id}: "
            f"mem_fraction_static={result.fraction:.6f} "
            f"max_total_tokens={result.max_total_tokens}"
        )
    return results, 0


def emit_json(
    results: list[AdmissionResult], rc: int, allocation_mode: str | None = None
) -> None:
    document = {
        "result": "ADMIT" if rc == 0 else "REFUSE",
        "exit_code": rc,
        "allocation_mode": allocation_mode,
        "models": [result.to_dict() for result in results],
    }
    print(json.dumps(document, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="joint SGLang memory-budget resolver")
    parser.add_argument("ledger", help="per-model budget ledger TOML")
    parser.add_argument("plan", help="configured residency plan TOML")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    try:
        with open(args.plan, "rb") as handle:
            plan_doc = tomllib.load(handle)
        allocation_mode = str(
            plan_doc.get("policy", {}).get("allocation_mode", "fixed_targets")
        )
        if args.format == "json":
            with contextlib.redirect_stdout(io.StringIO()):
                results, rc = resolve(args.ledger, args.plan, args.dry_run)
            emit_json(results, rc, allocation_mode)
            return rc
        _results, rc = resolve(args.ledger, args.plan, args.dry_run)
        return rc
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "result": "REFUSE",
                        "exit_code": 2,
                        "allocation_mode": None,
                        "models": [],
                        "error": str(error),
                    },
                    indent=2,
                )
            )
        else:
            print(f"REFUSING: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
