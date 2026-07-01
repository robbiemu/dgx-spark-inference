#!/usr/bin/env python3
"""
resolve_memory_plan.py — Phase 1 memory-budget resolver for v0.2 multi-role.

Turns the per-model measured budget ledger into the two launch knobs SGLang needs,
and runs the two admission gates that make co-residency order-independent and
fail-safe. This is a PLANNING / CONSISTENCY tool (like resolve_service_plan.py):
stdlib-only, no GPU, no live launch. The dispatcher may call it as a preflight.

DESIGN (committed in docs/v0_2_phase0_results.md):
  - static_required = weights + (target_kv_tokens * kv_bytes_per_token) + static_pad
  - fraction        = static_required / A_preload        # DERIVED from observed free
  - peak_required   = static_required + cuda_graph_peak + request_workspace
  Two gates (fail-safe — refuse to start rather than shrink a contract or wedge host):
    GPU gate:   A_preload - static_required  >= graph_peak + workspace + gpu_headroom
    Linux gate: MemAvailable_now             >= memavailable_floor + incremental_peak
  Both sglang knobs are emitted:
    mem_fraction_static = derived fraction (keeps the target pool feasible)
    max_total_tokens    = target_kv_tokens  (caps the pool; makes order safe)

WHY graph/workspace are NOT in the fraction numerator:
  SGLang computes KV capacity from the fraction first, then the pool is allocated
  within it. Putting graph memory into the fraction enlarges the static KV
  reservation and steals the very slack needed to capture graphs. So graph/workspace
  live in the ADMISSION CHECK, not the static budget.

INVOCATION:
  resolve_memory_plan.py <ledger.toml> <plan.toml> [--dry-run]

  ledger.toml: per-model budget records (measured values).
  plan.toml:   the proposed residency plan (which roles enabled, observed free mem).

See test_resolve_memory_plan.py for the measured-Phase-0 fixtures this must satisfy.
"""
from __future__ import annotations
import argparse
import json
import sys
import tomllib
from dataclasses import dataclass, field
from typing import Any

GIB = 1 << 30  # GiB in bytes


@dataclass
class Budget:
    """A single model's measured memory budget (combination-independent)."""
    model_id: str
    weights_gib: float
    target_kv_tokens: int
    kv_bytes_per_token: float
    # The three capacity concepts (review: target is a CEILING, not a promise).
    # The order-independence probe proved realized < requested (366,011 vs 376,048),
    # so the launcher must verify realized capacity after startup, not trust the cap.
    #   target_kv_tokens        — the cap passed as max_total_tokens (CEILING)
    #   minimum_admissible_pool_tokens — the floor a role contract REQUIRES; the
    #                             launcher refuses/marks-unhealthy if realized < this
    #   realized_pool_tokens    — measured post-startup via /get_server_info (verify)
    minimum_admissible_pool_tokens: int = 0   # 0 = no minimum enforced
    static_pad_gib: float = 0.5
    # Empirically-reserved-but-unaccounted static overhead: the portion of a
    # model's MEASURED fraction that is NOT weights + KV pool. For large models
    # (e.g. the 27B primary), CUDA-graph capture and hybrid-attention state are
    # reserved INTO the static budget, so the measured fraction (0.60) exceeds
    # the clean component sum (~0.42). This field captures that measured gap so
    # the DERIVED fraction reproduces the MEASURED one. It is measured, not
    # computed; re-measure per model/profile. See docs/v0_2_phase0_results.md.
    static_overhead_gib: float = 0.0
    cuda_graph_peak_gib: float = 0.0      # transient peak (graph capture beyond static)
    request_workspace_gib: float = 0.0    # transient peak (per-request activations)
    memavailable_floor_gib: float = 8.0   # GB10 unified-memory safety floor
    gpu_headroom_gib: float = 1.0         # safety slack for allocator effects

    @property
    def static_required_gib(self) -> float:
        # weights + target KV pool + measured static overhead + alignment pad.
        # (static_overhead captures graph/hybrid-attn state reserved INTO the
        # static budget for large models; cuda_graph_peak is the TRANSIENT part.)
        kv_gib = (self.target_kv_tokens * self.kv_bytes_per_token) / GIB
        return self.weights_gib + kv_gib + self.static_overhead_gib + self.static_pad_gib

    @property
    def peak_required_gib(self) -> float:
        # static + the transient peak (graphs + per-request workspace).
        return self.static_required_gib + self.cuda_graph_peak_gib + self.request_workspace_gib

    def derive_fraction(self, a_preload_gib: float) -> float:
        """The DERIVED mem_fraction_static for this model at launch time."""
        if a_preload_gib <= 0:
            raise ValueError(f"{self.model_id}: A_preload must be > 0 (got {a_preload_gib})")
        return self.static_required_gib / a_preload_gib


@dataclass
class AdmissionResult:
    model_id: str
    fraction: float
    max_total_tokens: int
    static_required_gib: float
    peak_required_gib: float
    a_preload_gib: float
    # gate verdicts (True = pass)
    gpu_gate_pass: bool
    gpu_gate_detail: str
    linux_gate_pass: bool
    linux_gate_detail: str
    overall_pass: bool
    notes: list[str] = field(default_factory=list)
    # The role's minimum admissible pool (from the ledger); the launcher verifies
    # the post-startup realized pool >= this. Set from the Budget at admit() time.
    minimum_admissible_pool_tokens: int = 0

    def to_dict(self) -> dict:
        """Machine-readable serialization for the --format json contract."""
        return {
            "model_id": self.model_id,
            "mem_fraction_static": round(self.fraction, 6),
            "max_total_tokens": int(self.max_total_tokens),
            "minimum_admissible_pool_tokens": int(self.minimum_admissible_pool_tokens),
            "overall_pass": bool(self.overall_pass),
            "gpu_gate_pass": bool(self.gpu_gate_pass),
            "linux_gate_pass": bool(self.linux_gate_pass),
            "static_required_gib": round(self.static_required_gib, 4),
            "peak_required_gib": round(self.peak_required_gib, 4),
            "a_preload_gib": round(self.a_preload_gib, 4),
        }


def admit(
    budget: Budget,
    a_preload_gib: float,
    memavailable_before_gib: float,
) -> AdmissionResult:
    """Run both admission gates for one model against observed free memory.

    a_preload_gib:        GPU free memory this model sees at its load moment
                          (device_total - peak of already-resident models).
    memavailable_before_gib: host MemAvailable just before this model loads
                          (already reduced by resident models' peaks).
    """
    fraction = budget.derive_fraction(a_preload_gib)
    # max_total_tokens is the target — the cap that makes load order safe.
    max_total_tokens = budget.target_kv_tokens

    # GPU gate: room for graphs + workspace + headroom AFTER the static budget.
    post_static = a_preload_gib - budget.static_required_gib
    graph_workspace_headroom = budget.cuda_graph_peak_gib + budget.request_workspace_gib + budget.gpu_headroom_gib
    gpu_pass = post_static >= graph_workspace_headroom
    gpu_detail = (f"A_preload({a_preload_gib:.2f}) - static({budget.static_required_gib:.2f}) "
                  f"= {post_static:.2f} >= graph+ws+headroom({graph_workspace_headroom:.2f})")

    # Linux gate: after THIS model loads, MemAvailable must still be >= floor.
    # memavailable_before is the running counter (each prior slot decremented it).
    memavailable_after = memavailable_before_gib - budget.peak_required_gib
    linux_pass = memavailable_after >= budget.memavailable_floor_gib
    linux_detail = (f"MemAvailable after load = {memavailable_before_gib:.2f} - "
                    f"peak({budget.peak_required_gib:.2f}) = {memavailable_after:.2f} "
                    f">= floor({budget.memavailable_floor_gib:.2f})")

    notes = []
    if fraction > 0.95:
        notes.append(f"WARN: derived fraction {fraction:.3f} > 0.95 — model barely fits; "
                     "consider lowering target_kv_tokens or freeing residency.")
    if fraction < 0.05:
        notes.append(f"WARN: derived fraction {fraction:.3f} very low — verify target_kv_tokens "
                     "isn't starved (max_total_tokens cannot grow a tiny pool).")

    return AdmissionResult(
        model_id=budget.model_id, fraction=fraction, max_total_tokens=max_total_tokens,
        static_required_gib=budget.static_required_gib, peak_required_gib=budget.peak_required_gib,
        a_preload_gib=a_preload_gib,
        gpu_gate_pass=gpu_pass, gpu_gate_detail=gpu_detail,
        linux_gate_pass=linux_pass, linux_gate_detail=linux_detail,
        overall_pass=gpu_pass and linux_pass, notes=notes,
        minimum_admissible_pool_tokens=budget.minimum_admissible_pool_tokens,
    )


def load_budgets(ledger_path: str) -> dict[str, Budget]:
    with open(ledger_path, "rb") as f:
        d = tomllib.load(f)
    out = {}
    # [[profiles]] array-of-tables, each with model_id + a [budget] sub-table.
    # (Array-of-tables so dots in model_id stay a plain string, not a table path.)
    for rec in d.get("profiles", []):
        b = rec.get("budget")
        if not b:
            continue
        mid = rec["model_id"]
        out[mid] = Budget(
            model_id=mid,
            weights_gib=float(b["weights_gib"]),
            target_kv_tokens=int(b["target_kv_tokens"]),
            minimum_admissible_pool_tokens=int(b.get("minimum_admissible_pool_tokens", 0)),
            kv_bytes_per_token=float(b["kv_bytes_per_token"]),
            static_pad_gib=float(b.get("static_pad_gib", 0.5)),
            static_overhead_gib=float(b.get("static_overhead_gib", 0.0)),
            cuda_graph_peak_gib=float(b.get("cuda_graph_peak_gib", 0.0)),
            request_workspace_gib=float(b.get("request_workspace_gib", 0.0)),
            memavailable_floor_gib=float(b.get("memavailable_floor_gib", 8.0)),
            gpu_headroom_gib=float(b.get("gpu_headroom_gib", 1.0)),
        )
    return out


def resolve(ledger_path: str, plan_path: str, dry_run: bool = False) -> int:
    budgets = load_budgets(ledger_path)
    with open(plan_path, "rb") as f:
        plan = tomllib.load(f)

    device_total_gib = float(plan["device"]["total_gib"])
    memavailable_start_gib = float(plan["observed"]["memavailable_now_gib"])
    # Live mode (Blocker 1 fix): when observed.gpu_free_now_gib is present, it is
    # the MEASURED free GPU memory immediately before launch — the true A_preload.
    # It ALREADY includes resident allocations, so residents must NOT be subtracted
    # again (only used for identity/revision/guard checks). When absent (offline/
    # planning mode), fall back to device_total_gib and subtract resident peaks.
    gpu_free_now = plan.get("observed", {}).get("gpu_free_now_gib")
    has_live_free = gpu_free_now is not None
    if has_live_free:
        gpu_free_now = float(gpu_free_now)

    # The MemAvailable floor is a HOST-WIDE policy (how much system memory this
    # machine must keep free under GB10 unified memory), NOT a per-model property.
    # It lives in the plan (operator deployment state) and overrides any value in
    # the per-model ledger. Plan > ledger > built-in default (8.0).
    floor_gib = float(plan.get("policy", {}).get(
        "memavailable_floor_gib",
        next(iter(budgets.values())).memavailable_floor_gib if budgets else 8.0,
    ))
    # Apply the plan-level floor to every budget (so admission uses one policy).
    for b in budgets.values():
        b.memavailable_floor_gib = floor_gib
    print(f"# policy: memavailable_floor = {floor_gib:.1f} GiB (host-wide, from plan)\n")

    results: list[AdmissionResult] = []
    any_fail = False
    # gpu_free is the A_preload each candidate sees. In LIVE mode it starts from the
    # MEASURED gpu_free_now_gib (which already includes resident allocations — do NOT
    # subtract residents again). In OFFLINE/planning mode it starts at device_total
    # and resident peaks are subtracted (reconstruction, less accurate).
    gpu_free_gib = gpu_free_now if has_live_free else device_total_gib
    memavailable_gib = memavailable_start_gib

    resident = plan.get("resident", [])
    if resident:
        print(f"# already resident: {[r['model_id'] for r in resident]}")
        for r0 in resident:
            mid = r0["model_id"]
            if mid not in budgets:
                print(f"REFUSING: resident references unknown model '{mid}'", file=sys.stderr)
                return ([], 2)
            if not has_live_free:
                # OFFLINE mode only: reconstruct gpu_free by subtracting resident peaks.
                # LIVE mode: gpu_free_now already includes them — do not double-subtract.
                gpu_free_gib -= budgets[mid].peak_required_gib
        if has_live_free:
            print(f"#   (residents present but NOT subtracted: gpu_free_now already includes them)")
        else:
            print(f"#   (their peak subtracted in offline mode; MemAvailable={memavailable_gib:.1f} is observed)")
        print()

    print(f"# memory plan — device {device_total_gib:.1f} GiB, "
          f"MemAvailable now {memavailable_gib:.1f} GiB\n")
    for slot in plan.get("admit", plan.get("slots", [])):
        role = slot["role"]
        mid = slot["model_id"]
        if mid not in budgets:
            print(f"REFUSING: slot '{role}' references unknown model '{mid}'", file=sys.stderr)
            return ([], 2)
        b = budgets[mid]
        # A_preload = GPU free at this model's load moment (residents' peaks already subtracted).
        a_preload = gpu_free_gib
        # Linux gate gets the MemAvailable just BEFORE this model loads (running counter).
        r = admit(b, a_preload, memavailable_gib)
        results.append(r)
        print(f"## role={role}  model={mid}")
        print(f"  A_preload            = {r.a_preload_gib:.2f} GiB")
        print(f"  static_required      = {r.static_required_gib:.2f} GiB  "
              f"(weights {b.weights_gib} + kv {b.target_kv_tokens}tok + pad {b.static_pad_gib})")
        print(f"  peak_required        = {r.peak_required_gib:.2f} GiB  "
              f"(+ graph {b.cuda_graph_peak_gib} + workspace {b.request_workspace_gib})")
        print(f"  -> mem_fraction_static = {r.fraction:.4f}")
        print(f"  -> max_total_tokens    = {r.max_total_tokens}")
        print(f"  GPU gate:   [{'PASS' if r.gpu_gate_pass else 'FAIL'}] {r.gpu_gate_detail}")
        print(f"  Linux gate: [{'PASS' if r.linux_gate_pass else 'FAIL'}] {r.linux_gate_detail}")
        for n in r.notes:
            print(f"  NOTE: {n}")
        # A model that fails admission must NOT reduce the counters for subsequent slots
        # (fail-safe: one bad record can't cascade a false fit).
        if r.overall_pass:
            gpu_free_gib -= r.peak_required_gib
            memavailable_gib -= r.peak_required_gib
        else:
            any_fail = True
        print()

    if any_fail:
        print("RESULT: REFUSE — at least one slot failed admission. Do not launch.")
        return (results, 1)
    print(f"RESULT: ADMIT — all {len(results)} slots pass. "
          f"GPU free after all: {gpu_free_gib:.2f} GiB, "
          f"MemAvailable after all: {memavailable_gib:.2f} GiB")
    print("\n# emit (for the dispatcher):")
    for r in results:
        print(f'  {r.model_id}: mem_fraction_static={r.fraction:.4f}  '
              f'max_total_tokens={r.max_total_tokens}')
    return (results, 0)


def emit_json(results: list[AdmissionResult], rc: int) -> None:
    """Machine-readable contract for the dispatcher (--format json).
    The dispatcher parses this with json.loads and validates each model's fields;
    it must NOT grep the human-readable mode."""
    doc = {
        "result": "ADMIT" if rc == 0 else "REFUSE",
        "exit_code": rc,
        "models": [r.to_dict() for r in results],
    }
    print(json.dumps(doc, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description="v0.2 memory-budget resolver (planner, no GPU)")
    ap.add_argument("ledger", help="per-model budget ledger TOML")
    ap.add_argument("plan", help="residency plan TOML")
    ap.add_argument("--dry-run", action="store_true", help="no-op flag (this tool never launches)")
    ap.add_argument("--format", choices=["text", "json"], default="text",
                    help="json = machine-readable contract for the dispatcher (default text)")
    a = ap.parse_args()

    if a.format == "json":
        # Suppress the human-readable prose resolve() prints; emit structured JSON instead.
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            results, rc = resolve(a.ledger, a.plan, a.dry_run)
        emit_json(results, rc)
        return rc
    results, rc = resolve(a.ledger, a.plan, a.dry_run)
    return rc


if __name__ == "__main__":
    sys.exit(main())
