#!/usr/bin/env python3
"""
Unit tests for resolve_memory_plan.py — validates the budget algebra against the
MEASURED Phase 0 numbers. No GPU; pure arithmetic. Run: python3 test_resolver.py
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
from resolve_memory_plan import Budget, admit, load_budgets

LEDGER = os.path.join(os.path.dirname(__file__), "budget_ledger.toml")
GIB = 1 << 30

passed = failed = 0
def check(name, cond, detail=""):
    global passed, failed
    if cond: passed += 1; print(f"  PASS  {name}")
    else: failed += 1; print(f"  FAIL  {name}  {detail}")

def approx(a, b, tol=0.02):  # 2% relative tol for GiB rounding
    return math.isclose(a, b, rel_tol=tol)


def main():
    budgets = load_budgets(LEDGER)
    qwen = budgets["qwen36-27b-fp8"]
    helper = budgets["ornith-1.0-9b-fp8"]

    print("=== T1: static_required matches measured footprints ===")
    # helper: weights 12.98 + kv(376048*65536/GiB) + pad 0.5
    helper_kv_gib = 376048 * 65536 / GIB
    check("helper static ~ weights+pool+pad",
          approx(helper.static_required_gib, 12.98 + helper_kv_gib + 0.5),
          f"got {helper.static_required_gib:.2f}, expected {12.98+helper_kv_gib+0.5:.2f}")
    print(f"    helper static_required = {helper.static_required_gib:.2f} GiB "
          f"(kv pool alone = {helper_kv_gib:.2f})")

    print("=== T2: helper co-resident fraction ~ 0.80 (matches measured probe) ===")
    # The measured co-resident probe: helper loaded second, saw ~43.78 GiB free.
    r = admit(helper, a_preload_gib=43.78, memavailable_before_gib=46.0)
    check("co-resident fraction ~0.80", approx(r.fraction, 0.80, 0.05),
          f"got {r.fraction:.4f}")
    print(f"    derived fraction = {r.fraction:.4f}  (measured probe used 0.80)")

    print("=== T3: helper SOLO fraction ~ 0.294 (the second-opinion derivation) ===")
    # Helper first against ~124 GiB free. This is the case that wedged with a
    # static 0.80 (reserving ~99 GiB). Derived: 36.48/124 = 0.294.
    r_solo = admit(helper, a_preload_gib=124.0, memavailable_before_gib=124.0)
    check("solo fraction ~0.294", approx(r_solo.fraction, 0.294, 0.05),
          f"got {r_solo.fraction:.4f}")
    print(f"    derived solo fraction = {r_solo.fraction:.4f}  (vs 0.80 static that wedged)")

    print("=== T4: max_total_tokens is the target (the cap, order-independent) ===")
    check("helper max_total_tokens = 376048", r.max_total_tokens == 376048,
          f"got {r.max_total_tokens}")
    check("solo cap identical to co-resident cap",
          r_solo.max_total_tokens == r.max_total_tokens,
          "cap must be invariant to load order")

    print("=== T5: the SOLO case would now ADMIT (not wedge) — GPU gate passes ===")
    # static ~36.5; free 124; post-static ~87.5 >= graph+ws+headroom (~3.05). PASS.
    check("solo GPU gate passes", r_solo.gpu_gate_pass, r_solo.gpu_gate_detail)
    print(f"    {r_solo.gpu_gate_detail}")

    print("=== T6: primary fraction reproduces measured 0.60 (with static_overhead) ===")
    # The primary's measured 0.60 includes allocator overhead reserved into the
    # static budget. With static_overhead_gib accounted, the derived fraction
    # reproduces the empirically-tuned 0.60. (Without it, it'd be ~0.42 — see
    # the static_overhead ledger note for why large models need this field.)
    r_q = admit(qwen, a_preload_gib=121.7, memavailable_before_gib=121.7)
    check("primary fraction ~0.60 (overhead-accounted)", approx(r_q.fraction, 0.60, 0.03),
          f"got {r_q.fraction:.4f}")
    print(f"    derived primary fraction = {r_q.fraction:.4f}  (PRE_PREP measured 0.60)")
    print(f"    primary static_required = {r_q.static_required_gib:.2f} GiB "
          f"(weights+kv+overhead {qwen.static_overhead_gib}+pad)")

    print("=== T9: helper needs NO static_overhead (pool-dominated; asymmetry is the finding) ===")
    # The helper's measured 0.80 reproduced from components alone (T2). Its graph
    # footprint is small and pool-dominated, so the allocator's static overhead
    # is ~0 for it. The primary needs static_overhead; the helper doesn't. That
    # asymmetry is real and model-size-dependent — record per-profile.
    check("helper static_overhead is 0 (pool-dominated)",
          helper.static_overhead_gib == 0.0,
          f"got {helper.static_overhead_gib}")
    print(f"    helper derived from components alone (no overhead field needed)")

    print("=== T7: combined primary-then-helper fits within device ===")
    # primary peak + helper peak must be <= device total (rough sanity).
    combined_peak = qwen.peak_required_gib + helper.peak_required_gib
    check("combined peak < device total", combined_peak < 121.7,
          f"combined peak {combined_peak:.2f} vs 121.7")
    print(f"    combined peak = {combined_peak:.2f} GiB (device 121.7)")

    print("=== T8: Linux gate (MemAvailable floor) catches the wedge condition ===")
    # After THIS model loads, MemAvailable must stay >= floor (8 GiB).
    # Simulate the post-wedge state: only 6 GiB free before load -> 6 - peak < 0 < 8.
    r_low = admit(helper, a_preload_gib=43.78, memavailable_before_gib=6.0)
    check("Linux gate FAILS when post-load MemAvailable < floor",
          not r_low.linux_gate_pass, "should refuse: 6 - peak drops below 8 floor")
    print(f"    {r_low.linux_gate_detail}")

    print("=== T10: --format json contract (machine-readable; for dispatch) ===")
    import json as _json, subprocess
    LEDGER_P = os.path.join(os.path.dirname(__file__), "budget_ledger.toml")
    PLAN_P = os.path.join(os.path.dirname(__file__), "plan_helper_first.toml")
    RESOLVER = os.path.join(os.path.dirname(__file__), "resolve_memory_plan.py")
    proc = subprocess.run([sys.executable, RESOLVER, LEDGER_P, PLAN_P, "--format", "json"],
                          capture_output=True, text=True)
    ok_json = True
    try:
        doc = _json.loads(proc.stdout)
    except Exception as e:
        ok_json = False; doc = {}
    check("json mode emits valid JSON", ok_json, f"stdout: {proc.stdout[:120]}")
    if ok_json:
        check("json result is ADMIT", doc.get("result") == "ADMIT", f"got {doc.get('result')}")
        check("json has models list", isinstance(doc.get("models"), list) and len(doc["models"]) == 2)
        # find the helper entry
        helper_m = next((m for m in doc["models"] if m["model_id"] == "ornith-1.0-9b-fp8"), {})
        f = helper_m.get("mem_fraction_static")
        check("json mem_fraction_static is finite in (0,1)",
              isinstance(f, (int, float)) and 0.0 < f < 1.0, f"got {f}")
        mtt = helper_m.get("max_total_tokens")
        check("json max_total_tokens is positive int",
              isinstance(mtt, int) and mtt > 0, f"got {mtt}")
        minp = helper_m.get("minimum_admissible_pool_tokens")
        check("json minimum_admissible_pool_tokens present",
              isinstance(minp, int) and minp >= 0, f"got {minp}")
        # json ADMIT exit code must be 0
        check("json exit code 0 on ADMIT", proc.returncode == 0, f"rc={proc.returncode}")
        # json matches the in-process derivation (cross-check the two paths agree)
        check("json helper fraction matches derive_fraction",
              approx(f, helper.derive_fraction(121.7), 0.01), f"json {f} vs derived")
        print(f"    helper json: frac={f} mtt={mtt} min={minp}")

    print("=== T11: REFUSE produces result=REFUSE in json ===")
    # A plan that starves MemAvailable below floor -> REFUSE.
    BAD_PLAN = os.path.join(os.path.dirname(__file__), "_bad_plan.toml")
    open(BAD_PLAN, "w").write(
        "device.total_gib = 121.7\n[policy]\nmemavailable_floor_gib = 50.0\n"
        "[observed]\nmemavailable_now_gib = 10.0\n[[admit]]\nrole='r'\nmodel_id='ornith-1.0-9b-fp8'\n")
    proc2 = subprocess.run([sys.executable, RESOLVER, LEDGER_P, BAD_PLAN, "--format", "json"],
                           capture_output=True, text=True)
    os.remove(BAD_PLAN)
    try:
        doc2 = _json.loads(proc2.stdout)
    except Exception:
        doc2 = {}
    check("REFUSE json result is REFUSE", doc2.get("result") == "REFUSE", f"got {doc2.get('result')}")
    check("REFUSE exit code 1", proc2.returncode == 1, f"rc={proc2.returncode}")

    print(f"\n=== {passed} passed, {failed} failed ===")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
