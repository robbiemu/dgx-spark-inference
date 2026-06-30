# Memory plan (schema notes)

The memory planner (`tools/memory_planner/resolve_memory_plan.py`) is a
**standalone consistency + planning tool**, like the capability resolver. It is
not yet on the live launch path; the plan is for `dispatch.sh` to call it as a
preflight (derive the two knobs, refuse on gate failure). See
`docs/v0_2_phase0_results.md` for the Phase 0 measurements this is built on and
the design rationale (why a *derived* fraction + `max_total_tokens` cap, not a
static fraction or enumerated residency sets).

The planner turns a per-model **budget ledger** + a **residency plan** into the
two launch knobs SGLang needs and runs two admission gates. It is stdlib-only,
no GPU.

## Budget ledger — `budget_ledger.toml`

One `[[profiles]]` entry per model (array-of-tables so dots in `model_id` stay a
plain string, not a TOML table path). Required `[profiles.budget]` fields:

- `model_id`
- `weights_gib` — measured cold-load persistent delta
- `target_kv_tokens` — the **measured realized pool** (the target the cap pins)
- `kv_bytes_per_token` — **measured per profile, not a timeless constant** (KV
  dtype, attention layout, TP, MTP/speculative config, and SGLang version can
  alter realized pool bytes; re-measure if any change)
- `static_overhead_gib` — the empirically-reserved-but-unaccounted portion of a
  model's measured fraction that is **not** weights + KV pool (see note below)
- `cuda_graph_peak_gib`, `request_workspace_gib` — transient peak (graph capture
  beyond the static budget; per-request activations)
- `static_pad_gib`, `gpu_headroom_gib` — alignment/allocator cushions

### `static_overhead_gib` — the large-model asymmetry (important)

For large models (e.g. the 27B primary), CUDA-graph capture and hybrid-attention
state are reserved **into** the static budget. The primary's measured
`mem_fraction_static=0.60` therefore includes ~21 GiB of overhead beyond
`weights + kv`, so the clean component sum derives only `~0.42`. `static_overhead_gib`
captures that measured gap so the **derived** fraction reproduces the
**measured** one. It is **measured, not computed**; record per profile.

The helper (9B, pool-dominated) needs none — its `0.80` reproduces from
components alone. That asymmetry is real and model-size-dependent.

> **Do not** fold `cuda_graph_peak_gib` / `request_workspace_gib` into the
> fraction numerator (via static_overhead or otherwise). Putting transient graph
> memory into the fraction makes SGLang enlarge the static KV reservation and
> steals the slack needed to capture graphs. Graph/workspace belong in the
> **admission check**, not the static budget. `static_overhead_gib` is for
> overhead that SGLang reserves *into* the static budget itself (the allocator's
> non-linear behavior for large models), which is a different thing.

## Residency plan — `plan_*.toml`

- `device.total_gib` — device total (SGLang's view; GB10 ≈ 121.7)
- `observed.memavailable_now_gib` — host MemAvailable **at plan time**, reflecting
  any already-resident models
- `[policy]` — host-wide policy (see below)
- `[[resident]]` — models already up (peak already consumed; not re-gated)
- `[[admit]]` — models to load now (`role` + `model_id`), in load order

### `[policy]` — host-wide tunables

- `memavailable_floor_gib` — the **hard refusal line**: the planner refuses to
  admit a model whose load would push system MemAvailable below this.

**Layering** (authoritative → fallback): plan `[policy]` > ledger per-model value
> built-in default `8.0`. The floor is host-wide policy, never hardcoded in the
resolver.

> **Gloss / tuning note.** `memavailable_floor_gib` is the line between "marginal
> but runs" and "refuse to start." On GB10 (121 GiB unified memory), fitting a 27B
> primary + a 9B helper steady-state lands at **~6.5 GiB** MemAvailable — so a
> floor of `8.0` *refuses* a configuration that empirically runs, while `6.0`
> admits it with ~0.5 GiB margin. **The floor's effective margin depends on
> `cuda_graph_peak_gib` and `request_workspace_gib`, which are currently inferred
> estimates.** Tightening those measurements is a Phase 1 task and may admit a
> higher floor later. Lower the floor to admit tighter residency (more risk of
> the host wedging under unexpected load); raise it to refuse sooner (safer).
> The wedge that prompted this design (helper-first with a static `0.80`
> fraction) reserved ~99 GiB and crashed the box — that is the failure mode the
> floor + cap exist to prevent.

## The two admission gates (fail-safe)

```
GPU gate:   A_preload − static_required  ≥  graph_peak + workspace + gpu_headroom
Linux gate: MemAvailable_after_load       ≥  memavailable_floor_gib
```
A model that fails either gate is refused **before the GPU is touched**. A failed
slot does not reduce the running counters for subsequent slots (one bad record
cannot cascade a false fit). The dispatcher emits, per admitted model:
`mem_fraction_static` (derived) + `max_total_tokens` (the target cap).
