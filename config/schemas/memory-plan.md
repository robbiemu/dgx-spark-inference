# Memory plan (schema notes)

The memory planner (`tools/memory_planner/resolve_memory_plan.py`) is wired into
the live launch path via `src/inferencectl/admission.sh` (the serialized
admission wrapper), invoked by `dispatch.sh` when a planner pair is enrolled.
The wrapper holds a global lock across discover → sample → resolve → launch →
**verified allocation** (closing the preflight↔allocation race), and refuses
(exit 75) on gate failure. The **capability** resolver (`resolve_service_plan.py`)
remains deliberately OFF the live path (forbidden anti-pattern). See
`docs/v0_2_phase0_results.md` for the Phase 0 measurements this is built on and
the design rationale (why a *derived* fraction + `max_total_tokens` cap, not a
static fraction or enumerated residency sets).

#
# For measuring new profiles step-by-step, see docs/measure-model-budget.md.
The planner turns a per-model **budget ledger** + the complete configured model
topology into the two launch knobs SGLang needs and runs two admission gates.
It is stdlib-only, no GPU.

## Budget ledger — `budget_ledger.toml`

One `[[profiles]]` entry per model (array-of-tables so dots in `model_id` stay a
plain string, not a TOML table path). Required `[profiles.budget]` fields:

- `model_id`
- `weights_gib` — measured cold-load persistent delta
- `kv_bytes_per_token` — **measured per profile, not a timeless constant** (KV
  dtype, attention layout, TP, MTP/speculative config, and SGLang version can
  alter realized pool bytes; re-measure if any change)
- `static_overhead_gib` — the empirically-reserved-but-unaccounted portion of a
  model's measured fraction that is **not** weights + KV pool (see note below)
- `mamba_kv_memory_ratio` — for hybrid allocators, proportional Mamba/linear
  state GiB per ordinary-KV GiB. This is deliberately separate from fixed
  overhead because it changes when the planned token pool changes.
- `fixed_mamba_cache_gib` — Mamba state held constant by an explicit
  `max_mamba_cache_size`; unlike the ratio field, this does not grow with KV.
- `minimum_admissible_pool_tokens` — hard per-slot floor.
- `target_kv_tokens` — expected-demand setting used to derive proportional
  total-token weights.
- `maximum_useful_pool_tokens` — hard allocation ceiling, normally
  `context_length × max_running_requests` from the launch configuration.
- `cuda_graph_peak_gib`, `request_workspace_gib` — transient peak (graph capture
  beyond the static budget; per-request activations)
- `static_pad_gib`, `gpu_headroom_gib` — alignment/allocator cushions
- `fraction_base` — "a_preload" (default) or "device_total". An SGLang
runtime-path calibration (not a model-intrinsic property) determining which
base the resolver derives mem_fraction_static against. Calibrate by comparing
predicted versus realized pool sizes; see docs/measure-model-budget.md.

### Fixed overhead versus proportional Mamba state

`static_overhead_gib` is only the measured residual that does not vary with the
planned token pool. Hybrid state must be classified from the actual allocator
setting: an explicit numeric `max_mamba_cache_size` is recorded in
`fixed_mamba_cache_gib`; ratio-based automatic sizing is recorded in
`mamba_kv_memory_ratio`, allowing the planner to recalculate it for every pool.

> **Do not** fold `cuda_graph_peak_gib` / `request_workspace_gib` into the
> fraction numerator (via static_overhead or otherwise). Putting transient graph
> memory into the fraction makes SGLang enlarge the static KV reservation and
> steals the slack needed to capture graphs. Graph/workspace belong in the
> **admission check**, not the static budget. `static_overhead_gib` is for
> overhead that SGLang reserves *into* the static budget itself (the allocator's
> non-linear behavior for large models), which is a different thing.

## Configured topology and host policy

On the live path, `active-models.toml` is the authoritative topology. Every
`[active.<role>]` entry contributes its configured `model_id` to one joint
calculation; the implementation does not assume a role name or a model count.
`memory_plan.toml` supplies host policy such as the system memory floor.

The resolver's standalone input also accepts:

- `device.total_gib` — device total (SGLang's view; GB10 ≈ 121.7)
- `observed.gpu_free_now_gib` and `observed.memavailable_now_gib` — cold live
  measurements
- `[policy].allocation_mode = "floor_weighted"`
- one `[[admit]]` per configured role/model

`fixed_targets` remains available for a single launch-time gate with an
already-chosen `allocated_kv_tokens`; it is how the wrapper turns one persisted
joint allocation into a fraction against that model's actual `A_preload`.

### Floor-first allocation

For every configured slot `i`:

```text
floor_tokens_i  = minimum_admissible_pool_tokens_i
raw_weight_i    = target_kv_tokens_i / floor_tokens_i
total_weight_i  = raw_weight_i / min(raw_weight)
```

The planner first reserves all weights, fixed overhead, fixed Mamba state,
floor KV, proportional Mamba state, pads, graph/workspace transients, GPU
headroom, and the system `MemAvailable` floor. If the complete set of floors
cannot fit, it refuses the entire cold plan before any model launches.

It then performs a bounded weighted water-fill over **total token pools**, not
over increments above each floor. A slot below its proportional share catches
up first; after that, total pools maintain the configured relationship. Token
counts are converted to bytes separately for every model. For
example, configured values of `512K/256K` and `1024K/256K` derive raw weights
`2` and `4`, normalized to a 1:2 **total-pool** relationship; that relationship
is not encoded by role. Different KV costs mean a token-weighted ratio is
intentionally not a raw-GiB ratio. Once a slot reaches its configured useful
ceiling, additional memory is redistributed among uncapped slots. If every
slot reaches its ceiling, the remainder stays free.

### `[policy]` — host-wide tunables

- `memavailable_floor_gib` — the **hard refusal line**: the planner refuses to
  admit a model whose load would push system MemAvailable below this.
- `reclaim_page_cache_before_probe` — optional boolean host policy for unified-
  memory systems such as GB10. When `true`, serialized admission runs `sync`
  and asks Linux to discard clean file-backed page cache before sampling CUDA
  free memory. This reconciles `torch.cuda.mem_get_info()` (immediately free
  pages) with `MemAvailable` (free plus safely reclaimable pages). It is off by
  default because discrete-GPU hosts do not need host page-cache reclamation.

**Layering** (authoritative → fallback): `DGX_MEMAVAILABLE_FLOOR_GIB` (env override)
> installed `memory_plan.toml [policy]` > built-in default `6.0`. Enforced on the
live path by `admission.sh::resolve_floor`, which parses the installed plan's
`[policy]`, validates the resolved value is a finite positive number, and never
silently substitutes the default. (An operator's configured 8.0 is honored.)

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
**`A_preload` is the MEASURED free GPU memory** (`observed.gpu_free_now_gib`, sampled
via `torch.cuda.mem_get_info` immediately before launch), NOT `device.total_gib`. In
live mode the measurement already includes resident allocations, so residents are NOT
subtracted again (they're for identity/revision/guard checks only). A GPU-probe
failure REFUSES in both auto and required modes once a pair exists — the resolver
must never derive a fraction from a synthetic/invented A_preload.

On a unified-memory host with `reclaim_page_cache_before_probe = true`, the live
probe occurs after clean page cache is reclaimed under the same serialized lock.
The probe remains real; the policy only makes reclaimable physical pages visible
to CUDA before the measurement.

A model that fails either gate is refused **before the GPU is touched**. A failed
slot does not reduce the running counters for subsequent slots (one bad record
cannot cascade a false fit). The dispatcher emits, per admitted model:
`mem_fraction_static` (derived) + `max_total_tokens` (the target cap).

## Admission wrapper — `src/inferencectl/admission.sh`

Enrollment (`DGX_MEMORY_PREFLIGHT`):
- **`auto`** (default): run the preflight only if a matched planner pair exists
  in `CONFIG_ROOT`; else legacy launch (current v0.1 behavior, unchanged).
- **`required`**: always run; **fail-closed** — missing/mismatched pair, GPU-probe
  failure, or an unmanaged GPU container → REFUSE (never silently downgrade to
  legacy; `/proc/meminfo` alone cannot derive the fraction SGLang will use).
- **`off`**: explicit manual bypass; loud warning, no resolver.

**Matched-pair atomicity.** `memory_ledger.toml` + `memory_plan.toml` are a
**pair**: both present (use both), both absent (legacy in `auto` / refuse in
`required`), or exactly-one present (**REFUSE in both modes** — a lone file
signals a half-edited deployment; never silently pair it with a repo copy of the
other, which could be from a different schema generation).

**Serialized admission (the race fix).** `flock` on `/run/dgx-inference-admission.lock`
spans discover → sample → resolve/reuse the joint allocation → launch → **verify realized allocation via
`/get_server_info`** (carrying the API key) → release. The adapter child is
launched with the lock fd closed (so a long-lived adapter cannot hold the lock
and deadlock co-residents). Two concurrent dispatchers cannot both pass while the
first candidate is between preflight and allocation commitment. After verified
admission, the wrapper execs into supervising the adapter (Type=simple requires
the tracked PID to stay alive).

On a cold start, the wrapper hashes the ledger, active topology, and host policy,
resolves all configured slots together, and atomically persists the allocation
under `/run/dgx-inference-memory-plan.json`. Subsequent serialized role launches
must match that revision and reuse their role+model allocation. If managed
residents exist but the state is missing or stale, admission refuses and
requires a coordinated cold reload rather than silently replanning around a
partial topology.

**Refusal exit code 75** (EX_TEMPFAIL): a deliberate admission refusal. The unit's
`StartLimitBurst=3`/`StartLimitIntervalSec=300` bounds any restart churn (no
infinite storm); an operator who wants a 75 to be non-retryable can adjust
`Restart=` policy (a future operator/systemd step). **Per-role fail-safe**: a
refusal for one role never takes down a healthy co-resident.

**Dispatch is the only accepted override source.** `admission.sh` clears any
inherited `DGX_MEM_*` env before exporting the resolver-derived values, so a stale
shell export or old systemd environment cannot bypass the budget contract.

## Resident discovery — durable labels (no name inference)

Co-residents are discovered by `io.inferencectl.managed=true` labels (set by the
adapter on every managed launch), NOT by container name. Each managed container
carries:
- `io.inferencectl.managed=true`
- `io.inferencectl.role=<ROLE>`
- `io.inferencectl.memory_profile=<MODEL_ID>` (the stable planner identity)
- `io.inferencectl.ledger_revision=<sha256[:16]>` (ties the resident to the exact
  budget-ledger generation; empty for legacy/unmanaged launches)

In `required` mode, an **unmanaged GPU-holding container** (not labeled) causes a
REFUSE — the plan cannot reason about unaccounted memory, and silently ignoring
it is fail-open.
