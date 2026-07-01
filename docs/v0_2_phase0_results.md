# v0.2 Phase 0 — Results

> **Status:** Phase 0 **COMPLETE** — 0a (co-residency, both BF16 and FP8), 0b
> (shape decision), 0c (capability contract), and the artifact-completeness
> audit are done, and the Phase 1 design (dynamic derived `mem_fraction_static`
> + `max_total_tokens` cap + two admission gates) is committed. Phase 1 begins
> with the order-independence verification probe. **Private** — lives on the
> host only; not published with the repo. Companion to `docs/V0_2_HANDOFF.md`.
>
> **Date:** 2026-06-30. **Operator/agent:** probe executed as user `robbie`; no
> sudo used. Supporting evidence: raw probe logs in `~/ornith-probe/probe*.log`
> and `capability_results.json`.
>
> **Recommended candidate:** **FP8 (`barryke/Ornith-1.0-9B-FP8-DYNAMIC`, hydrated)**
> — lighter footprint (12.98 GB vs 17.17 GB), larger KV pool (376,048 vs 268,319
> tokens), faster cold load, and clears every capability gate. BF16 remains a
> valid fallback. See "FP8 vs BF16" below.

## Outcome at a glance

| Gate | Result | Key evidence |
|---|---|---|
| 0a — co-residency at primary's 0.60 (BF16) | ✅ **PASS** | both `/health=200`; primary unchanged throughout |
| 0a — co-residency at primary's 0.60 (FP8) | ✅ **PASS** | both `/health=200`; lighter footprint, larger pool |
| 0a — Ornith footprint | measured | BF16 17.17 GB / 268,319 tok pool; **FP8 12.98 GB / 376,048 tok pool** |
| 0b — v0.2 shape decision | **multi-role (two simultaneous slots)** | co-residency succeeded at 0.60 → not B2 |
| 0c — capability contract | **all gates pass** | chat+reasoning, tool_calling, multiturn, structured_output (see below) |
| Artifact-completeness audit | BF16 complete; FP8 **incomplete as-shipped → hydrated** | byte-level provenance + manifest |

---

## 0a — Co-residency memory probe (the decision point)

**Setup.** The 27B primary (`qwen3.6-27b-agentic`) remained resident and serving
throughout at its required `mem_fraction_static=0.60`, 262144 context, 1/1 pool
— i.e. the *production baseline was the test fixture*, not a stand-in. Ornith-9B
(BF16) was launched alongside it on a throwaway container/port/key
(`ornith-probe` / `:30100`), isolated from the production slot via the same
discipline as the DFlash experiment (separate container name + port; never sourced
production `inference.env`; never edited `available.toml` / `active-models.toml`
/ the systemd unit; probe container `--rm`; throwaway probe-only API key).

**Healthy-before:** primary `active`, `/health=200`, scheduler 71,467 MiB,
`MemAvailable ≈ 45 GB`.

### The decisive test — both resident

Ornith loaded alongside the primary. After `mem_fraction_static` tuning (see
"fraction discovery" below), **both endpoints returned `/health=200` concurrently**:

```
primary :30000 /health = 200     ← unchanged, healthy throughout
ornith  :30100 /health = 200     ← loaded alongside, ready
```

The server auto-ran a chat completion during warmup
(`POST /v1/chat/completions 200 OK`); an explicit completion confirmed serving
(the 40 sampled tokens went into `reasoning_content`, as expected for a
`qwen3` reasoning-parser model at `max_tokens=40`).

**Steady-state co-resident memory:** `MemAvailable ≈ 10 GB` (tight but positive).

### Ornith BF16 footprint (from the passing load log, `probe4.log`)

```
Load weight end. elapsed=132.95 s, type=Qwen3_5ForConditionalGeneration, avail mem=24.12 GB, mem usage=17.17 GB.
Memory pool end. avail mem=12.13 GB
Capture target decode CUDA graph end. elapsed=12.47 s, mem usage=1.26 GB, avail mem=10.35 GB.
max_total_num_tokens=268319, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=1, context_len=32768, available_gpu_mem=10.35 GB
```

- Weights + load overhead: **17.17 GB**
- Realized KV pool: **268,319 tokens** (8× the 32,768 requested context — generous)
- Decode CUDA graph: **1.26 GB**

### Fraction discovery — and a correction worth recording

The pass was **not** first-try. Two earlier fractions failed at hybrid-state cache
allocation, and the *direction* of the fix was initially backwards:

| `mem_fraction_static` | result | evidence |
|---|---|---|
| 0.30 | FAIL | `total_rest_memory=-3.05 GB` (`probe2.log`) |
| 0.12 | FAIL | `total_rest_memory=-12.59 GB` (`probe3.log`) — *worse* |
| **0.80** | **PASS** | pool + CUDA graph allocated; `/health=200` (`probe4.log`) |

The first instinct (lower the fraction to "fit") was wrong. Reading the actual
allocator (`model_runner_kv_cache_mixin.py::_profile_available_bytes`):

```python
available_gpu_memory = get_available_gpu_memory(...)
rest_memory = available_gpu_memory - pre_model_load_memory * (1 - mem_fraction_static)
```

`mem_fraction_static` is **not** "fraction of total device." It scales the runtime
slack: **higher fraction → less slack reserved → more pool.** The error message
itself says "increase --mem-fraction-static." Ornith needs a *high* fraction
*because* it is the second resident model competing for free space — it must
claim most of what the primary leaves.

**`mem_fraction_static = 0.80` is a validated launch floor for THIS residency pair
and launch order ONLY, not a universal Ornith setting.** It is tied to:
the primary already resident at 0.60; the exact sglang build (0.5.14 / image
`d49152ee…`); Ornith BF16; 32K requested context; the observed allocator state;
and current system memory. Record it in the residency set as:

```toml
[sets.qwen27b_fp8_plus_ornith9b.members.agentic-helper]
model_id = "ornith-1.0-9b-bf16"
context_length = 32768
max_running_requests = 1
max_queued_requests = 1
mem_fraction_static = 0.80
validated_for = "qwen36-27b-fp8 primary resident at 0.60; Ornith BF16; 32K ctx; sglang 0.5.14 (image d49152ee); primary-loaded-first"
```

**⚠️ Open: restart-order testing (Phase 0c).** The 0.80 floor was measured with
the primary loaded first. Allocators turn a single successful boot into folklore.
Phase 0c must vary the launch order: helper-first-then-primary;
primary-first-then-helper (done); and helper launch while a primary workload is
active. If a different order needs a different fraction, record all of them.

### Healthy-after (baseline restored)

Probe container torn down (`--rm`); only `inference-agentic` remains. Primary
`active`, `/health=200`, scheduler 71,467 MiB, `MemAvailable ≈ 49.6 GB`.
The production slot was never touched.

---

## 0b — v0.2 shape decision

**Co-residency SUCCEEDS at the primary's required 0.60** → per the handoff's
decision tree, **v0.2 = the multi-role plan (two simultaneous slots)**. The
swappable-slot + residency-daemon path (B2) is **not** required for v0.2. The
residency daemon stub remains deferred (it was only load-bearing under B2).

---

## 0a (FP8) — co-residency gate on the recommended candidate

The BF16 pass (above) established feasibility. The **FP8 candidate** — the model
we'd actually prefer to deploy — was then probed identically: primary resident at
0.60, FP8 Ornith alongside on the throwaway slot. It required a **hydrated bundle**
first (see "Complete-artifact requirement" below), then loaded cleanly.

**Decisive test — both `/health=200` simultaneously:**
```
primary :30000 /health = 200     ← unchanged throughout
ornith  :30100 /health = 200     ← FP8, co-resident, ready
```

### Ornith FP8 footprint (from `probe_fp8_1.log`)

```
Load weight end. elapsed=76.95 s, type=Qwen3_5ForConditionalGeneration, quant=compressed-tensors, avail mem=30.80 GB, mem usage=12.98 GB.
Memory pool end. avail mem=8.36 GB
Capture target decode CUDA graph end. elapsed=11.07 s, mem usage=1.05 GB, avail mem=6.37 GB.
max_total_num_tokens=376048, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=1, context_len=32768, available_gpu_mem=6.37 GB
The server is fired up and ready to roll!
```

- Weights + load overhead: **12.98 GB** (compressed-tensors FP8, dynamic)
- Realized KV pool: **376,048 tokens** (11× the 32K context)
- Decode CUDA graph: **1.05 GB**
- `mem_fraction_static=0.80` (same floor as BF16; see caveat — not independently re-tuned for FP8)

### FP8 vs BF16 (both co-resident at the primary's 0.60)

| Metric | BF16 | FP8 | Winner |
|---|---|---|---|
| Weights + overhead | 17.17 GB | **12.98 GB** | FP8 (−4.2 GB) |
| Realized KV pool | 268,319 tok | **376,048 tok** | FP8 (+40%) |
| Cold load time | 132.95 s | **76.95 s** | FP8 (−42%) |
| Decode CUDA graph | 1.26 GB | 1.05 GB | FP8 |
| Steady-state free (MemAvailable) | ~10 GB | ~6 GB | BF16 (more slack) |
| Capability gates (0c) | not run | **all pass** | — |

**Recommendation: FP8.** The weight/pool/load wins outweigh the lower steady-state
slack (6 GB is still positive and stable under the 1/1 single-session helper load).
FP8 is the candidate to promote; BF16 stays as a documented fallback. Note: the
FP8 steady-state slack is tighter — if the helper ever needs a larger context or
>1 running request, re-measure; BF16's higher slack gives more headroom there.

---

## 0c — Capability contract (evidence-backed; run against FP8)

All gates run against the co-resident FP8 probe. Results in `capability_results.json`.

| Gate | Result | Evidence |
|---|---|---|
| C1 — chat completion + reasoning separation | **PASS** | answered "12"; `reasoning_content` present & separated; none leaked into content |
| C2 — tool calling (`qwen3_coder`) | **PASS** | emitted `calculator({"expression": "47 * 9"})` |
| C3 — multi-turn continuity | **PASS** | tool result 423 → "47 times 9 is 423." |
| C4 — structured_output (grammar-constrained) | **PASS (with caveat)** | `json_schema` enforced: asking for a list while schema demands `{name,age}` was forced to `{"name":"List 5 Numbers","age":25}` |

### Resulting `agentic-helper` capability contract

```toml
capabilities = ["chat_completion", "reasoning", "tool_calling", "multiturn_continuity", "structured_output"]
```

**`structured_output` IS included** — Phase 0c measured grammar-constrained
enforcement, so the helper *can* require it. This is the stronger outcome: a
client needing hard structured output may use the helper, not only the primary.

### ⚠️ Operational caveat — reasoning budget (same lesson as v0.1 Gate 8)

C4 initially appeared to **fail** with empty content. Diagnosis: Ornith reasons
heavily before emitting, and at `max_tokens=200` the reasoning trace consumed the
entire budget (303 reasoning tokens needed), so content came back empty. With
`max_tokens≥600` it returned valid schema-constrained JSON. This is the **same
reasoning-budget lesson** as v0.1's Gate 8 (200s+ thinking traces), not a
capability gap. **Operational rule for the helper contract: constrained /
structured requests must set `max_tokens` high enough to clear the reasoning
trace (≥600 observed), or clients will see empty content with `finish=length`.**
Document this in the helper's served contract.

### Clean SGLang launch + parser behavior (recorded)

- Pinned revision loaded cleanly; `reasoning_parser=qwen3`, `tool_call_parser=qwen3_coder`.
- Log note (informational, not an error): `Acceleration for non-quantized schemes
  is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod`
  — the loader correctly recognizing the compressed-tensors scheme.
- No silent fallback, no container conflict, no impact on the primary.

---

## Complete-artifact requirement (corrects the initial "VLM processor" note)

**Diagnosis (corrected).** Ornith's declared architecture is the **conditional
multimodal wrapper** (`Qwen3_5ForConditionalGeneration`), not a text-only CausalLM
class. It carries a `vision_config`. SGLang initializes the processor stack
(`AutoProcessor`) even for text-only serving, so a valid local checkpoint must
include the **full Hugging Face processor/tokenizer artifact set**, not just the
weight shards.

**The official BF16 distribution is a complete checkpoint.** `deepreinforce-ai/Ornith-1.0-9B`
@ `83dc1f5e…` ships all 16 files, including `preprocessor_config.json`,
`processor_config.json`, `video_preprocessor_config.json`, tokenizer files, and
`chat_template.jinja`. The initial probe failure (`OSError: Can't load image
processor ... preprocessor_config.json`, `probe.log`) was a **partial-download
artifact** — the acquisition used a shard-only `allow_patterns` filter and omitted
the processor configs. It was not a model defect. Re-fetching the full artifact
set resolved it.

### FP8 candidate is incomplete as-shipped — needs a hydrated bundle

`barryke/Ornith-1.0-9B-FP8-DYNAMIC` @ `01272cd6…` is **missing two files the
loader requires**: `preprocessor_config.json` and `video_preprocessor_config.json`.
This is not incidental: the FP8 repo *does* ship `processor_config.json`
(byte-identical to BF16), and that file **references both** an embedded
`image_processor` (`Qwen2VLImageProcessor`) and a `video_processor`
(`Qwen3VLVideoProcessor`) — so it declares a need the missing standalone configs
normally also satisfy. Given the BF16 crash was specifically on the missing
`preprocessor_config.json`, the FP8 repo will very likely fail identically as-shipped.

**It must not be promoted by casually sourcing the missing files at runtime** —
that produces a Frankenstein checkpoint with unclear provenance. Instead, for the
FP8 candidate, build a **hydrated local bundle** in the experiment directory:
record the exact BF16 base commit, copy only the verified-identical
`preprocessor_config.json` + `video_preprocessor_config.json`, hash every file in
an evidence manifest, and test it as a distinct artifact with its own provenance
record.

### Provenance of shared non-weight files (byte-level, BF16 vs FP8)

| File | BF16 vs FP8 | Classification |
|---|---|---|
| `tokenizer.json` | IDENTICAL | must-match |
| `chat_template.jinja` | IDENTICAL | must-match |
| `processor_config.json` | IDENTICAL | must-match |
| `tokenizer_config.json` | **DIFFERS** | may-legitimately-differ |
| `generation_config.json` | **DIFFERS** | may-legitimately-differ |

This is why the artifact gate must distinguish **must-match** (tokenizer,
chat_template, preprocessor configs) from **may-legitimately-differ**
(tokenizer_config, generation_config, processor_config contents, recipe.yaml).
A blanket "all non-weight files hash-identical" rule would false-fail on
legitimate quantization-derivative deltas.

### Artifact-completeness gate (new; for the resolver/profile validation)

Required for any `Qwen3_5ForConditionalGeneration` (and analogous multimodal)
candidate, scoped to **what `config.json`'s processor/auto_map references**
(architecture-aware, not a fixed universal list — a pure CausalLM checkpoint
has no vision tower and legitimately lacks `preprocessor_config.json`):

```text
required (qwen3_5 conditional/multimodal):
  config.json
  generation_config.json
  model.safetensors.index.json   (or single model.safetensors + index if sharded)
  all referenced weight shards
  tokenizer.json
  tokenizer_config.json
  chat_template.jinja
  preprocessor_config.json
  processor_config.json
  video_preprocessor_config.json
```

For a **derived** checkpoint (e.g. a quantization), require one of:
1. All **must-match** non-weight files present and hash-identical to the cited
   base revision; or
2. The conversion author documents every intentional difference to a
   must-match file and why it is compatible.

`may-legitimately-differ` files need not match the base but must be internally
valid (e.g. parse, reference existing processor classes).

---

## Phase 0 closed

**All research gates are satisfied.** 0a (co-residency, BF16 + FP8), 0b
(multi-role shape), and 0c (capability contract) are complete. Phase 1
implementation may proceed — gated on the verification probe below as its first
task, not on any further Phase 0 research.

## Corrections to the record (measurement-domain errors)

Two figures cited earlier in this doc and in `V0_2_HANDOFF.md`'s grounding
are **wrong** and must not be carried into Phase 1's budget ledger:

1. **"Primary consumes ~104 GiB under load" is not the steady-state footprint.**
   The live `nvidia-smi` process view shows the sglang scheduler at **~69.8 GiB**
   steady-state (primary resident). The ~104 GiB figure came from conflating
   `MemAvailable` with `MemFree` on GB10's **unified memory**: post-reboot
   diagnostics showed `MemFree ≈ 11.8 GiB` vs `MemAvailable ≈ 42.8 GiB` — a
   ~31 GiB gap of reclaimable cache/buffers that is *not* GPU pressure. The
   primary's static budget reconciles cleanly at **~73 GiB** (`f=0.60 × 121.7`),
   matching the observed footprint. The 104 GiB is an outlier from reading
   `MemAvailable`-derived occupancy, not a real footprint.

2. **"Weights sit outside `mem_fraction_static`" is wrong.** The SGLang allocator
   charges loaded weights against the static budget; the source algebra is
   `W + K = f × A_preload` (weights + KV pool = fraction × pre-load free). So
   `f × A_preload` **is** the full static allocation (weights + target KV pool).
   This means the fraction's meaning is unambiguous: it bounds weights+pool, not
   just the pool. (See the budget ledger + two-gate design below.)

## Design commitment for Phase 1 — dynamic (derived) `mem_fraction_static`

v0.2 scope is **N ≥ 2 roles** (not just two). Residency sets don't scale to N
(N-choose-k tested layouts; a fifth role would mean ~26 hand-maintained records).
The committed design is **per-model measured budgets + runtime-derived fraction +
absolute pool cap + two admission gates.** This is the unique design the
constraints force (sets rejected for combinatorics; static-fraction+order-constraint
rejected for not solving the N-role case).

**Per-model budget ledger** (measured, component-accurate, combination-independent):
```toml
[profiles.<model>.budget]
weights_gib            # measured cold-load persistent delta (e.g. FP8 12.98)
target_kv_tokens       # measured realized pool (e.g. 376048) — the target, not a guess
kv_bytes_per_token     # measured; NOT a timeless constant — record per profile,
                       #   re-measure if dtype/attention/sglang-version changes
static_pad_gib = 0.5   # page-alignment / allocator cushion
cuda_graph_peak_gib    # measured (graph memory goes in the ADMISSION CHECK, not f)
request_workspace_gib  # measured peak
memavailable_floor_gib = 8.0   # the MemAvailable floor (GB10 unified-memory safety)
```

**Two separate computations** (the footgun this avoids: graph/workspace memory in
the fraction numerator inflates the static KV reservation and steals graph slack):
```
static_required = weights + (target_kv_tokens × kv_bytes_per_token) + static_pad
fraction        = static_required / A_preload          # DERIVED at launch from observed free

peak_required   = static_required + cuda_graph_peak + request_workspace
```

**Two admission gates** (fail-safe: refuse to start the second service rather than
silently shrinking a role's contract or wedging the host):
```
GPU gate:   A_preload − static_required  ≥  graph_peak + workspace + GPU headroom
Linux gate: MemAvailable_now             ≥  memavailable_floor + incremental_peak
```

**Both sglang knobs are used** (max_total_tokens is a cap, not a reservation;
it cannot grow a too-small fraction's pool, only shrink an over-large one):
- `mem_fraction_static` = derived `fraction` (sized so the target pool is feasible)
- `max_total_tokens` = `target_kv_tokens` (caps the pool to the absolute target,
  so load order cannot inflate it — this is what makes the design order-independent)

### Why this makes load order safe
Loading first sees more free memory → a naive fraction reserves more. But
`max_total_tokens` clips the realized pool to the target regardless, so a
model loading first **cannot** over-claim the way the static-fraction probe did
(that probe, with `f=0.80` solo against ~124 GiB, reserved ~99 GiB static and
starved the primary → host wedged → reboot; see incident note). The cap is the
structural guard; the derived fraction keeps the pool *feasible*; the two gates
refuse an unsafe combination before the GPU is touched.

### Resolver — IMPLEMENTED and tested (`tools/memory_planner/`)

The planner exists and is unit-tested against the measured Phase 0 numbers
(`resolve_memory_plan.py`, `budget_ledger.toml`, `test_resolver.py`,
`plan_*.toml`). All 10 tests pass; both example plans ADMIT.

**Verified derivations (no GPU — pure arithmetic against measured ledger):**
- Primary: derived `0.6002` (PRE_PREP measured `0.60`) ✅
- Helper co-resident: derived `0.8159` (probe measured `0.80`) ✅
- Helper solo: derived `0.2994` (the `0.294` = 36.48/124 derivation) ✅
- `max_total_tokens` invariant to load order (the cap) ✅

**Finding the tests exposed (now captured in the ledger): large models need a
`static_overhead_gib` field.** The primary's measured `0.60` includes ~21 GiB
of allocator overhead (CUDA-graph capture, hybrid-attn state) reserved *into*
the static budget — clean `weights + kv` derives only `0.42`. The helper needs
none (pool-dominated; its `0.80` reproduces from components alone). That
asymmetry is real and model-size-dependent; the ledger records it honestly
rather than hiding it in a fudge factor.

### `memavailable_floor_gib` is a host-wide TUNABLE (in the plan, not the model)

The MemAvailable floor is a **host-wide policy** (how much system memory this
GB10 must keep free under unified memory), not a per-model property. It lives in
the plan under `[policy]` and overrides any ledger/default value:

```toml
[policy]
memavailable_floor_gib = 6.0   # operator-set; 6.0 observed safe in steady-state
```

**Why 6.0, not a higher default:** at the production plan (primary resident,
helper loads co-resident), the derived config lands at **6.52 GiB** MemAvailable
steady-state — so a floor of 8.0 would *refuse* a configuration that empirically
runs. 6.0 admits it with ~0.5 GiB margin. This is the marginal-but-acceptable
reality of fitting a 27B + 9B on 121 GiB; the tunable lets an operator tighten
(refuse sooner, safer) or loosen (admit tighter residency, more risk) per
deployment without touching the resolver.

**Layering:** plan `[policy]` (authoritative) > ledger per-model value (fallback)
> built-in default `8.0`. The floor is never hardcoded in the resolver logic.

## Order-independence — VERIFIED on GPU (2026-06-30)

The design's central claim — that a derived low fraction + `max_total_tokens`
cap makes the **helper-first load order safe** (the order that previously wedged
the host with a static `0.80` fraction) — was measured on the GB10.

**Procedure:** primary stopped → helper launched **solo** at the resolver's
derived `mem_fraction_static=0.2994` with `max_total_tokens=376048` (the cap) →
primary started on top → both verified.

**Result: PASS.** Both `/health=200`; the cap held and the device was not vacuumed.

| Step | Observation |
|---|---|
| Helper solo, weights loaded | `avail mem=101.99 GB` after weights (13.27 GB used) |
| Helper solo, pool allocated | `Memory pool end. avail mem=80.42 GB` |
| **Helper solo, realized pool** | **`max_total_num_tokens=366011`** (≈ the 376048 cap; not huge) ✅ |
| MemAvailable after helper load | ~78 GiB free (cap prevented over-allocation) |
| Primary loaded on top | reached `/health=200` |
| **Steady-state, both resident** | **MemAvailable ≈ 27.3 GiB** (≫ 6.0 floor) |

**Why this is the decisive proof:** with the static `0.80`, the solo helper saw
~124 GiB free and reserved ~99 GiB → starved the primary → host wedged → reboot.
With the derived `0.2994` + cap, the same solo helper reserved only ~36 GiB and
left ~78 GiB for the primary. The cap is the structural guard that makes load
order irrelevant — exactly as the planner predicted. **Order-independence is no
longer a derivation; it is a measured property of this configuration.**

**Bonus finding:** steady-state MemAvailable (~27 GiB both resident) is far higher
than the planner's predicted ~6.5 GiB. This means the ledger's
`request_workspace_gib` / `cuda_graph_peak_gib` estimates are conservative, and
the 6.0 floor has substantially more margin than modeled. Confirms the schema-doc
note that tightening those measurements may admit a higher floor later.

## Phase 1 — implementation status (the research is done)

1. **Resolver** — ✅ done. `tools/memory_planner/resolve_memory_plan.py` is built,
   tested (20/20: algebra + `--format json` contract), and its order-independence
   claim is **GPU-verified**. Derives `mem_fraction_static` + `max_total_tokens`
   from the measured budget ledger; `--format json` is the dispatch contract.
2. **Dispatcher wiring + serialized admission** — ✅ done.
   `src/inferencectl/admission.sh` is the serialized admission wrapper: holds a
   global `flock` across discover → sample → resolve → launch → **verified
   allocation** (closing the preflight↔allocation race), refuses (exit 75) on gate
   failure, clears inherited `DGX_MEM_*`, and verifies realized capacity via
   `/get_server_info` before releasing the lock. `dispatch.sh` routes through it
   when a planner pair is enrolled; legacy single-role launches unchanged. Adapter
   gained the `DGX_MEM_FRACTION_STATIC`/`DGX_MAX_TOTAL_TOKENS` env override tier +
   `max-total-tokens` emission + durable `io.inferencectl.*` labels. Test gate:
   T7–T9 (env override, serialization, matched-pair atomicity/fail-closed) — 9/9
   functional tests pass.
3. **Hermes real-workflow test** — ⏳ open. Phase 0c ran synthetic gates; run a
   real hermes session against the helper endpoint before final promotion.
4. **Budget-ledger tightening** — ⏳ open. `cuda_graph_peak_gib` and
   `request_workspace_gib` are currently conservative estimates (the order probe
   showed steady-state MemAvailable ~27 GiB vs the predicted ~6.5 — the
   workspace estimate especially over-counts). Measuring them precisely would
   tighten the gates and likely admit a higher `memavailable_floor_gib`.

### Not yet done (deferred / operator steps)
- **Production enablement**: dropping a matched `memory_ledger.toml` +
  `memory_plan.toml` into `CONFIG_ROOT` + setting `DGX_MEMORY_PREFLIGHT=required`
  on the unit (operator/systemd step). Until then the live service runs legacy
  (`auto`, no pair) — the wrapper is wired but inert.
- **`Restart=` policy** for exit 75: currently `Restart=always` + burst cap; an
  operator may make 75 non-retryable. Documented; not changed.

## State left behind

- Live service: unchanged — primary healthy, `qwen3.6-27b-agentic`, 0.60,
  262144, 1/1. Survived the helper-first incident + reboot; auto-starts on boot
  (`systemctl is-enabled` = enabled).
- Probe artifacts (private, on host): `~/ornith-probe/` — launcher
  (`run-ornith-probe.sh`, with `--solo` + `--max-tokens`), capability gates
  (`capability_gates.py` + `capability_results.json`), BF16 weights
  (`ornith-1.0-9b-bf16/`, 18.84 GB), FP8 **hydrated** bundle
  (`ornith-1.0-9b-fp8/` + `HYDRATION_MANIFEST.md`, 13.54 GB), probe logs
  (`probe.log`, `probe{2,3,4}.log`, `probe_fp8_1.log`).
- **Incident note:** the helper-first restart-order test (Variant A) with a
  static `mem_fraction_static=0.80` reserved ~99 GiB against the solo ~124 GiB
  free, starving the primary → host wedged → hard reboot. No data lost; primary
  recovered. This incident is the empirical evidence for the dynamic-fraction +
  cap design above (the static fraction's order-dependence is now a proven
  failure mode, not a theoretical one).
- Repo: clean — no production runtime/catalog/service files were modified.
  This doc is untracked (private).
