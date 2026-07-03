# v0.2 Handoff — multi-role + the cheap/fast helper role

> **Access:** Hostnames and operator accounts are deployment-specific and are not
> stored in this repository. Privileged operations require an operator to enter
> credentials interactively. Passwordless sudo is intentionally not assumed.

**Purpose.** Add a second role — a cheap/fast coding-and-tool helper (`agentic-helper`)
— alongside the strict `agentic` primary, served by `Ornith-1.0-9B` (BF16). **This
handoff starts with a research/feasibility gate (Phase 0) whose result decides the
entire v0.2 shape.** Do not implement anything in Phase 1+ until Phase 0 is complete
and its outcome recorded.

**Prerequisite:** v0.1 publication complete (see `PUBLICATION_HANDOFF.md`). This
handoff assumes the public `dgx-spark-inference` repo exists and the live service is
the v0.1 reference deployment. Companion docs on the host (private): `HANDBOOK.md`,
`PRE_PREP_RESULTS.md`, `contracts/role-model-binding-v2.md`.

**Governing principle (do not violate):** models earn admission through evidence.
Ornith is a candidate until its gates pass; nothing is auto-promoted.

---

## The load-bearing unknown (why Phase 0 exists)

**Two 27B+ class models cannot coexist on GB10** (121 GiB unified memory; the
primary alone at `mem_fraction_static=0.60` uses ~104 GiB under load). The question
Phase 0 answers: **can the 27B primary (at its required 0.60) and a 9B helper
(Ornith, ~18.82 GB weights BF16 + its own KV) be resident simultaneously?**

### Measured facts that ground the probe (do not re-derive; cite)
- **Primary Qwen3.6-27B-FP8** at `mem_fraction_static=0.60` → realized KV pool
  **346,485 tokens** (≥ 262144); per-token KV = 64.0 KiB/token. Under load the
  primary consumes ~104 GiB of 121 GiB (prep evidence `PRE_PREP_RESULTS.md`).
  The primary **needs 0.60** for full context: 0.40→148192 (fail), 0.50→245748 (fail).
- **Ornith-1.0-9B**: BF16 (no quantization), `Qwen3_5ForConditionalGeneration`,
  32 layers, hidden 4096. **Weights ≈ 18.82 GB** (4 safetensors shards: 4.94 + 4.99
  + 4.95 + 3.93 GB). HF repo `deepreinforce-ai/Ornith-1.0-9B`, public, not gated.
  Card documents SGLang support, `qwen3` reasoning parser, `qwen3_coder` tool-call
  parser, OpenAI-style tool calls; does NOT establish grammar-constrained JSON.

So the 9B weights (~18.8 GB) + its KV pool + CUDA-graph capture must fit in the
~17 GiB the primary leaves free. **This is physically marginal at best and most
likely impossible at the primary's required 0.60.** Phase 0 settles it empirically.

---

## Phase 0 — feasibility + capability probe (RESEARCH; no infra changes)

**This phase decides which v0.2 you build. Do all of it before any Phase 1 work.**

### 0a. Co-residency memory probe (the decision point)
A controlled, throwaway experiment — do NOT touch the live `agentic` service
configuration beyond a controlled stop/restore (same discipline as the DFlash
experiment; record healthy-before; restore the baseline afterward regardless of
outcome):
1. Acquire Ornith weights into a deterministic profile dir under `MODEL_CACHE_ROOT`
   (do NOT depend on HF snapshot-cache layout). Pin the revision; record it.
2. With the primary stopped, load **Ornith alone** at a modest context
   (e.g. 32768) on a throwaway port/container (`inference-ornith-probe` on
   `:30100`). Confirm it loads + serves a completion. Measure its realized
   memory footprint (weights + KV + CUDA graphs) from the `info`-level load log.
3. **The decisive test:** attempt to load the primary (27B @ 0.60, 262144) AND
   Ornith (9B, modest context) **simultaneously** (two containers, two ports).
   Record whether both reach `/health=200` or one/both OOM. **This is the gate.**

### 0b. Outcome → v0.2 shape (decided by 0a; record which)
- **If co-residency SUCCEEDS at the primary's 0.60:** v0.2 = the multi-role plan
  below (two simultaneous slots). Proceed to Phase 1.
- **If co-residency FAILS at 0.60:** two sub-options, owner decides from the 0a
  measurements:
  - **(B1) Lower the primary's fraction** only if a fraction exists that fits
    both AND still gives the primary ≥ 262144 tokens. Given prep calibration
    (0.50→245748 < 262144), this is unlikely — but **measure, don't assume**;
    a 9B KV pool is small, so the crossover might exist. If yes → still the
    multi-role plan, with the lower primary fraction recorded + the residency-set
    record (Phase 1) reflecting it.
  - **(B2) Co-residency is impossible at any primary fraction that keeps 262144:**
    v0.2 is **NOT "two simultaneous slots."** It is **one swappable slot + the
    residency daemon** (cold-load swap between agentic and helper, idle-eviction,
    ~4 min switch). That is the deferred `inference_residency_daemon` becoming
    load-bearing. The "two services, no router" architecture below does NOT apply;
    instead build the daemon (its README stub already exists) + `inference-cli use`
    swap semantics. **This is a legitimate v0.2 — but it is a different v0.2 than
    the multi-role plan. Commit to it explicitly; do not half-build both.**

### 0c. Ornith capability probe (independent of memory; do regardless)
Run Ornith through the helper-role gates (it determines the `agentic-helper`
capability contract, so the contract is evidence-backed, not asserted ahead):
- clean SGLang launch with pinned revision;
- reasoning separated into `reasoning_content` (not leaked into answer content);
- ordinary chat completion works;
- tool calls parse via `qwen3_coder`;
- multi-turn tool loop: request → tool call → tool result → useful continuation;
- the **actual hermes helper workflow** works (not a toy call);
- `response_format` behavior measured + recorded honestly (does it support
  grammar-constrained? if yes → helper can require `structured_output`; if no →
  helper contract omits it);
- no silent fallback, no container conflict.
Record outcomes. Treat the card's claims as reasons to test, not promotion evidence.

### Phase 0 deliverable
A results doc (`docs/v0_2_phase0_results.md`, private — on the host, not published)
recording: 0a measurements (Ornith-alone footprint; co-residency pass/fail at 0.60;
any crossover fraction), the 0b decision (which v0.2 shape), and the 0c gate
outcomes (the capability contract Ornith actually supports). **Phase 1+ is gated on
this doc existing.**

---

## Phase 1+ — ONLY if Phase 0 chose the multi-role (simultaneous) shape

(If Phase 0 chose B2 — swap + daemon — discard this section and instead implement
the residency daemon per its README + swap semantics. Do not build multi-role
infra for a topology that can't run.)

### Architecture: two services, no router
Do NOT add an API gateway / model router (scope creep; a third actor that must
understand role semantics). Use two independent systemd template instances:
```
dgx-spark-inference.target
 ├─ dgx-spark-inference@agentic.service        → :30000  container inference-agentic
 └─ dgx-spark-inference@agentic-helper.service → :30001  container inference-agentic-helper
```
Each role: its own systemd instance, container name, runtime YAML under `/run`,
health endpoint/port, stable served name, cold-swap lifecycle. Clients choose the
endpoint deliberately. **Both may share the existing bearer token** (trusted-LAN,
one user) — but docs must state explicitly the two endpoints are **not**
independently authorizable in v0.2.

### Served names (role-slot property, stable across swaps)
- `agentic` → `qwen3.6-27b-agentic` (unchanged from v0.1)
- `agentic-helper` → `agentic-helper` (role-shaped; NOT `ornith-1.0-9b` — keeps it swappable)

### Roles (capability contracts — written from Phase 0c evidence, not ahead)
`agentic` unchanged: `chat_completion`, `tool_calling`, `structured_output` (grammar-constrained).
`agentic-helper`: `chat_completion`, `reasoning`, `tool_calling`, `multiturn_continuity`.
**Include `structured_output` in the helper ONLY IF Phase 0c measured Ornith passes
an enforced `response_format`/grammar test.** Otherwise omit it — that preserves
`agentic`'s meaning (a client needing hard structured output stays on the primary).

### Config split (clean separation of concerns)
```
config/
  roles.toml              # version-controlled role contracts (policy)
  residency-sets.toml     # version-controlled APPROVED co-residency combinations
  examples/
    active-models.toml    # operator-owned selected candidates
    role-slots.toml       # operator-owned ports/names/enabled roles
    runtimes.toml         # operator-owned runtime roots
```
`roles.toml` = immutable policy. `role-slots.toml` = host deployment state (port,
container_name, enabled). `active-models.toml` = active selections (per role).
`residency-sets.toml` = **explicit tested co-residency records** (see below).

### Residency sets — explicit, tested, NOT fake arithmetic
A model can be individually eligible for a role yet not safely coexist with another
resident model. The capability resolver answers "can this model serve this role?"
It does NOT answer "can these two pairs inhabit one GB10 without an OOM fight?"
**Do NOT solve co-residency with an invented `mem_fraction_static <= 0.85` rule**
(SGLang allocation, CUDA graphs, weights, KV make that too hand-wavy for this repo).
Instead: an explicit **tested** co-residency record, approved only after the Phase 0
probe + a real simultaneous-load test:
```toml
[sets.qwen27b_fp8_plus_ornith9b]
promotion_state = "approved"            # only AFTER the simultaneous-load gate passes
roles = ["agentic", "agentic-helper"]
[sets.qwen27b_fp8_plus_ornith9b.members.agentic]
model_id = "qwen36-27b-fp8"
context_length = 262144
max_running_requests = 1
max_queued_requests = 1
[sets.qwen27b_fp8_plus_ornith9b.members.agentic-helper]
model_id = "ornith-1.0-9b-bf16"
context_length = 65536                  # Phase 0/measurement target, NOT a guess
max_running_requests = 1
max_queued_requests = 1
```

### Resolver extension (consistency tool — failure semantics matter)
Extend `resolve_service_plan.py` to verify: (1) every active role has an offered
candidate; (2) every candidate satisfies its role capability contract; (3) every
**enabled role combination matches an approved residency set**; (4) ports/container
names unique; (5) every selected role produces a unique unit + health endpoint.
**`dispatch.sh` may call this as a preflight, but the failure semantics must be
spelled out and fail-safe:** a validation failure on a role refuses to launch
*that role*; it must **never take down a healthy co-resident** because a different
role's record is mis-formed. Document this explicitly.

### Code changes (focused refactor, not a new control plane)
Refactor the global `inference.env` to hold only shared machine facts
(`INSTALL_ROOT`, `PROJECT_ROOT`, `MODEL_CACHE_ROOT`, `ACTIVE_MODELS`,
`RUNTIMES_INDEX`, `ROLE_SLOTS`); **remove** `ROLE`, `PORT`, `CONTAINER_NAME`
(those become per-role from `role-slots.toml`).
- `dispatch.sh <role>` reads the role slot config + validates the active residency plan.
- `adapter.sh` receives the role, reads its unique port/container name, writes `/run/<container>/sglang.yaml`.
- `inference-cli.sh` becomes role-aware: `status [role]`, `roles`, `candidates <role>`,
  `use <role> <candidate>`, `start <role>`, `unload <role>`, `reload <role>`.
- `use agentic-helper <x>` restarts **only** `dgx-spark-inference@agentic-helper.service`;
  swapping/restarting the helper must **not** interrupt the primary.

### Ornith promotion path (evidence-gated, like DFlash)
Keep Ornith outside the production catalog until it clears its gate. Create
`experiments/ornith-1.0-9b/` (run-experimental, evaluate, evidence) +
`profiles/ornith-1.0-9b-bf16/` (README, capability.toml, sglang.toml). The gate =
Phase 0c outcomes formalized. Only after it passes: add to
`available.toml[agentic-helper]`, mark capability `approved`, seed
`active-models.toml`, enable the helper unit.

### Safer rollout sequence
1. Multi-role infra with **only `agentic` enabled** — migrate the existing service
   to the template unit; prove old Qwen behavior unchanged.
2. Ornith in the isolated experimental path (throwaway port/container).
3. Create + test the approved residency set (the simultaneous-load gate).
4. Only then: add Ornith to `available.toml[agentic-helper]`, approve, seed, enable.
Installer: require an explicit `--migrate-v0.1` flag; timestamped backups; preserve
the existing primary selection; **never start the new target automatically**.

---

## Out of scope for v0.2
- A third role (embeddings/completion/memory).
- Independent per-role authorization (shared bearer in v0.2; documented).
- Automatic model routing / API gateway.
- Faster hot-swap (cold-load swaps remain; hot-swap is roadmap).
- DFlash 262144 fit / promotion (separate track).

## Notes / risks
- Sudo is password-gated — the agent cannot run the installer for real, start/stop
  services, or run the co-residency probe. Those are operator steps; the agent
  prepares scripts + docs and the operator executes the privileged bits.
- If Phase 0 → B2 (swap + daemon), the residency daemon README stub
  (`inference_residency_daemon/`) becomes the spec to implement; its open questions
  (activity source, status transport, adaptive-timer algorithm, lifecycle) must be
  resolved at that point.
- The 9B Ornith model is BF16 (no quantization); a future FP8/quantized variant
  would change the co-residency math materially — re-probe if that appears.
