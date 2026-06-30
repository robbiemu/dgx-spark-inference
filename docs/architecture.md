# Architecture — the binding system

This is the governing architecture for `dgx-spark-inference`. It describes how a
request gets served, what the parts are, and — honestly — where enforcement
happens and where it deliberately does not.

## The one launch path

```
                       ┌──────────────────────────────────────────┐
   client ─────HTTP────▶│  dgx-spark-inference.service (systemd)    │
   (e.g. an agent)      │  └─ dispatch.sh   (runtime-agnostic)      │
                       └──────────────────┬───────────────────────┘
                                          │ reads operator state
                  ┌───────────────────────┼──────────────────────────┐
                  ▼                       ▼                          ▼
        CONFIG_ROOT/             CONFIG_ROOT/                runtime/sglang/
        active-models.toml       runtimes.toml               ├ runtime-manifest.toml
        (resident now)           (runtime_id -> root)        ├ available.toml   (catalog)
                                                               └ adapters/sglang.sh (launch)

   dispatch.sh ──execs──▶ adapters/sglang.sh
                               │ merges manifest defaults + spec overrides
                               │ applies safety rails (image-id pin, api-key shape)
                               ▼
                       profiles/<id>/sglang.toml      (single model)
                       bundles/<id>/sglang.toml       (coordinated bundle)
                               ▼
                       docker run … sglang serve --config <rendered yaml>
```

There is exactly one launch path: `systemd → dispatch.sh → sglang adapter`. There
is no alternate launcher. (The lessons from the earlier v1 design — why a
dispatcher exists, why the in-container `--host 127.0.0.1` breaks Docker
`--publish`, why flashinfer is required on GB10 — are in
`docs/known-limitations.md`, as prose, never as runnable history files.)

## The parts

- **`src/inferencectl/dispatch.sh`** — runtime-agnostic role dispatcher. Knows
  nothing about sglang specifically. Reads operator state, validates the active
  candidate is offered, and execs the runtime's adapter.
- **`runtime/sglang/`** — the sglang mechanism: pinned image (Dockerfile),
  manifest, the candidate catalog (`available.toml`), and the launch adapter.
  Does not know about specific model pairings beyond what the catalog lists.
- **`profiles/<id>/`** — a single model: provenance + per-runtime launch config
  + a capability record. Profiles **describe**; they never contain weights.
- **`bundles/<id>/`** — a coordinated bundle (e.g. target+drafter): owns only the
  cross-model coordination + a capability record.
- **`tools/resolve_service_plan.py`** — a standalone consistency/planning tool
  (see "Capability enforcement" below). **Not on the live launch path.**

## The binding flow (how a request gets served)

1. systemd starts the unit → `dispatch.sh <role>`.
2. `dispatch.sh` reads `active-models.toml[<role>]` → `model_id` + `runtime_id`.
3. Resolves `runtime_id` → project root via `runtimes.toml`.
4. Loads that runtime's `available.toml`; validates `model_id` is offered for the
   role; resolves its `kind` + `spec` path.
5. Checks the candidate `kind` matches the spec's kind (refuses on mismatch).
6. Execs the runtime's adapter (`adapters/sglang.sh`) with the resolved candidate.
7. The adapter merges manifest defaults + spec overrides, applies safety rails
   (image-id pin, api-key shape check), renders the runtime YAML (mode 0600), and
   launches the container.

The served name (`qwen3.6-27b-agentic`) is a **role-slot property** — stable
across swaps, so a client keys on the name and never notices when the underlying
model changes.

## Configuration discovery (single mechanism)

Machine-specific values are not committed anywhere. systemd sets `CONFIG_ROOT`
(unit `Environment=`); `inference-cli` accepts `--config-root`; the generic
fallback is `/etc/dgx-spark-inference`. Scripts source
`$CONFIG_ROOT/inference.env` (written by the installer) for the seven resolved
values (`INSTALL_ROOT`, `PROJECT_ROOT`, `MODEL_CACHE_ROOT`, `ROLE`, `PORT`,
`CONTAINER_NAME`, …). `PROJECT_ROOT` defaults to `INSTALL_ROOT` — the live service
must **not** depend on a mutable Git checkout (someone could `git pull`
mid-service).

## Capability enforcement — the resolver is a consistency tool, NOT a live gate

**Be explicit and honest:** `dispatch.sh` does **not** call
`tools/resolve_service_plan.py`. There is no capability check at launch time.

Production safety comes from two things, both real:

1. **Capability validation in the repository test suite.** The resolver is a
   standalone tool (`tools/resolve_service_plan.py`, stdlib-only). It takes a role
   request + capability records and resolves whether an approved
   model×runtime pair satisfies the role's required capabilities
   (`required_capabilities.issubset(model_capabilities)`). The test suite
   exercises it (tests 1–3): the baseline resolves for `agentic`; DFlash does
   not.
2. **Only compatible candidates are enumerated in `available.toml`.** The live
   dispatcher simply accepts what the catalog offers — and the catalog only ever
   lists capability-compatible candidates (enforced by test 3, which guards
   against a future edit reintroducing an incompatible candidate).

The capability **vocabulary** is "things a role can require." `structured_output`
means **grammar-constrained** structured output (e.g.
`response_format: json_object`). There is deliberately **no**
`structured_output_prompt_only` identifier: a model either provides
grammar-constrained `structured_output` or it does not.

- The **baseline** claims `structured_output` → resolves for `agentic`.
- **DFlash** can produce prompt-only JSON but not grammar-constrained output, so
  its capability record **omits** `structured_output` → rejected for `agentic`.
  DFlash is intentionally absent from `available.toml[agentic]`.

This is what makes DFlash "included but correctly gated": its incompatibility is
*provable* (the resolver rejects it), and the production catalog simply doesn't
offer it (so the operator CLI cannot activate it). There is **no
`--allow-incompatible` bypass**, ever. To run DFlash you use the explicitly
labeled experimental launcher (`experiments/dflash/run-experimental.sh`), which is
deliberately separate from the production `inferencectl use` path.

## v1 → v2 (what changed, in prose)

The current design supersedes an earlier one that conflated three concerns in a
single version-controlled file: which models a runtime *can* serve, the
operator's chosen *default*, and the *currently-active* model. That made a swap
edit a version-controlled file and gave the operator no list of candidates.

The current design separates them:

- `runtime/sglang/available.toml` (version-controlled) — what this runtime *can*
  serve per role: the candidate list, the default/home, and the role-slot served
  name. Stable; changes only when a model is approved or retired.
- operator `active-models.toml` (`CONFIG_ROOT`, host-local) — what is resident
  *right now* per role. The only file a swap edits.

The older artifacts (`run-inference-agentic.sh`, `pairings.toml`,
`role-bindings.toml`, migration/cutover scripts, alternate service units,
`DEPLOYED_STATE.md`) are **not published**. Their lessons graduated into this
prose and into `docs/known-limitations.md` — never as runnable history files.

## Adding a new model or runtime (the promotion path)

Models earn deployment; they are not auto-promoted.

1. Produce a capability record that honestly states what the model provides.
2. Prove it satisfies the target role with the resolver (`tools/`).
3. Offer it: add the candidate to `runtime/sglang/available.toml[<role>].models`.
   (Test 3 will fail at build time if you add an incompatible one.)
4. Activate: `inferencectl use <role> <candidate_id>` (operator).

A new runtime (e.g. llama.cpp) = a new runtime dir + one row in `runtimes.toml`;
the dispatcher is untouched.
