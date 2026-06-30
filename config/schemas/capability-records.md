# Capability records (schema notes)

The resolver (`tools/resolve_service_plan.py`) is a **standalone consistency
tool**, not part of the live launch path (`dispatch.sh` does not call it).
Production safety comes from (1) capability validation in the repo test suite and
(2) the runtime catalog (`runtime/sglang/available.toml`) only ever listing
capability-compatible candidates. See `docs/architecture.md`.

The resolver considers only records whose `promotion_state` is `approved`.

## Model capability record

Lives beside its profile/bundle (e.g. `profiles/qwen36-27b-fp8/capability.toml`,
`bundles/experimental/.../capability.toml`). Required fields:

- `kind = "model-capability"`
- `capability_id`
- `promotion_state` (`approved` to be considered)
- `model_id`
- `roles` — list of roles this record offers for
- `capabilities` — list of capability identifiers it **provides**
- `compatible_runtime_ids`
- `launch_profile`

A role **requires** a set of capabilities (`config/examples/roles.toml`). A model
resolves for a role iff it is offered for that role **and** its capabilities are a
superset of the role's requirements (`required_capabilities.issubset(...)`).

### Capability vocabulary

The vocabulary is **"things a role can require."** `structured_output` means
**grammar-constrained** structured output (e.g. `response_format: json_object`).
There is deliberately **no** `structured_output_prompt_only` identifier: a model
either provides grammar-constrained `structured_output` or it does not. DFlash can
produce prompt-only JSON but not grammar-constrained output, so its capability
record simply **omits** `structured_output` — and is therefore rejected for the
`agentic` role, which requires it.

## Runtime capability record

Lives beside its runtime (e.g. `runtime/sglang/capability.toml`). Required fields:

- `kind = "runtime-capability"`
- `runtime_id`
- `promotion_state`
- `runtime`, `runtime_version`
- `derived_image`
- `container_cache_root`
- `approved_profiles`
- `[launch_requirements]` — device, context_length, parsers, speculation

The pinned image **ID** lives in exactly one place — `runtime-manifest.toml`
(`image_id`) — so a fresh build (`scripts/build-runtime.sh --update`) cannot leave
a stale pin in the capability record. Do **not** add `derived_image_id` here; the
resolver does not read it, and duplicating the pin creates two sources of truth.

## The capability records in this repo

| Record | Path | `structured_output`? |
|---|---|---|
| Baseline | `profiles/qwen36-27b-fp8/capability.toml` | **claimed** → resolves for `agentic` |
| DFlash bundle | `bundles/experimental/.../capability.toml` | **absent** → rejected for `agentic` |
| sglang runtime | `runtime/sglang/capability.toml` | (runtime record) |
