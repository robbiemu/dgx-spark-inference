# Roadmap

Explicitly out of scope for v0.1. These are recorded so the work is not lost; none
of them block the `v0.1.0-alpha` release.

## Automatic idle eviction (residency daemon)

A slot should be able to free itself when disused, so a second role (or a second
candidate) can use the memory. v0.1 ships `idle_behavior = "stay_up"` only; the
daemon that implements `"unload"` and adaptive idle timers is a platform roadmap
item. (DFlash full-context fit validation does *not* depend on this — it is a
manual stop/load/measure/restore, which is why the residency daemon is not a
prerequisite for that test.)

## CUDA-graph batch-size optimization

Cap CUDA graphs to `bs=1` (matches `max_running_requests=1`). Measured impact:
reclaims ~10 GiB, which would allow a much lower `mem_fraction_static` for the same
262144 context. This is **latency-sensitive** (CUDA graphs affect decode latency)
and belongs to a careful follow-up, not v0.1. The non-linear fraction→pool yield
noted in [`known-limitations.md`](known-limitations.md) is the direct consequence
of leaving this unoptimized.

## Broader roles

The architecture supports additional roles additively — a new role is a
`roles.toml` entry + capability records + catalog rows. v0.1 ships `agentic` only.
The cost depends on the role's modality:

- **Text-based roles** (embeddings, completion, memory) are genuinely additive:
  config + records + catalog, no code changes to the dispatcher or adapter.
- **Multimodal/audio roles** carry real caveats. The runtime image would need the
  relevant packages (e.g. `torchcodec` for audio — not in the base sglang image,
  so a `Dockerfile` change). The reasoning/tool parsers are currently read from
  the runtime manifest and are not per-profile overridable, so a model needing
  different parsers would require an adapter change. And sglang's current
  audio backends (`mimo_audio`, `mimo_v2_asr`) import FA3, which asserts SM≤90 —
  on GB10 (SM100) those backends will not load (see
  [`known-limitations.md`](known-limitations.md)). A non-FA3 multimodal backend
  would be fine; the present sglang audio stack is hardware-blocked here.

## DFlash full-context (262144) fit validation and promotion

DFlash's short-context (32768) quality and throughput gates passed, including a
measured 1.61× throughput improvement, but full-context 262144 memory fit has not
been validated because that disruptive hardware test has not been scheduled.
Promoting DFlash to a production candidate also depends on whether the `agentic`
role's `structured_output` requirement can be met — DFlash lacks
grammar-constrained structured output, so as specified today it cannot serve
`agentic`. Both the fit test and any role-spec change are separate future work.
