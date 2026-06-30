# Handbook — what this provides

A single, at-a-glance reference. For the governing architecture, see
[`architecture.md`](architecture.md).

## What this is

A local inference service on the DGX Spark (NVIDIA GB10, 121 GiB unified memory)
that serves one approved model at a time per logical *role*, swappable between
approved *candidates* without a reconfig-and-restart dance. For v0.1 it ships the
**agentic** role only.

| Property | Value |
|---|---|
| Endpoint (LAN-reachable) | `http://<host>:30000/v1` (bearer-auth) |
| Served model name (wire contract) | `qwen3.6-27b-agentic` |
| Resident profile | `qwen36-27b-fp8` (Qwen3.6-27B-FP8 @ pinned rev, FP8) |
| Context / pool | 262144 context, 1 running / 1 queued |
| Runtime | sglang v0.5.14 derived image |
| `mem-fraction-static` | 0.60 (measured → 346,485-token pool; ~32% margin) |

## Memory sizing (measured)

Verified from the model `config.json` (`text_config`): `num_hidden_layers=64`,
`full_attention_interval=4` → **16 full-attention + 48 linear-attention layers**
(64 total). `head_dim=256`, `num_key_value_heads=4`.

- Per-token KV = **64.0 KiB/token** (BF16, token-growing KV).
- 262144 tokens require **16 GiB** of token-growing KV.
- `mem_fraction_static=0.60` produced a **measured 346,485-token pool**
  (≥ 262144, ~32% margin). Calibration: `0.40 → 148,192` (fail),
  `0.50 → 245,748` (fail), `0.60 → 346,485` (pass).

`mem_fraction_static` controls SGLang's static allocation *target*; the realizable
pool is also constrained by weights, fixed hybrid-attention state, CUDA graphs,
activation workspace, and DGX Spark memory accounting. **0.60 is measured on this
configuration** — do not generalize it. The yield between fraction and pool is
non-linear (~9,760 tokens per +0.01, because CUDA-graph capture grows with the
fraction); the cap-CUDA-graphs-to-bs=1 optimization is on the
[`roadmap`](roadmap.md). See [`known-limitations.md`](known-limitations.md).

## Why one model at a time

GB10 has 121 GiB unified memory. A 27B model resident at 262144 context uses most
of it. **Only one model can be resident per role at a time.** Swapping is a full
cold load (~4 min), bounded by NVMe bandwidth — not something software can speed
up for dissimilar models. Automatic idle eviction (freeing a slot when disused) is
a documented [`roadmap`](roadmap.md) item, not a current capability.

## The promotion workflow

Models earn deployment; they are not auto-promoted.

1. Produce an honest capability record (what the model *provides*).
2. Prove it satisfies the role with the resolver (`tools/resolve_service_plan.py`).
3. Offer it in `runtime/sglang/available.toml[<role>]`.
4. Activate: `inferencectl use <role> <candidate_id>`.

## Read next

- [`architecture.md`](architecture.md) — the binding system; resolver honesty.
- [`operations.md`](operations.md) — operating the service.
- [`security.md`](security.md) — the firewall prerequisite.
- [`smoke-test.md`](smoke-test.md) — the release gate.
