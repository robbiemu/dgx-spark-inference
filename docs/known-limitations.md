# Known limitations

The honest constraints of this reference implementation on the DGX Spark (GB10).
Several of these are *lessons* from the earlier design, recorded as prose rather
than as runnable history.

## GB10 / attention backend — fa3 asserts SM ≤ 90; use flashinfer

The NVIDIA GB10 is SM100. SGLang's `fa3` attention backend asserts `SM ≤ 90`, so
it is not usable on GB10. **DFlash requires `attention-backend = flashinfer`** on
GB10 for this reason. The plain baseline uses `auto` (sglang selects an available
backend). This is a hardware fact, not a configuration choice.

## In-container `--host 127.0.0.1` breaks Docker `--publish`

An earlier launch attempt set the sglang `--host` to `127.0.0.1` *inside the
container*. That makes sglang listen only on the container's loopback, so Docker's
`--publish` cannot forward host traffic to it — the endpoint is unreachable from
the host/LAN. The adapter binds `0.0.0.0` inside the container (paired with a host
firewall + bearer auth; see [`security.md`](security.md)). This is why a host bind
appears in the manifest's `common_launch`.

## Why a dispatcher exists

The earlier design had the runtime directly coupled to a single model's launch
script. A dispatcher (`dispatch.sh`) that resolves the active candidate at launch
time is what makes *cold swaps* a one-file edit (`active-models.toml`) plus a
restart — no runtime file changes, no reimage. The trade-off is a little
indirection; the win is that swapping is safe and auditable.

## Memory: non-linear fraction → pool yield

`mem_fraction_static` controls SGLang's static allocation *target*; the realizable
KV pool is also constrained by model weights, fixed hybrid-attention state, CUDA
graphs, activation workspace, and DGX Spark memory accounting. Observed marginal
yield was approximately **9,760 tokens per +0.01 fraction** over the tested range
(0.40 → 148,192; 0.50 → 245,748; 0.60 → 346,485). CUDA graphs, hybrid-attention
state, weights, activation workspace, and unified-memory accounting all compete
with the KV pool; **their individual contributions were not isolated** by the
calibration. The published baseline pins `0.60` (measured → 346,485-token pool).
See [`handbook.md`](handbook.md) for the calibration points. **Do not generalize
0.60 to other models/hardware.**

## Thinking-mode latency (200–400s)

A full thinking trace at `max_tokens: 3000` runs ~200–400s end-to-end on this
hardware (~7.5 tok/s). Client request timeouts must exceed that. See
[`operations.md`](operations.md).

## DFlash limitations

DFlash is **experimental** and not a production candidate:

- **Grammar-constrained structured output: unsupported** (HTTP 400 for
  `response_format`/grammar). Prompt-only JSON works. This is why it is rejected
  for `agentic`.
- **Logprobs: unsupported** under DFLASH (`return_logprob` HTTP 400).
- **262144 memory fit: UNVALIDATED.** The validated context is 32768. The
  experimental launcher defaults to 32768 and requires an explicit dangerous
  override for 262144.
- **Reasoning-depth divergence:** DFlash reasoned ~4.6× more than baseline for the
  same seeded prompt (gate 8); both answers were correct.

See [`../bundles/experimental/qwen36-27b-fp8-dflash/README.md`](../bundles/experimental/qwen36-27b-fp8-dflash/README.md).

## One model resident at a time

Only one 27B model can be resident per role at a time (GB10, 121 GiB). Swapping is
a full cold load (~4 min, NVMe-bound). Automatic idle eviction is on the
[`roadmap`](roadmap.md).
