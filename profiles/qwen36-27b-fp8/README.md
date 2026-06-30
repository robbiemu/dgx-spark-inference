# Profile: `qwen36-27b-fp8` (agentic baseline)

The approved plain FP8 baseline for the **agentic** role. This profile
**describes** the model and its launch parameters; it **does not contain model
weights**. Weights are fetched on first use from Hugging Face into a
deterministic directory under `MODEL_CACHE_ROOT`.

## Provenance

| Field | Value |
|---|---|
| Model ID | `qwen36-27b-fp8` |
| Source repository | `Qwen/Qwen3.6-27B-FP8` |
| Pinned revision | `e89b16ebf1988b3d6befa7de50abc2d76f26eb09` |
| Quantization | fp8 |

## Fetch the weights (one-time, per host)

`Qwen/Qwen3.6-27B-FP8` is a **gated** repository. You must first accept its
license on the model page (https://huggingface.co/Qwen/Qwen3.6-27B-FP8) and
authenticate:

```bash
hf auth login            # paste a Read access token from huggingface.co/settings/tokens
```

The adapter resolves the weights from your Hugging Face **cache** by default — it
derives `<MODEL_CACHE_ROOT>/models--Qwen--Qwen3.6-27B-FP8/snapshots/<revision>`
from this profile's `source_repository` + `source_revision`, and mounts
`$MODEL_CACHE_ROOT` into the container at `/hf-cache` read-only. A plain cache
fetch (no `--local-dir`) is the expected, zero-copy path:

> **Mount design note (HF cache layout).** An HF cache snapshot is a tree of
> symlinks pointing into `blobs/` (e.g. `snapshots/<rev>/config.json ->
> ../../blobs/<hash>`). The adapter mounts `MODEL_CACHE_ROOT` itself into the
> container — **not** the snapshot directory — precisely so those symlinks
> resolve to real files inside the same mount root. If you ever narrow the mount
> to just the snapshot dir, the symlinks will break and sglang will fail to load.
> This is why the default is a cache fetch, not a relocated copy.

```bash
export MODEL_CACHE_ROOT=/srv/model-cache   # your host's cache root (see install.sh)
export HF_HOME="$MODEL_CACHE_ROOT"          # so hf writes into the cache the adapter reads
hf download Qwen/Qwen3.6-27B-FP8 \
  --revision e89b16ebf1988b3d6befa7de50abc2d76f26eb09
```

### Optional: a plain directory instead of the cache layout

If you prefer weights as a flat directory (what `--local-dir` produces), fetch it
and then set `model_dir` in `profiles/qwen36-27b-fp8/sglang.toml` to the directory
path **relative to `MODEL_CACHE_ROOT`**. The adapter uses `model_dir` verbatim
when set, and the preflight verifies it exists.

```bash
hf download Qwen/Qwen3.6-27B-FP8 \
  --revision e89b16ebf1988b3d6befa7de50abc2d76f26eb09 \
  --local-dir "$MODEL_CACHE_ROOT/qwen36-27b-fp8-plain"
# then in sglang.toml [identity]: model_dir = "qwen36-27b-fp8-plain"
```

No weights are redistributed with this repository.

## Launch parameters (model facts)

```toml
[launch]
context_length       = 262144
max_running_requests = 1
max_queued_requests  = 1
attention_backend    = "auto"
mem_fraction_static  = 0.60
```

`mem_fraction_static = 0.60` is a **measured** value for this exact configuration
(262144 context, 1/1 single-session pool, GB10), not a KV-only back-of-envelope.
It produced a realized KV pool of **346,485 tokens** (~32% margin over 262144).
Calibration points: `0.40 → 148,192` (fail), `0.50 → 245,748` (fail),
`0.60 → 346,485` (pass). See `docs/known-limitations.md` for the non-linear
fraction→pool yield (CUDA-graph capture grows with the fraction) and
`docs/roadmap.md` for the cap-CUDA-graphs-to-bs=1 optimization that would allow a
lower fraction for the same context.

## Memory sizing (measured)

Verified from the model `config.json` (`text_config`): `num_hidden_layers=64`,
`full_attention_interval=4` → **16 full-attention + 48 linear-attention layers**
(64 total). `head_dim=256`, `num_key_value_heads=4`.

- Per-token KV = **64.0 KiB/token** (BF16, token-growing KV).
- 262144 tokens require **16 GiB** of token-growing KV.

`mem_fraction_static` controls SGLang's static allocation *target*; the realizable
KV pool is also constrained by model weights, fixed hybrid-attention state, CUDA
graphs, activation workspace, and DGX Spark memory accounting. **The value 0.60 is
based on measured calibration on this configuration** — do not generalize it to
other models or hardware.
