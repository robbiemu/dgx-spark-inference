# dgx-spark-inference

> `dgx-spark-inference` is a reference implementation for operating one
> authenticated, swappable SGLang model slot on a single NVIDIA DGX Spark. It
> provides reproducible runtime and model profiles, explicit compatibility
> contracts, a systemd-managed endpoint, and operator-controlled cold swaps.
> **It is not a multi-tenant serving platform or an Ollama-style request-driven
> model loader.**


The topology is opinionated by its constraints: one GB10, one client,
operator-driven swaps, systemd. Multiple roles are out of scope for v0.1; the
release ships only the **`agentic`** role (coding/general agent work), served as
`qwen3.6-27b-agentic`. This is a reference implementation, not a generic platform.

<img width="100%" alt="ChatGPT Image 30 de jun  de 2026, 14_12_19" src="https://github.com/user-attachments/assets/4de0716b-1b83-4d32-8f0e-a49b2e91950e" />

## Tested environment

The measurements and behavior in this repo were validated on this configuration
(record the date you reproduce on your own):

| Component | Value |
|---|---|
| Hardware | NVIDIA DGX Spark (NVIDIA GB10, 121 GiB unified memory) |
| OS / kernel | Ubuntu 24.04.4 LTS · kernel `6.17.0-1021-nvidia` |
| GPU driver / CUDA | `580.159.03` / CUDA 13.0 |
| Container runtime | Docker with `--gpus all` |
| sglang runtime base image | `lmsysorg/sglang:v0.5.14-cu130-runtime` @ `sha256:9e436f44…0ad2` (see `runtime/sglang/Dockerfile`) |
| Served model | `Qwen/Qwen3.6-27B-FP8` @ revision `e89b16eb…6eb09` |
| Context / pool | 262144 context, 1 running / 1 queued, `mem-fraction-static=0.60` (measured → 346,485-token pool) |
| Measurements taken | 2026-06-29 (prep calibration + server-ready gates) |

## What's in the box

- A **single launch path**: `systemd → dispatch.sh → sglang adapter`. Exactly one;
  there is no alternate launcher.
- **Reproducible profiles** that *describe* a model (HF repo + pinned revision +
  quantization + launch params). **No model weights are redistributed** — fetch
  them with `hf` (the Hugging Face CLI; documented per profile).
- **Explicit compatibility contracts**: capability records + a resolver that
  proves a candidate satisfies a role's requirements. Production safety comes
  from capability validation in the test suite **and** the runtime catalog only
  listing capability-compatible candidates.
- A **thin, safe installer** with a dry-run mode that refuses to clobber existing
  files unless you pass `--replace` (and even then, never touches operator state).
- A systemd unit, an operator CLI (`status` / `candidates` / `use` / `unload` /
  `reload`), and an **experimental** DFlash speculative-decoding path that is
  deliberately not a production candidate.

## Install the service

This is a **reference blueprint to tailor locally**, not a clone-and-deploy
image. The one piece that is inherently host-specific is the **runtime image ID
pin**: the adapter refuses to launch any container whose image ID doesn't match
the committed pin in `runtime/sglang/runtime-manifest.toml`, and a fresh
`docker build` of the Dockerfile produces a *different* ID than the committed
one. So you must build and re-pin **before** install. `scripts/build-runtime.sh`
does both.

The phases below are short on commands but not on wall-clock: a 27B cold load is
~4 minutes, and the gated weight fetch is ~30 GB. See `docs/runbook.md` for
operating the service after install.

### 1. Prepare

```bash
# 1a. Build the runtime image and pin its ID into the manifest (REQUIRED first).
#     This rewrites runtime/sglang/runtime-manifest.toml:image_id to YOUR build.
scripts/build-runtime.sh --update

# 1b. Fetch the baseline weights into your model cache (one-time, ~30 GB).
#     Qwen3.6-27B-FP8 is gated: accept the license on the repo page, then `hf auth login`.
export MODEL_CACHE_ROOT=/srv/model-cache
export HF_HOME="$MODEL_CACHE_ROOT"
hf download Qwen/Qwen3.6-27B-FP8 \
  --revision e89b16ebf1988b3d6befa7de50abc2d76f26eb09

# 1c. Create your secret (operator-supplied; never handled by this repo).
install -d -m 0700 ~/.config/dgx-spark-inference
echo "SGLANG_API_KEY=$(openssl rand -hex 32)" > ~/.config/dgx-spark-inference/agent.env
chmod 600 ~/.config/dgx-spark-inference/agent.env
```

### 2. Install

```bash
# 2a. Preview the install (renders the unit + config plan; writes nothing).
deploy/install.sh --dry-run \
  --model-cache-root "$MODEL_CACHE_ROOT" \
  --agent-env ~/.config/dgx-spark-inference/agent.env

# 2b. Install (refuses to clobber existing files unless --replace; never starts the service).
deploy/install.sh \
  --model-cache-root "$MODEL_CACHE_ROOT" \
  --agent-env ~/.config/dgx-spark-inference/agent.env
```

### 3. Activate

```bash
# 3a. Start it (cold load ~4 min for 27B; `systemctl start` returns immediately,
#     readiness is when /health returns 200 — poll, don't trust the fast return).
sudo systemctl start dgx-spark-inference.service

# 3b. Verify (read-only smoke gate — full version in docs/smoke-test.md).
curl -s http://127.0.0.1:30000/health   # 200 = ready
```

## Where to read next

- [`docs/handbook.md`](docs/handbook.md) — what this provides (entry point).
- [`docs/architecture.md`](docs/architecture.md) — the binding system, request
  flow, and the honest role of the resolver.
- [`docs/operations.md`](docs/operations.md) — status/candidates/use/lifecycle/health.
- [`docs/runbook.md`](docs/runbook.md) — day-2 operations: diagnostics, rollback, upgrades.
- [`docs/security.md`](docs/security.md) — the LAN-only firewall prerequisite.
- [`docs/known-limitations.md`](docs/known-limitations.md) — GB10/DFlash/memory limits.
- [`docs/smoke-test.md`](docs/smoke-test.md) — the release gate.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
