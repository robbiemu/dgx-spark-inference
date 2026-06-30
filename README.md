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

## What's in the box

- A **single launch path**: `systemd → dispatch.sh → sglang adapter`. Exactly one;
  there is no alternate launcher.
- **Reproducible profiles** that *describe* a model (HF repo + pinned revision +
  quantization + launch params). **No model weights are redistributed** — fetch
  them with `huggingface-cli` (documented per profile).
- **Explicit compatibility contracts**: capability records + a resolver that
  proves a candidate satisfies a role's requirements. Production safety comes
  from capability validation in the test suite **and** the runtime catalog only
  listing capability-compatible candidates.
- A **thin, safe installer** with a dry-run mode that refuses to clobber existing
  files unless you pass `--replace` (and even then, never touches operator state).
- A systemd unit, an operator CLI (`status` / `candidates` / `use` / `unload` /
  `reload`), and an **experimental** DFlash speculative-decoding path that is
  deliberately not a production candidate.

## 5-minute install (operator)

Prerequisites: a DGX Spark (NVIDIA GB10), Docker with GPU support, the SGLang
runtime image built (see `runtime/sglang/Dockerfile`), and the Qwen weights
fetched (see `profiles/qwen36-27b-fp8/README.md`).

```bash
# 1. Fetch the baseline weights into your model cache (one-time).
export MODEL_CACHE_ROOT=/srv/model-cache
huggingface-cli download Qwen/Qwen3.6-27B-FP8 \
  --revision e89b16ebf1988b3d6befa7de50abc2d76f26eb09 \
  --local-dir "$MODEL_CACHE_ROOT/qwen36-27b-fp8"

# 2. Create your secret (operator-supplied; never handled by this repo).
install -d -m 0700 ~/.config/dgx-spark-inference
echo "SGLANG_API_KEY=$(openssl rand -hex 32)" > ~/.config/dgx-spark-inference/agent.env
chmod 600 ~/.config/dgx-spark-inference/agent.env

# 3. Preview the install (renders the unit + config plan; writes nothing).
deploy/install.sh --dry-run \
  --model-cache-root "$MODEL_CACHE_ROOT" \
  --agent-env ~/.config/dgx-spark-inference/agent.env

# 4. Install (refuses to clobber; never starts the service).
deploy/install.sh \
  --model-cache-root "$MODEL_CACHE_ROOT" \
  --agent-env ~/.config/dgx-spark-inference/agent.env

# 5. Start it (separate, reviewed operation).
sudo systemctl start dgx-spark-inference.service

# 6. Verify (read-only smoke gate — see docs/smoke-test.md).
curl -s http://127.0.0.1:30000/health   # 200 = ready
```

## Where to read next

- [`docs/handbook.md`](docs/handbook.md) — what this provides (entry point).
- [`docs/architecture.md`](docs/architecture.md) — the binding system, request
  flow, and the honest role of the resolver.
- [`docs/operations.md`](docs/operations.md) — status/candidates/use/lifecycle/health.
- [`docs/security.md`](docs/security.md) — the LAN-only firewall prerequisite.
- [`docs/known-limitations.md`](docs/known-limitations.md) — GB10/DFlash/memory limits.
- [`docs/smoke-test.md`](docs/smoke-test.md) — the release gate.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
