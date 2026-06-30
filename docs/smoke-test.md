# Smoke test — the release gate is READ-ONLY

The v0.1 release gate is a **read-only** smoke against the **live healthy
deployment**. It is evidence that the running service is healthy and that the
published artifact's topology (served name, context length, auth boundary)
reproduces what is live — with **zero disruption** to the resident baseline.

It is **not** a proof that the curated installer runs end-to-end: a second 27B
model cannot coexist with the resident baseline, so a fully isolated cold-load of
the published artifact is inherently disruptive (that is the *optional* staged
smoke below). Treat the read-only gate as strong live-health evidence; treat the
staged smoke as the artifact-install validation.

## The read-only gate (run this)

Against the running service (default `http://127.0.0.1:30000`). Read the bearer
token from wherever the **live** service keeps its secret — on the reference
deployment that is `$CONFIG_ROOT/agent.env` (default `/etc/dgx-spark-inference/agent.env`).
Adjust the path if your live unit points elsewhere.

```bash
: ${PORT:=30000}
: ${SECRET_FILE:=/etc/dgx-spark-inference/agent.env}   # match your live unit's EnvironmentFile
TOKEN="$(sudo grep -E '^SGLANG_API_KEY=' "$SECRET_FILE" | head -1 | cut -d= -f2-)"

# 1. Health: expect 200.
curl -s -o /dev/null -w "health=%{http_code}\n" http://127.0.0.1:$PORT/health

# 2. Health-generate: expect 200.
curl -s -o /dev/null -w "health_generate=%{http_code}\n" http://127.0.0.1:$PORT/health_generate

# 3. Auth boundary intact: expect 401 without a token.
curl -s -o /dev/null -w "models_no_token=%{http_code}\n" http://127.0.0.1:$PORT/v1/models

# 4. Authenticated models: expect qwen3.6-27b-agentic at max_model_len=262144.
curl -s http://127.0.0.1:$PORT/v1/models \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool

# 5. Semantic completion: expect a correct, coherent answer.
#    Use max_tokens>=1024: thinking-mode models spend tokens on the reasoning
#    trace (reasoning_content) BEFORE the visible answer. A tiny budget (e.g. 64)
#    returns content="" with finish_reason="length" — the model was still thinking,
#    not broken. (Alternatively, set "enable_thinking":false to skip reasoning.)
curl -s http://127.0.0.1:$PORT/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-agentic","max_tokens":1024,"messages":[{"role":"user","content":"What is the capital of France? Answer with just the city name."}]}' \
  | python3 -m json.tool
```

Pass criteria: `health=200`, `health_generate=200`, `models_no_token=401`, the
authenticated `/v1/models` response lists `qwen3.6-27b-agentic` with
`max_model_len=262144`, and the completion answers correctly.

This is **non-disruptive**: it only reads the live endpoint. Never point smoke
traffic at the production slot in a way that could disrupt it.

## Optional: a controlled staged smoke (artifact-install validation; NOT the gate)

A *controlled staged smoke* validates that the published artifact itself
cold-loads and serves — but it is disruptive and must **never run automatically**;
it requires an explicit maintenance window:

1. Stop the production service: `sudo systemctl stop dgx-spark-inference.service`.
2. Install (if not already) and launch the published artifact on a **throwaway
   unit + port** (e.g. `dgx-spark-inference-smoke.service` on `:30100`), using the
   same dispatcher and adapter the production path uses.
3. Run the read-only checks above against `:30100`.
4. Tear down the throwaway unit/container.
5. Restart production and verify the baseline restored (health 200, served name
   `qwen3.6-27b-agentic`).

**Never** smoke-test against `:30000` in a way that disrupts the production
baseline a client depends on.
