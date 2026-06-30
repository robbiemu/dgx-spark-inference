# Runbook — operating the service

Day-2 operations for `dgx-spark-inference` after install. For the binding system
and launch flow see [`architecture.md`](architecture.md); for status/swap command
reference see [`operations.md`](operations.md).

The operator CLI lives at
`/usr/local/lib/dgx-spark-inference/src/inferencectl/inference-cli.sh` (set as
`INFCTL` below). It accepts `--config-root` or the `CONFIG_ROOT` env var.

```bash
INFCTL=/usr/local/lib/dgx-spark-inference/src/inferencectl/inference-cli.sh
export CONFIG_ROOT=/etc/dgx-spark-inference   # default; change if you installed elsewhere
```

> **Passing `CONFIG_ROOT` through `sudo`.** `sudo` resets most environment
> variables by default (`env_reset`), so a bare `export CONFIG_ROOT=...` is
> silently dropped. Pass it inline on each command:
> `sudo CONFIG_ROOT="$CONFIG_ROOT" "$INFCTL" <subcommand>`. The table and
> examples below use this form.

## Commands and their side effects

| Command | What it does | Disruptive? | Cold load? |
|---|---|---|---|
| `sudo CONFIG_ROOT="$CONFIG_ROOT" "$INFCTL" status` | Prints resident model per role, systemd state, health code, served name. Read-only. | No | No |
| `sudo CONFIG_ROOT="$CONFIG_ROOT" "$INFCTL" candidates agentic` | Lists what `available.toml` offers for the role; marks the active one. Read-only. | No | No |
| `sudo CONFIG_ROOT="$CONFIG_ROOT" "$INFCTL" use agentic <id>` | Validates candidate is offered; **backs up** `active-models.toml`; edits it; restarts; waits for `/health=200`. **Refuses if not currently healthy.** | **Yes** (endpoint down during reload) | Yes (~4 min) |
| `sudo CONFIG_ROOT="$CONFIG_ROOT" "$INFCTL" reload agentic` | Restarts the unit — cold-loads the *currently active* candidate. No model change. | **Yes** (endpoint down during reload) | Yes (~4 min) |
| `sudo CONFIG_ROOT="$CONFIG_ROOT" "$INFCTL" unload agentic` | Stops the unit — frees the GPU slot. **No auto-reload**; the endpoint is down until you `reload`/`start`. | **Yes** (endpoint down until manual restart) | No (it stops) |

`use` only accepts candidates listed in `available.toml[agentic]`, and that
catalog only ever lists capability-compatible candidates (enforced by test 3).

## First look when something is wrong

Go in this order — each resolves a distinct failure class:

1. **Unit state + health.**
   ```bash
   systemctl status dgx-spark-inference.service
   curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:30000/health
   ```
   `activating` for >~5 min or `failed` → check the journal (below). `000` →
   container not serving yet, or crashed.
2. **The journal (most failures show their cause here).**
   ```bash
   sudo journalctl -u dgx-spark-inference.service -n 80 --no-pager
   ```
   Look for `REFUSING:` (the dispatcher/adapter refusing on purpose) vs a Python
   traceback (a real crash).
3. **Image-ID mismatch** — `REFUSING: image ID drifted`. Your built image differs
   from the manifest pin. Re-pin: `scripts/build-runtime.sh --update`, then
   `sudo systemctl restart dgx-spark-inference.service`.
4. **Missing model snapshot** — `REFUSING: model dir not found: <path>`. The
   resolved path under `MODEL_CACHE_ROOT` doesn't exist. Check
   `CONFIG_ROOT/inference.env` for `MODEL_CACHE_ROOT`, and confirm the HF cache
   layout (`models--Qwen--.../snapshots/<rev>`) is present. See
   [`profiles/qwen36-27b-fp8/README.md`](../profiles/qwen36-27b-fp8/README.md).
5. **Bad secret shape** — `REFUSING: SGLANG_API_KEY is not the expected 64-char
   lowercase hex`. Regenerate in `CONFIG_ROOT/agent.env` (mode 0600).
6. **Insufficient KV pool / OOM during load** — sglang crash-loops at
   `mamba-cache` or memory allocation. This is the `mem-fraction-static`
   calibration concern; see [`known-limitations.md`](known-limitations.md) and
   the measured 0.60 value.

## Rollback from a failed swap

`use` writes a timestamped backup before editing. If a swap fails (the CLI prints
a ROLLBACK hint), restore from the backup and restart:

```bash
sudo cp -a $CONFIG_ROOT/active-models.toml.pre-use.<timestamp> $CONFIG_ROOT/active-models.toml
sudo systemctl restart dgx-spark-inference.service
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:30000/health   # expect 200
```

Backups accumulate; prune old ones (`active-models.toml.pre-use.*`) as needed.

## Safe upgrade (new repo version)

To apply a newer checkout of this repo over an existing install:

```bash
cd <new checkout>
scripts/build-runtime.sh --update        # re-pin image ID for THIS build

# Use the same model-cache root used for the original install (the installer
# refuses an empty value; the live value is in $CONFIG_ROOT/inference.env).
export MODEL_CACHE_ROOT=/srv/model-cache

sudo deploy/install.sh --replace \
  --model-cache-root "$MODEL_CACHE_ROOT" \
  --agent-env "$CONFIG_ROOT/agent.env"
sudo systemctl restart dgx-spark-inference.service
```

`--replace` overwrites **program files + the unit only**. It **never** touches
operator state — `agent.env`, `active-models.toml`, `runtimes.toml` are preserved
(those change only via explicit operator action). Confirm with `status` + a
health check after restart.

## Thinking-mode timeouts (availability)

Thinking-mode requests can run **200–400s** end-to-end. Any reverse proxy or
client timeout shorter than that will drop long reasoning traces mid-generation.
For non-streaming thinking requests, set client timeouts above 400s. See
[`operations.md`](operations.md).
