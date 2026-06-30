# Operations

Operating the service after install. See [`architecture.md`](architecture.md) for
how a request gets served and [`security.md`](security.md) for the firewall
prerequisite.

The operator CLI is installed at:
`/usr/local/lib/dgx-spark-inference/src/inferencectl/inference-cli.sh`
(under `INSTALL_ROOT`; not the source checkout — the live service must not
depend on a mutable checkout). The examples below set it as `INFCTL`. The CLI
accepts `--config-root` or the `CONFIG_ROOT` env var (default
`/etc/dgx-spark-inference`).

## Status / inspection (read-only)

```bash
INFCTL=/usr/local/lib/dgx-spark-inference/src/inferencectl/inference-cli.sh
sudo CONFIG_ROOT=/etc/dgx-spark-inference "$INFCTL" status
sudo CONFIG_ROOT=/etc/dgx-spark-inference "$INFCTL" candidates agentic
```

`status` prints the resident model per role, the systemd state, the health code,
and the served name. `candidates` lists what the runtime catalog offers for the
role and marks the active one.

## Swap the resident model (cold load, ~4 min for 27B)

```bash
sudo CONFIG_ROOT=/etc/dgx-spark-inference "$INFCTL" use agentic <candidate_id>
```

`use` validates the candidate is offered, backs up `active-models.toml`, edits it,
restarts the service, and waits for `/health=200` (with a rollback hint on
failure). It refuses to swap if the service is not currently healthy. Only
candidates listed in `available.toml[agentic]` can be activated — and that catalog
only ever lists capability-compatible candidates (see
[`architecture.md`](architecture.md)).

For v0.1 the only production candidate is `qwen36-27b-fp8`.

## Manual lifecycle

```bash
sudo systemctl restart dgx-spark-inference.service   # cold reload of active candidate
sudo systemctl stop   dgx-spark-inference.service    # free the slot (no auto-reload)
```

## Health check

```bash
curl -s http://127.0.0.1:30000/health          # 200 = ready
curl -s http://127.0.0.1:30000/v1/models       # 401 without token (auth boundary intact)
```

## Timeout behavior — important for clients

Thinking-mode requests on this service can take a **long** time. Measured
end-to-end at ~7.5 tok/s, a full thinking trace at `max_tokens: 3000` runs roughly
**200–400s**. **Client request timeouts must exceed that**, or long reasoning
traces will be dropped mid-generation.

- For streaming clients: ensure the read/idle timeout is generous, and rely on the
  steady token stream as a liveness signal.
- For non-streaming clients: set the request timeout above 400s for thinking-mode
  requests.

This is an inherent property of serving a 27B reasoning model on this hardware, not
a defect. See [`known-limitations.md`](known-limitations.md).

## Running the experimental DFlash path

DFlash is **not** a production candidate and cannot be activated with `use`. It is
run via the explicitly-labeled experimental launcher, which defaults to its
validated 32768 context, requires an explicit dangerous override for 262144, and
**refuses to launch while the production service or container is active**:

```bash
experiments/dflash/run-experimental.sh
# or, accepting the unvalidated full-context risk:
# experiments/dflash/run-experimental.sh --i-understand-262144-fit-is-unvalidated
```

See [`../bundles/experimental/qwen36-27b-fp8-dflash/README.md`](../bundles/experimental/qwen36-27b-fp8-dflash/README.md).
