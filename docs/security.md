# Security

## The firewall prerequisite — `0.0.0.0` is not itself LAN security

The adapter binds `0.0.0.0` *inside the container* so Docker can publish the port
to the host. **Binding `0.0.0.0` is not a security measure.** It must be paired
with a host firewall that admits only your intended LAN/client, plus bearer-token
authentication on every request.

Required, in order:

1. **Host firewall.** Admit the published port only from your trusted LAN (or a
   single client). Drop it from everything else. If you skip this, the endpoint is
   reachable by anything that can route to the host.
2. **Bearer authentication.** Every request must present a valid `SGLANG_API_KEY`
   (the adapter validates it is a 64-char lowercase hex value and passes it as the
   sglang `api-key`). `GET /v1/models` returns 401 without it — verify that.
3. **Secret hygiene.** The key lives in `CONFIG_ROOT/agent.env`, root-owned, mode
   `0600`. It is never printed, never appears in command arguments or logs, and the
   rendered runtime YAML is mode `0600`. The manifest `log_level` is `warning`,
   which suppresses the startup `server_args=` record that can otherwise log the
   resolved key. (If you ever raise `log_level` to `info`, audit the journal
   afterward — see the note below.)
4. **Read-only mounts.** The model cache and the rendered YAML are mounted into the
   container read-only.

## The installer never handles the secret

`deploy/install.sh` checks that the operator-supplied `agent.env` exists, contains
`SGLANG_API_KEY=`, and has safe permissions — then copies it (once, never
overwritten, even by `--replace`) into `CONFIG_ROOT`. The installer never reads or
prints its value. `--replace` may overwrite program files and the unit **only**;
it never overwrites `agent.env`, `active-models.toml`, or `runtimes.toml`.

## Journal safety

The service runs at `log_level = warning`. If you temporarily raise it to `info`
for debugging, sglang may log the full `server_args=` (including the key). After
such debugging, purge the key-bearing journal lines and restore `log_level`. Do
not commit raw journal dumps as evidence.

## Thinking-mode timeouts

A thinking trace can run **200–400s** end-to-end. This is an operational property,
not a security issue, but it matters for availability: a reverse proxy or client
with a short timeout will drop long reasoning traces mid-generation. See
[`operations.md`](operations.md) and [`known-limitations.md`](known-limitations.md).

## What is explicitly out of threat model

- This is a single-tenant, single-slot reference service on a trusted LAN. It is
  not hardened for hostile multi-tenant access.
- TLS termination is assumed to be handled by a reverse proxy in front of the
  service if you need it; the service itself serves plain HTTP on the LAN.
