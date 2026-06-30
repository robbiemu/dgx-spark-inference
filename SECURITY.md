# Security policy

## Reporting

This is a reference implementation. Report security issues via the project's
private vulnerability reporting channel on GitHub (Security tab → "Report a
vulnerability"). Do not open a public issue for security problems.

## Boundary

This service is designed for a **single-tenant, single-slot** deployment on a
trusted LAN. It is **not** hardened for hostile multi-tenant access.

The security boundary, in order of importance:

1. **Host firewall (required).** The adapter binds `0.0.0.0` inside the container
   so Docker can publish the port. **`0.0.0.0` is not itself LAN security.** You
   must run a host firewall that admits the published port only from your trusted
   LAN (or a single client). See [`docs/security.md`](docs/security.md).
2. **Bearer authentication (required).** Every request must present a valid
   `SGLANG_API_KEY`. `GET /v1/models` returns 401 without it — verify this after
   every activation.
3. **Secret hygiene.** The key lives in `CONFIG_ROOT/agent.env`, root-owned, mode
   `0600`, never printed, never in command arguments, never in logs. The rendered
   runtime YAML is mode `0600`. The manifest `log_level` is `warning`.

## Secrets are never in this repository

No API keys, bearer tokens, private keys, or passwords are committed. The
contextual secret scanner (`tests/scan_secrets.py`, run in `tests/run_all.sh`)
rejects credential assignments, PEM headers, user-specific paths
(`/home/<user>`), private IPs, and `.local` hostnames. It deliberately allows
documented loopback/bind (`127.0.0.1`, `0.0.0.0`), image digests (`sha256:…`), and
model revisions. The operator supplies the secret at install time; the installer
never handles its value.

## Timeout note for availability

Thinking-mode requests can take **200–400s** end-to-end. A reverse proxy or client
with a short timeout will drop long reasoning traces mid-generation. Size timeouts
accordingly. See [`docs/operations.md`](docs/operations.md).
