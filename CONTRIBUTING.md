# Contributing

This is a reference implementation for a specific topology (one DGX Spark, one
resident SGLang slot, operator-driven swaps, systemd). Contributions that keep
that scope honest are welcome.

## Before you change anything

1. **Run the gate.** `bash tests/run_all.sh` must pass (TOML validation, the five
   no-GPU tests, and the contextual secret scan). It needs only python3 (3.12+).
2. **Read [`docs/architecture.md`](docs/architecture.md)**, especially the
   "Capability enforcement" section: the resolver is a consistency tool, **not** a
   live gate. Do not wire `resolve_service_plan.py` into `dispatch.sh`. Production
   safety comes from capability validation in the tests **and** the runtime catalog
   only listing compatible candidates.

## Things to keep in mind

- **No machine-specific paths.** No `/home/<user>`, no private IPs, no `.local`
  hostnames, no fixed ports/names in committed files. The seven machine values are
  installer inputs surfaced via `$CONFIG_ROOT/inference.env`. The scanner will fail
  the gate if you add one.
- **Profiles describe, never contain weights.** A profile carries HF repo + pinned
  revision + quantization + launch params. Fetch weights with `hf` (the Hugging
  Face CLI; documented per profile). No weights are redistributed.
- **One launch path.** `systemd → dispatch.sh → sglang adapter`. Don't add an
  alternate launcher.
- **Capability honesty.** A capability record states what a model *provides*.
  `structured_output` means grammar-constrained; there is no
  `structured_output_prompt_only`. If you add a candidate to
  `available.toml[<role>]`, test 3 will fail unless it actually satisfies the role.
- **No `--allow-incompatible` bypass**, ever. Incompatible candidates run only via
  an explicitly labeled experimental path.

## External checks (run before release, not part of run_all.sh)

ShellCheck on every `.sh`; Gitleaks and TruffleHog over the tree. The maintainer
runs these after tool install. The contextual scanner in `run_all.sh` is the
in-tree check; the external scanners are the belt-and-suspenders for publication.

## Licensing

By contributing you agree your contributions are licensed Apache-2.0, like the rest
of this repository.
