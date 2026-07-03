# Ornith 1.0 9B FP8 helper profile

This directory defines the managed `agentic-helper` profile and records its
qualification evidence. It does **not** distribute Ornith model weights,
tokenizer files, processor files, or a hydrated model bundle.

Operators must obtain the model artifacts from their upstream repositories and
hydrate the FP8 bundle as documented in the repository's Ornith setup
walkthrough. The configured `identity.model_dir` is resolved below the
operator-supplied `MODEL_CACHE_ROOT`; it is not populated by the installer.

The pinned sources are:

- `barryke/Ornith-1.0-9B-FP8-DYNAMIC` revision
  `01272cd6c8228e82897c08826ef83c86b3787a0d`;
- processor metadata from `deepreinforce-ai/Ornith-1.0-9B` revision
  `83dc1f5e24ef8527af019a6b3bf66ac0f1c2c999`.

See `evidence/2026-06-30.md` and `docs/v0_2_phase0_results.md` for the measured
capability and memory results.
