# Qwen3.6 27B NVFP4 with FP8 E4M3 KV

Autoregressive fallback candidate for the `agentic` role.

The pinned NVIDIA ModelOpt checkpoint is the production candidate. SGLang
0.5.14 currently rejects it during weight construction because a 48-wide Gated
DeltaNet projection is incompatible with the detected 128-wide FP8 block. That
runtime incompatibility must be resolved without substituting a third-party
checkpoint.

This profile is offered only by the isolated
`runtime/sglang-nvfp4fix/available.toml` catalog. The original FP8 runtime and
catalog remain unchanged.

- chat completion;
- tool calling;
- grammar-constrained structured output;
- three concurrent primary requests at the intended request shape.

`mem_fraction_static = 0.60` produced an 827,041-token pool during the
qualification capacity test. The deployed profile subsequently caps the shared
pool at 524,288 tokens, preserving three running-request slots while reserving
memory for the co-resident helper.

The explicit `identity.model_dir` records the `HF_HOME=/cache` layout produced
by Hugging Face Hub 1.21, where snapshots live below `hub/`.

See `evidence/2026-07-03.md` for the launch, capability, concurrency, memory,
and cancellation evidence.
