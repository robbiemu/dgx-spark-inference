# DFlash experimental bundle — `qwen36-27b-fp8-dflash`

> ⚠️ **EXPERIMENTAL — not a production candidate for any role.**
>
> This bundle is included as a runnable *experiment* with a validated speedup. It
> is **not** offered as a production candidate for the `agentic` role: the runtime
> catalog (`runtime/sglang/available.toml[agentic]`) does not list it, and it
> fails capability matching for `agentic` because it lacks grammar-constrained
> `structured_output` (proven by `tests/test_dflash_rejected_for_agentic.py`).
>
> There is **no `--allow-incompatible` bypass**, ever. To run DFlash you use the
> honestly-labeled experimental launcher (`experiments/dflash/run-experimental.sh`),
> which is deliberately separate from the production `inferencectl use` path.

## What this bundle is

A coordinated speculative-decoding bundle: the `qwen36-27b-fp8` target served by
SGLang with a `qwen36-27b-dflash` drafter under the `DFLASH` algorithm. The
adapter launches it via the bundle branch of `runtime/sglang/adapter.sh`. See
`sglang.toml` for the coordination parameters.

## Experimental status

> DFlash passed its short-context quality and throughput gates, including a
> measured 1.61× final-answer throughput improvement. Full-context 262144 memory
> fit has not yet been validated because that disruptive hardware test has not
> been scheduled. Automatic idle eviction is a separate platform roadmap item.

## Results summary (sanitized)

Methodology and outcomes from the experiment (no raw evidence, no prompts, no
machine-specific artifacts are published):

| Gate | What | Result |
|---|---|---|
| 1 | DFlash CLI + internals | pass |
| 2 | load (at 32768 ctx; 262144 fit is the open load experiment) | pass |
| 3 | health | pass |
| 4 | coherent completion | pass |
| 5a | structured output | **limitation**: grammar-constrained HTTP 400; prompt-only pass |
| 5b | tool call | pass |
| 6 | multi-turn | pass |
| 7a | token equivalence | strong pass (50/50 first tokens identical, 80/100) |
| 7b | throughput | **pass: DFlash 11.99 vs baseline 7.44 tok/s (1.61×)** |
| 8 | end-to-end | pass (both correct); reasoning-depth divergence noted |

## Known limitations

- **Grammar-constrained structured output: unsupported under DFLASH** (HTTP 400
  for `response_format`/grammar). Prompt-only JSON works. This is why DFlash is
  not a production candidate for `agentic`.
- **Logprobs: unsupported under DFLASH** (`return_logprob` HTTP 400).
- **262144 memory fit: UNVALIDATED.** Requires the load experiment before any
  live binding.
- **Reasoning-depth divergence:** DFlash reasoned ~4.6× more than baseline for
  the same seeded prompt (gate 8); both answers were correct.

## Fetch the drafter (one-time, per host)

The bundle adapter mounts the drafter from `MODEL_CACHE_ROOT` read-only. The
drafter resolves the same way the baseline does: by default from the Hugging Face
cache layout derived from its `source_repository` + `source_revision` (set
`model_dir` in the drafter's `sglang.toml` for a `--local-dir` plain directory).
Fetch via the cache:

```bash
export MODEL_CACHE_ROOT=/srv/model-cache
export HF_HOME="$MODEL_CACHE_ROOT"
hf download z-lab/Qwen3.6-27B-DFlash \
  --revision 0919688658996800f86b895034249700e9481106
```

The target weights are the baseline profile (`profiles/qwen36-27b-fp8/`).

## How to run it

Use the experimental launcher (`experiments/dflash/run-experimental.sh`), which
defaults to the validated **32768** context, requires an explicit dangerous
override for 262144, and refuses to launch while the production service or its
container is active. Do not point it at the production port or container.
