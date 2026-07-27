# Laguna S 2.1 NVFP4 + DFlash experimental bundle

This bundle pairs `poolside/Laguna-S-2.1-NVFP4` with its matched
`poolside/Laguna-S-2.1-DFlash-NVFP4` drafter. It is isolated from the production
candidate catalog and defaults to a 32768-token context for the first DGX Spark
fit and quality trial.

The launch settings follow Poolside's model card and SGLang's merged Laguna
DFlash guidance and the DGX Spark qualification:

- a 16-token DFlash block: 15 proposed draft tokens plus the verifier bonus
- `flashinfer` target attention for single-GPU GB10 (SM121); the upstream
  `trtllm_mha` prefill kernel currently accepts SM100 only
- page size 1 and `mem-fraction-static=0.70`
- one global BF16 KV cache for both target and drafter; SGLang 0.5.15.post1
  cannot express target FP8 KV with drafter BF16 KV
- Poolside reasoning and tool-call parsers

## Current validation status

Keep this bundle in the `agentic-experimental` role. At 32768 tokens on one DGX
Spark, the clean runtime passed clean target-only-versus-DFlash output-set parity,
repeated tool-call selection, tool-result continuation, cached-prefix reuse, and
a 1498-token prompt. Weather was byte-identical. The multiply probe had three
valid greedy variants, and all three appeared identically in the target-only and
DFlash controls, including after cache flush. Every measured DFlash run proposed
exactly 15 tokens per verification call and accepted draft tokens. CUDA graphs
and overlap scheduling were enabled.

SGLang's DFlash scheduler rejects grammar-constrained structured output and
`return_logprob`, so this bundle deliberately does not claim either capability.
The model's 262144 context remains unqualified. The runtime is also a documented
compatibility override: SGLang 0.5.15.post1 declares FlashInfer 0.6.12, while the
qualified image uses a matched 0.6.15.post1 Python/cubin/JIT-cache triplet plus
the Laguna per-attention-group query-head and sliding-window planning fix.

The production runtime manifest does not reference this image. The isolated
launcher pins it independently:

```bash
docker build \
  --file runtime/sglang/Dockerfile.laguna-flashinfer-0.6.15-post1-clean.experimental \
  --tag local/sglang-runtime:v0.5.15.post1-laguna-flashinfer0.6.15.post1-clean-experimental \
  runtime/sglang
docker image inspect \
  local/sglang-runtime:v0.5.15.post1-laguna-flashinfer0.6.15.post1-clean-experimental \
  --format '{{.Id}}'
```

The required image ID is
`sha256:50ce969dd51c2c95575e5417b4509dbca27289b38389b550839939b64875684f`.

### TRTLLM MHA SM121 control

TP=1 does permit TRTLLM MHA decode on SM121. A split control using FlashInfer
prefill and TRTLLM MHA decode loaded and reached HTTP readiness, but retained
the same prompt substitutions. A reproduction-only image then widened
SGLang's prefill guard from exact SM100 to SM100-or-SM120. Full TRTLLM MHA
reached CUDA-graph initialization, where FlashInfer's `TllmGenFmhaRunner`
failed with `Unsupported architecture`. This is a kernel-level SM121 rejection,
not the TP>=4 constraint, so the guard override is not a viable serving path.

Fetch both pinned snapshots into a writable Hugging Face cache, then launch from
the repository checkout:

```bash
export MODEL_CACHE_ROOT="$HOME/.cache/huggingface/hub"
export PROJECT_ROOT="$HOME/projects/dgx-spark-inference"
export DFLASH_BUNDLE_ID="laguna-s-2.1-nvfp4-dflash"
export DFLASH_SERVED_NAME="laguna-s-2.1-agentic-experimental"
export DFLASH_ROLE="agentic-experimental"
export SGLANG_API_KEY="$(openssl rand -hex 32)"
experiments/dflash/run-experimental.sh
```

The launcher refuses contexts other than 32768. Qualify a larger context in a
separate guarded workflow before changing that boundary.
