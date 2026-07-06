# Measuring a model's memory budget — enrolling the planner for a new profile

The memory planner (`tools/memory_planner/resolve_memory_plan.py`) derives
`mem_fraction_static` and `max_total_tokens` dynamically at launch time,
accounting for co-resident models. To use it, you need a **budget ledger**
entry for each model — and that entry must be **measured**, not guessed.

This tutorial shows how to measure a model's budget from a sglang startup log
and enroll the planner. It takes one restart cycle per model.

> **When to do this:** when adding a new model/profile, changing quantization,
> changing KV dtype, or updating sglang version. The `kv_bytes_per_token` field
> is not a timeless constant — re-measure if any of these change.

## Prerequisites

- The model loads and generates correctly (do this *after* the profile is
  qualified for correctness, not before).
- A few minutes of GPU time for one launch with `--log-level info`.
- The measurement tool: `tools/memory_planner/measure_model_budget.py`.

## Step 1 — Capture an info-level startup log

Launch the model once with `--log-level info` and capture the full startup
output. The detailed memory lines (`Load weight end`, `KV Cache is allocated`,
`Mamba Cache is allocated`, CUDA graph capture) only appear at `info` level.

If your model is managed by the dispatch system, temporarily set `log_level` to
`"info"` in the runtime manifest's `[common_launch]`, restart, capture, then
revert:

```bash
# Temporarily enable info logging (in the installed manifest)
sudo sed -i 's/log_level         = "warning"/log_level         = "info"/' \
  /usr/local/lib/dgx-spark-inference/runtime/sglang/runtime-manifest.toml

# Restart and wait for health
sudo systemctl restart dgx-spark-inference.service
# ... wait for the service to come up ...

# Capture the full startup log
docker logs inference-agentic > /tmp/model_info.log 2>&1

# Revert logging
sudo sed -i 's/log_level         = "info"/log_level         = "warning"/' \
  /usr/local/lib/dgx-spark-inference/runtime/sglang/runtime-manifest.toml
```

If launching manually (not via the dispatch system), pipe sglang's output
directly:

```bash
python3 -m sglang.launch_server ... --log-level info 2>&1 | tee /tmp/model_info.log
```

## Step 2 — Run the measurement tool

```bash
python3 tools/memory_planner/measure_model_budget.py \
  --log /tmp/model_info.log \
  --model-id my-model-id \
  --mem-fraction 0.60 \
  > /tmp/new_profile.toml
```

The tool prints a **derivation trace** to stderr (every field with its source
log line) and a TOML `[[profiles]]` entry to stdout. Review the trace:

```
============================================================
DERIVATION TRACE for my-model-id
(mem_fraction_static=0.6)
============================================================
  weights_gib                    = 27.57
    ← 22.11 (Qwen3_5ForConditionalGeneration); 5.46 (MTP draft)
  target_kv_tokens               = 313966
    ← KV Cache is allocated. #tokens: 313966, K size: ... V size: ...
  kv_bytes_per_token             = 32673.8
    ← (K_size + V_size) GiB / tokens
  mamba_cache_gib                = 5.56
    ← Mamba Cache is allocated. ssm_state: 3.23GB ...
  static_overhead_gib            = 5.56
    ← (0.6 × A_preload) - (weights + kv + pad)
============================================================
```

### Fields to verify

- **`weights_gib`**: should match the model's known weight size. If the model
  loads in multiple passes (e.g. main + MTP draft), the tool sums them — check
  the trace shows each.
- **`kv_bytes_per_token`**: for hybrid attention models (GDN/Mamba), this is
  smaller than dense models because only full-attention layers hold KV. Verify
  it matches `(num_full_attn_layers × 2 × num_kv_heads × head_dim × dtype_size)`.
- **`static_overhead_gib`**: the measured gap between the fraction budget and
  the clean components. For large models this is substantial (CUDA graph
  capture, allocator fragmentation). **If it's negative**, the fraction is too
  low — see Troubleshooting below.
- **`cuda_graph_peak_gib`**: the transient graph-capture peak. Should be small
  relative to weights.

### Adjusting the output

The tool emits sensible defaults for `static_pad_gib` (0.5),
`request_workspace_gib` (2.0), and `gpu_headroom_gib` (1.0). Override them
with `--static-pad`, `--request-workspace`, `--gpu-headroom` if your
measurements differ.

Set `--minimum-pool` to the role's contract floor (the minimum pool the role
requires to function — e.g. 262144 for a full-context primary).

## Step 3 — Add the entry to the budget ledger

Review the generated entry, adjust if needed, then append:

```bash
cat /tmp/new_profile.toml >> tools/memory_planner/budget_ledger.toml
```

Verify it parses and the resolver can find it:

```bash
python3 tools/memory_planner/resolve_memory_plan.py \
  tools/memory_planner/budget_ledger.toml \
  <(echo 'device.total_gib = 121.7
[policy]
memavailable_floor_gib = 8.0
[observed]
gpu_free_now_gib = 121.7
memavailable_now_gib = 121.7
[[admit]]
role = "agentic"
model_id = "my-model-id"')
```

Expect `ADMIT` with a derived `mem_fraction_static` and `max_total_tokens`.

## Step 4 — Remove the hardcoded fraction from the profile

If the profile has `mem_fraction_static` hardcoded, remove it. The planner
sets it dynamically via the `DGX_MEM_FRACTION_STATIC` environment variable at
launch time. A hardcoded value takes precedence and defeats the planner.

```bash
# In profiles/my-model-id/sglang.toml, remove or comment out:
# mem_fraction_static = 0.60
```

## Step 5 — Enroll the planner (place the ledger + plan in /etc)

The planner activates when it finds a **matched pair** in `CONFIG_ROOT`:
`memory_ledger.toml` + `memory_plan.toml`. Without both, admission runs in
"legacy launch" mode (no preflight).

```bash
# The budget ledger (operator state — copy from the repo)
sudo install -m 0640 tools/memory_planner/budget_ledger.toml \
  /etc/dgx-spark-inference/memory_ledger.toml

# The memory plan (operator state — write once, device-specific)
sudo tee /etc/dgx-spark-inference/memory_plan.toml > /dev/null << 'EOF'
device.total_gib = 121.7   # your device total (check: python3 -c "import torch; print(torch.cuda.mem_get_info()[1]/(1<<30))")
[policy]
memavailable_floor_gib = 8.0   # host MemAvailable safety floor
EOF
```

## Step 6 — Restart and verify

```bash
sudo systemctl restart dgx-spark-inference.service
```

Check the journal — admission should say "resolving memory plan" (not "legacy
launch"):

```bash
journalctl -eu dgx-spark-inference.service --since "5 min ago" | grep admission
# expect: [admission] resolving memory plan ...
#         [admission] admitted: mem_fraction_static=0.XXXX max_total_tokens=YYYYYY
#         [admission] allocation verified: realized=YYYYYY tokens
```

Verify the realized pool:

```bash
curl -s http://127.0.0.1:30000/get_server_info \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['max_total_num_tokens'])"
```

The realized pool should be within `[minimum_admissible_pool_tokens,
max_total_tokens]`. If it falls outside, admission kills the adapter — see
Troubleshooting.

## The `fraction_base` parameter

The resolver derives `mem_fraction_static = static_required / base`, where the
base depends on how sglang interprets the fraction for a given model. Most
models use the default:

```toml
fraction_base = "a_preload"   # default — derived against available GPU memory
```

If a model's realized pool consistently falls short of the target despite the
fraction being correct, try `device_total`:

```toml
fraction_base = "device_total"  # derived against full device total
```

**How to tell which base is correct:** the `a_preload` base is correct when
sglang's effective budget tracks the available (post-co-resident) memory. The
`device_total` base is correct when sglang applies the fraction against the
full device regardless of co-residents. When in doubt, use `a_preload` (the
default) — it has been verified on hybrid GDN models with large allocator
overhead.

## Troubleshooting

### Realized pool is much smaller than the target

The `static_overhead_gib` may be too low. The overhead captures sglang's
allocator behavior (CUDA graph capture, fragmentation) which is
model-dependent and can be large for hybrid attention models. Check the
realized VRAM usage (`nvidia-smi`) vs the budget the fraction implies.

### `static_overhead_gib` computes negative

The `mem_fraction` used during measurement was too low for the model's
overhead. Re-run the measurement with a higher `--mem-fraction` (e.g. 0.85)
and use that log.

### Admission refuses ("a gate failed")

The model doesn't fit in the available GPU memory alongside co-residents.
Either lower `target_kv_tokens`, reduce the co-resident's footprint, or run
solo. The resolver's output shows exactly which gate failed and the numbers.

### `kv_bytes_per_token` seems wrong

For hybrid attention models (GDN/Mamba), only full-attention layers hold KV
cache. The per-token cost is `2 × num_full_attn_layers × num_kv_heads ×
head_dim × dtype_size`, NOT `2 × total_layers × ...`. The measurement tool
derives it from the actual KV cache allocation, which is correct — but verify
against the architecture if it seems surprisingly small.

## See also

- `config/schemas/memory-plan.md` — field definitions and the overhead model
- `docs/v0_2_phase0_results.md` — the Phase 0 measurement methodology
- `tools/memory_planner/resolve_memory_plan.py` — the resolver (run with no
  args for usage)
