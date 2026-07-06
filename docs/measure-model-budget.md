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

## Variables

Set these to match your deployment. The defaults below are the reference
deployment's primary (`agentic`) role; adjust for your role/unit.

```bash
# The systemd unit, container, and port for the role you are measuring.
UNIT=dgx-spark-inference.service
CONTAINER_NAME=inference-agentic
PORT=30000

# The installed runtime manifest (where log_level lives).
MANIFEST=/usr/local/lib/dgx-spark-inference/runtime/sglang/runtime-manifest.toml

# Where this role keeps its API key (for the verification step).
SECRET_FILE=/etc/dgx-spark-inference/agent.env
TOKEN="$(sudo awk -F= '/^SGLANG_API_KEY=/{print $2; exit}' "$SECRET_FILE")"

# Output paths.
LOG=/tmp/model_info.log
ENTRY=/tmp/new_profile.toml
```

> **Discovering alternatives:** `systemctl list-units --type=service | grep
> inference` lists managed inference units. Each unit's `ExecStart` names the
> container and role. The manifest path is always under the install root's
> `runtime/sglang/runtime-manifest.toml`.

## Step 1 — Capture an info-level startup log

Launch the model once with `--log-level info` and capture the startup output.
The detailed memory lines (`Load weight end`, `KV Cache is allocated`,
`Mamba Cache is allocated`, CUDA graph capture) only appear at `info` level.

Use a backup-and-restore sequence so the temporary edit is always reverted,
then restart so the running process returns to its normal log level:

```bash
# Back up the manifest.
sudo cp "$MANIFEST" "${MANIFEST}.before-measurement"

# Temporarily enable info logging.
sudo sed -i 's/^log_level         = "warning"/log_level         = "info"/' "$MANIFEST"

# Restart and wait for health.
sudo systemctl restart "$UNIT"
# Poll until healthy (timeout 5 min):
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" \
    -H "Authorization: Bearer $TOKEN")
  [ "$code" = "200" ] && break
  sleep 5
done

# Capture THIS startup's log (not the container's full history).
docker logs "$CONTAINER_NAME" --since "$(systemctl show -p ActiveEnterTimestamp "$UNIT" | cut -d= -f2)" > "$LOG" 2>&1

# Restore the manifest and restart so the running process uses normal logging.
sudo mv "${MANIFEST}.before-measurement" "$MANIFEST"
sudo systemctl restart "$UNIT"
```

## Step 2 — Run the measurement tool

```bash
python3 tools/memory_planner/measure_model_budget.py \
  --log "$LOG" \
  --model-id my-model-id \
  --mem-fraction 0.60 \
  > "$ENTRY"
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
  low — re-run with a higher `--mem-fraction`.
- **`cuda_graph_peak_gib`**: the transient graph-capture peak (optional; may
  be 0 if graph capture is disabled).

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
cat "$ENTRY" >> tools/memory_planner/budget_ledger.toml
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

## Step 4 — Keep the profile value as a legacy fallback

**Do not remove `mem_fraction_static` from the profile.** The planner's
`DGX_MEM_FRACTION_STATIC` environment override (exported by `admission.sh`
when a planner pair is active) takes precedence over the profile value at
launch time. The profile value serves as the **validated fallback** when the
planner is not active (legacy/auto mode, or when `DGX_MEMORY_PREFLIGHT=off`).

The precedence chain:

```
DGX_MEM_FRACTION_STATIC (env, exported only by admission.sh's planner path)
  > profile launch.mem_fraction_static (the fallback)
  > key omitted → sglang computes its own default
```

Keep the profile's `mem_fraction_static` at its previously validated value.
If a launch must refuse rather than fall back without a planner decision, set
`DGX_MEMORY_PREFLIGHT=required` in the unit's environment.

## Step 5 — Enroll the planner (place the ledger + plan in /etc)

The planner activates when it finds a **matched pair** in `CONFIG_ROOT`:
`memory_ledger.toml` + `memory_plan.toml`. Without both, admission runs in
"legacy launch" mode (no preflight).

```bash
# The budget ledger (operator state — copy from the repo).
sudo install -m 0640 tools/memory_planner/budget_ledger.toml \
  /etc/dgx-spark-inference/memory_ledger.toml

# The memory plan (operator state — write once, device-specific).
sudo tee /etc/dgx-spark-inference/memory_plan.toml > /dev/null << 'EOF'
device.total_gib = 121.7   # your device total
[policy]
memavailable_floor_gib = 8.0   # host MemAvailable safety floor
EOF
```

## Step 6 — Restart and verify

```bash
sudo systemctl restart "$UNIT"
```

Check the journal — admission should say "resolving memory plan" (not "legacy
launch"):

```bash
journalctl -eu "$UNIT" --since "5 min ago" | grep admission
# expect: [admission] resolving memory plan ...
#         [admission] admitted: mem_fraction_static=0.XXXX max_total_tokens=YYYYYY
#         [admission] allocation verified: realized=YYYYYY tokens
```

Verify the realized pool:

```bash
curl -s "http://127.0.0.1:$PORT/get_server_info" \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['max_total_num_tokens'])"
```

The realized pool should be within `[minimum_admissible_pool_tokens,
max_total_tokens]`. If it falls outside, admission kills the adapter — see
Troubleshooting below.

## The `fraction_base` calibration

`fraction_base` determines which base the resolver derives
`mem_fraction_static` against. It is an **SGLang runtime-path calibration**,
not a model-intrinsic property — it depends on how sglang's allocator
interprets the fraction for a given model/runtime combination.

The default is `"a_preload"` (derived against available GPU memory at launch
time). This is correct for most models. To determine whether `"device_total"`
is needed for yours:

1. Keep `"a_preload"` (the default) and launch under a controlled, known
   residency state.
2. Compare the resolver's predicted `max_total_tokens` against the realized
   pool reported by `/get_server_info`.
3. Change to `"device_total"` only when repeated runs show that its prediction
   matches the realized allocation better than `"a_preload"`.
4. Re-validate under the intended co-resident state.

Do **not** swap bases based on a single symptom (e.g. "pool was small"). The
relationship runs both ways: assuming `a_preload` when sglang uses
`device_total` makes the fraction too large; assuming `device_total` when
sglang uses `a_preload` makes it too small. Calibrate by comparing predicted
versus realized across multiple runs.

## Troubleshooting

### Multiple KV pools (composite KV allocation)

If the measurement tool reports "multiple KV pools found," it refuses to emit
a TOML entry. **Do not enroll this profile in planner mode.** The current
ledger schema and resolver do not support composite KV allocation (e.g. MTP
with separate target + draft pools). Keep the profile on the validated legacy
path (hardcoded `mem_fraction_static`) until explicit multi-pool accounting
is implemented.

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

- `config/schemas/memory-plan.md` — field definitions, the overhead model, and
  the `fraction_base` parameter
- `docs/v0_2_phase0_results.md` — the Phase 0 measurement methodology
- `tools/memory_planner/resolve_memory_plan.py` — the resolver (run with no
  args for usage)
- `tools/memory_planner/measure_model_budget.py` — the measurement tool
