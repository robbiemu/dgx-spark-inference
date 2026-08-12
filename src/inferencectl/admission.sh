#!/usr/bin/env bash
# admission.sh — serialized admission wrapper for the v0.2 memory preflight.
#
# WHY THIS EXISTS (the correctness fix): a naive preflight that samples free
# memory, then execs the adapter, has a RACE WINDOW between "preflight passes"
# and "allocation committed". Two roles starting near-simultaneously both see
# the same free-memory snapshot and both pass — recreating the over-admission
# the resolver exists to prevent. This wrapper holds a global lock across the
# whole window: discover residents -> sample memory -> resolve -> launch ->
# VERIFY ALLOCATION COMMITTED -> release lock. Two concurrent dispatchers cannot
# both pass while the first candidate is between preflight and allocation.
#
# LAUNCH PATH (v0.2): systemd -> dispatch.sh -> admission.sh -> adapter -> docker
# dispatch.sh execs THIS instead of the adapter directly when the memory preflight
# is enrolled (DGX_MEMORY_PREFLIGHT). Legacy v0.1 roles (no planner pair) still
# exec the adapter directly from dispatch.sh — this wrapper is only on the path
# when enrollment is active.
#
# INVOCATION (same 7 args dispatch.sh passes the adapter, plus it knows how to
# reach the resolver/ledger/plan via CONFIG_ROOT/PROJECT_ROOT):
#   admission.sh <ROLE> <RUNTIME_ROOT> <PROJECT_ROOT> <MODEL_ID> <KIND> <SPEC> <SERVED> <ADAPTER>
#
# The 8th arg (ADAPTER abs path) is added by dispatch.sh so this wrapper can exec
# the real adapter after admission, without re-deriving its path.
#
# TYPE=simple implication: systemd tracks THIS process's PID as the service main
# process. So after admission we must STAY ALIVE supervising the adapter child
# (we exec->wait on it), not exit — otherwise systemd marks the unit failed.
set -Eeuo pipefail

# ---- args ------------------------------------------------------------------
ROLE="$1"; RUNTIME_ROOT="$2"; PROJECT_ROOT="$3"; MODEL_ID="$4"
KIND="$5"; SPEC="$6"; SERVED="$7"; ADAPTER="$8"

# Preserve per-unit slot identity before loading shared primary defaults.
CALLER_ROLE="$ROLE"
CALLER_PORT="${PORT:-}"
CALLER_CONTAINER_NAME="${CONTAINER_NAME:-}"

CONFIG_ROOT="${CONFIG_ROOT:-/etc/dgx-spark-inference}"
[ -f "$CONFIG_ROOT/inference.env" ] && . "$CONFIG_ROOT/inference.env"

# Per-unit values win over inference.env's primary-slot defaults.
ROLE="$CALLER_ROLE"
[ -n "$CALLER_PORT" ] && PORT="$CALLER_PORT"
[ -n "$CALLER_CONTAINER_NAME" ] && CONTAINER_NAME="$CALLER_CONTAINER_NAME"
export ROLE PORT CONTAINER_NAME

# Resolve the memory planner (repo-shipped; installed alongside the adapter).
PLANNER="${DGX_MEMORY_PLANNER:-$PROJECT_ROOT/tools/memory_planner/resolve_memory_plan.py}"
LEDGER="${DGX_MEMORY_LEDGER:-$CONFIG_ROOT/memory_ledger.toml}"
PLAN="${DGX_MEMORY_PLAN:-$CONFIG_ROOT/memory_plan.toml}"
ACTIVE_MODELS="${ACTIVE_MODELS:-$CONFIG_ROOT/active-models.toml}"
PREFLIGHT="${DGX_MEMORY_PREFLIGHT:-auto}"   # auto | required | off
# FLOOR is resolved AFTER the pair check (Blocker 3): layered
#   DGX_MEMAVAILABLE_FLOOR_GIB (env) > installed memory_plan.toml [policy] > default 6.0
LOCK="${DGX_ADMISSION_LOCK:-/run/dgx-inference-admission.lock}"
JOINT_STATE="${DGX_MEMORY_PLAN_STATE:-/run/dgx-inference-memory-plan.json}"
DROP_CACHES_PATH="${DGX_DROP_CACHES_PATH:-/proc/sys/vm/drop_caches}"
PORT="${PORT:-30000}"
ADMISSION_READY_TIMEOUT="${DGX_ADMISSION_READY_TIMEOUT:-900}"  # sec to verify allocation

die() { echo "ERROR: REFUSING: $*" >&2; exit 75; }   # 75 = deliberate refusal (EX_TEMPFAIL)
log() { echo "[admission] $*"; }

case "$ADMISSION_READY_TIMEOUT" in
  ''|*[!0-9]*) die "DGX_ADMISSION_READY_TIMEOUT must be a positive integer" ;;
esac
[ "$ADMISSION_READY_TIMEOUT" -gt 0 ] \
  || die "DGX_ADMISSION_READY_TIMEOUT must be a positive integer"

# ---- enrollment: decide whether to run the preflight at all ----------------
# auto  : run only if a matched planner pair exists in CONFIG_ROOT; else legacy.
# required : always run; missing pair / probe failure -> REFUSE (fail-closed).
# off   : skip entirely (explicit manual bypass; loud warning).
if [ "$PREFLIGHT" = "off" ]; then
  log "WARN: DGX_MEMORY_PREFLIGHT=off — bypassing memory preflight (manual override)"
  exec "$ADAPTER" "$ROLE" "$RUNTIME_ROOT" "$PROJECT_ROOT" "$MODEL_ID" "$KIND" "$SPEC" "$SERVED"
fi

# Matched-pair check (atomic — never mix CONFIG_ROOT file with a repo copy).
# ledger + plan are a MATCHED PAIR: both present (use both), both absent (legacy
# in auto / refuse in required), or exactly-one present (REFUSE in BOTH modes —
# a lone file signals a half-edited deployment and must never be silently paired
# with a repo copy of the other, which could be from a different schema generation).
has_ledger=0; has_plan=0
[ -f "$LEDGER" ] && has_ledger=1
[ -f "$PLAN" ] && has_plan=1
has_pair=0; [ "$has_ledger" = "1" ] && [ "$has_plan" = "1" ] && has_pair=1
if [ "$has_pair" = "0" ]; then
  if [ "$has_ledger" = "1" ] && [ "$has_plan" = "0" ]; then
    die "ledger present but plan missing (refuse; never mix roots / pair a lone file)"
  fi
  if [ "$has_plan" = "1" ] && [ "$has_ledger" = "0" ]; then
    die "plan present but ledger missing (refuse; never mix roots / pair a lone file)"
  fi
  # neither present.
  if [ "$PREFLIGHT" = "required" ]; then
    die "managed mode: no planner pair at CONFIG_ROOT (need $LEDGER + $PLAN)"
  fi
  log "auto mode: no planner pair — legacy launch (no preflight)"
  exec "$ADAPTER" "$ROLE" "$RUNTIME_ROOT" "$PROJECT_ROOT" "$MODEL_ID" "$KIND" "$SPEC" "$SERVED"
fi

[ -x "$PLANNER" ] || [ -f "$PLANNER" ] || die "planner not found: $PLANNER"
[ -f "$ACTIVE_MODELS" ] || die "managed mode: active-models topology missing: $ACTIVE_MODELS"

# The caller must be exactly one of the configured active slots.  The joint
# planner reads every slot from this same file; no role count or role name is
# embedded in the allocator.
python3 - "$ACTIVE_MODELS" "$ROLE" "$MODEL_ID" <<'PY' \
  || die "caller '$ROLE/$MODEL_ID' does not match active-models topology"
import sys, tomllib
path, role, model = sys.argv[1:4]
active = tomllib.load(open(path, "rb")).get("active", {})
slot = active.get(role)
if not isinstance(slot, dict) or slot.get("model_id") != model:
    sys.exit(1)
if not active:
    sys.exit(1)
PY

# ---- resolve the MemAvailable floor (Blocker 3: layered, honors installed plan) -
# Layering: DGX_MEMAVAILABLE_FLOOR_GIB (env override) > installed memory_plan.toml
# [policy].memavailable_floor_gib > default 6.0. An operator's configured floor in
# the installed plan must NOT be silently replaced by the default.
resolve_floor() {
  local plan_floor
  plan_floor="$(python3 - "$PLAN" <<'PY'
import sys, tomllib
try:
    p = tomllib.load(open(sys.argv[1],"rb"))
    v = p.get("policy", {}).get("memavailable_floor_gib")
    if v is None: sys.exit(1)
    f = float(v)
    if not (f == f and f > 0):  # NaN or non-positive -> invalid
        sys.exit(2)
    print(f)
except Exception:
    sys.exit(1)
PY
)" || plan_floor=""
  if [ -n "${DGX_MEMAVAILABLE_FLOOR_GIB:-}" ]; then
    printf '%s' "$DGX_MEMAVAILABLE_FLOOR_GIB"
  elif [ -n "$plan_floor" ]; then
    printf '%s' "$plan_floor"
  else
    printf '6.0'
  fi
}
FLOOR="$(resolve_floor)"
# validate the resolved floor is a finite positive number before it reaches TOML.
python3 -c "f=float('$FLOOR'); assert f==f and f>0" 2>/dev/null \
  || die "resolved memavailable_floor_gib is not a finite positive number: '$FLOOR'"

# GB10 uses one physical memory pool for Linux and CUDA. Loading a model also
# populates clean file-backed page cache (checkpoint shards, container layers,
# shared libraries). MemAvailable counts those pages as reclaimable, while
# torch.cuda.mem_get_info() reports only immediately free pages. Without an
# explicit reclaim between serialized launches, the CUDA probe can therefore
# report only a few GiB even though tens of GiB are safely reclaimable, and a
# valid co-resident model is refused before it reaches Docker.
#
# This is an explicit host policy because dropping the host page cache is not
# appropriate on every CUDA system. It is performed under the admission lock,
# before every live probe, so both the cold joint plan and later fractions use
# the same reclaimable-memory view. The Linux MemAvailable floor remains the
# hard safety gate after reclaim.
resolve_page_cache_reclaim() {
  python3 - "$PLAN" <<'PY'
import sys, tomllib
try:
    value = tomllib.load(open(sys.argv[1], "rb")).get("policy", {}).get(
        "reclaim_page_cache_before_probe", False
    )
except Exception:
    raise SystemExit(1)
if not isinstance(value, bool):
    raise SystemExit(2)
print("true" if value else "false")
PY
}
RECLAIM_PAGE_CACHE="$(resolve_page_cache_reclaim)" \
  || die "memory plan has invalid policy.reclaim_page_cache_before_probe (must be true or false)"

# ---- the serialized admission lock -----------------------------------------
# Hold across discover->sample->resolve->launch->VERIFY. flock is released when
# the holding fd closes (on exec-via-wait or exit). We open it on fd 9.
exec 9>"$LOCK"
log "acquiring admission lock ($LOCK)..."
flock 9
log "lock held"

if [ "$RECLAIM_PAGE_CACHE" = "true" ]; then
  memfree_before_kib="$(awk '/^MemFree:/ {print $2}' /proc/meminfo)"
  memavail_before_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  sync
  printf '3\n' > "$DROP_CACHES_PATH" \
    || die "cannot reclaim page cache through $DROP_CACHES_PATH"
  memfree_after_kib="$(awk '/^MemFree:/ {print $2}' /proc/meminfo)"
  log "reclaimed clean page cache before GPU probe (MemFree ${memfree_before_kib}KiB -> ${memfree_after_kib}KiB; MemAvailable ${memavail_before_kib}KiB)"
fi

# ---- discover residents (label-based, for guards only) ---------------------
# In LIVE mode (Blocker 1) residents are NOT subtracted from A_preload — the
# measured gpu_free_now_gib already includes them. Discovery remains for:
# identity/revision checks and the unmanaged-GPU-tenant guard.
EXPECTED_REV=""
RESIDENT_LINES=""
if command -v docker >/dev/null 2>&1; then
  # Ledger-revision check: export DGX_MEMORY_LEDGER so the adapter stamps the
  # io.inferencectl.ledger_revision label; here we compute the expected revision
  # and refuse in required mode if any resident's revision is absent/mismatched
  # (a stale resident from a different ledger generation would invalidate the plan).
  [ -f "$LEDGER" ] && EXPECTED_REV="$(sha256sum "$LEDGER" 2>/dev/null | cut -c1-16)"
  export DGX_MEMORY_LEDGER="$LEDGER"   # so the adapter labels the new container
  RESIDENT_LINES="$(docker ps --filter "label=io.inferencectl.managed=true" \
      --format '{{.Label "io.inferencectl.role"}}|{{.Label "io.inferencectl.memory_profile"}}|{{.Label "io.inferencectl.ledger_revision"}}' \
      2>/dev/null || true)"
  # Every running managed container must be a configured role/model pair.  A
  # stale or experimental managed tenant cannot silently consume part of a plan
  # computed for a different topology.
  resident_error="$(DGX_ACTIVE_MODELS="$ACTIVE_MODELS" DGX_RESIDENT_LINES="$RESIDENT_LINES" python3 <<'PY'
import os, tomllib
active = tomllib.load(open(os.environ["DGX_ACTIVE_MODELS"], "rb")).get("active", {})
seen = set()
errors = []
for line in os.environ.get("DGX_RESIDENT_LINES", "").splitlines():
    if not line.strip():
        continue
    role, model, _revision = (line.split("|", 2) + ["", ""])[:3]
    if role in seen:
        errors.append(f"duplicate managed resident role {role!r}")
    seen.add(role)
    configured = active.get(role, {}).get("model_id")
    if configured != model:
        errors.append(
            f"resident {role!r}/{model!r} is not configured "
            f"(active model is {configured!r})"
        )
print("; ".join(errors))
PY
)"
  [ -z "$resident_error" ] || die "managed resident topology mismatch: $resident_error"
  if [ -n "$EXPECTED_REV" ] && [ "$PREFLIGHT" = "required" ]; then
    bad_rev="$(printf '%s\n' "$RESIDENT_LINES" \
      | awk -F'|' -v expected_rev="$EXPECTED_REV" \
          'NF && $3 != expected_rev {print "mismatch"}')"
    if [ -n "$bad_rev" ]; then
      die "managed mode: a resident's ledger_revision differs from the current ledger (stale resident; restart it under the new ledger)"
    fi
  fi
  # Unmanaged GPU-tenant guard (required mode): inspect DeviceRequests (the real
  # GPU allocation) — only containers that actually request a GPU and are NOT
  # io.inferencectl.managed are refused. A CPU-only sidecar/monitor does not
  # consume GPU memory and is already accounted for in MemAvailable.
  if [ "$PREFLIGHT" = "required" ]; then
    gpu_unmanaged=$( { for c in $(docker ps --format '{{.Names}}'); do
          dr=$(docker inspect "$c" --format '{{json .HostConfig.DeviceRequests}}' 2>/dev/null || echo "null")
          managed=$(docker inspect "$c" --format '{{index .Config.Labels "io.inferencectl.managed"}}' 2>/dev/null || true)
          # DeviceRequests non-null/non-empty AND not managed -> unmanaged GPU tenant
          if [ "$dr" != "null" ] && [ -n "$dr" ] && [ "$dr" != "[]" ] && [ "$managed" != "true" ]; then
            echo "$c"
          fi
        done; } 2>/dev/null || true)
    if [ -n "$gpu_unmanaged" ]; then
      die "managed mode: unmanaged GPU container(s) present ($(echo "$gpu_unmanaged" | tr '\n' ' ')); cannot reason about unaccounted GPU memory"
    fi
  fi
fi

RESIDENT_COUNT="$(printf '%s\n' "$RESIDENT_LINES" | awk 'NF {n++} END {print n+0}')"

# ---- sample live free memory (GPU via torch.cuda.mem_get_info + Linux floor)-
# GPU probe via a throwaway container (verified working). In required mode a
# probe failure REFUSES (fail-closed): /proc/meminfo alone cannot derive the
# fraction SGLang will use, so silently downgrading is fail-open — rejected.
probe_gpu() {  # prints "FREE_GIB TOTAL_GIB" or returns nonzero
  IMAGE="$(python3 - "$RUNTIME_ROOT/runtime-manifest.toml" <<'PY'
import sys, tomllib; print(tomllib.load(open(sys.argv[1],"rb"))["image"])
PY
)"
  docker run --rm --gpus all --entrypoint /bin/sh "$IMAGE" -c \
    'python3 -c "import torch; f,t=torch.cuda.mem_get_info(); print(\"FREE_GIB %.2f TOTAL_GIB %.2f\" % (f/1073741824, t/1073741824))"' \
    2>/dev/null | grep -E "FREE_GIB"
}
gpu_line="$(probe_gpu || true)"
if [ -z "$gpu_line" ]; then
  # Blocker 4: once a pair exists and preflight has begun, a GPU-probe failure
  # REFUSES in BOTH auto and required modes. Feeding the resolver a synthetic
  # total would look like success with an invented A_preload (fail-open). The
  # only mode difference is the no-pair case (handled above: auto legacy-launches).
  die "GPU free-memory probe failed (refuse; cannot derive fraction from a synthetic value)"
else
  gpu_free="$(echo "$gpu_line" | awk '{print $2}')"
  gpu_total="$(echo "$gpu_line" | awk '{print $4}')"
fi
memavail_kib="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
memavail_gib="$(python3 -c "print(${memavail_kib}/1048576)")"

# ---- resolve/reuse one cold joint topology plan ----------------------------
# The active-models file is the topology source.  On a cold start the resolver
# proves that ALL configured minimums fit, derives each growth weight from that
# model's target/floor, and spends the remaining usable memory jointly.  Its
# absolute token allocations are cached under /run.  Later serialized launches
# reuse those allocations but re-derive the fraction from their LIVE A_preload.
JOINT_REV="$(python3 - "$LEDGER" "$ACTIVE_MODELS" "$PLAN" <<'PY'
import hashlib, sys
h = hashlib.sha256()
for path in sys.argv[1:]:
    data = open(path, "rb").read()
    h.update(len(data).to_bytes(8, "big"))
    h.update(data)
print(h.hexdigest())
PY
)"

JOINT_PLAN_TMP="$(mktemp "${TMPDIR:-/tmp}/dgx-joint-plan.XXXXXX")"
CURRENT_PLAN_TMP="$(mktemp "${TMPDIR:-/tmp}/dgx-current-plan.XXXXXX")"
PLANNER_ERR="$(mktemp)"
trap 'rm -f "$JOINT_PLAN_TMP" "$CURRENT_PLAN_TMP" "$PLANNER_ERR"' EXIT

if [ "$RESIDENT_COUNT" -eq 0 ]; then
  cat > "$JOINT_PLAN_TMP" <<EOF
device.total_gib = ${gpu_total:-121.7}
[policy]
allocation_mode = "floor_weighted"
memavailable_floor_gib = ${FLOOR}
[observed]
gpu_free_now_gib = ${gpu_free}
memavailable_now_gib = ${memavail_gib}
EOF
  python3 - "$ACTIVE_MODELS" >> "$JOINT_PLAN_TMP" <<'PY'
import json, sys, tomllib
active = tomllib.load(open(sys.argv[1], "rb")).get("active", {})
if not active:
    raise SystemExit("active-models contains no [active.*] slots")
for role, slot in active.items():
    model = slot.get("model_id") if isinstance(slot, dict) else None
    if not model:
        raise SystemExit(f"active role {role!r} has no model_id")
    print("[[admit]]")
    print(f"role = {json.dumps(role)}")
    print(f"model_id = {json.dumps(model)}")
PY
  log "resolving cold joint plan for every configured model (floor=${FLOOR}G, memavail=${memavail_gib}G, gpu_free=${gpu_free}G)..."
  set +e
  JOINT_JSON="$(python3 "$PLANNER" "$LEDGER" "$JOINT_PLAN_TMP" --format json 2>"$PLANNER_ERR")"
  JOINT_RC=$?
  set -e
  [ "$JOINT_RC" -eq 0 ] || die "joint memory plan REFUSED; all configured minimums must fit before any model launches"
  STATE_TMP="$(mktemp "${JOINT_STATE}.tmp.XXXXXX")" \
    || die "cannot create joint-plan state beside $JOINT_STATE"
  DGX_JOINT_JSON="$JOINT_JSON" DGX_JOINT_REV="$JOINT_REV" \
    python3 > "$STATE_TMP" <<'PY'
import json, os
allocation = json.loads(os.environ["DGX_JOINT_JSON"])
if allocation.get("result") != "ADMIT":
    raise SystemExit("joint allocation is not ADMIT")
print(json.dumps({
    "schema_version": 1,
    "input_revision": os.environ["DGX_JOINT_REV"],
    "allocation": allocation,
}, indent=2))
PY
  chmod 0600 "$STATE_TMP"
  mv -f "$STATE_TMP" "$JOINT_STATE"
  log "joint plan committed: ${JOINT_STATE}"
else
  [ -f "$JOINT_STATE" ] \
    || die "managed residents exist but the cold joint-plan state is missing; coordinated cold reload required"
  log "reusing cold joint plan for ${RESIDENT_COUNT} configured resident(s)"
fi

# Select by role AND model.  Matching only model_id is ambiguous when a single
# profile is assigned to multiple configured slots.
set +e
ALLOCATED="$(DGX_JOINT_REV="$JOINT_REV" python3 - "$JOINT_STATE" "$ROLE" "$MODEL_ID" <<'PY'
import json, os, sys
state_path, role, model = sys.argv[1:4]
try:
    state = json.load(open(state_path))
except Exception:
    sys.exit(1)
if state.get("schema_version") != 1:
    sys.exit(2)
if state.get("input_revision") != os.environ["DGX_JOINT_REV"]:
    sys.exit(3)
allocation = state.get("allocation", {})
if allocation.get("result") != "ADMIT":
    sys.exit(4)
matches = [
    item for item in allocation.get("models", [])
    if item.get("role") == role and item.get("model_id") == model
]
if len(matches) != 1:
    sys.exit(5)
item = matches[0]
tokens = item.get("max_total_tokens")
minimum = item.get("minimum_admissible_pool_tokens")
if not isinstance(tokens, int) or tokens <= 0:
    sys.exit(6)
if not isinstance(minimum, int) or minimum <= 0 or tokens < minimum:
    sys.exit(7)
print(tokens, minimum)
PY
)"
ALLOC_RC=$?
set -e
[ "$ALLOC_RC" -eq 0 ] \
  || die "joint-plan state is stale or has no allocation for '$ROLE/$MODEL_ID' (rc=$ALLOC_RC); coordinated cold reload required"
ALLOCATED_TOKENS="$(printf '%s' "$ALLOCATED" | awk '{print $1}')"

# Re-run the ordinary single-slot gates against the current live measurements,
# pinning the token allocation chosen by the cold joint plan.  This is where the
# launch-time mem_fraction_static is derived; no fraction is stored or hardcoded.
cat > "$CURRENT_PLAN_TMP" <<EOF
device.total_gib = ${gpu_total:-121.7}
[policy]
allocation_mode = "fixed_targets"
memavailable_floor_gib = ${FLOOR}
[observed]
gpu_free_now_gib = ${gpu_free}
memavailable_now_gib = ${memavail_gib}
[[admit]]
role = "${ROLE}"
model_id = "${MODEL_ID}"
allocated_kv_tokens = ${ALLOCATED_TOKENS}
EOF
log "deriving live fraction for joint allocation (tokens=${ALLOCATED_TOKENS}, memavail=${memavail_gib}G, gpu_free=${gpu_free}G)..."
# Capture stdout and rc SEPARATELY (correction): the resolver exits nonzero for a
# valid REFUSE, and `|| JSON_OUT=""` would discard the structured JSON, losing the
# ability to distinguish an intentional gate failure from malformed output.
set +e
JSON_OUT="$(python3 "$PLANNER" "$LEDGER" "$CURRENT_PLAN_TMP" --format json 2>"$PLANNER_ERR")"
set -e
# parse the JSON for THIS model's derived knobs (stdlib; no grep on prose).
# JSON passed via env (DGX_PARSE_JSON), not stdin — a bash heredoc would consume
# stdin as the script source and break json.load.
read_knobs() {  # prints "FRACTION MAXTOKENS MINTOKENS"; exit 0 on success
  DGX_PARSE_JSON="$JSON_OUT" DGX_PARSE_ROLE="$ROLE" DGX_PARSE_MODEL="$MODEL_ID" python3 <<'PY'
import os, sys, json
role = os.environ["DGX_PARSE_ROLE"]
mid = os.environ["DGX_PARSE_MODEL"]
try:
    doc = json.loads(os.environ["DGX_PARSE_JSON"])
except Exception:
    sys.exit(1)
if doc.get("result") != "ADMIT":
    sys.exit(2)
for m in doc.get("models", []):
    if m.get("role") == role and m.get("model_id") == mid:
        f = m.get("mem_fraction_static"); mtt = m.get("max_total_tokens")
        if not (isinstance(f,(int,float)) and 0.0 < f < 1.0): sys.exit(3)
        if not (isinstance(mtt,int) and mtt > 0): sys.exit(3)
        print(f"{f} {mtt} {m.get('minimum_admissible_pool_tokens',0)}")
        sys.exit(0)
sys.exit(4)
PY
}
KNOBS="$(read_knobs)" || rc=$?
rc=${rc:-0}
if [ "$rc" != "0" ]; then
  case "$rc" in
    2) die "memory preflight REFUSED role '$ROLE' (a gate failed; co-residents untouched)";;
    *) die "memory preflight produced no valid admission for '$MODEL_ID' (rc=$rc)";;
  esac
fi
FRACTION="$(echo "$KNOBS" | awk '{print $1}')"
MAXTOKENS="$(echo "$KNOBS" | awk '{print $2}')"
MINTOKENS="$(echo "$KNOBS" | awk '{print $3}')"
log "admitted: mem_fraction_static=${FRACTION} max_total_tokens=${MAXTOKENS} min=${MINTOKENS}"

# ---- clear inherited overrides (dispatch is the only accepted source) ------
unset DGX_MEM_FRACTION_STATIC DGX_MAX_TOTAL_TOKENS
export DGX_MEM_FRACTION_STATIC="$FRACTION"
export DGX_MAX_TOTAL_TOKENS="$MAXTOKENS"

# ---- launch the adapter as a CHILD (retain control to verify allocation) ----
# The lock (fd 9) must stay held by THIS process across verify, but the adapter
# child must NOT inherit it — otherwise the child keeps the lock open after the
# parent releases its copy, deadlocking subsequent admissions. Close fd 9 in the
# subshell that launches the child (the child inherits the closed fd).
log "launching adapter (allocation verification pending)..."
( exec 9>&-; "$ADAPTER" "$ROLE" "$RUNTIME_ROOT" "$PROJECT_ROOT" "$MODEL_ID" "$KIND" "$SPEC" "$SERVED" ) &
ADAPTER_PID=$!

# ---- verify allocation committed before releasing the lock -----------------
# /health=200 is bare liveness (proven insufficient). The realized pool is behind
# 401-gated /get_server_info. Poll it (carrying the API key) and confirm the
# realized pool is >= the role's minimum AND <= the requested cap. Only then
# release the lock. systemd's EnvironmentFile gives us SGLANG_API_KEY.
verify_ready() {  # returns 0 when realized pool is within [min, cap]
  local i info realized
  for ((i=0; i<ADMISSION_READY_TIMEOUT; i+=5)); do
    # adapter child crashed? then it can never become ready.
    if ! kill -0 "$ADAPTER_PID" 2>/dev/null; then return 1; fi
    if curl -sf -o /dev/null "http://127.0.0.1:${PORT}/health" 2>/dev/null; then
      # liveness up — now verify realized capacity via the authenticated endpoint.
      info="$(curl -sf -H "Authorization: Bearer ${SGLANG_API_KEY:-}" \
                  "http://127.0.0.1:${PORT}/get_server_info" 2>/dev/null || true)"
      realized="$(printf '%s' "$info" | python3 -c '
import sys, json
try:
    d=json.load(sys.stdin); print(int(d.get("max_total_num_tokens",0)))
except Exception:
    sys.exit(1)
' 2>/dev/null || true)"
      # Distinguish: (a) realized reported AND in band -> verified; (b) realized
      # reported AND out of band -> unhealthy contract, kill; (c) not yet reported
      # (empty/0/unparseable) -> keep polling. Treating 0 as "reported" would
      # mis-kill during the startup window before /get_server_info has the value.
      if [ -n "$realized" ] && [ "$realized" -gt 0 ] 2>/dev/null; then
        if [ "$realized" -ge "$MINTOKENS" ] 2>/dev/null \
           && [ "$realized" -le "$MAXTOKENS" ] 2>/dev/null; then
          log "allocation verified: realized=${realized} tokens (in [${MINTOKENS}, ${MAXTOKENS}])"
          return 0
        fi
        log "ERROR: realized pool ${realized} outside contract [${MINTOKENS},${MAXTOKENS}] — killing"
        kill "$ADAPTER_PID" 2>/dev/null || true
        return 1
      fi
      # realized not yet reported (0/empty) -> keep polling.
    fi
    sleep 5
  done
  return 1
}

if ! verify_ready; then
  log "ERROR: allocation not verified within ${ADMISSION_READY_TIMEOUT}s — killing adapter"
  kill "$ADAPTER_PID" 2>/dev/null || true
  # Hardening: deterministically remove the candidate container. The adapter runs
  # docker with --rm, but killing the foreground docker client may not propagate
  # cleanly, leaving an unverified model resident. Remove the candidate BEFORE
  # waiting for the client: Docker's foreground client can ignore/absorb TERM
  # while the container continues starting, which would otherwise deadlock this
  # cleanup path. docker rm -f is bounded to THIS container name, never a
  # co-resident.
  if [ -n "${CONTAINER_NAME:-}" ] && command -v docker >/dev/null 2>&1; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
  wait "$ADAPTER_PID" 2>/dev/null || true
  # release the lock (fd 9 closes on exit)
  die "role '$ROLE' failed admission verification (co-residents untouched)"
fi

# ---- release the lock + become the long-lived supervisor -------------------
# Allocation is committed and verified. Release the lock so a co-resident may
# now admit, then exec into supervising the adapter child so systemd's tracked
# PID (this one) stays alive for the service lifetime (Type=simple requirement).
log "admission complete; releasing lock; supervising adapter (pid $ADAPTER_PID)"
exec 9>&-   # release flock

wait "$ADAPTER_PID"
exit $?
