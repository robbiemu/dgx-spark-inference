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
PREFLIGHT="${DGX_MEMORY_PREFLIGHT:-auto}"   # auto | required | off
# FLOOR is resolved AFTER the pair check (Blocker 3): layered
#   DGX_MEMAVAILABLE_FLOOR_GIB (env) > installed memory_plan.toml [policy] > default 6.0
LOCK="${DGX_ADMISSION_LOCK:-/run/dgx-inference-admission.lock}"
PORT="${PORT:-30000}"
ADMISSION_READY_TIMEOUT="${DGX_ADMISSION_READY_TIMEOUT:-300}"  # sec to verify allocation

die() { echo "ERROR: REFUSING: $*" >&2; exit 75; }   # 75 = deliberate refusal (EX_TEMPFAIL)
log() { echo "[admission] $*"; }

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

# ---- the serialized admission lock -----------------------------------------
# Hold across discover->sample->resolve->launch->VERIFY. flock is released when
# the holding fd closes (on exec-via-wait or exit). We open it on fd 9.
exec 9>"$LOCK"
log "acquiring admission lock ($LOCK)..."
flock 9
log "lock held"

# ---- discover residents (label-based, for guards only) ---------------------
# In LIVE mode (Blocker 1) residents are NOT subtracted from A_preload — the
# measured gpu_free_now_gib already includes them. Discovery remains for:
# identity/revision checks and the unmanaged-GPU-tenant guard.
if command -v docker >/dev/null 2>&1; then
  # Ledger-revision check: export DGX_MEMORY_LEDGER so the adapter stamps the
  # io.inferencectl.ledger_revision label; here we compute the expected revision
  # and refuse in required mode if any resident's revision is absent/mismatched
  # (a stale resident from a different ledger generation would invalidate the plan).
  EXPECTED_REV=""
  [ -f "$LEDGER" ] && EXPECTED_REV="$(sha256sum "$LEDGER" 2>/dev/null | cut -c1-16)"
  export DGX_MEMORY_LEDGER="$LEDGER"   # so the adapter labels the new container
  if [ -n "$EXPECTED_REV" ] && [ "$PREFLIGHT" = "required" ]; then
    bad_rev=$( { docker ps --filter "label=io.inferencectl.managed=true" \
                   --format '{{.Label "io.inferencectl.ledger_revision"}}' \
                 | awk -v exp="$EXPECTED_REV" '$1 != exp {print "mismatch"}'; } 2>/dev/null || true)
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
      die "managed mode: unmanaged GPU container(s) present ($(echo $gpu_unmanaged | tr '\n' ' ')); cannot reason about unaccounted GPU memory"
    fi
  fi
fi

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

# ---- build the transient plan + run the resolver (--format json) -----------
# Blocker 1: pass the MEASURED gpu_free_now_gib as the A_preload. The resolver
# uses it directly and does NOT subtract residents again (the measurement already
# includes resident allocations). Residents are for identity/revision/guard checks.
PLAN_TMP="$(mktemp --suffix=.toml)"
cat > "$PLAN_TMP" <<EOF
device.total_gib = ${gpu_total:-121.7}
[policy]
memavailable_floor_gib = ${FLOOR}
[observed]
gpu_free_now_gib = ${gpu_free}
memavailable_now_gib = ${memavail_gib}
[[admit]]
role = "${ROLE}"
model_id = "${MODEL_ID}"
EOF
log "resolving memory plan (floor=${FLOOR}G, memavail=${memavail_gib}G, gpu_free=${gpu_free}G)..."
PLANNER_ERR="$(mktemp)"
trap 'rm -f "$PLAN_TMP" "$PLANNER_ERR"' EXIT
# Capture stdout and rc SEPARATELY (correction): the resolver exits nonzero for a
# valid REFUSE, and `|| JSON_OUT=""` would discard the structured JSON, losing the
# ability to distinguish an intentional gate failure from malformed output.
set +e
JSON_OUT="$(python3 "$PLANNER" "$LEDGER" "$PLAN_TMP" --format json 2>"$PLANNER_ERR")"
RESOLVER_RC=$?
set -e
# parse the JSON for THIS model's derived knobs (stdlib; no grep on prose).
# JSON passed via env (DGX_PARSE_JSON), not stdin — a bash heredoc would consume
# stdin as the script source and break json.load.
read_knobs() {  # prints "FRACTION MAXTOKENS MINTOKENS"; exit 0 on success
  DGX_PARSE_JSON="$JSON_OUT" DGX_PARSE_MODEL="$MODEL_ID" python3 <<'PY'
import os, sys, json
mid = os.environ["DGX_PARSE_MODEL"]
try:
    doc = json.loads(os.environ["DGX_PARSE_JSON"])
except Exception:
    sys.exit(1)
if doc.get("result") != "ADMIT":
    sys.exit(2)
for m in doc.get("models", []):
    if m.get("model_id") == mid:
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
  wait "$ADAPTER_PID" 2>/dev/null || true
  # Hardening: deterministically remove the candidate container. The adapter runs
  # docker with --rm, but killing the foreground docker client may not propagate
  # cleanly, leaving an unverified model resident. docker rm -f is bounded to THIS
  # container name (the candidate), never a co-resident.
  if [ -n "${CONTAINER_NAME:-}" ] && command -v docker >/dev/null 2>&1; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
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
