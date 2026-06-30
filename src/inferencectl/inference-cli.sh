#!/usr/bin/env bash
# inference-cli v2 — operator residency/swap tool (binding v2). De-machine-speced.
# See docs/architecture.md.
#
# Thin orchestrator over dispatch.sh + systemd. Does NOT talk to any runtime
# directly. Swaps edit CONFIG_ROOT/active-models.toml only (the runtime catalog
# available.toml is never touched by swaps).
#
# Subcommands:
#   status                       resident per role + health + served name
#   candidates [role]            available candidates (from available.toml) + active
#   use <role> <candidate_id>    swap: validate offered, edit active-models, restart, verify
#   unload <role>                stop the role's container (free the slot)
#   reload <role]                restart the role's active candidate (cold); no arg = home
#
# CONFIG DISCOVERY: --config-root (default /etc/dgx-spark-inference). Resolved
# operator values are read from $CONFIG_ROOT/inference.env. No host path is
# hard-coded here.
set -Eeuo pipefail

CONFIG_ROOT="${CONFIG_ROOT:-/etc/dgx-spark-inference}"
while [ $# -gt 0 ]; do
  case "$1" in
    --config-root) CONFIG_ROOT="$2"; shift 2;;
    *) break;;
  esac
done
# shellcheck disable=SC1090,SC1091
[ -f "$CONFIG_ROOT/inference.env" ] && . "$CONFIG_ROOT/inference.env"

ACTIVE_MODELS="${ACTIVE_MODELS:-$CONFIG_ROOT/active-models.toml}"
RUNTIMES_INDEX="${RUNTIMES_INDEX:-$CONFIG_ROOT/runtimes.toml}"
PROJECTS_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required (set in $CONFIG_ROOT/inference.env)}"
HEALTH_PORT="${PORT:-${HEALTH_PORT:-30000}}"
CONTAINER_NAME="${CONTAINER_NAME:-inference-agentic}"
SYSTEMD_UNIT="${SYSTEMD_UNIT:-dgx-spark-inference.service}"

die() { echo "ERROR: $*" >&2; exit 1; }
health_code() { curl -sS -o /dev/null -w "%{http_code}" --max-time 8 "http://127.0.0.1:${HEALTH_PORT}/health" 2>/dev/null || echo "000"; }

# Resolve runtime root for a role (via active-models -> runtimes).
runtime_root_for() {
  python3 - "$ACTIVE_MODELS" "$RUNTIMES_INDEX" "$1" <<'PY'
import sys, os, tomllib
active_p, runtimes_p, role = sys.argv[1:4]
a = tomllib.load(open(active_p,"rb")).get("active",{}).get(role,{})
rid = a.get("runtime_id","")
runtimes = tomllib.load(open(runtimes_p,"rb")).get("runtimes",{})
root = None
try:
    cur = runtimes
    for part in rid.split("."): cur = cur[part]
    root = cur.get("project_root")
except (KeyError, TypeError):
    root = runtimes.get(rid,{}).get("project_root")
print(root or "")
PY
}

cmd_status() {
  echo "=== inference service status ==="
  echo "systemd unit : $SYSTEMD_UNIT  ($(systemctl is-active "$SYSTEMD_UNIT" 2>/dev/null || echo unknown))"
  echo "health :${HEALTH_PORT} : $(health_code)"
  echo
  [ -f "$ACTIVE_MODELS" ] || { echo "(no active-models at $ACTIVE_MODELS)"; return; }
  python3 - "$ACTIVE_MODELS" "$RUNTIMES_INDEX" "$PROJECTS_ROOT" <<'PY'
import sys, os, tomllib
active_p, runtimes_p, proot = sys.argv[1:4]
active = tomllib.load(open(active_p,"rb")).get("active",{})
runtimes = tomllib.load(open(runtimes_p,"rb")).get("runtimes",{})
def root_for(rid):
    try:
        cur = runtimes
        for p in rid.split("."): cur = cur[p]
        return cur.get("project_root")
    except (KeyError,TypeError): return runtimes.get(rid,{}).get("project_root")
print("=== roles ===")
for role, a in active.items():
    mid, rid = a.get("model_id","?"), a.get("runtime_id","?")
    root = root_for(rid) or "?"
    served = "?"
    if root and root != "?" and os.path.isfile(os.path.join(root,"available.toml")):
        av = tomllib.load(open(os.path.join(root,"available.toml"),"rb")).get("roles",{}).get(role,{})
        served = av.get("served_model_name","?")
    print(f"  {role:12s} active={mid}  runtime={rid}  served={served}")
PY
}

cmd_candidates() {
  local role="${1:-}"; [ -n "$role" ] || die "usage: candidates <role>"
  local root; root="$(runtime_root_for "$role")"
  [ -n "$root" ] || die "no runtime root for role '$role' (active-models missing?)"
  local avail="$root/available.toml"
  [ -f "$avail" ] || die "no available.toml at $avail"
  local active_mid; active_mid="$(python3 - "$ACTIVE_MODELS" "$role" <<'PY'
import sys, tomllib
a = tomllib.load(open(sys.argv[1],"rb")).get("active",{}).get(sys.argv[2],{})
print(a.get("model_id",""))
PY
)"
  echo "=== candidates for role '$role' (runtime available.toml) ==="
  python3 - "$avail" "$role" "$active_mid" <<'PY'
import sys, tomllib
av = tomllib.load(open(sys.argv[1],"rb")).get("roles",{}).get(sys.argv[2],{})
active = sys.argv[3]
print(f"  default (home) : {av.get('default','?')}")
print(f"  served_name    : {av.get('served_model_name','?')}")
print(f"  active now     : {active or '?'}")
print("  candidates:")
for m in av.get("models",[]):
    mark = " <- ACTIVE" if m["id"]==active else ""
    print(f"    {m['id']:28s} kind={m.get('kind','?'):8s} spec={m.get('spec','?')}{mark}")
PY
}

cmd_use() {
  local role="${1:-}"; local cand="${2:-}"
  if [ -z "$role" ] || [ -z "$cand" ]; then die "usage: use <role> <candidate_id>"; fi
  echo "[use] role='$role' candidate='$cand'"
  local h; h="$(health_code)"
  [ "$h" = "200" ] || die "REFUSING: service not healthy (health=$h). Swap only from known-good."
  echo "[use] healthy-before: /health=200"
  local root; root="$(runtime_root_for "$role")"
  [ -n "$root" ] || die "no runtime root for '$role'"
  local avail="$root/available.toml"
  # validate candidate is offered
  python3 - "$avail" "$role" "$cand" <<'PY'
import sys, tomllib
av = tomllib.load(open(sys.argv[1],"rb")).get("roles",{}).get(sys.argv[2],{})
ids = {m["id"] for m in av.get("models",[])}
if sys.argv[3] not in ids:
    print(f"REFUSE: '{sys.argv[3]}' not offered in available[{sys.argv[2]}].models (have: {sorted(ids)})"); sys.exit(1)
PY
  local current; current="$(python3 - "$ACTIVE_MODELS" "$role" <<'PY'
import sys, tomllib
print(tomllib.load(open(sys.argv[1],"rb")).get("active",{}).get(sys.argv[2],{}).get("model_id",""))
PY
)"
  echo "[use] current active: $current"
  [ "$cand" = "$current" ] && { echo "[use] already active; nothing to do."; exit 0; }
  # backup + edit active-models.toml
  cp -a "$ACTIVE_MODELS" "${ACTIVE_MODELS}.pre-use.$(date -u +%Y%m%dT%H%M%SZ)"
  python3 - "$ACTIVE_MODELS" "$role" "$cand" <<'PY'
import sys, tomllib
path, role, cand = sys.argv[1:4]
# rewrite the [active.<role>] model_id, preserving runtime_id
data = tomllib.load(open(path,"rb"))
rid = data.get("active",{}).get(role,{}).get("runtime_id","")
# textual replace of the model_id line within [active.<role>]
text = open(path).read()
import re
# replace model_id = "..."  (the first occurrence after [active.<role>])
pat = re.compile(r'(\[active\.' + re.escape(role) + r'\][^\[]*?model_id\s*=\s*")[^"]*(")', re.S)
new, n = pat.subn(rf'\g<1>{cand}\g<2>', text, count=1)
if n != 1:
    print("REFUSE: could not edit model_id"); sys.exit(1)
tomllib.loads(new)  # validate
open(path,"w").write(new)
print(f"[use] active-models: {role} -> {cand} (runtime {rid} unchanged)")
PY
  echo "[use] restarting $SYSTEMD_UNIT (cold load ~4min)..."
  sudo systemctl restart "$SYSTEMD_UNIT"
  echo "[use] waiting up to 360s for /health 200..."
  local start; start=$(date +%s); local ok=0
  for _ in $(seq 1 72); do
    sleep 5
    if [ "$(health_code)" = "200" ]; then ok=1; break; fi
  done
  local elapsed; elapsed=$(( $(date +%s) - start ))
  if [ "$ok" = "1" ]; then
    echo "[use] SUCCESS: $role -> $cand, healthy after ${elapsed}s"
  else
    echo "[use] FAILED: not healthy after ${elapsed}s. ROLLBACK: restore ${ACTIVE_MODELS}.pre-use.* + sudo systemctl restart $SYSTEMD_UNIT" >&2
    exit 1
  fi
}

cmd_unload() {
  local role="${1:-}"; [ -n "$role" ] || die "usage: unload <role>"
  echo "[unload] stopping $SYSTEMD_UNIT (frees the slot; no auto-reload)"
  sudo systemctl stop "$SYSTEMD_UNIT"
  echo "[unload] stopped."
}

cmd_reload() {
  local role="${1:-}"
  [ -n "$role" ] || die "usage: reload <role>"
  echo "[reload] restarting $SYSTEMD_UNIT (cold load of active candidate)..."
  sudo systemctl restart "$SYSTEMD_UNIT"
  echo "[reload] restart issued; poll: inference-cli status"
}

case "${1:-}" in
  status)     shift; cmd_status "$@";;
  candidates) shift; cmd_candidates "$@";;
  use)        shift; cmd_use "$@";;
  unload)     shift; cmd_unload "$@";;
  reload)     shift; cmd_reload "$@";;
  ""|-h|--help|help) sed -n '2,18p' "$0";;
  *) die "unknown subcommand: $1 (try: status|candidates|use|unload|reload)";;
esac
