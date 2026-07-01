#!/usr/bin/env bash
# dispatch.sh — runtime-agnostic role dispatcher (binding v2). De-machine-speced.
# See docs/architecture.md (the binding system).
#
# Knows nothing about any specific runtime. Reads:
#   1. active-models.toml[<role>]   -> model_id, runtime_id  (what's resident now)
#   2. runtimes.toml[<runtime_id>]  -> project_root          (where the runtime lives)
#   3. <root>/available.toml[<role>] -> candidate list (validates model_id is offered)
# Then execs the runtime's adapter with: role, runtime_root, projects_root,
# and the resolved candidate (id + kind + spec path) for the adapter to launch.
#
# CONFIG DISCOVERY: operator state lives under CONFIG_ROOT (default
# /etc/dgx-spark-inference), surfaced by systemd. Resolution paths can be
# overridden via the operator env file or the environment. No host path is
# hard-coded here.
#
# Invocation (systemd, per-role unit):
#   dispatch.sh <ROLE>
set -Eeuo pipefail

ROLE="${1:?usage: dispatch.sh <role>}"

# Operator configuration discovery (single mechanism).
CONFIG_ROOT="${CONFIG_ROOT:-/etc/dgx-spark-inference}"
# shellcheck disable=SC1090,SC1091
[ -f "$CONFIG_ROOT/inference.env" ] && . "$CONFIG_ROOT/inference.env"

ACTIVE_MODELS="${ACTIVE_MODELS:-$CONFIG_ROOT/active-models.toml}"
RUNTIMES_INDEX="${RUNTIMES_INDEX:-$CONFIG_ROOT/runtimes.toml}"
PROJECTS_ROOT="${PROJECT_ROOT:-${PROJECTS_ROOT:-}}"

die() { echo "ERROR: $*" >&2; exit 1; }

[ -n "$PROJECTS_ROOT" ] || die "PROJECT_ROOT is required (set in $CONFIG_ROOT/inference.env)"

# Resolve: read active + available, validate, print "id|kind|spec|runtime_root|served_name"
IFS='|' read -r MODEL_ID KIND SPEC RUNTIME_ROOT SERVED <<EOF
$(python3 - "$ACTIVE_MODELS" "$RUNTIMES_INDEX" "$PROJECTS_ROOT" "$ROLE" <<'PY'
import sys, os, tomllib
active_p, runtimes_p, proot, role = sys.argv[1:5]
active = tomllib.load(open(active_p,"rb")).get("active",{}).get(role)
if not active:
    print(f"REFUSE: role '{role}' has no entry in {active_p}", file=sys.stderr); sys.exit(1)
mid, rid = active.get("model_id"), active.get("runtime_id")
# runtimes.toml keys may be bare dotted (parsed as nested) OR quoted (flat).
# Traverse the dotted form first; fall back to flat lookup.
runtimes = tomllib.load(open(runtimes_p,"rb")).get("runtimes",{})
root = None
try:
    cur = runtimes
    for part in rid.split("."):
        cur = cur[part]
    root = cur.get("project_root")
except (KeyError, TypeError):
    root = runtimes.get(rid, {}).get("project_root")
if not root or not os.path.isdir(root):
    print(f"REFUSE: runtime '{rid}' not registered / missing root", file=sys.stderr); sys.exit(1)
avail = tomllib.load(open(os.path.join(root,"available.toml"),"rb")).get("roles",{}).get(role)
if not avail:
    print(f"REFUSE: role '{role}' not in {root}/available.toml", file=sys.stderr); sys.exit(1)
cands = {m["id"]: m for m in avail.get("models",[])}
if mid not in cands:
    print(f"REFUSE: active model_id '{mid}' not offered in available[{role}].models", file=sys.stderr); sys.exit(1)
c = cands[mid]
served = avail.get("served_model_name","")
_kind = c.get("kind","model")
_spec = c.get("spec","")
print(f"{mid}|{_kind}|{_spec}|{root}|{served}")
PY
)
EOF

[ -n "$MODEL_ID" ] || die "dispatch resolution failed for role '$ROLE'"
[ -f "$PROJECTS_ROOT/$SPEC" ] || die "spec missing: $PROJECTS_ROOT/$SPEC"

# Load the manifest to find the adapter + validate kind matches spec kind.
MANIFEST="$RUNTIME_ROOT/runtime-manifest.toml"
ADAPTER_REL="$(python3 - "$MANIFEST" <<'PY'
import sys, tomllib
print(tomllib.load(open(sys.argv[1],"rb"))["launch_adapter"])
PY
)"
ADAPTER="$RUNTIME_ROOT/$ADAPTER_REL"
[ -x "$ADAPTER" ] || die "adapter missing/not executable: $ADAPTER"

# Kind-match check (available.kind must equal spec kind's prefix).
SPEC_KIND="$(python3 - "$PROJECTS_ROOT/$SPEC" <<'PY'
import sys, tomllib
k = tomllib.load(open(sys.argv[1],"rb")).get("kind","")
print("model" if k=="model-runtime-config" else "bundle" if k=="bundle-runtime-config" else k)
PY
)"
[ "$KIND" = "$SPEC_KIND" ] || die "kind mismatch: available says '$KIND', spec says '$SPEC_KIND'"

echo "[dispatch] role=$ROLE model=$MODEL_ID kind=$KIND runtime=$(basename "$RUNTIME_ROOT") served=$SERVED"

# Memory preflight (v0.2): when enrolled (DGX_MEMORY_PREFLIGHT), route through
# admission.sh — the serialized admission wrapper that holds a global lock across
# discover->sample->resolve->launch->VERIFY-allocation, closing the preflight<->
# allocation race. Passes the adapter path as the 8th arg so admission.sh can
# exec the real adapter after verified admission. Legacy v0.1 roles (preflight
# unset/auto with no planner pair) exec the adapter directly, unchanged.
# Only the MEMORY resolver is wired (capability resolver stays off the live path).
ADMISSION="${DGX_ADMISSION_SH:-$PROJECTS_ROOT/src/inferencectl/admission.sh}"
if [ "${DGX_MEMORY_PREFLIGHT:-auto}" != "auto" ] || {
     # auto mode: enroll only if a matched planner pair exists in CONFIG_ROOT.
     [ -f "${CONFIG_ROOT}/memory_ledger.toml" ] && [ -f "${CONFIG_ROOT}/memory_plan.toml" ]
   }; then
  if [ -x "$ADMISSION" ] || [ -f "$ADMISSION" ]; then
    exec "$ADMISSION" "$ROLE" "$RUNTIME_ROOT" "$PROJECTS_ROOT" "$MODEL_ID" "$KIND" "$SPEC" "$SERVED" "$ADAPTER"
  fi
  echo "[dispatch] WARN: admission.sh not found ($ADMISSION); legacy launch" >&2
fi
exec "$ADAPTER" "$ROLE" "$RUNTIME_ROOT" "$PROJECTS_ROOT" "$MODEL_ID" "$KIND" "$SPEC" "$SERVED"
