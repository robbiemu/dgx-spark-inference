#!/usr/bin/env bash
# install.sh — thin installer for dgx-spark-inference. De-machine-speced.
#
# Deliberately NOT fully idempotent (full idempotency is a documented roadmap
# goal, not v0.1). "Alpha reduces scope, not safety":
#   - refuses to overwrite an existing PROGRAM target unless --replace is passed;
#   - --replace touches PROGRAM files + the unit ONLY — it NEVER overwrites
#     operator state (agent.env, active-models.toml, runtimes.toml);
#   - never starts or restarts the service (operator starts it).
#
# v0.1 scope is fixed and honest: exactly ONE role (agentic), ONE unit name
# (dgx-spark-inference.service), and PROJECT_ROOT = INSTALL_ROOT (immutable
# installed artifacts, never a mutable checkout). Broader role/unit/project
# customization is a roadmap item, post multi-role support.
#
# Usage:
#   install.sh [options]
# Options (all take values except --dry-run/--replace):
#   --install-root DIR    default /usr/local/lib/dgx-spark-inference
#   --config-root DIR     default /etc/dgx-spark-inference
#   --model-cache-root DIR  REQUIRED (no default; your host's HF cache)
#   --port N              default 30000
#   --container-name NAME default inference-agentic
#   --agent-env FILE      operator-created secret file (REQUIRED; not handled here)
#   --dry-run             render + print plan; write nothing
#   --replace             allow overwriting PROGRAM files + the unit
#
# Fixed (not configurable in v0.1): ROLE=agentic, UNIT_NAME=dgx-spark-inference.service,
# PROJECT_ROOT=$INSTALL_ROOT.
set -Eeuo pipefail

DRY_RUN=0
REPLACE=0
INSTALL_ROOT="/usr/local/lib/dgx-spark-inference"
CONFIG_ROOT="/etc/dgx-spark-inference"
MODEL_CACHE_ROOT=""
PORT=30000
CONTAINER_NAME="inference-agentic"
AGENT_ENV=""

# Fixed v0.1 constants.
ROLE="agentic"
UNIT_NAME="dgx-spark-inference.service"

usage() { sed -n '3,32p' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --install-root)     INSTALL_ROOT="$2"; shift 2;;
    --config-root)      CONFIG_ROOT="$2"; shift 2;;
    --model-cache-root) MODEL_CACHE_ROOT="$2"; shift 2;;
    --port)             PORT="$2"; shift 2;;
    --container-name)   CONTAINER_NAME="$2"; shift 2;;
    --agent-env)        AGENT_ENV="$2"; shift 2;;
    --dry-run)          DRY_RUN=1; shift;;
    --replace)          REPLACE=1; shift;;
    -h|--help)          usage; exit 0;;
    *) echo "install.sh: unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done

# v0.1: the resolution root is the install location (immutable). The live service
# must NOT depend on a mutable Git checkout.
PROJECT_ROOT="$INSTALL_ROOT"
die() { echo "install.sh: $*" >&2; exit 1; }
# In dry-run we preview commands; otherwise run with sudo (program files install
# under a root-owned path by default).
run() { if [ "$DRY_RUN" = 1 ]; then printf '  (dry-run) %s\n' "$*"; else "$@"; fi; }
srun() { if [ "$DRY_RUN" = 1 ]; then printf '  (dry-run) sudo %s\n' "$*"; else sudo "$@"; fi; }

[ -n "$MODEL_CACHE_ROOT" ] || die "--model-cache-root is required (no default)."
[ -n "$AGENT_ENV" ] || die "--agent-env is required (operator supplies the secret file)."

# Resolve the source tree (the checkout we are installing from).
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- validate the operator secret (presence + shape; never handle its value) ---
[ -f "$AGENT_ENV" ] || die "agent.env not found: $AGENT_ENV"
if ! grep -qE '^[[:space:]]*SGLANG_API_KEY=' "$AGENT_ENV"; then
  die "REFUSING: $AGENT_ENV lacks SGLANG_API_KEY"
fi
# Safe perms on the secret (do not print its contents).
if [ "$(stat -c '%a' "$AGENT_ENV" 2>/dev/null || stat -f '%Lp' "$AGENT_ENV")" != "600" ]; then
  echo "install.sh: WARNING: $AGENT_ENV is not mode 0600 (expected for the secret)." >&2
fi

# ---- operator state files are NEVER overwritten by --replace ------------------
OPERATOR_STATE=(active-models.toml runtimes.toml)

# ---- what this install places under INSTALL_ROOT (PROGRAM files) -------------
install_program_tree() {
  srun install -d -m 0755 "$INSTALL_ROOT"
  srun install -d -m 0755 "$INSTALL_ROOT/src/inferencectl" \
                  "$INSTALL_ROOT/tools" "$INSTALL_ROOT/tools/memory_planner" \
                  "$INSTALL_ROOT/runtime/sglang/adapters" \
                  "$INSTALL_ROOT/experiments/dflash"
  srun install -m 0755 "$SOURCE_DIR/src/inferencectl/dispatch.sh"        "$INSTALL_ROOT/src/inferencectl/dispatch.sh"
  srun install -m 0755 "$SOURCE_DIR/src/inferencectl/inference-cli.sh"   "$INSTALL_ROOT/src/inferencectl/inference-cli.sh"
  srun install -m 0755 "$SOURCE_DIR/src/inferencectl/admission.sh"       "$INSTALL_ROOT/src/inferencectl/admission.sh"
  srun install -m 0755 "$SOURCE_DIR/tools/resolve_service_plan.py"       "$INSTALL_ROOT/tools/resolve_service_plan.py"
  srun install -m 0755 "$SOURCE_DIR/tools/bind_active_runtime.py"        "$INSTALL_ROOT/tools/bind_active_runtime.py"
  srun install -m 0755 "$SOURCE_DIR/tools/validate_agentic_endpoint.py"  "$INSTALL_ROOT/tools/validate_agentic_endpoint.py"
  srun install -m 0755 "$SOURCE_DIR/tools/wait_for_health.sh"            "$INSTALL_ROOT/tools/wait_for_health.sh"
  srun install -m 0755 "$SOURCE_DIR/tools/memory_planner/resolve_memory_plan.py" \
                       "$INSTALL_ROOT/tools/memory_planner/resolve_memory_plan.py"
  srun install -m 0644 "$SOURCE_DIR/runtime/sglang/runtime-manifest.toml" "$INSTALL_ROOT/runtime/sglang/runtime-manifest.toml"
  srun install -m 0644 "$SOURCE_DIR/runtime/sglang/Dockerfile"            "$INSTALL_ROOT/runtime/sglang/Dockerfile"
  srun install -m 0644 "$SOURCE_DIR/runtime/sglang/available.toml"        "$INSTALL_ROOT/runtime/sglang/available.toml"
  srun install -m 0644 "$SOURCE_DIR/runtime/sglang/capability.toml"       "$INSTALL_ROOT/runtime/sglang/capability.toml"
  # adapter lives at adapters/sglang.sh in the source tree (matching the manifest's
  # launch_adapter declaration), so install copies it in-place — no rename.
  srun install -m 0755 "$SOURCE_DIR/runtime/sglang/adapters/sglang.sh" "$INSTALL_ROOT/runtime/sglang/adapters/sglang.sh"
  # profiles + bundles + their capability records (immutable artifacts). Copy
  # WITHOUT preserving source ownership: these install root-owned (the live
  # service must not depend on user-owned files under a root-owned path).
  srun cp -a "$SOURCE_DIR/profiles" "$INSTALL_ROOT/"
  srun cp -a "$SOURCE_DIR/bundles"  "$INSTALL_ROOT/"
  srun chown -R root:root "$INSTALL_ROOT/profiles" "$INSTALL_ROOT/bundles"
  # experimental launcher (lives under INSTALL_ROOT so it resolves from there)
  srun install -m 0755 "$SOURCE_DIR/experiments/dflash/run-experimental.sh" \
                       "$INSTALL_ROOT/experiments/dflash/run-experimental.sh"
}

render_unit() {  # stdin template -> stdout with placeholders substituted
  sed -e "s|__CONFIG_ROOT__|$CONFIG_ROOT|g" \
      -e "s|__INSTALL_ROOT__|$INSTALL_ROOT/src/inferencectl|g" \
      -e "s|__ROLE__|$ROLE|g" \
      -e "s|__CONTAINER_NAME__|$CONTAINER_NAME|g"
}

# Render runtimes.toml from INSTALL_ROOT (project_root must match where the
# runtime was installed, not a hardcoded path).
render_runtimes() {
  cat <<EOF
# runtimes.toml — registry of approved runtime projects.
# OPERATOR state under CONFIG_ROOT ($CONFIG_ROOT). Seeded from this file by
# install.sh on first install; operator-owned thereafter (--replace never
# overwrites it). project_root points at the INSTALLED runtime tree under
# INSTALL_ROOT, not a Git checkout. See docs/architecture.md.
[runtimes.sglang-v0.5.14-cu130-runtime-distro1.9.0]
project_root = "$INSTALL_ROOT/runtime/sglang"

EOF
}

emit_inference_env() {  # the operator env file (machine-specific values)
  cat <<EOF
# Generated by install.sh. Machine-specific resolved values (single source).
# Sourced by the dispatcher, CLI, and adapter via CONFIG_ROOT/inference.env.
INSTALL_ROOT="$INSTALL_ROOT"
PROJECT_ROOT="$PROJECT_ROOT"
MODEL_CACHE_ROOT="$MODEL_CACHE_ROOT"
ROLE="$ROLE"
PORT="$PORT"
CONTAINER_NAME="$CONTAINER_NAME"
SYSTEMD_UNIT="$UNIT_NAME"
EOF
}

mode_label() {
  local m="APPLY"
  [ "$DRY_RUN" = 1 ] && m="DRY-RUN"
  [ "$REPLACE" = 1 ] && m="$m + --replace"
  printf '%s' "$m"
}

print_plan() {
  cat <<EOF
=== dgx-spark-inference install plan ===
  mode            : $(mode_label)
  install-root    : $INSTALL_ROOT   (program/profile/runtime files; immutable)
  project-root    : $PROJECT_ROOT   (= install-root; v0.1 fixed)
  config-root     : $CONFIG_ROOT    (operator state + generated inference.env)
  model-cache-root: $MODEL_CACHE_ROOT
  role            : $ROLE   port: $PORT   container: $CONTAINER_NAME   unit: $UNIT_NAME
  secret          : $AGENT_ENV (mode checked, never handled)
  fixed (v0.1)    : role=agentic, unit=$UNIT_NAME, project-root=install-root
  operator state never overwritten: ${OPERATOR_STATE[*]}
EOF
}
print_plan

# ---- existing-target safety --------------------------------------------------
# A program install is refused if EITHER program target already exists, unless
# --replace is given. We check both the install root and the unit file, so a
# partial prior install (root gone, unit present — or vice versa) cannot silently
# overwrite the unit without an explicit --replace.
check_existing_program() {
  local conflicts=()
  [ -d "$INSTALL_ROOT" ] && conflicts+=("install-root ($INSTALL_ROOT)")
  [ -f "/etc/systemd/system/$UNIT_NAME" ] && conflicts+=("unit (/etc/systemd/system/$UNIT_NAME)")
  if [ "${#conflicts[@]}" -gt 0 ] && [ "$REPLACE" = 0 ]; then
    die "REFUSING: existing program target(s): ${conflicts[*]}. Re-run with --replace to overwrite PROGRAM files + the unit (operator state is still preserved)."
  fi
}

# ---- apply (skipped in --dry-run) --------------------------------------------
if [ "$DRY_RUN" = 1 ]; then
  echo "--- rendered unit (preview) ---"
  render_unit < "$SOURCE_DIR/deploy/systemd/inference-agentic.service.in"
  echo "--- rendered inference.env (preview, no secrets) ---"
  emit_inference_env
  echo "--- rendered runtimes.toml (preview) ---"
  render_runtimes
  echo "--- (dry-run: nothing written) ---"
  exit 0
fi

check_existing_program
install_program_tree

# operator config dir + generated env (root:root 0750 dir; env 0640; no secret)
srun install -d -o root -g root -m 0750 "$CONFIG_ROOT"
# agent.env: copied ONLY if absent (never overwritten, even by --replace).
if [ ! -f "$CONFIG_ROOT/agent.env" ]; then
  srun install -o root -g root -m 0600 "$AGENT_ENV" "$CONFIG_ROOT/agent.env"
else
  echo "install.sh: $CONFIG_ROOT/agent.env exists; leaving it (operator rotates the secret separately)." >&2
fi
# generated inference.env (no secret) — overwritten freely; it is derived.
emit_inference_env | srun tee "$CONFIG_ROOT/inference.env" >/dev/null
srun chmod 0640 "$CONFIG_ROOT/inference.env"

# seed operator state from examples ONLY if absent (never overwritten).
# active-models.toml is a static template; runtimes.toml is rendered from
# INSTALL_ROOT so project_root matches the install location.
if [ ! -f "$CONFIG_ROOT/active-models.toml" ]; then
  srun install -o root -g root -m 0640 "$SOURCE_DIR/config/examples/active-models.toml" "$CONFIG_ROOT/active-models.toml"
else
  echo "install.sh: $CONFIG_ROOT/active-models.toml exists; leaving operator state untouched." >&2
fi
if [ ! -f "$CONFIG_ROOT/runtimes.toml" ]; then
  render_runtimes | srun install -o root -g root -m 0640 /dev/stdin "$CONFIG_ROOT/runtimes.toml"
else
  echo "install.sh: $CONFIG_ROOT/runtimes.toml exists; leaving operator state untouched." >&2
fi

# render + install the unit (PROGRAM target — --replace covers it).
render_unit < "$SOURCE_DIR/deploy/systemd/inference-agentic.service.in" \
  | srun install -o root -g root -m 0644 /dev/stdin "/etc/systemd/system/$UNIT_NAME"
srun systemctl daemon-reload
srun systemctl enable "$UNIT_NAME"

cat <<EOF

Installed and enabled $UNIT_NAME. The service has NOT been started.
Start it as a separate reviewed operation:
  sudo systemctl start $UNIT_NAME
Then verify health: curl -s http://127.0.0.1:$PORT/health  (expect 200)
See docs/smoke-test.md (read-only gate) and SECURITY.md (firewall prerequisite).
EOF
