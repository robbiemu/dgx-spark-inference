#!/usr/bin/env bash
# run-experimental.sh — EXPERIMENTAL DFlash launcher. NOT a production path.
# See bundles/experimental/qwen36-27b-fp8-dflash/README.md.
#
# Safety (load-bearing):
#   - defaults to the VALIDATED 32768 context (the only fit validated so far);
#   - refuses any bundle whose context differs from the validated 32768 setting;
#   - REFUSES to launch if the production service is active or the production
#     primary/helper containers exist (no collision or unplanned co-residency);
#   - launches on a SEPARATE container name + port so it can never take over the
#     production endpoint;
#   - pins its own experimental image and image ID independently of production;
#   - there is NO --allow-incompatible bypass and no inferencectl use path here.
#
# This is a manual research tool, not a managed service. Run it in a terminal you
# are watching; Ctrl-C tears it down.
set -Eeuo pipefail

CONFIG_ROOT="${CONFIG_ROOT:-/etc/dgx-spark-inference}"
# shellcheck disable=SC1090,SC1091
[ -f "$CONFIG_ROOT/inference.env" ] && . "$CONFIG_ROOT/inference.env"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:?MODEL_CACHE_ROOT is required (set in $CONFIG_ROOT/inference.env)}"
PROJECT_ROOT="${PROJECT_ROOT:-/usr/local/lib/dgx-spark-inference}"
PROD_CONTAINER="${CONTAINER_NAME:-inference-agentic}"
PROD_PORT="${PORT:-30000}"
PROD_UNIT="${SYSTEMD_UNIT:-dgx-spark-inference.service}"
PROD_HELPER_UNIT="${HELPER_SYSTEMD_UNIT:-inference-agentic-helper.service}"
PROD_HELPER_CONTAINER="${HELPER_CONTAINER_NAME:-inference-agentic-helper}"

EXP_CONTAINER="dflash-experimental"
EXP_PORT="${EXP_PORT:-30100}"
: "${DFLASH_IMAGE:?DFLASH_IMAGE is required for an experimental run}"
: "${DFLASH_IMAGE_ID:?DFLASH_IMAGE_ID is required for an experimental run}"

usage() { sed -n '2,23p' "$0"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --port) EXP_PORT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "run-experimental.sh: unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done

# The committed bundle spec declares the validated 32768 context. Refuse profile
# drift rather than silently expanding the experiment's safety envelope.
BUNDLE_SPEC="$PROJECT_ROOT/bundles/experimental/qwen36-27b-fp8-dflash/sglang.toml"
[ -f "$BUNDLE_SPEC" ] || { echo "missing bundle spec: $BUNDLE_SPEC" >&2; exit 1; }
CONTEXT="$(python3 - "$BUNDLE_SPEC" <<'PY'
import sys, tomllib
print(tomllib.load(open(sys.argv[1], "rb"))["launch"]["context_length"])
PY
)"
[ "$CONTEXT" = "32768" ] || {
  echo "REFUSING: experimental bundle context is $CONTEXT; only 32768 is qualified" >&2
  exit 1
}
echo "[dflash-experimental] requested context: $CONTEXT"

# ---- safety: refuse to collide with production -------------------------------
if systemctl is-active --quiet "$PROD_UNIT" 2>/dev/null; then
  echo "REFUSING: production unit '$PROD_UNIT' is active. Stop it first" >&2
  echo "  (sudo systemctl stop $PROD_UNIT) within a maintenance window." >&2
  exit 1
fi
if systemctl is-active --quiet "$PROD_HELPER_UNIT" 2>/dev/null; then
  echo "REFUSING: production helper unit '$PROD_HELPER_UNIT' is active. Stop it first." >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$PROD_CONTAINER"; then
  echo "REFUSING: production container '$PROD_CONTAINER' exists. Remove it first." >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$PROD_HELPER_CONTAINER"; then
  echo "REFUSING: production helper container '$PROD_HELPER_CONTAINER' exists. Remove it first." >&2
  exit 1
fi
if [ "$EXP_PORT" = "$PROD_PORT" ]; then
  echo "REFUSING: experimental port ($EXP_PORT) equals the production port ($PROD_PORT)." >&2
  exit 1
fi
ACTUAL_IMAGE_ID="$(docker image inspect "$DFLASH_IMAGE" --format '{{.Id}}' 2>/dev/null || true)"
[ "$ACTUAL_IMAGE_ID" = "$DFLASH_IMAGE_ID" ] || {
  echo "REFUSING: experimental image mismatch for $DFLASH_IMAGE" >&2
  echo "expected=$DFLASH_IMAGE_ID actual=${ACTUAL_IMAGE_ID:-missing}" >&2
  exit 1
}

# ---- launch via the bundle branch of the real adapter ------------------------
# The adapter's bundle branch renders the runtime YAML and launches docker. We
# set CONTAINER_NAME/PORT to the experimental values so it cannot touch the
# production slot. SGLANG_API_KEY must be present (the adapter validates it).
ADAPTER="$PROJECT_ROOT/runtime/sglang/adapters/sglang.sh"
RUNTIME_ROOT="$PROJECT_ROOT/runtime/sglang"
[ -x "$ADAPTER" ] || { echo "adapter missing/not executable: $ADAPTER" >&2; exit 1; }
: "${SGLANG_API_KEY:?SGLANG_API_KEY is required (export it for the experimental run)}"

echo "[dflash-experimental] launching: container=$EXP_CONTAINER port=$EXP_PORT"
echo "[dflash-experimental] Ctrl-C to tear down. This is NOT a managed service."
# Isolation: tell the adapter this is experimental so it does NOT source the
# production inference.env (which would clobber our isolated port/container and
# route the run onto the production slot). Pass MODEL_CACHE_ROOT explicitly.
export DGX_INFERENCE_EXPERIMENTAL=1
export DGX_RUNTIME_IMAGE="$DFLASH_IMAGE"
export DGX_RUNTIME_IMAGE_ID="$DFLASH_IMAGE_ID"
export MODEL_CACHE_ROOT
export CONTAINER_NAME="$EXP_CONTAINER"
export PORT="$EXP_PORT"
export DGX_RUNTIME_CONFIG_DIR="${XDG_RUNTIME_DIR:-/tmp}/${EXP_CONTAINER}-${UID}"
# The bundle spec is an installed immutable artifact under PROJECT_ROOT.
SPEC_ARG="$BUNDLE_SPEC"
exec "$ADAPTER" "dflash-experimental" "$RUNTIME_ROOT" "$PROJECT_ROOT" \
     "qwen36-27b-fp8-dflash" "bundle" "$SPEC_ARG" \
     "qwen3.6-27b-dflash-experimental"
