#!/usr/bin/env bash
# run-experimental.sh — EXPERIMENTAL DFlash launcher. NOT a production path.
# See bundles/experimental/qwen36-27b-fp8-dflash/README.md.
#
# Safety (load-bearing):
#   - defaults to the VALIDATED 32768 context (the only fit validated so far);
#   - 262144 requires an explicit, separately-named dangerous override
#     (--i-understand-262144-fit-is-unvalidated);
#   - REFUSES to launch if the production service is active or the production
#     container exists (no collision on the production slot);
#   - launches on a SEPARATE container name + port so it can never take over the
#     production endpoint;
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

EXP_CONTAINER="dflash-experimental"
EXP_PORT="${EXP_PORT:-30100}"
FORCE_262144=0

usage() { sed -n '2,23p' "$0"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --port) EXP_PORT="$2"; shift 2;;
    --i-understand-262144-fit-is-unvalidated) FORCE_262144=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "run-experimental.sh: unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done

# The committed bundle spec already declares the validated 32768 default, so the
# default path uses it directly. The 262144 override writes a corrected spec to a
# TEMP file (never under the installed PROJECT_ROOT, which is immutable).
BUNDLE_SPEC="$PROJECT_ROOT/bundles/experimental/qwen36-27b-fp8-dflash/sglang.toml"
[ -f "$BUNDLE_SPEC" ] || { echo "missing bundle spec: $BUNDLE_SPEC" >&2; exit 1; }
OVERRIDE_SPEC=""
if [ "$FORCE_262144" = 1 ]; then
  OVERRIDE_SPEC="$(mktemp --suffix=.toml)"
  python3 - "$BUNDLE_SPEC" "$OVERRIDE_SPEC" 262144 <<'PY'
import sys
src, dst, ctx = sys.argv[1], sys.argv[2], sys.argv[3]
# Re-emit the bundle spec with context_length overridden. Only the scalar/table
# shapes present in the committed bundle are handled; invalid input aborts.
try:
    import tomllib
    d = tomllib.load(open(src, "rb"))
except Exception as e:
    raise SystemExit(f"cannot read bundle spec {src}: {e}")
d["launch"]["context_length"] = int(ctx)
def q(v):  # quote scalars for the structure used here
    return v if isinstance(v, (int, float)) else f'"{v}"'
out = [
    f'kind = "{d["kind"]}"',
    f'bundle_id = "{d["bundle_id"]}"',
    f'runtime_id = "{d["runtime_id"]}"',
    "",
    "[components.target]",
    f'model_id = "{d["components"]["target"]["model_id"]}"',
    "",
    "[components.drafter]",
    f'model_id = "{d["components"]["drafter"]["model_id"]}"',
    f'role = "{d["components"]["drafter"]["role"]}"',
    "",
    "[coordination]",
    f'algorithm = "{d["coordination"]["algorithm"]}"',
    f'num_draft_tokens = {d["coordination"]["num_draft_tokens"]}',
    "",
    "[launch]",
]
for k, v in d["launch"].items():
    out.append(f"{k} = {q(v)}")
open(dst, "w").write("\n".join(out) + "\n")
PY
  trap 'rm -f "$OVERRIDE_SPEC"' EXIT
  echo "[dflash-experimental] WARNING: launching at 262144 — fit is UNVALIDATED."
fi
echo "[dflash-experimental] requested context: $([ "$FORCE_262144" = 1 ] && echo 262144 || echo 32768)"

# ---- safety: refuse to collide with production -------------------------------
if systemctl is-active --quiet "$PROD_UNIT" 2>/dev/null; then
  echo "REFUSING: production unit '$PROD_UNIT' is active. Stop it first" >&2
  echo "  (sudo systemctl stop $PROD_UNIT) within a maintenance window." >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$PROD_CONTAINER"; then
  echo "REFUSING: production container '$PROD_CONTAINER' exists. Remove it first." >&2
  exit 1
fi
if [ "$EXP_PORT" = "$PROD_PORT" ]; then
  echo "REFUSING: experimental port ($EXP_PORT) equals the production port ($PROD_PORT)." >&2
  exit 1
fi

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
export MODEL_CACHE_ROOT
export CONTAINER_NAME="$EXP_CONTAINER"
export PORT="$EXP_PORT"
# Commit-issued bundle spec is relative to PROJECT_ROOT; the 262144 override is
# an absolute temp path. The adapter resolves both (relative->PROJECT_ROOT).
SPEC_ARG="${OVERRIDE_SPEC:-$BUNDLE_SPEC}"
exec "$ADAPTER" "dflash-experimental" "$RUNTIME_ROOT" "$PROJECT_ROOT" \
     "qwen36-27b-fp8-dflash" "bundle" "$SPEC_ARG" \
     "qwen3.6-27b-dflash-experimental"
