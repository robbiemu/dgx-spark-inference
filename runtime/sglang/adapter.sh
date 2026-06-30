#!/usr/bin/env bash
# shellcheck disable=SC2034  # ROLE/MODEL_ID held for the 7-arg positional contract
# sglang launch adapter (binding v2). De-machine-speced for the public reference.
# See docs/architecture.md (the binding system) and SECURITY.md.
#
# INVOCATION (by dispatch.sh):
#   adapter.sh <ROLE> <RUNTIME_ROOT> <PROJECT_ROOT> <MODEL_ID> <KIND> <SPEC> <SERVED_NAME>
#
# The dispatcher has already resolved the active candidate (validated it is
# offered and the kind matches the spec). This adapter:
#   - reads the spec (model-runtime-config or bundle-runtime-config) at the spec path
#   - merges manifest defaults + spec overrides
#   - applies safety rails (image-id pin, api-key 64-hex) -> render config -> launch
# Branches on KIND: model (single) vs bundle (target+drafter+speculation).
#
# CONFIG DISCOVERY: ports, container name, and the model-cache root come from the
# operator environment file written by install.sh and surfaced by systemd as env:
#   $CONFIG_ROOT/inference.env   (INSTALL_ROOT, CONFIG_ROOT, MODEL_CACHE_ROOT,
#                                  PORT, CONTAINER_NAME, ROLE, PROJECT_ROOT)
# This file is the single source of machine-specific values; no host path, port,
# or container name is hard-coded here.
set -Eeuo pipefail

# Positional contract (see dispatch.sh). ROLE and MODEL_ID are held to keep the
# 7-arg signature legible/aligned even though this adapter resolves launch values
# from the spec rather than those two names directly.
ROLE="$1"
RUNTIME_ROOT="$2"
PROJECT_ROOT="$3"
MODEL_ID="$4"
KIND="$5"
SPEC="$6"
SERVED_NAME="$7"

# ---- operator configuration discovery --------------------------------------
# SAFETY: a caller that supplies its own PORT/CONTAINER_NAME/etc. (e.g. the
# experimental DFlash launcher) sets DGX_INFERENCE_EXPERIMENTAL=1. In that mode we
# do NOT source the production inference.env (it would clobber the caller's
# isolated values and route the run onto the production slot). The caller is then
# responsible for MODEL_CACHE_ROOT/PORT/CONTAINER_NAME.
CONFIG_ROOT="${CONFIG_ROOT:-/etc/dgx-spark-inference}"
if [ "${DGX_INFERENCE_EXPERIMENTAL:-0}" != "1" ]; then
  # shellcheck disable=SC1090,SC1091
  [ -f "$CONFIG_ROOT/inference.env" ] && . "$CONFIG_ROOT/inference.env"
fi
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:?MODEL_CACHE_ROOT is required (export it, or set it in $CONFIG_ROOT/inference.env)}"
PORT="${PORT:-30000}"
CONTAINER_NAME="${CONTAINER_NAME:-inference-agentic}"

# Runtime config dir is on tmpfs; the YAML is rendered 0600 root:root here.
RUNTIME_CONFIG_DIR="/run/${CONTAINER_NAME}"
RUNTIME_CONFIG_HOST="${RUNTIME_CONFIG_DIR}/sglang.yaml"
RUNTIME_CONFIG_CONTAINER="/etc/sglang-runtime/sglang.yaml"
# SPEC may be relative (resolved against PROJECT_ROOT, the normal case) or
# absolute (the experimental launcher passes a temp override spec).
case "$SPEC" in
  /*) SPEC_PATH="$SPEC" ;;
  *)  SPEC_PATH="$PROJECT_ROOT/$SPEC" ;;
esac

toml_get() {  # toml_get <file> <dotted.key>
  python3 - "$1" "$2" <<'PY'
import sys, tomllib
f, key = sys.argv[1], sys.argv[2]
d = tomllib.load(open(f,"rb")); cur=d
for part in key.split("."):
    if not isinstance(cur,dict) or part not in cur: sys.exit(44)
    cur=cur[part]
print(cur if not isinstance(cur,list) else " ".join(cur))
PY
}
pick() {  # pick <cfg> <manifest> <cfg.key> <manifest.key>
  m=$(toml_get "$1" "$3" 2>/dev/null) && { printf '%s' "$m"; return; }
  printf '%s' "$(toml_get "$2" "$4")"
}
# OPTIONAL vocabulary a model may omit AND the manifest may leave without a
# default. Prints the value if either source sets it; returns 3 (prints nothing)
# when neither does, so emit_yaml can OMIT the YAML key entirely (sglang then
# computes it = unchanged behavior for un-pinned units).
pick_optional() {  # pick_optional <cfg> <manifest> <cfg.key> <manifest.key>
  m=$(toml_get "$1" "$3" 2>/dev/null) && { printf '%s' "$m"; return 0; }
  m=$(toml_get "$2" "$4" 2>/dev/null) && { printf '%s' "$m"; return 0; }
  return 3
}

[ -f "$SPEC_PATH" ] || { echo "REFUSING: spec missing: $SPEC_PATH" >&2; exit 1; }
MANIFEST="$RUNTIME_ROOT/runtime-manifest.toml"
IMAGE="$(toml_get "$MANIFEST" image)"
EXPECTED_IMAGE_ID="$(toml_get "$MANIFEST" image_id)"
HOST_BIND="$(toml_get "$MANIFEST" common_launch.host_bind)"
CACHE_ROOT="$(toml_get "$MANIFEST" common_launch.container_cache_root)"

# ---- emit_yaml: the ONE render path (used live AND by the --emit-yaml probe) -
# Writes (or prints, see mode) the runtime YAML from merged values.
#   emit_yaml <mode: write|print> <ctx> <mr> <mq> <mfs> <api_key_or_placeholder>
emit_yaml() {
  local mode="$1" ctx="$2" mr="$3" mq="$4" mfs="$5" key="$6" out
  out=$({
    printf 'served-model-name: "%s"\n' "$SERVED_NAME"
    printf 'host: "%s"\n' "$HOST_BIND"
    printf 'port: %s\n' "$PORT"
    printf 'context-length: %s\n' "$ctx"
    printf 'max-running-requests: %s\n' "$mr"
    printf 'max-queued-requests: %s\n' "$mq"
    # Optional: only emitted when the unit pinned mem_fraction_static (otherwise
    # the key is absent and sglang computes the fraction = unchanged behavior).
    [ -n "$mfs" ] && printf 'mem-fraction-static: %s\n' "$mfs"
    printf 'reasoning-parser: "%s"\n' "$(toml_get "$MANIFEST" common_launch.reasoning_parser)"
    printf 'tool-call-parser: "%s"\n' "$(toml_get "$MANIFEST" common_launch.tool_call_parser)"
    printf 'log-level: "%s"\n' "$(toml_get "$MANIFEST" common_launch.log_level)"
    printf 'log-level-http: "%s"\n' "$(toml_get "$MANIFEST" common_launch.log_level_http)"
    printf 'api-key: "%s"\n' "$key"
  })
  if [ "$mode" = print ]; then
    printf '%s\n' "$out"
    return
  fi
  install -d -o root -g root -m 0700 "$RUNTIME_CONFIG_DIR"
  local tmp="${RUNTIME_CONFIG_HOST}.tmp.$$"
  printf '%s\n' "$out" > "$tmp"
  install -o root -g root -m 0600 "$tmp" "$RUNTIME_CONFIG_HOST"
  rm -f "$tmp"
}

# ---- emit-yaml mode: a no-launch probe (for tests / dry inspection) ----------
# Renders the same YAML the live path would write, using a non-secret placeholder
# key, and prints it to stdout. Performs NO safety rails and launches nothing.
# Usage: adapter.sh ... emit-yaml   (set as 8th arg)
if [ "${8:-}" = "emit-yaml" ]; then
  CTX="$(pick "$SPEC_PATH" "$MANIFEST" launch.context_length common_launch.context_length)"
  MR="$(pick "$SPEC_PATH" "$MANIFEST" launch.max_running_requests common_launch.max_running_requests)"
  MQ="$(pick "$SPEC_PATH" "$MANIFEST" launch.max_queued_requests common_launch.max_queued_requests)"
  MFS="$(pick_optional "$SPEC_PATH" "$MANIFEST" launch.mem_fraction_static common_launch.mem_fraction_static || true)"
  emit_yaml print "$CTX" "$MR" "$MQ" "$MFS" "REDACTED-PLACEHOLDER"
  exit 0
fi

# ---- safety rails (shared, live path only) ----------------------------------
: "${SGLANG_API_KEY:?SGLANG_API_KEY is required (systemd EnvironmentFile)}"
if [[ ! "$SGLANG_API_KEY" =~ ^[0-9a-f]{64}$ ]]; then
  echo "REFUSING: SGLANG_API_KEY is not the expected 64-char lowercase hex." >&2; exit 1
fi
ACTUAL_IMAGE_ID="$(/usr/bin/docker image inspect "$IMAGE" --format '{{.Id}}')"
[ "$ACTUAL_IMAGE_ID" = "$EXPECTED_IMAGE_ID" ] || {
  echo "REFUSING: image ID drifted (expected $EXPECTED_IMAGE_ID, got $ACTUAL_IMAGE_ID)." >&2; exit 1; }

umask 077

# ===========================================================================
if [ "$KIND" = "model" ]; then
  # ---- SINGLE MODEL ---------------------------------------------------------
  # Profiles describe, never contain weights. local_dir is a deterministic
  # subdir of MODEL_CACHE_ROOT (documented in the profile README). The adapter
  # mounts MODEL_CACHE_ROOT -> CACHE_ROOT read-only and points sglang at it.
  LOCAL_DIR="$(toml_get "$SPEC_PATH" identity.local_dir)"
  CONTAINER_MODEL_PATH="${CACHE_ROOT%/}/${LOCAL_DIR#/}"
  CTX="$(pick "$SPEC_PATH" "$MANIFEST" launch.context_length common_launch.context_length)"
  MR="$(pick "$SPEC_PATH" "$MANIFEST" launch.max_running_requests common_launch.max_running_requests)"
  MQ="$(pick "$SPEC_PATH" "$MANIFEST" launch.max_queued_requests common_launch.max_queued_requests)"
  MFS="$(pick_optional "$SPEC_PATH" "$MANIFEST" launch.mem_fraction_static common_launch.mem_fraction_static || true)"
  emit_yaml write "$CTX" "$MR" "$MQ" "$MFS" "$SGLANG_API_KEY"
  exec /usr/bin/docker run \
    --rm --name "$CONTAINER_NAME" --gpus all --ipc host \
    --publish "${HOST_BIND}:${PORT}:${PORT}" \
    --volume "${MODEL_CACHE_ROOT}:${CACHE_ROOT}:ro" \
    --volume "${RUNTIME_CONFIG_HOST}:${RUNTIME_CONFIG_CONTAINER}:ro" \
    --entrypoint /bin/sh "$IMAGE" \
    -ceu 'exec sglang serve --model-path "$1" --config "$2"' \
    sh "$CONTAINER_MODEL_PATH" "$RUNTIME_CONFIG_CONTAINER"

elif [ "$KIND" = "bundle" ]; then
  # ---- COORDINATED BUNDLE (target + drafter + speculation) ------------------
  # NOTE: production does not offer any bundle for agentic (DFlash is excluded
  # from available.toml). This branch exists for the experimental path only.
  TARGET_ID="$(toml_get "$SPEC_PATH" components.target.model_id)"
  DRAFTER_ID="$(toml_get "$SPEC_PATH" components.drafter.model_id)"
  TARGET_CFG="$PROJECT_ROOT/profiles/qwen36-27b-${TARGET_ID#qwen36-27b-}/sglang.toml"
  DRAFTER_CFG="$PROJECT_ROOT/profiles/experimental/qwen36-27b-${DRAFTER_ID#qwen36-27b-}/sglang.toml"
  [ -f "$TARGET_CFG" ] || { echo "REFUSING: target config missing: $TARGET_CFG" >&2; exit 1; }
  [ -f "$DRAFTER_CFG" ] || { echo "REFUSING: drafter config missing: $DRAFTER_CFG" >&2; exit 1; }
  TARGET_LOCAL_DIR="$(toml_get "$TARGET_CFG" identity.local_dir)"
  DRAFTER_LOCAL_DIR="$(toml_get "$DRAFTER_CFG" identity.local_dir)"
  ALGO="$(toml_get "$SPEC_PATH" coordination.algorithm)"
  NUM_DRAFT="$(toml_get "$SPEC_PATH" coordination.num_draft_tokens)"
  ATTN="$(pick "$SPEC_PATH" "$MANIFEST" launch.attention_backend common_launch.attention_backend)"
  CTX="$(pick "$SPEC_PATH" "$MANIFEST" launch.context_length common_launch.context_length)"
  MR="$(pick "$SPEC_PATH" "$MANIFEST" launch.max_running_requests common_launch.max_running_requests)"
  MQ="$(pick "$SPEC_PATH" "$MANIFEST" launch.max_queued_requests common_launch.max_queued_requests)"
  MFS="$(pick_optional "$SPEC_PATH" "$MANIFEST" launch.mem_fraction_static common_launch.mem_fraction_static || true)"
  emit_yaml write "$CTX" "$MR" "$MQ" "$MFS" "$SGLANG_API_KEY"
  exec /usr/bin/docker run \
    --rm --name "$CONTAINER_NAME" --gpus all --ipc host \
    --publish "${HOST_BIND}:${PORT}:${PORT}" \
    --volume "${MODEL_CACHE_ROOT}:${CACHE_ROOT}:ro" \
    --volume "${MODEL_CACHE_ROOT}/${DRAFTER_LOCAL_DIR#/}:/drafter:ro" \
    --volume "${RUNTIME_CONFIG_HOST}:${RUNTIME_CONFIG_CONTAINER}:ro" \
    --entrypoint /bin/sh "$IMAGE" \
    -ceu 'exec sglang serve --model-path "$1" --config "$2" \
        --speculative-algorithm "'"${ALGO}"'" \
        --speculative-draft-model-path /drafter \
        --speculative-num-draft-tokens "'"${NUM_DRAFT}"'" \
        --trust-remote-code --attention-backend "'"${ATTN}"'"' \
    sh "${CACHE_ROOT%/}/${TARGET_LOCAL_DIR#/}" "$RUNTIME_CONFIG_CONTAINER"

else
  echo "REFUSING: unknown kind '$KIND' (expected model|bundle)." >&2; exit 1
fi
