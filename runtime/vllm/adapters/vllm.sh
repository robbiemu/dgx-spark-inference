#!/usr/bin/env bash
# Same dispatcher contract and operator configuration precedence as SGLang.
set -Eeuo pipefail
caller_port="${PORT:-}"
caller_name="${CONTAINER_NAME:-}"
CONFIG_ROOT="${CONFIG_ROOT:-/etc/dgx-spark-inference}"
if [ "${DGX_INFERENCE_EXPERIMENTAL:-0}" != 1 ]; then
  [ ! -f "$CONFIG_ROOT/inference.env" ] || . "$CONFIG_ROOT/inference.env"
fi
[ -z "$caller_port" ] || PORT="$caller_port"
[ -z "$caller_name" ] || CONTAINER_NAME="$caller_name"
export PORT CONTAINER_NAME MODEL_CACHE_ROOT
exec python3 "$(dirname -- "${BASH_SOURCE[0]}")/adapter.py" "$@"
