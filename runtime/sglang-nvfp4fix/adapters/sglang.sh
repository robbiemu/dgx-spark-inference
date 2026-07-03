#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${3:?PROJECT_ROOT positional argument is required}"
exec "$PROJECT_ROOT/runtime/sglang/adapters/sglang.sh" "$@"
