#!/usr/bin/env bash
set -Eeuo pipefail

URL="${1:?usage: wait_for_health.sh URL [TIMEOUT_SECONDS]}"
TIMEOUT_SECONDS="${2:-600}"
INTERVAL_SECONDS="${WAIT_HEALTH_INTERVAL_SECONDS:-2}"

[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "wait_for_health.sh: timeout must be a positive integer" >&2
  exit 2
}

deadline=$((SECONDS + TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  code="$(curl --silent --show-error --output /dev/null \
    --write-out '%{http_code}' --max-time 2 "$URL" 2>/dev/null || true)"
  if [ "$code" = 200 ]; then
    echo "wait_for_health.sh: ready: $URL"
    exit 0
  fi
  sleep "$INTERVAL_SECONDS"
done

echo "wait_for_health.sh: timed out after ${TIMEOUT_SECONDS}s: $URL" >&2
exit 75
