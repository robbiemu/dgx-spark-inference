#!/usr/bin/env bash
# build-runtime.sh — build the local sglang runtime image and print its image ID
# for pinning into runtime-manifest.toml.
#
# Why this exists: the adapter pins an EXACT image ID (sha256:...) and refuses to
# launch if the local image's ID drifts from the pin (a supply-chain safety rail).
# That means building the Dockerfile produces a *different* ID than the committed
# pin, so a fresh build must record its own ID before the adapter will accept it.
#
# Usage:
#   scripts/build-runtime.sh            # builds, prints the image_id to pin
#   scripts/build-runtime.sh --update   # builds AND rewrites the manifest pin
#
# This script does NOT push the image anywhere. It is local-only.
set -Eeuo pipefail

UPDATE=0
case "${1:-}" in
  --update) UPDATE=1;;
  -h|--help) sed -n '2,18p' "$0"; exit 0;;
  "") ;;
  *) echo "build-runtime.sh: unknown option: $1" >&2; exit 2;;
esac

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/runtime/sglang/runtime-manifest.toml"
DOCKERFILE="$ROOT/runtime/sglang/Dockerfile"
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST" >&2; exit 1; }
[ -f "$DOCKERFILE" ] || { echo "missing $DOCKERFILE" >&2; exit 1; }

# Read the declared image tag from the manifest (single source of truth for the name).
TAG="$(grep -E '^image[[:space:]]*=' "$MANIFEST" | head -1 | sed -E 's/.*=\s*"([^"]+)".*/\1/')"
[ -n "$TAG" ] || { echo "could not read 'image' from $MANIFEST" >&2; exit 1; }

echo "Building $TAG from $DOCKERFILE ..."
docker build --tag "$TAG" --file "$DOCKERFILE" "$ROOT/runtime/sglang"

# The image ID the adapter pins is docker's content-addressable ID (.Id), not the
# RepoDigest. Print it in the exact form the manifest expects.
ID="$(docker image inspect "$TAG" --format '{{.Id}}')"
echo
echo "Built: $TAG"
echo "image_id = \"$ID\""

if [ "$UPDATE" = 1 ]; then
  python3 - "$MANIFEST" "$ID" <<'PY'
import sys, re
path, new_id = sys.argv[1], sys.argv[2]
text = open(path).read()
new_text, n = re.subn(r'(?m)^image_id\s*=\s*".*"', f'image_id = "{new_id}"', text)
if n != 1:
    print(f"could not update image_id in {path} (found {n} matches)", file=sys.stderr); sys.exit(1)
open(path, "w").write(new_text)
print(f"updated image_id in {path}")
PY
else
  echo
  echo "To pin this build, either:"
  echo "  re-run: scripts/build-runtime.sh --update"
  echo "  or edit runtime/sglang/runtime-manifest.toml: image_id = \"$ID\""
fi
