# runtime/sglang — the sglang runtime

The sglang *mechanism*: the pinned Docker image, the runtime manifest, the
candidate catalog (`available.toml`), the capability record, and the launch
adapter. This directory does not know about specific model pairings beyond what
the catalog lists. See [`../../docs/architecture.md`](../../docs/architecture.md).

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the local sglang runtime image (a thin layer over the pinned upstream image). |
| `runtime-manifest.toml` | What the runtime *is*: image tag + **pinned image ID**, launch defaults, supported config kinds, known limitations. |
| `available.toml` | The candidate catalog per role (production; DFlash is intentionally absent from `agentic`). |
| `capability.toml` | The runtime capability record (resolver input). |
| `adapters/sglang.sh` | The launch adapter (lives at the path the manifest declares). |

## Building the runtime image (important)

The adapter **pins an exact image ID** (`image_id` in `runtime-manifest.toml`) and
refuses to launch if the local image's ID does not match. This is a supply-chain
safety rail — but it means: **if you build the Dockerfile yourself, the resulting
image has a *different* ID than the committed pin, and the adapter will reject it
until you update the pin.**

The committed pin corresponds to *our* build of this Dockerfile. If you reproduce
the build (same Dockerfile, same base digest), record *your* image ID before
launching. Use the helper:

```bash
# Build, print the image_id to pin:
scripts/build-runtime.sh

# Build AND rewrite the manifest pin in one step:
scripts/build-runtime.sh --update
```

`scripts/build-runtime.sh` reads the declared `image` tag from the manifest,
builds, inspects the built image's content-addressable ID
(`docker image inspect … --format '{{.Id}}'`), and either prints it or writes it
back into `runtime-manifest.toml`. It does not push the image — it is local-only.

### Manual equivalent

```bash
docker build --tag local/sglang-runtime:v0.5.14-cu130-runtime-distro1.9.0 \
             --file runtime/sglang/Dockerfile runtime/sglang
docker image inspect local/sglang-runtime:v0.5.14-cu130-runtime-distro1.9.0 \
             --format '{{.Id}}'
# paste the printed sha256:... into runtime-manifest.toml as image_id
```

## Why the image ID pin exists

The pin guards against an unintended image replacing the runtime between releases
(docker pull, tag drift, a stale local build). The trade-off is that reproduction
requires re-pinning, which is why `scripts/build-runtime.sh` exists.
