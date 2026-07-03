# SGLang NVFP4 compatibility overlay

This runtime is an isolated, immutable overlay on the repository's pinned
SGLang v0.5.14 image. It contains the five SGLang source-file fixes required by
the official NVIDIA Qwen3.6-27B-NVFP4 checkpoint.

The qualified image is identified authoritatively by `image_id` in
`runtime-manifest.toml`. It was built from:

- SGLang v0.5.14 source commit
  `49e384ce9d304648e9959666ecb8ce8cd98d0deb`;
- the ModelOpt mixed-precision changes submitted upstream as SGLang commit
  `8b6fe539f1`;
- the base image and digest recorded in `runtime-manifest.toml`.

`Dockerfile.overlay` records the overlay shape. Its build context must be a
reviewed v0.5.14 SGLang checkout containing the backported source changes:

```bash
docker build \
  -f Dockerfile.overlay \
  -t local/sglang-runtime:v0.5.14-cu130-runtime-distro1.9.0-nvfp4fix \
  .
```

The Dockerfile alone does not assert byte-for-byte reproduction of the
qualified image; verify the resulting image ID before changing the manifest.
