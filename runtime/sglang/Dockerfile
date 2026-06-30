FROM lmsysorg/sglang:v0.5.14-cu130-runtime@sha256:9e436f44523e9f53519c6175fefd1e0d373322bf54b8154bb331a2f5e4840ad2

RUN python3 -m pip install \
      --no-cache-dir \
      --disable-pip-version-check \
      "distro==1.9.0" \
    && python3 -c 'import distro; print("distro=" + distro.__version__)'
