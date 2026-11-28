# CPU-only rli-server image:
#   docker build -t ezrankayamba/rli-server:latest .
# The model is not in the image: mount it and point RLI_MODEL at it, or set RLI_MODEL_URL to
# download it on first start (see docker/entrypoint.sh).

ARG RUST_VERSION=1.99.1

# Kernels pick their SIMD at run time, so a generic build serves every CPU of the architecture.
FROM rust:${RUST_VERSION}-slim-trixie AS build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p rli-server --no-default-features \
 && cp target/release/rli-server /rli-server

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends tini curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --no-create-home rli \
 && install -d -o rli -g rli /models
COPY --from=build /rli-server /usr/local/bin/rli-server
COPY docker/entrypoint.sh /usr/local/bin/rli-entrypoint
ENV RLI_MODEL=/models/model.gguf RLI_HOST=0.0.0.0 RLI_PORT=8080
USER rli
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD ["bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/${RLI_PORT} && printf 'GET /healthz HTTP/1.0\\r\\n\\r\\n' >&3 && grep -q ok <&3"]
ENTRYPOINT ["tini", "--", "rli-entrypoint"]
