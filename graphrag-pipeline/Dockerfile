# ── Build stage ────────────────────────────────────────────────────────────
FROM rust:1.85-bookworm AS builder

WORKDIR /app

# Copy manifests and patched orp first (for dependency caching)
COPY Cargo.toml Cargo.lock ./
COPY orp-patched orp-patched

# Create dummy src so cargo can resolve deps
RUN mkdir src && echo 'fn main() { println!("stub"); }' > src/main.rs

# Build dependencies only (cached unless Cargo.toml/lock/orp-patched change)
RUN cargo build --release 2>/dev/null || true
RUN rm -rf src

# Now copy real source and rebuild (only the binary, deps are cached)
COPY src src
RUN touch src/main.rs && cargo build --release

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime 1.20.1 (matches ort 2.0.0-rc.9 which requires 1.20.x)
ARG ORT_VERSION=1.20.1
RUN curl -fsSL \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    | tar xz -C /opt \
    && ln -s /opt/onnxruntime-linux-x64-${ORT_VERSION} /opt/onnxruntime

ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib

WORKDIR /app

# Copy the built binary
COPY --from=builder /app/target/release/graphrag-pipeline .

# Copy NER-only model (small, fast)
COPY models/gliner_small-v2.1/tokenizer.json        models/gliner_small-v2.1/tokenizer.json
COPY models/gliner_small-v2.1/gliner_config.json     models/gliner_small-v2.1/gliner_config.json
COPY models/gliner_small-v2.1/config.json            models/gliner_small-v2.1/config.json
COPY models/gliner_small-v2.1/onnx/model_int8.onnx   models/gliner_small-v2.1/onnx/model_int8.onnx

# Copy multitask model for NER + relation extraction (q4f16 quantized)
COPY models/gliner-multitask-large-v0.5/tokenizer.json       models/gliner-multitask-large-v0.5/tokenizer.json
COPY models/gliner-multitask-large-v0.5/gliner_config.json    models/gliner-multitask-large-v0.5/gliner_config.json
COPY models/gliner-multitask-large-v0.5/config.json           models/gliner-multitask-large-v0.5/config.json
COPY models/gliner-multitask-large-v0.5/onnx/model_quantized.onnx models/gliner-multitask-large-v0.5/onnx/model_quantized.onnx

# Sources are mounted at runtime
VOLUME ["/app/sources"]

ENTRYPOINT ["./graphrag-pipeline"]
CMD ["--help"]
