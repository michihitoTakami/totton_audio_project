# Raspberry Pi PCM Bridge container (ARM)

ARG BASE_IMAGE=debian:12-slim

FROM ${BASE_IMAGE} AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY raspberry_pi/ ./raspberry_pi/
RUN cmake -S raspberry_pi -B raspberry_pi/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    && cmake --build raspberry_pi/build --target rpi_pcm_bridge_app -j"$(nproc)"

ARG RUNTIME_IMAGE=${BASE_IMAGE}
FROM ${RUNTIME_IMAGE} AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libasound2 \
        ca-certificates \
        bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/pcm-bridge
COPY --from=builder /app/raspberry_pi/build/rpi_pcm_bridge /usr/local/bin/rpi_pcm_bridge
COPY raspberry_pi/docker/entrypoint.sh /usr/local/bin/pcm-bridge-entrypoint.sh
RUN chmod +x /usr/local/bin/pcm-bridge-entrypoint.sh

ENV PCM_BRIDGE_DEVICE=hw:0,0 \
    PCM_BRIDGE_HOST=127.0.0.1 \
    PCM_BRIDGE_PORT=46001 \
    PCM_BRIDGE_RATE=48000 \
    PCM_BRIDGE_FORMAT=S16_LE \
    PCM_BRIDGE_FRAMES=4096 \
    PCM_BRIDGE_LOG_LEVEL=info \
    PCM_BRIDGE_ITERATIONS=-1

ENTRYPOINT ["/usr/local/bin/pcm-bridge-entrypoint.sh"]
CMD []
