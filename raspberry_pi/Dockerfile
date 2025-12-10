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
    && cmake --build raspberry_pi/build --target rpi_pcm_bridge_app rpi_rtp_sender -j"$(nproc)"

FROM ${BASE_IMAGE} AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libasound2 \
        gstreamer1.0-alsa \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        libzmq3-dev \
        python3 \
        python3-pip \
        curl \
        ca-certificates \
        bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/pcm-bridge
COPY --from=builder /app/raspberry_pi/build/src/rpi_pcm_bridge /usr/local/bin/rpi_pcm_bridge
COPY --from=builder /app/raspberry_pi/build/src/rpi_rtp_sender /usr/local/bin/rpi_rtp_sender
COPY raspberry_pi/docker/entrypoint.sh /usr/local/bin/pcm-bridge-entrypoint.sh
RUN chmod +x /usr/local/bin/pcm-bridge-entrypoint.sh
RUN pip3 install --no-cache-dir pyzmq
COPY raspberry_pi/rtp_receiver /opt/pcm-bridge/raspberry_pi/rtp_receiver
COPY raspberry_pi/__init__.py /opt/pcm-bridge/raspberry_pi/__init__.py
ENV PYTHONPATH=/opt/pcm-bridge:${PYTHONPATH}

ENV PCM_BRIDGE_DEVICE=hw:0,0 \
    PCM_BRIDGE_HOST=127.0.0.1 \
    PCM_BRIDGE_PORT=46001 \
    PCM_BRIDGE_RATE=48000 \
    PCM_BRIDGE_FORMAT=S16_LE \
    PCM_BRIDGE_FRAMES=4096 \
    PCM_BRIDGE_LOG_LEVEL=info \
    PCM_BRIDGE_ITERATIONS=-1 \
    RTP_SENDER_DEVICE=hw:0,0 \
    RTP_SENDER_HOST=127.0.0.1 \
    RTP_SENDER_RTP_PORT=46000 \
    RTP_SENDER_RTCP_PORT=46001 \
    RTP_SENDER_RTCP_LISTEN_PORT=46002 \
    RTP_SENDER_PAYLOAD_TYPE=96 \
    RTP_SENDER_POLL_MS=250 \
    RTP_SENDER_LOG_LEVEL=warn

ENTRYPOINT ["/usr/local/bin/pcm-bridge-entrypoint.sh"]
CMD []
