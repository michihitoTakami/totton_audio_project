FROM debian:12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libasound2 \
        alsa-utils \
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
COPY rtp_receiver /opt/pcm-bridge/raspberry_pi/rtp_receiver
COPY rtp_sender.py /opt/pcm-bridge/raspberry_pi/rtp_sender.py
COPY __init__.py /opt/pcm-bridge/raspberry_pi/__init__.py
COPY docker/entrypoint.sh /usr/local/bin/pcm-bridge-entrypoint.sh
RUN chmod +x /usr/local/bin/pcm-bridge-entrypoint.sh
RUN pip3 install --no-cache-dir --break-system-packages pyzmq

ENV PYTHONPATH=/opt/pcm-bridge:${PYTHONPATH}

ENV RTP_SENDER_DEVICE=hw:2,0 \
    RTP_SENDER_HOST=192.168.55.1 \
    RTP_SENDER_RTP_PORT=46000 \
    RTP_SENDER_RTCP_PORT=46001 \
    RTP_SENDER_RTCP_LISTEN_PORT=46002 \
    RTP_SENDER_PAYLOAD_TYPE=96 \
    RTP_SENDER_SAMPLE_RATE=44100 \
    RTP_SENDER_CHANNELS=2 \
    RTP_SENDER_FORMAT=S24_3BE \
    RTP_SENDER_LATENCY_MS=100 \
    RTP_SENDER_AUTO_SAMPLE_RATE=true \
    RTP_SENDER_RATE_POLL_INTERVAL_SEC=2.0 \
    RTP_BRIDGE_STATS_PATH=/tmp/rtp_receiver_stats.json

ENTRYPOINT ["/usr/local/bin/pcm-bridge-entrypoint.sh"]
CMD []
