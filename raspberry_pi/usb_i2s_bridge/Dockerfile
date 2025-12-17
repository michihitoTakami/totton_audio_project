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
        python3 \
        python3-zmq \
        python3-gi \
        gir1.2-gstreamer-1.0 \
        gir1.2-gst-plugins-base-1.0 \
        ca-certificates \
        bash \
        util-linux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/usb-i2s-bridge

# Copy only what we need (RTP実装とは独立)
COPY raspberry_pi/__init__.py /opt/usb-i2s-bridge/raspberry_pi/__init__.py
COPY raspberry_pi/usb_i2s_bridge /opt/usb-i2s-bridge/raspberry_pi/usb_i2s_bridge
COPY raspberry_pi/usb_i2s_bridge/docker/entrypoint.sh /usr/local/bin/usb-i2s-bridge-entrypoint.sh
RUN chmod +x /usr/local/bin/usb-i2s-bridge-entrypoint.sh

ENV PYTHONPATH=/opt/usb-i2s-bridge:${PYTHONPATH}

# Defaults (override via env / compose)
ENV USB_I2S_CAPTURE_DEVICE=hw:2,0 \
    USB_I2S_PLAYBACK_DEVICE=hw:0,0 \
    USB_I2S_CHANNELS=2 \
    USB_I2S_FALLBACK_RATE=48000 \
    USB_I2S_PREFERRED_FORMAT=S32_LE \
    USB_I2S_PASSTHROUGH=true \
    USB_I2S_ALSA_BUFFER_TIME_US=200000 \
    USB_I2S_ALSA_LATENCY_TIME_US=20000 \
    USB_I2S_QUEUE_TIME_NS=100000000 \
    USB_I2S_FADE_MS=80 \
    USB_I2S_POLL_INTERVAL_SEC=1.0 \
    USB_I2S_RESTART_BACKOFF_SEC=0.5 \
    USB_I2S_KEEP_SILENCE=true \
    USB_I2S_STATUS_PATH=/var/run/usb-i2s-bridge/status.json

ENTRYPOINT ["/usr/local/bin/usb-i2s-bridge-entrypoint.sh"]
CMD []
