FROM debian:12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libasound2 \
        alsa-utils \
        python3 \
        python3-gi \
        python3-zmq \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-alsa \
        gir1.2-gstreamer-1.0 \
        ca-certificates \
        bash \
        util-linux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/usb-i2s-bridge

# Copy only what we need (RTP実装とは独立)
COPY raspberry_pi/__init__.py /opt/usb-i2s-bridge/raspberry_pi/__init__.py
COPY raspberry_pi/usb_i2s_bridge /opt/usb-i2s-bridge/raspberry_pi/usb_i2s_bridge
COPY raspberry_pi/usb_i2s_bridge/usb-i2s-bridge.env /opt/usb-i2s-bridge/raspberry_pi/usb_i2s_bridge/usb-i2s-bridge.env
COPY raspberry_pi/usb_i2s_bridge/docker/entrypoint.sh /usr/local/bin/usb-i2s-bridge-entrypoint.sh
RUN chmod +x /usr/local/bin/usb-i2s-bridge-entrypoint.sh

ENV PYTHONPATH=/opt/usb-i2s-bridge:${PYTHONPATH}

# Defaults (override via env / compose)
ENV USB_I2S_CONFIG_PATH=/var/lib/usb-i2s-bridge/config.env

ENTRYPOINT ["/usr/local/bin/usb-i2s-bridge-entrypoint.sh"]
CMD []
