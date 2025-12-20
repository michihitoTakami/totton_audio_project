FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/raspi-control
COPY raspberry_pi /opt/raspi-control/raspberry_pi

RUN pip install --no-cache-dir fastapi uvicorn docker pyzmq

ENV PYTHONPATH=/opt/raspi-control

ENTRYPOINT ["python3", "-m", "raspberry_pi.control_api"]
