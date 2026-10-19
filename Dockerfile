# Unikin als Container. LM Studio läuft nativ auf dem Host und wird über
# host.docker.internal:1234 angesprochen (siehe docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY unikin.py mcp_server.py mcp_client.py ./

# Nicht als root laufen. UID 1000 entspricht dem VM-Benutzer von Colima bzw.
# dem ersten Benutzer auf Linux-Hosts, damit Bind-Mounts beschreibbar sind.
RUN useradd --uid 1000 --user-group --no-create-home --shell /usr/sbin/nologin unikin \
    && mkdir -p /app/memory /app/logs \
    && chown unikin:unikin /app/memory /app/logs
USER unikin

VOLUME ["/app/memory", "/app/logs"]

CMD ["python", "unikin.py"]
