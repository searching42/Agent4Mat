FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Default image uses deterministic CPU profile.
RUN bash -lc "./scripts/install_profile.sh cpu"

CMD ["bash", "-lc", "PYTHONPATH=src python3 -m oled_agent.cli doctor --workspace-root /app"]
