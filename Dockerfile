FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip \
    && pip install .[web]

RUN mkdir -p /app/runs /app/outputs /app/logs

ENV CLAWLEARN_WEB_HOST=0.0.0.0 \
    CLAWLEARN_WEB_PORT=7860

EXPOSE 7860

CMD ["clawlearn-web"]
