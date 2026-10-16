FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY service.py app.py ./
RUN mkdir models && useradd --create-home --uid 10001 appuser
USER appuser

CMD ["python", "-m", "uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
