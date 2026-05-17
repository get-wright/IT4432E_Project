FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY application/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY training_pipeline /app/training_pipeline
COPY application /app/application

ENV APP_CKPT=/app/application/models/best.pt
ENV APP_DATA_DIR=/app/application/embeddings
ENV APP_THRESHOLD=0.5
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "application.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
