FROM python:3.11-slim

# Install system dependencies (libgomp1 is required by ctranslate2/faster-whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set PYTHONPATH to include the src folder so that ml_models can be imported directly
ENV PYTHONPATH=/app/src

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.web.api:app", "--host", "0.0.0.0", "--port", "8000"]
