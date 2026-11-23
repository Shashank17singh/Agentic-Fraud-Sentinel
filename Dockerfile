#Base image
FROM python:3.11-slim


#Set working directory 
# All subsequent commands run from /app inside the container
# This is where your code lives inside the container

WORKDIR /app

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/

COPY notebooks/data/models/ ./data/models/


ENV PYTHONAPP=/app

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

# Run as a regular user, not root. It owns /app so a log or checkpoint file
# set through MONITOR_LOG_PATH or CHECKPOINT_DB_PATH can still be written.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn","api.main:app", "--host", "0.0.0.0", "--port", "8000"]





