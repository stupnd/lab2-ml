FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Note: .env is NOT copied into the image intentionally.
# Secrets are injected at runtime via environment variables
# e.g. docker run -e API_KEY=secret ...
# or via docker-compose environment section.

EXPOSE 8080

ENV PRODUCTION=true

CMD ["python", "app.py"]
