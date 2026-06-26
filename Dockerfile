FROM python:3.11-slim

WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir --timeout 120 -r requirements-web.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["sh", "-c", "uvicorn dashboard.server:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 120"]