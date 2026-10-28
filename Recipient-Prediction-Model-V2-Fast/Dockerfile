FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir . gunicorn>=22,<24
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "gunicorn webapp:app --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 180"]
