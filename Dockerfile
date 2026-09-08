FROM python:3.13.7-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/app/.cache/huggingface
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/uploads /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app/.cache
COPY --chown=app:app src ./src
COPY --chown=app:app frontend ./frontend
USER app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
