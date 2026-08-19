FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install ".[serve]"

EXPOSE 8000

# Mount a versioned checkpoint and set BIOAI_CHECKPOINT=/models/student.pt.
CMD ["uvicorn", "bioai.api:app", "--host", "0.0.0.0", "--port", "8000"]
