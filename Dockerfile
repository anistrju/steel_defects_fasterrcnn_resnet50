
# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

LABEL maintainer="anianirudha2007@gmail.com"
LABEL description="Streamlit Steel Defect Detection App"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PORT=8501  

WORKDIR /app

# System deps
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app and model
COPY app.py .
  

EXPOSE ${PORT}

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "app.py", "--server.port=${PORT}", "--server.address=0.0.0.0"]