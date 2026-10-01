FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.4.1 torchvision==0.19.1

COPY api.py ./
COPY ml/ ./ml/
COPY models/ ./models/

ENV PORT=8080
CMD exec gunicorn -w 1 -b 0.0.0.0:$PORT --timeout 120 api:app
