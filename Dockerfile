# Containerizes the FastAPI service (the "production" surface — see api/main.py).
# Build:  docker build -t youtube-sentiment-api .
# Run:    docker run -p 8000:8000 youtube-sentiment-api
#
# Note: this Dockerfile was written to standard practice but NOT build-tested in
# the environment this project was created in (no Docker daemon available there —
# see the project guide). Test it with the two commands above before you rely on
# it for a demo; if anything's off it's almost certainly a path issue in COPY.

FROM python:3.10-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer across rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY src/ src/
COPY api/ api/
COPY data/ data/
COPY train.py .

# Train the model at build time so the image is ready to serve immediately.
# For a real production setup you'd instead mount a pre-trained model volume or
# pull it from a model registry — baking it into the image is the simple version.
RUN python train.py

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
