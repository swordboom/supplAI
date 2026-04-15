# SupplAI — Dockerfile for Google Cloud Run
# -------------------------------------------------
# Build:  docker build -t supplai .
# Run:    docker run -p 8080:8080 supplai
# Deploy: cloudbuild.yaml handles push → Artifact Registry → Cloud Run

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system build tools (needed by XGBoost, LightGBM, scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Streamlit server config — headless, listen on 0.0.0.0:8080 for Cloud Run
RUN mkdir -p /app/.streamlit
RUN printf '[server]\nheadless = true\naddress = "0.0.0.0"\nport = 8080\nenableCORS = false\nenableXsrfProtection = false\n\n[theme]\nbase = "dark"\n' \
    > /app/.streamlit/config.toml

# Cloud Run sets PORT env var — default to 8080
ENV PORT=8080

EXPOSE 8080

# Entrypoint
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
