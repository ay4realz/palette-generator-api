# ==========================
# Palette Generator API Dockerfile
# For Hugging Face Spaces (FastAPI backend)
# ==========================

# --- Base image ---
FROM python:3.10-slim

# --- Set working directory ---
WORKDIR /app

# --- Copy project files ---
COPY . /app

# --- Install system dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran wget git && \
    rm -rf /var/lib/apt/lists/*

# --- Upgrade pip and install Python packages ---
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# --- Expose the port Hugging Face expects ---
EXPOSE 7860

# --- Default startup command (Hugging Face detects this) ---
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
