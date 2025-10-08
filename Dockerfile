# Use a stable Python image compatible with TensorFlow and SciPy
FROM python:3.10-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies (build tools + gfortran + git)
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy your project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Render expects
EXPOSE 10000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
