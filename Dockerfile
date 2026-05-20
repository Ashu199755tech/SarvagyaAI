# Use Python 3.12 slim image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Step 1: Install heavy AI libraries (CPU-only versions) FIRST
# This saves ~5GB of space compared to the default GPU-enabled versions.
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 3: Install NLP models
RUN pip install --no-cache-dir spacy nltk
RUN python -m spacy download en_core_web_sm

# Step 4: Copy the application code
COPY . .

# Step 5: Setup directories
RUN mkdir -p data/inbox data/converted data/chromadb logs && \
    chmod -R 777 data logs

EXPOSE 8001

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8001

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
