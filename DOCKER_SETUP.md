# Setup Guide for ResoAI

Yo! Here’s how you can get the project running on your system using Docker. It’s pretty straightforward, but you’ll need a few things set up first.

## 1. Prerequisites
Before you start, make sure you have these installed:
*   **Docker Desktop** (or Docker Engine on Linux)
*   **Ollama** (Running locally on your machine)

## 2. Get the Code
Clone the repo and jump into the folder:
```bash
git clone https://github.com/Ashu199755tech/SarvagyaAI.git
cd SarvagyaAI
```

## 3. Environment Setup
You need to tell the Docker container where your Ollama is. Create a `.env` file in the root folder and add this:
*   **If you are on Windows or Mac:**
    `OLLAMA_BASE_URL=http://host.docker.internal:11434`
*   **If you are on Linux:**
    `OLLAMA_BASE_URL=http://172.17.0.1:11434`

## 4. Spin up Docker
Run this to build and start the server:
```bash
docker-compose up -d --build
```
*Wait a minute for it to finish building.*

## 5. Load the Knowledge Base (Don't skip this!)
The database starts empty, so you need to "seed" it with the data. Run these two commands:

**First, consolidate the employees and projects:**
```bash
docker-compose exec resoai-api python3 -m src.consolidator.run
```

**Second, run the RAG ingestion pipeline:**
```bash
docker-compose exec resoai-api python3 -m src.ingestion.pipeline
```

## 6. Test it out
Once the ingestion is done, you can test if the AI is awake:
```bash
curl -X POST http://localhost:8000/api/ask \
-H "Content-Type: application/json" \
-d '{"question": "How many employees are in devops?"}'
```

That’s it! The API is running on port 8000. Let me know if you hit any snags.
