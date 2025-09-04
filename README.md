# Hikma - Saudi Labor Law AI Assistant Backend

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-latest-blue)](https://www.docker.com/)
[![VLLM](https://img.shields.io/badge/VLLM-0.5.4-orange)](https://github.com/vllm-project/vllm)

---
[Watch Hikma Prototype Testing Video](media/hikma_prototype_testing_video.webm)

---

## Overview

**Hikma** is an AI-powered chatbot for **Saudi Labor Law** queries.  
It leverages a **Retrieval-Augmented Generation (RAG)** system to provide accurate legal guidance by searching through uploaded PDF law documents.  

The backend is built with **FastAPI** and uses **VLLM** Docker containers for LLM inference. **Qdrant** is used for vector storage of embeddings.

---

## Features

- Chatbot with **RAG system** for Saudi Labor Law
- Fast and scalable **LLM inference** with VLLM Docker
- **PDF ingestion** and chunking
- **Vector search** via Qdrant
- Chat history support per session
- Auto-summarized context for LLM responses
- Handles multiple chats and messages
- Supports **interactive and dynamic prompts**

---

## Architecture

```text
User ---> FastAPI Backend ---> RAG System ---> Qdrant Vector DB
                                      |
                                      ---> VLLM Docker for LLM inference
```
FastAPI: Handles REST API calls (/ask) for chat.

RAG System: Embeds queries, searches Qdrant, and generates prompts.

Qdrant: Stores embeddings and metadata for fast similarity search.

VLLM Docker: Runs LLM model inference in a containerized environment.

## Getting Started

Prerequisites

    Docker & Docker Compose
    Python 3.10+
    Git

## Clone Repository
```
git clone https://github.com/omumarvaishya005/Hikma-Fast-API-backend.git
cd Hikma-Fast-API-backend
```

## Install Python Dependencies
```
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Setting Up Qdrant
```
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

    //Stores your document embeddings for vector search.

    //Persistent storage is in qdrant_storage folder.

## Setting Up VLLM Docker
```
docker pull vllm/vllm-openai:v0.5.4
docker run -d \
  --name vllm-server \
  -p 9090:8000 \
  -v $(pwd)/models:/app/models \
  vllm/vllm-openai:v0.5.4 \
  --max-model-len 4096
```
Port 9090 is used to serve LLM inference requests.

Place your downloaded models in models/ directory.

--max-model-len can be increased for larger context windows.

## Running the Backend
```
uvicorn app.main:app --reload --host 0.0.0.0 --port 7500

   // Endpoint: POST /ask
```
## Body Example:

```
{
  "question": "What are the working hours in Saudi Arabia?"
}
```

## Response Example:

```
{
  "answer": "The standard working hours in Saudi Arabia are ...",
  "context": [ ... ]
}
```

## Project Structure

```
Hikma-Fast-API-backend/
│
├─ app/
│  ├─ api.py              # API routes
│  ├─ services/
│  │   ├─ embeddings.py   # Handles embeddings generation
│  │   ├─ qdrant.py       # Qdrant client & collection
│  │   └─ pdf_loader.py   # PDF ingestion and chunking
│  └─ models/
│      └─ generated.py    # Any generated model schemas
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ qdrant_storage/        # Persistent vector DB
└─ README.md
```


