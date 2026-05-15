# tp2-rag-arquitecture
In this project we set to implement an LLM with RAG arquitecture

# Setup

### Create the virtual environment

```bash
python3 -m venv .venv
```

### Install dependencies

```bash
.venv/bin/pip install -r requirements.txt
```

# Running the API

```bash
.venv/bin/uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

# Scripts

### Populate MongoDB

```bash
.venv/bin/python scripts/seed_users.py
```

### Ingest songs dataset

```bash
.venv/bin/python scripts/ingest_songs.py <max_songs>
```

Example — ingest the first 5 000 songs:

```bash
.venv/bin/python scripts/ingest_songs.py 5000
```

### Fetch songs from mongoDB

```bash
.venv/bin/python scripts/fetch_songs.py
```

# Indexing Pipeline

Reads songs from MongoDB → splits lyrics into chunks → embeds with `sentence-transformers/all-MiniLM-L6-v2` → upserts vectors into Pinecone.

### Prerequisites

1. MongoDB running with songs ingested (see Scripts above)
2. `.env` contains `PINECONE_API_KEY` and `PINECONE_INDEX_NAME`

### Trigger via API (recommended)

With the API running, call:

```bash
curl -X POST http://localhost:8000/index
```

Response:
```json
{"total_docs": 5000, "total_chunks": 31200, "total_indexed": 31200}
```

### Run directly (without API)

```bash
.venv/bin/python -m app.controllers.indexing_controller
```

# Full Setup — Step by Step

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill in environment variables
cp .env.example .env

# 4. Start MongoDB
bash scripts/mongo-start.sh

# 5. Ingest songs into MongoDB (only needed once — data persists in Docker volume)
python scripts/ingest_songs.py 5000

# 6. Start the API
uvicorn app.main:app --reload

# 7. Run the indexing pipeline (separate terminal)
curl -X POST http://localhost:8000/index
```