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

### Seed user profiles

```bash
.venv/bin/python scripts/seed_users.py
```

### Ingest joined songs dataset

Streams the full Spotify tracks dataset into memory, then streams Genius song lyrics up to `<max_songs>`, joining both by title and artist. Results are stored in the `songs` collection with Spotify audio features populated where a match exists and `null` otherwise.

```bash
.venv/bin/python scripts/ingest_songs_joined.py <max_songs>
```

Example — ingest the first 5 000 songs:

```bash
.venv/bin/python scripts/ingest_songs_joined.py 5000
```

# Tests

```bash
.venv/bin/python -m pytest tests/ -v
```