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

### Start MongoDB

Creates the Docker container on first run, or starts it if it already exists:

```bash
bash scripts/mongo-start.sh
```

### Seed user profiles

```bash
.venv/bin/python scripts/seed_users.py
```

### Ingest songs dataset

Streams the full Spotify tracks dataset into memory, then streams Genius song lyrics up to `<max_songs>`, joining both by title and artist. Results are stored in the `songs` collection with Spotify audio features populated where a match exists and `null` otherwise.

```bash
.venv/bin/python scripts/ingest_songs_joined.py <max_songs>
```

Example — ingest the first 5 000 songs:

```bash
.venv/bin/python scripts/ingest_songs_joined.py 5000
```

### Drop songs collection (to re-ingest)

```bash
.venv/bin/python scripts/clear_mongo.py
```

### Inspect MongoDB container

Enter the container shell:

```bash
docker exec -it mongo-rag mongosh -u admin -p secret
```

Once inside, switch to the database and query songs:

```js
use rag_db

// count total songs
db.songs.countDocuments()

// see first 5 songs (without lyrics for readability)
db.songs.find({}, { lyrics: 0 }).limit(5)

// find a song with audio features populated
db.songs.findOne({ loudness: { $ne: null } }, { lyrics: 0 })

// find songs by artist
db.songs.find({ artist: "JAY-Z" }, { title: 1, year: 1, _id: 0 })

// find songs by genre
db.songs.find({ tag: "rap" }, { title: 1, artist: 1, _id: 0 }).limit(10)
```

# Indexing Pipeline

Reads songs from the `songs` collection → splits lyrics into chunks → embeds with `sentence-transformers/all-MiniLM-L6-v2` → upserts vectors into Pinecone.

Each vector is stored with this metadata shape:

```json
{
  "song_id": 123,
  "track_name": "Its Like That",
  "artist_name": "JAY-Z",
  "genres": ["rap"],
  "release_date": "1998",
  "song_characteristics_chunk": "{\"popularity\": 72, \"duration_ms\": 280000, \"explicit\": true, \"tempo\": 84.115, \"time_signature\": 4, \"language\": \"en\"}",
  "audio_features_chunk": "{\"danceability\": 0.773, \"energy\": 0.54, \"key\": 6, \"loudness\": -7.123, \"mode\": 1, \"speechiness\": 0.103, \"acousticness\": 0.371, \"instrumentalness\": 0, \"liveness\": 0.131, \"valence\": 0.322}"
}
```

Songs without Spotify data will have empty `{}` for both chunk fields.

### Prerequisites

1. MongoDB running with songs ingested (see Scripts above)
2. `.env` contains `PINECONE_API_KEY` and `PINECONE_INDEX_NAME`

### Run the pipeline

With the API running, call:

```bash
curl -X POST http://localhost:8000/index
```

Response:

```json
{"total_docs": 5000, "total_chunks": 31200, "total_indexed": 31200}
```

### Clear the Pinecone index

To wipe all vectors before a re-index (e.g. after a metadata schema change):

```bash
.venv/bin/python scripts/clear_pinecone.py
```

# Full Setup — Step by Step

```bash
# 1. Create virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Copy and fill in environment variables
cp .env.example .env

# 3. Start MongoDB
bash scripts/mongo-start.sh

# 4. Ingest songs (only needed once — data persists in Docker volume)
.venv/bin/python scripts/ingest_songs_joined.py 5000

# 5. Start the API
.venv/bin/uvicorn app.main:app --reload

# 6. Run the indexing pipeline (separate terminal)
curl -X POST http://localhost:8000/index
```
