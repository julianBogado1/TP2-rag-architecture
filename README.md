# tp2-rag-arquitecture
In this project we set to implement an LLM with RAG arquitecture.

A natural-language prompt → ranked song recommendations, using song **lyrics** as
the semantic corpus (Pinecone, MiniLM 384-dim) and Spotify **audio features** as a
second ranking signal. FastAPI + MongoDB + Pinecone + OpenAI `gpt-4o-mini`.

> **Full architecture, diagrams and workflows:** see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
> Worked metric examples: [`docs/metrics.md`](docs/metrics.md).
> How the evaluation metrics evolved (with charts): [`docs/metrics-evolution-2026-05-29.md`](docs/metrics-evolution-2026-05-29.md).

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

Streams the full Spotify tracks dataset into memory, then streams Genius song lyrics up to `<max_songs>`, joining both on a normalized title + artist key (accent/case-folded, `feat.`/cosmetic suffixes stripped, multi-artist any-match; see `app/services/song_matcher.py`). Results are stored in the `songs` collection with Spotify audio features populated where a match exists and `null` otherwise.

```bash
.venv/bin/python scripts/ingest_songs_joined.py <max_songs>
```

Example — ingest the first 5 000 songs:

```bash
.venv/bin/python scripts/ingest_songs_joined.py 5000
```

### Drop songs collection (to re-ingest)

Destructive — requires confirmation. Pass `--yes` to skip the prompt:

```bash
.venv/bin/python scripts/clear_mongo.py --yes
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
  "song_characteristics_chunk": "{\"popularity\": 72, \"duration_ms\": 280000, \"explicit\": true, \"time_signature\": 4, \"language\": \"en\"}",
  "audio_features_chunk": "{\"danceability\": 0.773, \"energy\": 0.54, \"key\": 6, \"loudness\": -7.123, \"mode\": 1, \"speechiness\": 0.103, \"acousticness\": 0.371, \"instrumentalness\": 0, \"liveness\": 0.131, \"valence\": 0.322, \"tempo\": 84.115}"
}
```

`tempo` lives in `audio_features_chunk` (the retrieval reader looks for it there).
Songs without a Spotify match have empty `{}` for both chunk fields — they are still
indexed and remain retrievable, scored on lyrics alone at recommendation time.

Lyrics are split into **400-char chunks (50 overlap)** — verse-sized, and inside the
embedding model's 256-token limit so chunk tails aren't silently truncated.

### Prerequisites

1. MongoDB running with songs ingested (see Scripts above)
2. `.env` contains `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, and `OPENAI_API_KEY`

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
#   Edit .env to set:
#     PINECONE_API_KEY=pcsk_...
#     OPENAI_API_KEY=sk-...

# 3. Start MongoDB
bash scripts/mongo-start.sh

# 4. Ingest songs (only needed once — data persists in Docker volume)
.venv/bin/python scripts/ingest_songs_joined.py 5000

# 5. Seed user profiles (required for /recommend)
.venv/bin/python scripts/seed_users.py

# 6. Start the API
.venv/bin/uvicorn app.main:app --reload

# 7. Run the indexing pipeline (separate terminal)
curl -X POST http://localhost:8000/index

# 8. Make a recommendation
curl -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "user_001", "raw_prompt": "quiero canciones felices :)"}'
```

# Recommendation endpoint

Once songs are indexed in Pinecone and a user profile exists in Mongo, the recommendation pipeline turns a natural-language prompt into 10 ranked songs.

```bash
curl -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "user_001", "raw_prompt": "quiero canciones que vayan con mi mood feliz :)"}'
```

Returns a `RecommendationResponse` with a `message` and up to 10 `recommendations`.

Pipeline stages (see `docs/2026-05-24-rag-recommendation-pipeline-example.md` for a full step-by-step trace):

1. `PromptParserService` — OpenAI structured-output → `PromptScore`.
2. `UserProfileRepository.get_by_user_id` — fetches `UserProfileData` from Mongo.
3. `RecommendationRequestBuilder` — derives filters and weight overrides.
4. `QueryEmbedderService` — embeds the cleaned `semantic_query` with the same model used at index time.
5. `VectorRetrievalService` — Pinecone search → JSON metadata parse → Python filtering.
6. `CandidateAggregatorService` — groups chunks by song, fetches evidence lyrics from Mongo.
7. `HybridRerankerService` — additive weighted-sum scoring (lyrics + audio + profile + popularity + recency); a missing axis contributes 0 (no weight redistribution). Audio similarity is a **normalized inverted Euclidean** distance over the prompt-implied axes (not cosine — cosine can't distinguish "low" vs "high" or single-axis targets).
8. `TopNSelectorService` — sort, dedup by title, cap per artist.
9. `ResponseGeneratorService` — OpenAI structured-output → `RecommendationResponse`. **Enabled by default** (`app/main.py` constructs it with `skip_llm=False`): OpenAI writes a short opener plus a per-song explanation citing evidence lyrics and matched moods. Construct the service with `skip_llm=True` to synthesise the response locally (no OpenAI call) — each `recommendation` then gets `explanation=""` and `matched_mood`/`matched_audio_features` derived from `PromptScore` fields > 0.5.

### Skip flag — what calls OpenAI today

| Stage                       | OpenAI call?                                 |
|-----------------------------|----------------------------------------------|
| `PromptParserService` (step 1) | Yes — always (parses raw prompt → `PromptScore`) |
| `ResponseGeneratorService` (step 9) | Yes — enabled by default (`skip_llm=False`)  |

To run without the response LLM (local synthesis, no OpenAI call for step 9), construct `ResponseGeneratorService(..., skip_llm=True)` in `app/main.py`.

### Smoke test (hits real Mongo + Pinecone + OpenAI for step 1)

```bash
.venv/bin/python scripts/smoke_recommend.py "happy upbeat songs"
.venv/bin/python scripts/smoke_recommend.py "songs about heartbreak" user_007
```

Prints the full `RecommendationResponse` JSON, including per-song `explanation` text from the response LLM (empty only if you run with `skip_llm=True`).

# Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

# Evaluation

Five metrics evaluate the pipeline end-to-end: **Context Precision@K**, **Recall@K** and **NDCG@K** measure retrieval/ranking quality; **Faithfulness** and **Answer Relevance** measure generation quality. See [docs/metrics.md](docs/metrics.md) for formulas and [docs/metrics-evolution-2026-05-29.md](docs/metrics-evolution-2026-05-29.md) for how they were tuned.

Ground truth has **two families**, each test case tagged with a `kind`:
- **audio** — relevance = a soft audio threshold (`valence>0.6` Happy, `energy<0.4` Calm, …); evaluated **only against songs that have audio features**.
- **genre** — one case per canonical genre (`pop, rap, rock, rb, country, misc`); relevance = `tag` membership ("ask for pop → get pop"); covers audio and no-audio songs.

### Prerequisites

- MongoDB running with songs ingested and Pinecone indexed (see Full Setup above)
- `OPENAI_API_KEY` set in `.env`
- Response LLM is enabled by default (`skip_llm=False`), so faithfulness/answer-relevance reflect real generations

### Step 1 — Build ground truth

Builds the audio + genre test cases from MongoDB, intersects each GT set with the indexed-ids file, and creates a neutral eval user with no preferences (to isolate retrieval quality from profile bias).

```bash
.venv/bin/python scripts/build_gt.py
```

Output: `data/gt_test_cases.json` — one entry per case with its `kind` and list of `gt_song_ids`.

### Step 2 — Run evaluation

Runs the full recommendation pipeline for each test case and computes all five metrics, **kind-aware**: audio cases are scored over the audio-bearing subset of the top-K; genre cases over all returned songs. Reports per-kind and overall averages.

```bash
.venv/bin/python scripts/run_evaluation.py
```

Output: printed metrics table + `data/evaluation_results.json`.

Tune the evaluation window and retrieval depth from `.env` (no code edits):

```
RETRIEVAL_TOP_K=150   # candidate pool the reranker reorders (lever for surfacing audio songs)
OUTPUT_TOP_N=10       # songs the pipeline returns (must be >= EVAL_K)
EVAL_K=10             # the @K window for CP@K / Rec@K / NDCG@K
```

Example output (illustrative numbers, not a measured run):

```
Label            Kind     CP@K  Rec@K   NDCG  Faith  AnsRel
-----------------------------------------------------------
Happy            audio   1.000  0.500  0.481  1.000   0.632
Sad              audio   0.838  0.580  0.668  1.000   0.633
Calm             audio   0.812  0.450  0.420  1.000   0.520
Genre:pop        genre   1.000  1.000  1.000  0.900   0.591
Genre:rap        genre   1.000  1.000  1.000  1.000   0.520
-----------------------------------------------------------
avg [audio]              0.90+  0.50+  0.50+  0.94    0.55
avg [genre]              1.000  1.000  1.000  0.90    0.50
```
