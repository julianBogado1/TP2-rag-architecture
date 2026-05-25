# Evaluation Metrics

This document describes the four metrics used to evaluate the RAG music recommendation pipeline.
Metrics are split into two categories: **Retrieval** (quality of what Pinecone + reranker surfaces)
and **Generation** (quality of what the LLM produces from those results).

---

## 1. Context Precision@K (Retrieval)

**What it measures:** Whether relevant songs are ranked higher than irrelevant ones in the top K results.

**Why it matters for recommendations:** A good recommender should not just retrieve relevant songs — it should surface them early. A system that buries the best matches at position 9 is worse than one that puts them at position 1.

**Formula:**

$$\text{CP@K} = \frac{\sum_{k=1}^{K} \text{Precision@k} \times v_k}{\text{total relevant in top } K}$$

$$\text{Precision@k} = \frac{TP@k}{k}, \quad v_k \in \{0, 1\}$$

where $v_k = 1$ if the song at rank $k$ belongs to the ground truth set, and 0 otherwise.

**Ground truth:** A song is considered relevant if its audio feature satisfies the threshold for the test category (e.g., `valence > 0.7` for the "Happy" prompt).

**Example:** For a "Happy songs" query with K=3 and GT = {song_A, song_B}:
- Rank 1: song_X (irrelevant) → v₁=0, Precision@1=0
- Rank 2: song_A (relevant) → v₂=1, Precision@2=1/2
- Rank 3: song_B (relevant) → v₃=1, Precision@3=2/3
- CP@3 = (0.5×1 + 0.67×1) / 2 = **0.58**

---

## 2. Recall@K (Retrieval — adapted)

**What it measures:** How many of the relevant songs were actually surfaced in the top K results.

**Why it matters for recommendations:** Precision tells us if the top songs are good; Recall tells us if we are missing important songs. Both together give a fuller picture.

**Adaptation from classic RAG:** The standard Context Recall formula divides by the total number of GT sentences. In a recommendation system, the GT set can contain thousands of eligible songs — a system returning K=10 results would always score ~0.001 even if it's perfect. We cap the denominator at K to make the metric meaningful.

**Formula:**

$$\text{Recall@K} = \frac{|\{top\text{-}K\} \cap GT|}{min(K, |GT|)}$$

**Example:** K=10, |GT|=500, 7 of the top 10 songs are in GT:
- Recall@10 = 7 / min(10, 500) = 7/10 = **0.70**

---

## 3. Faithfulness (Generation)

**What it measures:** Whether the LLM-generated explanation for each song is grounded in the retrieved lyric evidence, or whether it hallucinates.

**Why it matters for recommendations:** The LLM writes a short explanation (~40 words) citing song lyrics. If it invents claims not present in the lyrics, the explanation is misleading — even if the song itself is a good recommendation.

**Formula:**

$$\text{Faithfulness} = \frac{|\text{explanations supported by evidence\_chunks}|}{|\text{total recommendations}|}$$

**Implementation:** A GPT-4o-mini judge receives the `evidence_chunks` (top lyric snippets) and the generated `explanation`. It decides whether every claim in the explanation can be inferred from the context. Score = fraction of recommendations that pass.

**Example:**
- Context: *"baby I'm dancing in the dark with you between my arms"*
- Explanation: *"A romantic dance track about loving connection"* → supported ✓
- Explanation: *"A heavy metal anthem about rebellion"* → not supported ✗
- Faithfulness = 1/2 = **0.50**

---

## 4. Answer Relevance (Generation)

**What it measures:** Whether the recommendation response actually addresses what the user asked. A high score means the response is on-topic; a low score means it drifted off or gave an incomplete answer.

**Why it matters for recommendations:** The LLM could return a grammatically correct response that ignores the mood/genre requested. This metric catches that drift.

**Formula (from RAGAS):**

$$\text{AR} = \frac{1}{N} \sum_{i=1}^{N} \cos(E_{g_i}, E_o)$$

where:
- $E_o$ = embedding of the original user prompt
- $E_{g_i}$ = embedding of the $i$-th reverse question generated from the response
- $N$ = number of generated questions (3)

**Implementation:** GPT-4o-mini generates 3 questions that a user could have asked to receive the response. Each is embedded with `sentence-transformers/all-MiniLM-L6-v2` (same model used for lyrics). Cosine similarity is computed against the embedded original prompt and averaged.

**Example:**
- Prompt: *"I want happy upbeat songs"*
- Generated questions: *"What mood songs do you want?"*, *"Can you recommend cheerful tracks?"*, *"Looking for energetic music?"*
- Each embeds close to the original → AR ≈ **0.88**

---

## Summary Table

| Metric | Stage | Needs LLM | Needs GT |
|--------|-------|-----------|----------|
| Context Precision@K | Retrieval | No | Yes |
| Recall@K | Retrieval | No | Yes |
| Faithfulness | Generation | Yes (judge) | No |
| Answer Relevance | Generation | Yes (reverse Q) | No |

## Ground Truth Construction

GT is built from the MongoDB song collection using Spotify audio features (normalized 0–1):

| Category | Feature | Condition |
|----------|---------|-----------|
| Happy | valence | > 0.7 |
| Sad | valence | < 0.3 |
| Energetic | energy | > 0.8 |
| Calm | energy | < 0.3 |
| Danceable | danceability | > 0.7 |
| Acoustic | acousticness | > 0.7 |
| Instrumental | instrumentalness | > 0.5 |
