# rag-starter — RAG Pipeline for Semantic Search and Grounded Generation

A production-ready RAG (Retrieval-Augmented Generation) pipeline that ingests documents, embeds them into a vector database, retrieves semantically relevant chunks for a user query, and generates grounded answers using Claude. Built with Chroma, Anthropic, and Python.

![CLI requesting a grounded answer from Claude based on chunks (RAG) from Chroma](./assets/rag-starter.png)

---

## What it does

- Ingests markdown or text documents from a local directory
- Chunks documents with configurable chunk size and overlap
- Embeds chunks using Chroma's default embedding function (all-MiniLM-L6-v2, runs locally)
- Stores embeddings in a persistent Chroma vector database
- Retrieves top-K semantically similar chunks for any user query
- Generates Claude-grounded answers using only retrieved context
- Returns both the answer and source attribution (which documents were used)
- Prevents hallucination by constraining Claude to say "I don't know" when context is insufficient

---

## How it works

A user query flows through four stages:

**Document ingestion and chunking:** Raw markdown files are loaded from `./data/`, split into chunks (default: 500 tokens with 50-token overlap using recursive chunking), and prepared for embedding. Chunking strategy balances retrieval granularity (smaller chunks = more precise results) against context preservation (larger chunks = more surrounding context per result).

**Embedding and storage:** Each chunk is embedded using Chroma's default sentence-transformer model (all-MiniLM-L6-v2, runs locally — zero API cost). Embeddings are stored in a persistent Chroma collection at `./chroma_db/` alongside metadata (source file, chunk index). The persistent collection survives between runs and can be queried without re-ingesting.

**Retrieval:** When a user asks a question, the question is embedded using the same model and used as a query vector against the stored embeddings. Chroma returns the top-K most similar chunks (default: 3) ranked by cosine similarity. This semantic search retrieves relevant context even if keywords don't match exactly.

**Grounded generation:** Retrieved chunks are formatted with source attribution and passed to Claude as context. A system prompt explicitly instructs Claude to answer only from the provided context. Claude generates an answer grounded in the retrieved chunks and cites which chunks support the answer. If the context doesn't contain the answer, Claude says "I don't know based on the provided context" rather than hallucinating.

---

## Architecture: RAG vs long-context

This implementation uses chunked retrieval (RAG) — splitting the corpus into pieces, embedding each, and selectively retrieving only the relevant chunks for each query.

**Why RAG for rag-starter?**
- Works with corpora larger than a single LLM context window
- Reduces per-query latency (retrieve 3 chunks, not 100K tokens)
- Reduces per-query cost (pay for retrieved context only, not the entire corpus)
- Scales gracefully as the corpus grows

**When long-context beats RAG:**
As of 2025, Anthropic's Sonnet 4.6 and Opus 4.6 support 1M-token context at flat-rate pricing. For corpora that fit entirely in context (most documents under ~800K tokens), a single long-context call is often simpler and sometimes cheaper than chunking + retrieval. 

If you're building a similar system for a different corpus, benchmark both approaches:
- Time and cost to ingest and maintain RAG pipeline
- Per-query latency and cost for RAG retrieval + generation
- Per-query cost for long-context call with full corpus

For production, measure against your actual usage patterns before committing to either.

---

## Limitations

- Embedding quality depends on the corpus — retrieval is only as good as the embeddings. For domain-specific jargon (medical, legal, technical), consider fine-tuned or domain-specific embedding models in production.
- Chunk size and overlap are fixed — no adaptive chunking based on document structure
- No query expansion or reranking — retrieved chunks are returned in embedding similarity order without additional refinement
- No eval harness in W4 — retrieval quality is tested manually. Automated eval harness (precision, recall, hallucination rate) added in W5E.
- Persistent collection stored locally — not suitable for multi-user or distributed scenarios without additional infrastructure
- **Prompt caching not yet implemented** — Anthropic prompt caching will be
  applied at production hardening (W13) once a token-cost baseline is established.


---

## Quick start

**Clone and enter the project:**

    git clone https://github.com/digitalrower/rag-starter.git
    cd rag-starter

**Pin Python version (requires pyenv):**

    pyenv local 3.13.3
    python --version              # should show Python 3.13.3

**Create and activate a virtual environment:**

    python -m venv .venv
    source .venv/bin/activate     # Mac/Linux
    # Windows: .venv\Scripts\activate

**Install dependencies:**

    pip install -r requirements.txt

**Set up environment variables:**

    cp .env.example .env

Open `.env` and replace the placeholder with your actual Anthropic API key:

    ANTHROPIC_API_KEY=your_actual_api_key_here

---

## Ingest documents

Documents should be placed in `./data/` as markdown or text files.

**Run the ingestion pipeline:**

    python -m src.ingest

This will:
1. Load all `.md` files from `./data/`
2. Chunk them (500 tokens, 50-token overlap)
3. Embed and store in `./chroma_db/`
4. Print a summary: "Ingested X chunks from Y documents"

The collection is persistent — run this once, then query as many times as you want without re-ingesting.

---

## Query the pipeline

**From the command line:**

    python -m src.query "What are agent skills?"

**Output:**

```
{
    'answer': "Based on the provided documentation, **Agent Skills** are modular capabilities that extend Claude's functionality. Here are the key points:\n\n## What They Are\nEach Skill packages instructions, metadata, and optional resources (scripts, templates) that Claude uses automatically when relevant.\n\n## Why Use Them\nSkills are reusable, filesystem-based resources that provide Claude with domain-specific expertise: workflows, context, and best practices that transform general-purpose agents into specialists. Key benefits include:\n\n- **Specialize Claude**: Tailor capabilities for domain-specific tasks\n- **Reduce repetition**: Create once, use automatically across multiple conversations\n- **Compose capabilities**: Combine Skills to build complex workflows\n\n## Types of Skills\n\n1. **Pre-built Agent Skills**: Anthropic provides pre-built Skills for common document tasks (PowerPoint, Excel, Word, PDF). These are available on claude.ai, the Claude API, Claude Platform on AWS, and Microsoft Foundry.\n\n2. **Custom Skills**: You can create your own Skills to package domain expertise and organizational knowledge. These are available across Claude's products and can be created in Claude Code, uploaded through the Claude API, or added in claude.ai settings.\n\nUnlike prompts (which are conversation-level instructions for one-off tasks), Skills load on-demand and eliminate the need to repeatedly provide the same guidance across multiple conversations.", 
    'sources': ['agent-skills.md', 'managed-agents-overview.md']
}

```




**Test multiple queries:**

For testing grounding, try:
- A query answerable from your corpus (verify Claude cites sources)
- A query NOT in your corpus (verify Claude says "I don't know" — no hallucination)
- An ambiguous query (verify Claude acknowledges ambiguity and uses context)

---

## Requirements

- Python 3.13+
- Git
- An Anthropic API key ([get one here](https://console.anthropic.com))

Dependencies are listed in `requirements.txt`. See [Tech stack](#tech-stack) below.

---

## Project structure

    rag-starter/
    ├── src/
    │   ├── ingest.py        # Load, chunk, embed, store
    │   └── query.py         # Retrieve, generate, return grounded answer
    ├── data/                # Source documents (markdown/text)
    ├── chroma_db/           # Persistent vector database (gitignored)
    ├── assets/
    ├── .env.example         # Environment variable template
    ├── .gitignore
    ├── .python-version
    ├── requirements.txt
    └── README.md

---

## Ingest configuration

Edit these parameters in `src/ingest.py` to tune ingestion behavior:

| Parameter | Default | Impact |
|-----------|---------|--------|
| `chunk_size` | 500 tokens | Larger = more context per chunk, fewer total chunks. Smaller = more granular retrieval, more chunks. |
| `chunk_overlap` | 50 tokens | Overlap between adjacent chunks. Prevents context loss at chunk boundaries. |

---

## Retrieval configuration

Edit these parameters in `src/query.py` to tune retrieval behavior:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_results` | 3 | Retrieved chunk count per query. Start at 3, increase if Claude needs more context. |
| `max_tokens` | 500 | Max response length. Increase if answers are truncated; decrease to reduce cost. |

---

## API reference

### Ingest

**Purpose:** Load documents, chunk, embed, and store in Chroma.

**Command:**

    python -m src.ingest

**Output:**

    Ingested 145 chunks from 4 documents into collection 'anthropic_docs'

**Effect:** Creates or updates `./chroma_db/` with the persistent collection.

---

### Query

**Purpose:** Retrieve relevant chunks and generate a grounded answer.

**Command:**

    python -m src.query "your question here"

**Output:**

```json
{
  "answer": "Claude's grounded answer based on retrieved context...",
  "sources": ["source-file-1.md", "source-file-2.md"]
}
```

**Behavior:**
- Returns top-3 semantically similar chunks
- Constrains Claude to answer only from context
- Returns sources for transparency and verification

---

## Implementation highlights

- **Local embeddings:** Uses Chroma's default embedding function (sentence-transformers all-MiniLM-L6-v2), which runs locally with zero API cost. Swappable in production for OpenAI, Voyage, or other providers.
- **Persistent storage:** Chroma collection persists to disk, allowing multiple queries without re-ingestion.
- **Source attribution:** Every answer includes which documents the chunks came from, enabling verification and trust.
- **Grounding constraints:** System prompt explicitly prevents hallucination by instructing Claude to say "I don't know" when context is insufficient.
- **Modular functions:** Separate `retrieve_chunks()`, `build_prompt()`, and `generate_answer()` functions are importable for use in other projects (Streamlit demos, FastAPI services, eval harnesses).

---

## Error handling

Common issues and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| `Collection not found` | `ingest.py` hasn't been run yet | Run `python -m src.ingest` first |
| `ANTHROPIC_API_KEY not set` | Missing `.env` file or key | Copy `.env.example` to `.env` and add your key |
| `AuthenticationError` | Invalid API key | Verify key at console.anthropic.com |
| `RateLimitError` | Too many requests to Claude | Wait a moment and retry |
| Empty retrieval results | No chunks match the query | Try a different query or check corpus relevance |

---

## Tech stack

- [Chroma](https://docs.trychroma.com/) — Vector database (local persistence)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) — Claude API client
- [sentence-transformers](https://www.sbert.net/) — Embedding model (runs locally via Chroma)
- [python-dotenv](https://github.com/theskumar/python-dotenv) — Environment variable management

---

## Testing grounding

Before moving to W5E (eval harness), manually test grounding with these scenarios:

**Test 1 — Answerable query:**

    python -m src.query "What are agent skills?"

Expected: Claude answers confidently and cites source chunks.

**Test 2 — Unanswerable query:**

    python -m src.query "what is the capital of mars"

Expected: Claude says "I don't know based on the provided context" (no hallucination).

**Test 3 — Before/after comparison:**

Run the same query through `src/query.py` (with retrieval) and compare to Claude's answer without retrieval (just the system prompt and question, no context). 
- Does retrieval change the answer?
- Is the grounded answer more accurate or more cautious?
- Does Claude cite sources when retrieval is used?

---

## Next steps

- **W5E:** Build eval harness with 30+ golden Q/A pairs and multi-dimensional scoring (precision, recall, hallucination rate)
- **W9:** Adapt ingestion and retrieval logic for Project 1 (Docs Copilot) — same architecture, different corpus
- **W13:** Add prompt caching to reduce redundant token cost on repeated queries
- **W28+:** Benchmark against OpenAI's long-context APIs to decide when RAG is overkill

---

## License

MIT
