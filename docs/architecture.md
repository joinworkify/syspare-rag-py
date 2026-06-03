# Architecture Overview

## What This Is

A **multimodal RAG (Retrieval-Augmented Generation)** API server. It ingests PDF manuals, extracts text and images, builds vector embeddings for both, stores them in a local cache (synced to S3), and answers natural-language questions by retrieving the most relevant text chunks and images and feeding them into Gemini.

## High-Level Flow

```
PDF files
    │
    ▼
[Cache Build]
    ├── Text extraction (fitz/PyMuPDF) → chunking → text embeddings (Vertex text-embedding-004)
    └── Image extraction (fitz) → auto-crop (OpenCV) → Gemini description → image + text embeddings
    │
    ▼
Cache (pkl + csv)  ──────────────► S3 bucket
    │  img_path = S3 URL                   (images uploaded individually, metadata only in bulk sync)
    │
    ▼
[Query]
  User question
    ├── embed question (text-embedding-004)
    ├── cosine search → top-k text chunks
    ├── cosine search → top-k images (by description embedding)
    └── Gemini multimodal prompt (question + text chunks + images) → answer
```

## Component Map

```
rag_server.py          FastAPI app — all HTTP endpoints, orchestration
pipeline.py            MultimodalRAGPipeline — cache save/load, search, answer generation
utils.py               PDF extraction, embeddings, Gemini calls, get_document_metadata
s3_storage.py          S3 sync helpers (upload/download cache + PDFs + individual images)
syspare_rag/
  config.py            RagConfig, ManualConfig, ManualRegistry, env loader
  indexing/embedder.py Vertex AI embedding wrappers (text-embedding-004, multimodalembedding@001)
  indexing/validation.py Embedding dimension guard
  retrieval/similarity.py Cosine similarity helpers
  _retry.py            Exponential backoff for Vertex API calls
```

## Data Flow: Cache Build

1. `POST /api/build-cache?manual_id=<id>`
2. `_sync_from_s3(manual)` — pulls PDF + metadata (pkl/csv, **not** images) from S3
3. `MultimodalRAGPipeline.build_metadata(...)` — calls `get_document_metadata()` in `utils.py`
   - Pages processed **concurrently** (default 4 workers, set `RAG_BUILD_WORKERS` env)
   - Per page: text extract → embed; image extract → crop → Gemini describe → embed
4. `_upload_images_to_s3_and_rewrite(rag, manual)` — uploads each image to S3, rewrites `img_path` in DataFrame to S3 public URL, re-saves pkl/csv
5. `_sync_to_s3(manual)` — uploads pkl/csv to S3 (skips `images/` subdir)

## Data Flow: Query

1. `POST /api/query` (or `/api/query-myanmar`, `/api/query-japanese`, `/api/v1/diagnose`)
2. `_get_rag(manual_id)` — loads pipeline from memory cache, or loads pkl from disk (pulling S3 if needed)
3. `search_text()` — cosine similarity on text chunk embeddings
4. `search_images_by_description_text()` — cosine similarity on image description text embeddings
5. `answer_multimodal_query()` — builds Gemini prompt with retrieved context, returns answer

## Cache Structure

```
manuals/<manual_id>/
  pdf/                  ← PDF source files
  cache/
    text_metadata_df.pkl    ← text chunks + embeddings (768d)
    text_metadata_df.csv    ← human-readable version
    image_metadata_df.pkl   ← image descriptions + embeddings + S3 URLs
    image_metadata_df.csv
    images/             ← local extracted images (NOT uploaded to S3 in bulk)
                           img_path in pkl = S3 public URL
```

## S3 Layout

```
{S3_RAG_PREFIX}/manuals/<manual_id>/
  pdf/           → same as local manuals/<manual_id>/pdf/
  cache/         → pkl + csv only (no images/)
  cache/images/  → each image uploaded individually during build
```

## Models Used

| Purpose | Model |
|---|---|
| Text embedding (chunks, queries, descriptions) | `text-embedding-004` (768d) |
| Image pixel embedding | `multimodalembedding@001` (1408d) |
| Image description + answer generation | `gemini-2.5-flash` |

## Language Support

- **English** — `/api/query`
- **Myanmar** — `/api/query-myanmar` (rewrites question to English → RAG → translates answer back)
- **Japanese** — `/api/query-japanese` (same pattern)
- All languages: `answer_language` param on `/api/query`

## Multi-Manual Support

Manuals are registered in `ManualRegistry` (loaded from `MANUALS_JSON` env file or hardcoded defaults in `config.py`). Each manual has isolated pdf/cache dirs and its own S3 prefix. Every API endpoint accepts `?manual_id=<id>` to route to the correct pipeline.
