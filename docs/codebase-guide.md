# Codebase Guide

For engineers picking up this project.

---

## Entry Points

| File | Role |
|---|---|
| `rag_server.py` | The entire FastAPI app. Start here. |
| `pipeline.py` | `MultimodalRAGPipeline` — cache I/O + search + answer generation |
| `utils.py` | PDF extraction, Gemini calls, embedding helpers, `get_document_metadata` |
| `s3_storage.py` | All S3 operations |
| `syspare_rag/config.py` | Config dataclasses and manual registry |

---

## Key Classes

### `MultimodalRAGPipeline` (`pipeline.py`)

The main pipeline object. One instance per manual, held in `_pipelines` dict in `rag_server.py`.

**Important methods:**

| Method | What it does |
|---|---|
| `build_metadata(pdf_folder_path, cache_dir, ...)` | Full extraction pipeline. Calls `get_document_metadata` → appends OCR chunks → `save_cache` |
| `save_cache(cache_dir)` | Writes `text_metadata_df` + `image_metadata_df` to pkl + csv |
| `load_cache(cache_dir)` | Reads pkl files. Returns `False` if missing |
| `search_text(query, top_n)` | Cosine similarity over text chunk embeddings |
| `search_images_by_description_text(query, top_n)` | Cosine similarity over image description embeddings |
| `answer_multimodal_query(question, ...)` | Builds Gemini prompt + calls Gemini |
| `answer_diagnostic_query(question, ...)` | Like above but returns structured JSON (for `/v1/diagnose`) |

**State:**
- `self.text_metadata_df` — DataFrame with columns: `file_name`, `page_num`, `text`, `chunk_text`, `text_embedding_chunk`, etc.
- `self.image_metadata_df` — DataFrame with columns: `file_name`, `page_num`, `img_path`, `img_desc`, `mm_embedding_from_img_only`, `text_embedding_from_image_description`

### `ManualRegistry` (`syspare_rag/config.py`)

Holds all `ManualConfig` objects. Loaded once at startup from `MANUALS_JSON` or hardcoded defaults.

```python
registry.get("YM358_service")    # → ManualConfig
registry.list()                  # → [ManualConfig, ...]
registry.default                 # → default ManualConfig
```

---

## Key Functions

### `get_document_metadata` (`utils.py:596`)

Processes all PDFs in a folder. Returns `(text_df, image_df)`.

**Parallel by default** (`RAG_BUILD_WORKERS=4`). Each worker:
1. Opens its own `fitz.Document` (thread-safe)
2. Extracts text → embeds
3. Extracts images → crops → Gemini describe → embeds
4. Returns `(page_num, text_meta, image_meta)`

### `get_image_for_gemini` (`utils.py:~325`)

Extracts one image from a PDF page:
1. `fitz.Pixmap` → save as `.jpeg`
2. `find_and_crop_blocks` → OpenCV contour detection → crop to largest block → delete original **only if crops found**
3. `Image.load_from_file` → returns Gemini `Image` object + path

### `_upload_images_to_s3_and_rewrite` (`rag_server.py:~199`)

Called after `build_metadata`. Iterates `image_metadata_df["img_path"]`, uploads each file to S3, rewrites path to public URL (`https://<bucket>.s3.<region>.amazonaws.com/<key>`), then re-saves pkl/csv.

### `_get_rag` (`rag_server.py:209`)

Lazy loader for `MultimodalRAGPipeline`. On first access:
1. `_sync_from_s3` — pulls PDFs + metadata from S3
2. `load_cache` — tries to load pkl
3. If no cache: `build_metadata` + `_upload_images_to_s3_and_rewrite` + `_sync_to_s3`
4. `_remap_image_paths` — fixes stale local paths (skips S3 URLs)
5. Stores in `_pipelines[manual_id]`

---

## DataFrame Schemas

### `text_metadata_df`

| Column | Type | Description |
|---|---|---|
| `file_name` | str | Source PDF filename |
| `page_num` | int | 1-indexed page number |
| `text` | str | Full page text |
| `chunk_text` | str | One text chunk |
| `text_embedding_chunk` | list[float] | 768d embedding of `chunk_text` |
| `text_embedding_page` | list[float] | 768d embedding of full page text |

### `image_metadata_df`

| Column | Type | Description |
|---|---|---|
| `file_name` | str | Source PDF filename |
| `page_num` | int | 1-indexed page number |
| `img_path` | str | **S3 public URL** (after build) or local path (legacy) |
| `img_desc` | str | Gemini-generated description |
| `mm_embedding_from_img_only` | list[float] | 1408d image pixel embedding |
| `text_embedding_from_image_description` | list[float] | 768d text embedding of `img_desc` |

---

## S3 Image URL Strategy

**Before (legacy):** `img_path` = local file path. On pull-from-S3, all images had to be downloaded.

**Now:** During `build_metadata`, each image is uploaded to:
```
s3://<bucket>/rag-data/manuals/<manual_id>/cache/images/<filename>
```
And `img_path` in the pkl is rewritten to the public HTTPS URL. The bulk S3 cache sync (`upload_manual_cache_to_s3` / `download_manual_cache_from_s3`) skips `images/` subdir — only pkl/csv are synced in bulk.

At query time, `Image.from_bytes(requests.get(url).content)` is used when `img_path` starts with `http`.

---

## Adding a Language

Pattern established for Myanmar and Japanese:

1. Add `_rewrite_<lang>_to_english_query(question, rag, manual_id)` — Gemini call
2. Add `_english_answer_to_<lang>(answer, rag)` — Gemini call
3. Add `<Lang>QueryRequest` and `<Lang>QueryResponse` Pydantic models
4. Add `@app.post("/api/query-<lang>")` endpoint following the existing pattern

---

## Retry Logic

All Vertex AI calls are wrapped in `retry_call` from `syspare_rag/_retry.py`. Exponential backoff on `429`, `503`, transient network errors. Gemini calls additionally wrapped in `_gemini_generate_with_retry`.

Do not add bare `vertexai` calls without wrapping in retry.

---

## Common Gotchas

**fitz thread safety**: `fitz.Document` is not thread-safe. Each worker in `get_document_metadata` opens its own handle via `fitz.open(pdf_path)` and closes it in `finally`. Never pass a shared doc across threads.

**Stale pkl paths**: Old caches have local paths in `img_path`. `_remap_image_paths` rebases them to `manual.image_dir`. New caches have S3 URLs — `_remap_image_paths` skips those (starts with `http`).

**image_object column**: PIL Image objects are loaded into memory in `_rebuild_image_objects_from_paths` (pipeline.py). The server always uses `rebuild_image_objects=False` and loads on demand. This column is dropped when saving CSV.

**Embedding dimension guard**: `validate_embedding_dimension` in `syspare_rag/indexing/validation.py` raises on mismatch. If you change models, flush the cache.
