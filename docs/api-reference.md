# API Reference

Base URL: `http://localhost:8000` (local) or your deployed URL.

All JSON endpoints accept/return `application/json`. Image uploads use `multipart/form-data`.

---

## Health

### `GET /health`
Returns `{"status": "ok"}`. Use as Render/load-balancer health check.

---

## Query Endpoints

### `POST /api/query`
Main RAG query. Retrieves relevant text + images from the manual and answers via Gemini.

**Request body:**
```json
{
  "question": "How do I change the engine oil?",
  "manual_id": "YM358_service",
  "top_k_text": 5,
  "top_k_img": 6,
  "temp": 0.5,
  "answer_language": "English"
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `question` | string | required | |
| `manual_id` | string | default manual | |
| `top_k_text` | int | 5 | text chunks to retrieve |
| `top_k_img` | int | 6 | images to retrieve |
| `temp` | float | 0.5 | Gemini temperature |
| `answer_language` | string | "English" | answer language |

**Response:**
```json
{
  "answer": "...",
  "texts": [{"chunk_text": "...", "page_num": 12, "score": 0.87}],
  "images": [{"img_url": "https://...s3.amazonaws.com/...", "caption": "...", "score": 0.91}],
  "manual_id": "YM358_service"
}
```

---

### `POST /api/query-myanmar`
Myanmar question → English RAG → Myanmar answer.

**Request body:** same as `/api/query` plus:
```json
{ "include_intermediate_english": true }
```

**Response:** same as `/api/query` plus `english_answer` field if `include_intermediate_english=true`.

---

### `POST /api/query-japanese`
Japanese question → English RAG → Japanese answer. Same pattern as Myanmar.

---

### `POST /api/query-with-image`
Query with an optional reference image (base64).

**Request body:**
```json
{
  "question": "What part is this?",
  "image_base64": "<base64 string>",
  "manual_id": "YM358_service"
}
```

---

### `POST /api/query-upload`
Same as `query-with-image` but accepts `multipart/form-data` with an image file upload.

---

## Diagnostic Endpoint

### `POST /api/v1/diagnose`
Structured diagnostic response for equipment issues. Returns `summary`, `confidence`, `urgency`, `actions`, `torque_spec`, `warnings`, `references`, `images`.

**Request body:**
```json
{
  "question": "Engine makes knocking noise at startup",
  "manual_id": "YM358_service",
  "top_k_text": 5,
  "top_k_img": 6,
  "temp": 0.2,
  "answer_language": "English"
}
```

**Response** (key fields):
```json
{
  "summary": "...",
  "confidence": {"score": 0.85, "label": "high"},
  "urgency": {"level": "medium", "label": "Check within 1 week"},
  "actions": [{"step": 1, "instruction": "...", "tool": "..."}],
  "torque_spec": {"value": 45.0, "unit": "Nm", "component": "..."},
  "warnings": ["..."],
  "images": [...],
  "manual_id": "YM358_service"
}
```

### `POST /api/v1/diagnose-upload`
Same as `/api/v1/diagnose` but accepts a `multipart/form-data` image upload alongside the JSON payload.

---

## Cache Management

### `POST /api/build-cache?manual_id=<id>`
Rebuild metadata cache from PDFs. Slow (minutes). Does full extraction + Gemini + embeddings.

- Pulls latest PDFs from S3 first
- Processes pages concurrently (`RAG_BUILD_WORKERS`, default 4)
- Uploads extracted images to S3, stores public URLs in pkl
- Syncs metadata (pkl/csv) back to S3
- Clears in-memory pipeline so next query reloads fresh cache

**Response:**
```json
{
  "ok": true,
  "manual_id": "YM358_service",
  "message": "Cache for YM358_service rebuilt and synced to S3.",
  "s3_uploaded": {"cache": 4, "pdfs": 1, "images_to_s3": 387}
}
```

### `POST /api/clean-cache?manual_id=<id>`
Delete local cache files and in-memory pipeline for a manual. Next query triggers rebuild.

---

## S3 Sync

### `POST /api/sync-to-s3?manual_id=<id>`
Push local cache metadata (pkl/csv) + PDFs to S3. Skips `images/` subdir.

### `POST /api/pull-from-s3?manual_id=<id>`
Pull cache metadata (pkl/csv) from S3. Skips `images/` (S3 URLs are in the pkl). Clears in-memory state.

---

## PDF Upload

### `POST /api/upload-pdf`
Upload a PDF to a manual's local folder and S3.

**Form fields:**
- `file` — PDF file
- `manual_id` — (optional) target manual

---

## Manual Registry

### `GET /api/manuals`
List all registered manuals.

**Response:**
```json
{
  "default_manual_id": "YM358_service",
  "manuals": [
    {
      "manual_id": "YM358_service",
      "display_name": "YM358 Service Manual",
      "description": "...",
      "is_default": true
    }
  ]
}
```

---

## UI Pages

| Route | Description |
|---|---|
| `GET /` | Legacy single-turn query UI (HTML) |
| `GET /v1` | Internal diagnostic dashboard UI |
| `GET /chat` | Standalone chat and feedback UI |
| `GET /debug-generator` | Developer question generator debug page |
| `GET /manage` | Admin manual manager UI |

---

## Other

### `GET /api/models`
List available Gemini models.

### `POST /api/generate-random-question`
Generate a random test question from the manual context.

### `POST /api/chat`
Multi-turn chat interface using the RAG pipeline.
