# Environment Variables

Copy `.env` from the repo and fill in the values. Never commit real secrets.

---

## Required

| Variable | Example | Description |
|---|---|---|
| `PROJECT_ID` | `fortunaii` | Google Cloud project ID (Vertex AI billing) |
| `LOCATION` | `us-central1` | Vertex AI region |
| `GOOGLE_APPLICATION_CREDENTIALS` | `./rag-service-account.json` | Path to GCP service account JSON key |

---

## AWS S3 (required for S3 sync)

| Variable | Example | Description |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | `AKIA...` | AWS IAM access key |
| `AWS_SECRET_ACCESS_KEY` | `...` | AWS IAM secret key |
| `AWS_DEFAULT_REGION` | `us-east-2` | S3 bucket region |
| `S3_BUCKET_NAME` | `syspare-vercel` | Bucket name |
| `S3_RAG_PREFIX` | `rag-data` | Prefix inside bucket (default: `rag-data`) |

**Required S3 bucket setup**: bucket must have public read access (or presigned URLs — not yet implemented). Images are stored at `https://<bucket>.s3.<region>.amazonaws.com/<key>` and linked directly in the pkl metadata.

---

## Manual Configuration

| Variable | Example | Description |
|---|---|---|
| `DEFAULT_MANUAL_ID` | `YM358_service` | Which manual loads by default |
| `MANUALS_JSON` | `./manuals.json` | Path to JSON file defining manuals (overrides hardcoded defaults) |

**`MANUALS_JSON` format:**
```json
[
  {
    "manual_id": "YM358_service",
    "display_name": "YM358 Service Manual",
    "pdf_folder": "manuals/YM358_service/pdf",
    "cache_dir": "manuals/YM358_service/cache",
    "image_dir": "manuals/YM358_service/cache/images",
    "ocr_lang": "eng",
    "description": "..."
  }
]
```

---

## OCR

| Variable | Example | Description |
|---|---|---|
| `OCR_LANG` | `mya+eng` | Tesseract language codes (plus-separated). Use `mya+eng` for Myanmar+English, `jpn+eng` for Japanese+English, `eng` for English only |

Requires `tesseract-ocr` binary + language packs installed.

---

## Paths (usually leave as defaults)

| Variable | Default | Description |
|---|---|---|
| `PDF_FOLDER` | `manuals/<default_id>/pdf` | PDF source directory |
| `CACHE_DIR` | `manuals/<default_id>/cache` | Metadata cache directory |
| `IMAGE_DIR` | `manuals/<default_id>/cache/images` | Extracted images directory |

---

## Server Behaviour

| Variable | Default | Description |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:5173,...` | Comma-separated allowed CORS origins |
| `DISABLE_S3_SYNC` | `0` | Set to `1` to skip all S3 sync (local dev without AWS creds) |
| `INIT_RAG_ON_STARTUP` | `0` | Set to `1` to load the default pipeline at startup (slower start, no cold-start on first query) |
| `RAG_BUILD_WORKERS` | `4` | Concurrent page workers during cache build. Lower if hitting Vertex quota errors |

---

## Render-specific notes

On Render, set all variables in **Dashboard → Environment**. For `GOOGLE_APPLICATION_CREDENTIALS`:
1. Add the service account JSON as a **Secret File** (e.g. filename `/etc/secrets/gcp-key.json`)
2. Set `GOOGLE_APPLICATION_CREDENTIALS=/etc/secrets/gcp-key.json`

Set `DISABLE_S3_SYNC=0` so cache persists across deploys (Render filesystem is ephemeral).
