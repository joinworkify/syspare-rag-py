# Operations Guide

## Local Development

```bash
# 1. Install dependencies
make install

# 2. Set env vars
cp .env.example .env   # fill in PROJECT_ID, LOCATION, GOOGLE_APPLICATION_CREDENTIALS

# 3. Authenticate GCP (if not using service account key)
gcloud auth application-default login
gcloud auth application-default set-quota-project fortunaii

# 4. Start dev server (hot-reload)
make dev

# 5. Open http://localhost:8000
```

---

## First-time Cache Build

Cache must exist before queries work. Two ways:

**A — Pull from S3 (fastest if cache already built):**
```bash
make pull-s3                          # default manual
make pull-s3-operation                # YM358_operation
```

**B — Build from PDFs (full rebuild, takes minutes):**
```bash
# Server must be running first
make dev   # in another terminal

make build-cache                      # YM358_service (default)
make build-cache-operation            # YM358_operation
```

Build time scales with PDF size and `RAG_BUILD_WORKERS`. A 400-page PDF takes ~15-30 min at 4 workers.

---

## Adding a New Manual

1. Add entry to `_DEFAULT_MANUALS` in `syspare_rag/config.py` (or set `MANUALS_JSON` env pointing to a JSON file)
2. Create local dirs: `manuals/<new_id>/pdf/` and `manuals/<new_id>/cache/`
3. Drop PDFs into `manuals/<new_id>/pdf/`
4. Build cache: `curl -X POST "http://localhost:8000/api/build-cache?manual_id=<new_id>"`

---

## Routine S3 Sync

```bash
# After local cache build — push metadata to S3
make sync-s3

# On a new machine / after deploy — pull metadata from S3
make pull-s3
```

**What gets synced:**
- `sync-to-s3`: uploads `pkl + csv` (skips `images/` — already uploaded per-image during build)
- `pull-from-s3`: downloads `pkl + csv` only (images are S3 URLs in the pkl, served directly)

---

## Deploying on Render

1. Connect repo to Render, create Web Service
2. Set all env vars (see [environment-variables.md](./environment-variables.md))
3. Add `GOOGLE_APPLICATION_CREDENTIALS` as a Secret File
4. Set `DISABLE_S3_SYNC=0`
5. Deploy — on first startup, server pulls cache from S3 automatically

**Cold starts on free tier**: Render free plan sleeps after 15 min idle. Use [cron-job.org](https://cron-job.org) to ping `GET /health` every 14 minutes.

---

## Tuning Build Speed

`RAG_BUILD_WORKERS` controls concurrent page processing. Each worker makes 3 Vertex AI API calls (Gemini describe + image embed + text embed).

| Workers | Notes |
|---|---|
| 1 | Serial, safest, slowest |
| 4 | Default. Good for paid Vertex tier |
| 6-8 | Only if you have high Vertex QPM quota |

If you see `429 RESOURCE_EXHAUSTED` errors during build, lower the workers:
```bash
RAG_BUILD_WORKERS=2 make dev
```

---

## Troubleshooting

### Queries return 503 "RAG not available"
- Cache missing: run `make pull-s3` or `make build-cache`
- GCP creds invalid: check `GOOGLE_APPLICATION_CREDENTIALS` path and service account roles
- Vertex AI API not enabled: `gcloud services enable aiplatform.googleapis.com`

### Images 404 in query responses
Cache was built before the S3-URL feature. Rebuild: `make build-cache`.

### `No such file or directory: ...image...jpeg` during build
This is a known crop-detection edge case (now fixed). Images with no detectable blocks fall back to the original jpeg. Safe to ignore if it continues.

### S3 sync not working
- Check `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET_NAME` in `.env`
- Confirm bucket is in correct region (`AWS_DEFAULT_REGION`)
- Confirm bucket has public read access (needed for image URLs)
- Test: `curl https://<bucket>.s3.<region>.amazonaws.com/<any-key>`

### Quota errors from Vertex AI
- Lower `RAG_BUILD_WORKERS` (default 4 → try 2)
- Check quota in Google Cloud Console → Vertex AI → Quotas

### OCR text missing for scanned pages
- Confirm `tesseract-ocr` is installed: `tesseract --version`
- Install language packs: `sudo apt install tesseract-ocr-mya` (Myanmar)
- Check `OCR_LANG` env matches installed language packs

---

## Useful Endpoints for Ops

```bash
# Health check
curl http://localhost:8000/health

# List manuals
curl http://localhost:8000/api/manuals

# Force rebuild
curl -X POST "http://localhost:8000/api/build-cache?manual_id=YM358_service"

# Clean cache (forces rebuild on next query)
curl -X POST "http://localhost:8000/api/clean-cache?manual_id=YM358_service"

# Push to S3
curl -X POST "http://localhost:8000/api/sync-to-s3?manual_id=YM358_service"

# Pull from S3
curl -X POST "http://localhost:8000/api/pull-from-s3?manual_id=YM358_service"
```
