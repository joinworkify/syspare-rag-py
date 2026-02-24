# Multimodal RAG (Vertex AI Gemini)

This project runs a Multimodal RAG pipeline (PDF text + images) using
Google Vertex AI Gemini.

- **Next.js app (TypeScript)** — Full-stack app in `web/`: RAG backend and modern UI. No Python required. See [web/README.md](web/README.md).
- **Python (UV + VS Code)** — Original pipeline and FastAPI server (see below).

---

## Next.js app (recommended)

From the repo root:

```bash
cd web
cp .env.example .env.local   # set GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION
npm install && npm run dev
```

Then build the RAG cache (once): `POST http://localhost:3000/api/build-metadata` with body `{"includePageImages": true}`. After that, open http://localhost:3000 and run queries.

---

# Python setup (UV + VS Code)

Originally a Colab notebook, adapted to run locally in VS Code with `uv`.

---

## Prerequisites

- Python 3.11+ (3.12 recommended)
- `uv` installed
- Google Cloud SDK (`gcloud`) installed and authenticated
- Access to a Google Cloud project with Vertex AI enabled

---

## 1) Create project & install dependencies (UV)

From the repo root:

```bash
uv init
uv add google-cloud-aiplatform pymupdf rich colorama pandas pillow numpy
uv sync

brew install tesseract
```

---

## 2) Google Cloud authentication (local dev)

Login to Google Cloud:

```bash
gcloud auth login
gcloud auth application-default login
```

Set your project and region:

```bash
gcloud config set project fortunaii
gcloud config set ai/region us-central1
```

Important: set the ADC quota project:

```bash
gcloud auth application-default set-quota-project fortunaii
```

Verify:

```bash
gcloud config list
gcloud auth application-default print-access-token >/dev/null && echo "ADC OK"
```

---

## 3) Enable required APIs

```bash
gcloud services enable aiplatform.googleapis.com
```

---

## 4) Project structure

```text
syspare-rag-python/
├── README.md
├── __pycache__
│   ├── multimodal_rag_pipeline.cpython-312.pyc
│   ├── pipeline.cpython-312.pyc
│   ├── rag_server.cpython-312.pyc
│   └── utils.cpython-312.pyc
├── cache_ym358a
│   ├── image_metadata_df.csv
│   ├── image_metadata_df.pkl
│   ├── images
│   │   ├── ym358a-service-manual-pdf-split-ocr.pdf_image_0_0_16.jpeg
│   │   ├── ym358a-service-manual-pdf-split-ocr.pdf_image_10_0_68.jpeg
│   │   ├── ym358a-service-manual-pdf-split-ocr.pdf_image_1_0_23.jpeg
│   │   ├── ym358a-service-manual-pdf-split-ocr.pdf_image_2_0_28.jpeg
│   │   ├── ym358a-service-manual-pdf-split-ocr.pdf_image_3_0_33.jpeg
│   │   ├── ym358a-service-manual-pdf-split-ocr.pdf_image_4_0_38.jpeg
│   │   ├── ym358a-service-manual-pdf-split-ocr.pdf_image_5_0_43.jpeg
│   │   ├── ym358a-service-manual-pdf-split-ocr.pdf_image_6_0_48.jpeg
│   │   ├── ym358a-service-manual-pdf-split-ocr.pdf_image_7_0_53.jpeg
│   │   ├── ym358a-service-manual-pdf-split-ocr.pdf_image_8_0_58.jpeg
│   │   └── ym358a-service-manual-pdf-split-ocr.pdf_image_9_0_63.jpeg
│   ├── text_metadata_df.csv
│   └── text_metadata_df.pkl
├── data
│   └── ym358a-service-manual-pdf-split-ocr.pdf
├── main.py
├── pipeline.py
├── pyproject.toml
├── rag_server.py
├── utils.py
└── uv.lock
```

---

## 5) Run

```bash
uvicorn rag_server:app --reload --port 8000
```

---

## Deploy on Render

The Python FastAPI app can be deployed on [Render](https://render.com) using the repo’s `render.yaml` and `requirements.txt`.

1. **Connect the repo** in the Render Dashboard and create a new Web Service (or use the Blueprint from `render.yaml`).

2. **Environment variables** (Dashboard → Environment):
   - `PROJECT_ID` — Google Cloud project ID (Vertex AI).
   - `LOCATION` — e.g. `us-central1`.
   - Optional: `PDF_FOLDER`, `CACHE_DIR`, `IMAGE_DIR` (defaults: `./data/`, `./cache_ym358a`, `./cache_ym358a/images`).

3. **Google Cloud auth** — Use a [service account](https://cloud.google.com/iam/docs/service-accounts) and set `GOOGLE_APPLICATION_CREDENTIALS` to the path of the JSON key file. On Render, add the key as a **Secret File** and set `GOOGLE_APPLICATION_CREDENTIALS` to that path (e.g. `/etc/secrets/gcp-key.json`).

4. **Data and cache** — Render’s filesystem is ephemeral. Either:
   - Use **AWS S3** (see below), or
   - Commit a pre-built `cache_ym358a/` (and optionally `data/`) into the repo, or
   - Use a [Render Persistent Disk](https://render.com/docs/disks) and set `CACHE_DIR` / `PDF_FOLDER` to paths on the disk.

5. **Health check** — Use `GET /health` for Render’s health check URL.

---

## AWS S3 storage (cache + PDFs)

Cache and PDFs can be stored in an S3 bucket so they persist across deploys and are shared across instances.

**S3 layout** (prefix `rag-data` by default):

- `rag-data/pdfs/` — uploaded PDFs (synced to local `PDF_FOLDER` at startup).
- `rag-data/cache/` — RAG cache (metadata + extracted images; synced to local `CACHE_DIR`).

**Environment variables** (set in Render or `.env`; never commit real keys):

| Variable | Description |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_DEFAULT_REGION` | e.g. `us-east-2` |
| `S3_BUCKET_NAME` | Bucket name (e.g. `syspare-vercel`) |
| `S3_RAG_PREFIX` | Optional; default `rag-data` |

**Behavior:**

- On startup, if S3 is configured, the app downloads `rag-data/pdfs/` and `rag-data/cache/` into local `PDF_FOLDER` and `CACHE_DIR`.
- After building metadata (when cache was missing), the app uploads the local cache (and PDFs) back to S3.
- **Upload PDF via API:** `POST /api/upload-pdf` with `multipart/form-data` and a PDF file. The file is saved locally and uploaded to `rag-data/pdfs/`. Restart or run a new query (with cache cleared) to re-index.

**Security:** If your AWS keys were ever pasted in chat or committed, rotate them in the AWS IAM console and update the env vars.

---

## 6) Quick Vertex AI smoke test

Create `test_vertex.py`:

```python
import vertexai
from vertexai.generative_models import GenerativeModel

PROJECT_ID = "fortunaii"
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.0-flash")

resp = model.generate_content("Say 'ok' in one word.")
print(resp.text)
```

Run:

```bash
uv run python test_vertex.py
```

---

## Troubleshooting

### A) quota project mismatch

```bash
gcloud auth application-default set-quota-project fortunaii
```

### B) DefaultCredentialsError

```bash
gcloud auth application-default login
```

### C) API not enabled

```bash
gcloud services enable aiplatform.googleapis.com
```

### D) Permission denied / 403

Ensure your account has Vertex AI permissions in the project.
