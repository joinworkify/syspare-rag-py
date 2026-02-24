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
