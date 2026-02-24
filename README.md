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
multi-rag-syspare/
  main.py
  intro_multimodal_rag_utils.py
  data/
    your_pdf_1.pdf
    your_pdf_2.pdf
  images/
  README.md
  pyproject.toml
```

---

## 5) Run

```bash
uv run python main.py
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
