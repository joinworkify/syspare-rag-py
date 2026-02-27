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

Then build the RAG cache (once): `POST http://localhost:3000/api/build-metadata` with body `{"includePageImages": true}`. After that, open [http://localhost:3000](http://localhost:3000) and run queries.

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

## 3b) Service account & GOOGLE_APPLICATION_CREDENTIALS (optional)

Use this when you want to authenticate with a **JSON key file** (e.g. on Render, or without `gcloud` on your machine).

### Step 1: Create a service account in Google Cloud

1. Open [Google Cloud Console](https://console.cloud.google.com/) and select your project (e.g. `fortunaii`).
2. Go to **IAM & Admin** → **Service Accounts** (or search “Service accounts” in the top bar).
3. Click **+ Create Service Account**.
4. **Service account name:** e.g. `rag-vertex-ai`.
5. Click **Create and Continue**.
6. **Grant access (optional):** add role **Vertex AI User** (or “Vertex AI Administrator” if you need full access). Click **Continue** → **Done**.

### Step 2: Create and download the JSON key

1. On the Service accounts list, click the service account you just created.
2. Open the **Keys** tab.
3. Click **Add key** → **Create new key** → choose **JSON** → **Create**.
4. A JSON file downloads. **Keep it secret** (never commit it to git). Example name: `your-project-abc123.json`.

### Step 3: Use it locally

1. Move the file somewhere safe, e.g.:
  ```bash
   mkdir -p ~/.config/gcloud
   mv ~/Downloads/your-project-abc123.json ~/.config/gcloud/rag-service-account.json
  ```
2. In your project, set the path in `.env`:
  ```bash
   GOOGLE_APPLICATION_CREDENTIALS=/Users/yourusername/.config/gcloud/rag-service-account.json
  ```
   Or export in the shell before running:

### Step 4: Use it on Render

1. In Render Dashboard → your Web Service → **Environment**.
2. Add a **Secret File**:
  - **Key:** `GOOGLE_APPLICATION_CREDENTIALS`
  - **Filename:** path the app will read, e.g. `/etc/secrets/gcp-key.json`
  - **Contents:** paste the **entire contents** of your service account JSON file.
3. Save. Render writes that content to `/etc/secrets/gcp-key.json` at runtime, so the app finds the key at that path.

If you use **Environment** only (no Secret File), add a variable:

- **Key:** `GOOGLE_APPLICATION_CREDENTIALS`
- **Value:** `/etc/secrets/gcp-key.json` (only if you also added the Secret File with that filename above).

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

## Vercel vs Render (and keeping the service awake)

### Can this Python app run on Vercel?

**No.** This RAG server is a **stateful, long-running** app: it loads the pipeline and cache into memory and keeps them there. Vercel runs **serverless functions** — short-lived, stateless, with strict time limits (e.g. 10–60 seconds). Each request can hit a new instance, so you can’t keep the RAG pipeline in memory. Cold starts would re-run heavy init on every request, and long RAG queries could hit the execution limit. So **deploy this Python API on Render** (or another host that runs a persistent process), not on Vercel.

### Recommended: API on Render, frontend on Vercel

- **Render** — Deploy the Python RAG server (this repo, `render.yaml`). You get a URL like `https://syspare-rag-py.onrender.com`.
- **Vercel** — Deploy only the **frontend** (e.g. `rag-viewer-tsx`). Build with `VITE_RAG_API_URL=https://syspare-rag-py.onrender.com` (or your Render URL). Point the UI at the API.
- On the **RAG server** (Render), set **CORS** so Vercel can call it: in Render Environment add `CORS_ORIGINS=https://your-app.vercel.app` (your Vercel app’s URL).

That way you get Vercel’s fast, global frontend and Render’s always-on (or wake-on-request) API.

### Render free tier: service sleeps when idle

On Render’s **free** plan, the service **spins down after ~15 minutes of no traffic**. The next request wakes it (cold start can take 30–60 seconds). So “Render only starts the service when someone enters the link” is expected on free tier.

**Options:**

1. **Ping to keep it awake** — Use a free uptime/cron service (e.g. [cron-job.org](https://cron-job.org), [UptimeRobot](https://uptimerobot.com)) to hit `GET https://your-app.onrender.com/health` every 10–15 minutes. That reduces (but doesn’t always prevent) spin-down, and can still sleep between pings.
2. **Upgrade Render** — A paid plan keeps the service **always on** (no spin-down), so the link responds quickly every time.

---

## Deploy on PythonAnywhere

**Yes.** PythonAnywhere runs **long-running Python web apps** (unlike Vercel), so this RAG server is a good fit. You get a persistent process and can use the free tier for light use or a paid plan for always-on.

### Steps (outline)

1. **Sign up** at [pythonanywhere.com](https://www.pythonanywhere.com) and create a **Web** app.
2. **Clone the repo** (or upload files) into your home directory, e.g. `/home/yourusername/multi-rag-syspare`.
3. **Virtualenv:** In the Web app configuration, set the virtualenv to a path like `/home/yourusername/multi-rag-syspare/venv`, then create it and install deps:
  ```bash
   python -m venv /home/yourusername/multi-rag-syspare/venv
   source venv/bin/activate
   pip install -r requirements.txt
  ```
4. **Run FastAPI:** PythonAnywhere supports ASGI (beta) or WSGI.
  - **ASGI (beta):** Contact [support@pythonanywhere.com](mailto:support@pythonanywhere.com) to enable ASGI, then in the Web tab set the ASGI app to your app (e.g. `rag_server:app`) and the run command to use uvicorn. See [ASGI on PythonAnywhere](https://help.pythonanywhere.com/pages/ASGICommandLine/).
  - **WSGI:** Use a WSGI adapter so their server can run FastAPI. In the **WSGI configuration file** (e.g. `/var/www/yourusername_pythonanywhere_com_wsgi.py`), point to your app:
    ```python
    import sys
    sys.path.insert(0, '/home/yourusername/multi-rag-syspare')
    from a2wsgi import ASGIMiddleware
    from rag_server import app
    application = ASGIMiddleware(app)
    ```
    Install the adapter: `pip install a2wsgi`.
5. **Static files:** In the Web tab, add a **Static files** mapping: URL `/static` → Directory `/home/yourusername/multi-rag-syspare/cache_ym358a/images` (or whatever `IMAGE_DIR` is), so retrieved images load correctly.
6. **Environment variables:** Set `PROJECT_ID`, `LOCATION`, `GOOGLE_APPLICATION_CREDENTIALS` (path to your service account JSON in your home dir), and optionally `PDF_FOLDER`, `CACHE_DIR`, `CORS_ORIGINS`, and S3 vars. You can put them in a `.env` file in the project root or set them in the Web app setup if PythonAnywhere allows.
7. **Reload** the web app from the PythonAnywhere Web tab.

Free tier limits (e.g. one web app, limited CPU) apply; paid plans give more resources and always-on behavior. If you use S3 for cache and PDFs, the app can sync on startup so the ephemeral filesystem is less of an issue.

---

## AWS S3 storage (cache + PDFs)

Cache and PDFs can be stored in an S3 bucket so they persist across deploys and are shared across instances.

**S3 layout** (prefix `rag-data` by default):

- `rag-data/pdfs/` — uploaded PDFs (synced to local `PDF_FOLDER` at startup).
- `rag-data/cache/` — RAG cache (metadata + extracted images; synced to local `CACHE_DIR`).

**Environment variables** (set in Render or `.env`; never commit real keys):


| Variable                | Description                         |
| ----------------------- | ----------------------------------- |
| `AWS_ACCESS_KEY_ID`     | AWS access key                      |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key                      |
| `AWS_DEFAULT_REGION`    | e.g. `us-east-2`                    |
| `S3_BUCKET_NAME`        | Bucket name (e.g. `syspare-vercel`) |
| `S3_RAG_PREFIX`         | Optional; default `rag-data`        |


**Behavior:**

- On startup, if S3 is configured, the app downloads `rag-data/pdfs/` and `rag-data/cache/` into local `PDF_FOLDER` and `CACHE_DIR`.
- After building metadata (when cache was missing), the app uploads the local cache (and PDFs) back to S3.
- **Upload PDF via API:** `POST /api/upload-pdf` with `multipart/form-data` and a PDF file. The file is saved locally and uploaded to `rag-data/pdfs/`. Restart or run a new query (with cache cleared) to re-index.

**Security:** If your AWS keys were ever pasted in chat or committed, rotate them in the AWS IAM console and update the env vars.

---

## Deploy on AWS EC2 (with S3 cache)

You can run the Python FastAPI server on a small EC2 instance and let it read PDFs/cache from S3.

### 1) Create EC2 instance

- Launch an **Ubuntu 22.04** instance (you can start small and scale later).
- Create a new **key pair** (e.g. `ec2-syspare-pass`); AWS will download `ec2-syspare-pass.pem` to your machine.
- In the **Security Group**, allow:
  - TCP `22` from your IP (SSH).
  - TCP `8000` (or `80/443` if you later add nginx) from the IPs that should access the app.

You will use the `.pem` file to SSH into the instance and copy your service-account JSON.

### 2) Clone repo into `/opt/syspare-rag-py`

This block shows the full sequence you used to:

- SSH into EC2.
- Install system packages.
- Clone the repo.
- Create a virtualenv and install Python dependencies.
- Copy the GCP service-account JSON onto the server.
- Set up `.env` and systemd.

```bash
ssh -i ~/Downloads/ec2-syspare-pass.pem ubuntu@ubuntu@YOUR_EC2_PUBLIC_IP

# update the dependencies
sudo apt update
sudo apt install -y python3.12 python3.12-venv git tesseract-ocr
sudo apt-get update
sudo apt-get install -y libgl1

# if asked for systemctl restart networkd-dispatcher.service do this:
sudo systemctl restart networkd-dispatcher.service #optional

# download the source code
cd /opt
sudo git clone https://github.com/joinworkify/syspare-rag-py.git
sudo chown -R ubuntu:ubuntu syspare-rag-py
cd /opt/syspare-rag-py

# create env
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# exit the ssh and do these
chmod 600 ~/Downloads/ec2-syspare-pass.pem
scp -i ~/Downloads/ec2-syspare-pass.pem ~/Downloads/rag-service-account.json ubuntu@YOUR_EC2_PUBLIC_IP:/tmp/rag-service-account.json
sudo mv /tmp/rag-service-account.json /opt/syspare-rag-py/rag-service-account.json

# login back to the ssh
ssh -i ~/Downloads/ec2-syspare-pass.pem ubuntu@ubuntu@YOUR_EC2_PUBLIC_IP

# move the rag-service-account to the /opt
sudo mv /tmp/rag-service-account.json /opt/syspare-rag-py/rag-service-account.json
sudo chown ubuntu:ubuntu /opt/syspare-rag-py/rag-service-account.json
sudo chmod 600 /opt/syspare-rag-py/rag-service-account.json

# after that set environment
sudo nano .env
```

The `.env` contents you used:

```
PROJECT_ID=fortunaii
LOCATION=us-central1

GOOGLE_APPLICATION_CREDENTIALS=/opt/syspare-rag-py/rag-service-account.json

PDF_FOLDER=./data
CACHE_DIR=./cache
IMAGE_DIR=./cache/images

# S3 (only if you want S3 sync in prod)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=your-bucket
S3_RAG_PREFIX=rag-data

# For EC2 you usually WANT S3 sync, so leave this at 0 or omit it
DISABLE_S3_SYNC=0
```

Then you verified the key file and exported extra environment variables directly in the shell:

```
ls -l /opt/syspare-rag-py/rag-service-account.json
# do this in the ssh terminal
export GOOGLE_APPLICATION_CREDENTIALS=/opt/syspare-rag-py/rag-service-account.json
export GOOGLE_CLOUD_PROJECT="fortuneaii"
export GOOGLE_CLOUD_LOCATION="us-central1" 

# run the source code and test
uvicorn rag_server:app --host 0.0.0.0 --port 8000

# if any error with (often libgthread-2.0.so.0), also install:
sudo apt-get update
sudo apt-get install -y libglib2.0-0

# if the source code run correctly we will make that autorun
sudo nano /etc/systemd/system/rag.service

```
Add this in the `rag.service` file
```
[Unit]
Description=Syspare RAG FastAPI service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/syspare-rag-py
Environment=PYTHONUNBUFFERED=1
# Load env vars from .env
EnvironmentFile=/opt/syspare-rag-py/.env
ExecStart=/opt/syspare-rag-py/.venv/bin/uvicorn rag_server:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```
Lastly run this command to enable auto start
```
sudo systemctl daemon-reload
sudo systemctl enable rag.service
sudo systemctl start rag.service
sudo systemctl status rag.service

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