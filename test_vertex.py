"""Quick Vertex AI smoke test (see README)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import vertexai
from vertexai.generative_models import GenerativeModel

PROJECT_ID = os.environ.get("PROJECT_ID", "fortunaii")
LOCATION = os.environ.get("LOCATION", "us-central1")
CREDS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

if CREDS:
    cred_path = Path(CREDS)
    if not cred_path.is_absolute():
        cred_path = Path(__file__).resolve().parent / cred_path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path.resolve())

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-3.5-flash")

resp = model.generate_content("Say 'ok' in one word.")
print(resp.text)
