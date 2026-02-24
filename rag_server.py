# rag_server.py
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Template

from vertexai.generative_models import GenerationConfig

from pipeline import RagConfig, MultimodalRAGPipeline


# -----------------------------
# CONFIG (edit these)
# -----------------------------
PDF_FOLDER = "./data/"
CACHE_DIR = "./cache_ym358a"
IMAGE_DIR = "./cache_ym358a/images"  # must match RagConfig.image_save_dir

PROJECT_ID = "fortunaii"
LOCATION = "us-central1"


# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Multimodal RAG Browser UI")

# Serve images folder in browser as /static/...
Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=IMAGE_DIR), name="static")


# -----------------------------
# Initialize pipeline once
# -----------------------------
cfg = RagConfig(
    project_id=PROJECT_ID,
    location=LOCATION,
    model_name="gemini-2.0-flash",
    embedding_size=1408,
    embedding_model_name="multimodalembedding@001",
    image_save_dir=IMAGE_DIR,
    enable_ocr_fallback=True,
    ocr_min_chars=40,
    ocr_dpi=200,
    ocr_lang="eng",
)

rag = MultimodalRAGPipeline(cfg)

# Check if metadata is already loaded or cached
if not rag.text_metadata_df or not rag.image_metadata_df:
    print("Metadata not loaded. Building metadata...")
    rag.build_metadata(
        pdf_folder_path=PDF_FOLDER,
        cache_dir=CACHE_DIR,
        force_rebuild=False,
        generation_config=GenerationConfig(temperature=0.2),
        ocr_fallback=True,
    )
else:
    print("Metadata already loaded. Skipping build.")


# -----------------------------
# Helpers
# -----------------------------
def _safe_image_url(img_path: str) -> str:
    """
    Convert absolute/relative img_path to URL under /static/.
    This assumes img_path is inside IMAGE_DIR.
    """
    p = Path(img_path)
    # If stored path is absolute, just use filename
    return f"/static/{p.name}"


def _normalize_image_matches(
    image_matches: Dict[Any, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Your utils often return a dict-like structure whose values look like:
      { "img_path": ..., "image_description": ..., "score": ..., ... }
    We'll normalize it for HTML rendering.
    """
    out: List[Dict[str, Any]] = []
    for _, v in image_matches.items():
        img_path = v.get("img_path") or v.get("image_path") or v.get("path")
        if not img_path:
            continue
        out.append(
            {
                "img_path": img_path,
                "img_url": _safe_image_url(str(img_path)),
                "caption": v.get("image_description", ""),
                "score": v.get("score"),
                "page": v.get("page") or v.get("page_number"),
                "doc": v.get("doc_name") or v.get("source") or v.get("file_name"),
            }
        )
    return out


def _normalize_text_matches(
    text_matches: Dict[Any, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for _, v in text_matches.items():
        out.append(
            {
                "chunk_text": v.get("chunk_text") or v.get("text") or "",
                "score": v.get("score"),
                "page": v.get("page") or v.get("page_number"),
                "doc": v.get("doc_name") or v.get("source") or v.get("file_name"),
            }
        )
    return out


# -----------------------------
# HTML template (single-file)
# -----------------------------
PAGE = Template(
    r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Multimodal RAG Viewer</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
    .row { display: flex; gap: 18px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 14px; background: #fff; }
    .left { flex: 1.2; }
    .right { flex: 1; }
    textarea, input { width: 100%; padding: 10px; border-radius: 10px; border: 1px solid #ccc; }
    button { padding: 10px 14px; border-radius: 10px; border: 0; background: #111; color: #fff; cursor: pointer; }
    pre { white-space: pre-wrap; word-break: break-word; }
    .muted { color: #666; font-size: 13px; }
    .imggrid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .imgitem img { width: 100%; border-radius: 10px; border: 1px solid #eee; }
    .chunk { margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px dashed #ddd; }
    .answer { font-size: 15px; line-height: 1.45; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; border: 1px solid #ddd; font-size: 12px; margin-right: 6px; }
  </style>
</head>

<body>
  <h2>Multimodal RAG Viewer</h2>
  <div class="muted">Query → retrieve text/images → Gemini answer</div>

  <div class="card" style="margin-top: 16px;">
    <form method="post" action="/query">
      <label class="muted">Your question</label>
      <input name="q" value="{{ q|e }}" placeholder="Ask something..." />

      <div class="row" style="margin-top: 10px;width: 100%;">
        <div style="flex:1;">
          <label class="muted">Top-K text</label>
          <input name="top_k_text" value="{{ top_k_text }}" />
        </div>
        <div style="flex:1;">
          <label class="muted">Top-K images</label>
          <input name="top_k_img" value="{{ top_k_img }}" />
        </div>
        <div style="flex:1;">
          <label class="muted">Temperature</label>
          <input name="temp" value="{{ temp }}" />
        </div>
      </div>

      <div style="margin-top: 12px;">
        <button type="submit">Run RAG</button>
      </div>
    </form>
  </div>

  {% if ran %}
  <div class="row" style="margin-top: 18px;">
    <div class="card left">
      <h3>Gemini Answer</h3>
      <div class="answer"><pre>{{ answer }}</pre></div>
    </div>

    <div class="card right">
      <h3>Retrieved Images</h3>
      <div class="imggrid">
        {% for img in images %}
          <div class="imgitem">
            <img src="{{ img.img_url }}" />
            <div class="muted" style="margin-top:6px;">
              {% if img.doc %}<span class="pill">{{ img.doc }}</span>{% endif %}
              {% if img.page %}<span class="pill">p{{ img.page }}</span>{% endif %}
              {% if img.score is not none %}<span class="pill">score {{ "%.3f"|format(img.score) }}</span>{% endif %}
            </div>
            <div class="muted"><pre>{{ img.caption }}</pre></div>
          </div>
        {% endfor %}
      </div>
    </div>
  </div>

  <div class="card" style="margin-top: 18px;">
    <h3>Retrieved Text Chunks</h3>
    {% for t in texts %}
      <div class="chunk">
        <div class="muted">
          {% if t.doc %}<span class="pill">{{ t.doc }}</span>{% endif %}
          {% if t.page %}<span class="pill">p{{ t.page }}</span>{% endif %}
          {% if t.score is not none %}<span class="pill">score {{ "%.3f"|format(t.score) }}</span>{% endif %}
        </div>
        <pre>{{ t.chunk_text }}</pre>
      </div>
    {% endfor %}
  </div>
  {% endif %}
</body>
</html>
"""
)


# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    html = PAGE.render(
        ran=False,
        q="Every 2 years what should we do for the safety precuations of YM358A tractor?",
        top_k_text=5,
        top_k_img=6,
        temp=0.5,
        answer="",
        texts=[],
        images=[],
    )
    return HTMLResponse(html)


@app.post("/query", response_class=HTMLResponse)
def query(
    request: Request,
    q: str = Form(...),
    top_k_text: int = Form(5),
    top_k_img: int = Form(6),
    temp: float = Form(0.5),
):
    # 1) Retrieve text + images (no need to print citations; we display results directly)
    text_matches = rag.search_text(q, top_n=top_k_text, chunk_text=True)
    image_matches = rag.search_images_by_description_text(q, top_n=top_k_img)

    # 2) Build a multimodal prompt and ask Gemini (NON-streaming for clean UI)
    #    We'll reuse your rag.answer_multimodal_query but set stream=False
    #    (If your get_gemini_response always streams, we can implement direct generate_content,
    #     but usually stream=False returns the full text.)
    out = rag.answer_multimodal_query(
        q,
        top_n_text=top_k_text,
        top_n_images=top_k_img,
        temperature=temp,
        stream=False,
        include_step_by_step=False,
    )

    answer = out["response"]
    # Some utils return a rich object; ensure string
    if not isinstance(answer, str):
        answer = str(answer)

    html = PAGE.render(
        ran=True,
        q=q,
        top_k_text=top_k_text,
        top_k_img=top_k_img,
        temp=temp,
        answer=answer,
        texts=_normalize_text_matches(text_matches),
        images=_normalize_image_matches(image_matches),
    )
    return HTMLResponse(html)
