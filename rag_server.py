# rag_server.py
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

from vertexai.generative_models import GenerationConfig

from new_pipeline import Pipeline

load_dotenv()

try:
    from s3_storage import (
        download_cache_from_s3,
        download_pdfs_from_s3,
        is_s3_configured,
        upload_cache_to_s3,
        upload_pdf_file_to_s3,
        upload_pdfs_to_s3,
    )
except ImportError:
    is_s3_configured = lambda: False
    download_pdfs_from_s3 = lambda _: 0
    download_cache_from_s3 = lambda _: 0
    upload_cache_to_s3 = lambda _: 0
    upload_pdfs_to_s3 = lambda _: 0
    upload_pdf_file_to_s3 = lambda *a, **k: False


# -----------------------------
# CONFIG (env for Render; fallback for local)
# -----------------------------
def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


PDF_FOLDER = _env("PDF_FOLDER", "./data/")
CACHE_DIR = _env("CACHE_DIR", "./cache")
IMAGE_DIR = _env("IMAGE_DIR", "./cache/images")  # must match RagConfig.image_save_dir

PROJECT_ID = _env("PROJECT_ID", "fortunaii")
LOCATION = _env("LOCATION", "us-central1")


# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Syspare RAG Python")

# CORS: allow TSX/viewer on different origin (e.g. localhost:5173 or your frontend)
_cors_origins = _env("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").strip()
_cors_list = [o.strip() for o in _cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Templates (HTML in templates/)
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)

# Serve images folder in browser as /static/...
Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=IMAGE_DIR), name="static")


# -----------------------------
# Initialize pipeline once (lazy on first use if init fails)
# -----------------------------
_rag: Optional[Pipeline] = None
_rag_error: Optional[str] = None


def _sync_from_s3() -> None:
    """Pull PDFs and cache from S3 into local dirs (if S3 configured)."""
    if not is_s3_configured():
        return
    Path(PDF_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    n_pdfs = download_pdfs_from_s3(PDF_FOLDER)
    n_cache = download_cache_from_s3(CACHE_DIR)
    if n_pdfs or n_cache:
        print(f"S3 sync: downloaded {n_pdfs} PDF(s), {n_cache} cache file(s).")


def _sync_to_s3() -> Dict[str, int]:
    """Push cache (and PDFs) to S3 (if S3 configured). Returns counts."""
    if not is_s3_configured():
        return {"cache": 0, "pdfs": 0}
    n_cache = upload_cache_to_s3(CACHE_DIR)
    n_pdfs = upload_pdfs_to_s3(PDF_FOLDER)
    if n_cache or n_pdfs:
        print(f"S3 sync: uploaded {n_cache} cache file(s), {n_pdfs} PDF(s).")
    return {"cache": n_cache, "pdfs": n_pdfs}


def _clear_rag_state() -> None:
    """Reset in-memory RAG so next request will load or rebuild."""
    global _rag, _rag_error
    _rag = None
    _rag_error = None


def _get_rag() -> Pipeline:
    global _rag, _rag_error
    if _rag is not None:
        return _rag
    if _rag_error:
        raise RuntimeError(_rag_error)
    try:
        _sync_from_s3()
        rag_instance = Pipeline()
        try:
            # Try to load existing cache first
            rag_instance.load_cache(cache_dir=CACHE_DIR)
            print("Metadata cache loaded from disk.")
        except Exception:
            print("Metadata not loaded. Building metadata...")
            rag_instance.build_metadata(
                pdf_folder_path=PDF_FOLDER,
                cache_dir=CACHE_DIR,
                force_rebuild=True,
                generation_config=GenerationConfig(temperature=0.2),
                ocr_fallback=True,
                image_save_dir=IMAGE_DIR,
            )
            _sync_to_s3()
        _rag = rag_instance
        return _rag
    except Exception as e:
        _rag_error = str(e)
        raise RuntimeError(_rag_error)


@app.on_event("startup")
def _ensure_rag():
    """Optionally init pipeline at startup (fails gracefully)."""
    try:
        _get_rag()
    except Exception as e:
        print(f"RAG not ready at startup: {e}")


# -----------------------------
# Helpers
# -----------------------------
def _safe_image_url(img_path: str) -> str:
    """
    Convert absolute/relative img_path to URL under /static/.
    This assumes img_path is inside IMAGE_DIR (possibly in subdirectories).
    """
    p = Path(img_path)
    root = Path(IMAGE_DIR).resolve()

    try:
        # Prefer path relative to IMAGE_DIR so subdirectories are preserved
        rel = p.resolve().relative_to(root)
    except Exception:
        # Fallback: just use the basename
        rel = p.name

    return f"/static/{rel.as_posix()}"


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
# API models (JSON RAG API)
# -----------------------------
class TextChunk(BaseModel):
    chunk_text: str
    score: Optional[float] = None
    page: Optional[int] = None
    doc: Optional[str] = None


class ImageMatch(BaseModel):
    img_url: str
    caption: str
    score: Optional[float] = None
    page: Optional[int] = None
    doc: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    top_k_text: int = 5
    top_k_img: int = 6
    temp: float = 0.5


class QueryResponse(BaseModel):
    answer: str
    texts: List[TextChunk]
    images: List[ImageMatch]


class QueryWithOptionalImageRequest(BaseModel):
    """
    JSON API request that mirrors the notebook's
    `test_question_with_optional_input_image` helper:
    - Always has a text question
    - Optionally provides an image path that will be used for
      image-embedding-based retrieval instead of description-text search.
    """

    question: str
    image_query_path: Optional[str] = None
    top_k_text: int = 5
    top_k_img: int = 1
    temp: float = 0.2
    embedding_size: int = 128


# -----------------------------
# Template render helper
# -----------------------------
def _render_page(**kwargs: Any) -> str:
    tpl = jinja_env.get_template("index.html")
    return tpl.render(**kwargs)


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    """For Render (and other platforms) health checks."""
    return {"status": "ok"}


@app.get("/app", response_class=HTMLResponse)
def app_page():
    """API-driven RAG viewer (same UI as index.html, uses POST /api/query)."""
    path = TEMPLATES_DIR / "app.html"
    if not path.exists():
        return HTMLResponse("<p>app.html not found</p>", status_code=404)
    return FileResponse(path, media_type="text/html")


@app.get("/app/rag-app.js")
def app_js():
    """Serve the API-driven RAG app script."""
    path = TEMPLATES_DIR / "rag-app.js"
    if not path.exists():
        return JSONResponse({"error": "rag-app.js not found"}, status_code=404)
    return FileResponse(path, media_type="application/javascript")


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF: save to local PDF_FOLDER and to S3 (if configured)."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            {"ok": False, "error": "Only PDF files allowed"},
            status_code=400,
        )
    Path(PDF_FOLDER).mkdir(parents=True, exist_ok=True)
    dest = Path(PDF_FOLDER) / file.filename
    try:
        content = await file.read()
        dest.write_bytes(content)
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": str(e)},
            status_code=500,
        )
    if is_s3_configured():
        upload_pdf_file_to_s3(str(dest))
    return JSONResponse({"ok": True, "filename": file.filename})


@app.post("/api/clean-cache")
def api_clean_cache():
    """Delete local cache files and reset RAG. Next query will rebuild from PDFs (or S3)."""
    global _rag, _rag_error
    cache_path = Path(CACHE_DIR)
    if cache_path.exists():
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    _clear_rag_state()
    return JSONResponse({"ok": True, "message": "Local cache cleared. Run a query or Build cache to rebuild."})


@app.post("/api/build-cache")
def api_build_cache():
    """Force rebuild metadata from PDFs and upload cache + PDFs to S3."""
    global _rag, _rag_error
    _clear_rag_state()
    try:
        _sync_from_s3()
        rag_instance = Pipeline()
        rag_instance.build_metadata(
            pdf_folder_path=PDF_FOLDER,
            cache_dir=CACHE_DIR,
            force_rebuild=True,
            generation_config=GenerationConfig(temperature=0.2),
            ocr_fallback=True,
            image_save_dir=IMAGE_DIR,
        )
        counts = _sync_to_s3()
        _rag = rag_instance
        return JSONResponse({
            "ok": True,
            "message": "Cache rebuilt and synced to S3.",
            "s3_uploaded": counts,
        })
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": str(e)},
            status_code=500,
        )


@app.post("/api/sync-to-s3")
def api_sync_to_s3():
    """Upload current local cache and PDFs to S3."""
    counts = _sync_to_s3()
    if not is_s3_configured():
        return JSONResponse(
            {"ok": False, "error": "S3 not configured. Set AWS_* and S3_BUCKET_NAME."},
            status_code=400,
        )
    return JSONResponse({
        "ok": True,
        "message": f"Uploaded {counts['cache']} cache file(s), {counts['pdfs']} PDF(s) to S3.",
        "uploaded": counts,
    })


@app.post("/api/query", response_model=QueryResponse)
def api_query(payload: QueryRequest):
    """
    JSON RAG endpoint for reuse from other services.

    Request body:
      {
        "question": "...",
        "top_k_text": 5,
        "top_k_img": 6,
        "temp": 0.5
      }
    """
    try:
        rag = _get_rag()
    except RuntimeError as e:
        return JSONResponse(
            {
                "detail": f"RAG not available: {e}. "
                "Check PROJECT_ID / LOCATION / GOOGLE_APPLICATION_CREDENTIALS and data/cache.",
            },
            status_code=503,
        )

    text_matches = rag.search_text(
        payload.question,
        top_n=payload.top_k_text,
        chunk_text=True,
    )
    image_matches = rag.search_images_by_description_text(
        payload.question,
        top_n=payload.top_k_img,
    )
    out = rag.answer_multimodal_query(
        payload.question,
        top_n_text=payload.top_k_text,
        top_n_images=payload.top_k_img,
        temperature=payload.temp,
        stream=False,
        include_step_by_step=False,
    )
    answer = out["response"]
    if not isinstance(answer, str):
        answer = str(answer)

    texts_norm = _normalize_text_matches(text_matches)
    images_norm = _normalize_image_matches(image_matches)

    return QueryResponse(
        answer=answer,
        texts=[TextChunk(**t) for t in texts_norm],
        images=[ImageMatch(**img) for img in images_norm],
    )


@app.post("/api/query-with-image", response_model=QueryResponse)
def api_query_with_optional_image(payload: QueryWithOptionalImageRequest):
    """
    JSON RAG endpoint that mirrors the notebook helper
    `test_question_with_optional_input_image`.

    Request body:
      {
        "question": "...",
        "image_query_path": "./out_images/ym358a/page_011/page_011_img_01_crop_01.png" | null,
        "top_k_text": 5,
        "top_k_img": 1,
        "temp": 0.2,
        "embedding_size": 128
      }
    """
    try:
        rag = _get_rag()
    except RuntimeError as e:
        return JSONResponse(
            {
                "detail": f"RAG not available: {e}. "
                "Check PROJECT_ID / LOCATION / GOOGLE_APPLICATION_CREDENTIALS and data/cache.",
            },
            status_code=503,
        )

    # Text-side answer: same as notebook's `answer_text_query`
    text_result = rag.answer_text_query(
        payload.question,
        top_n=payload.top_k_text,
        temperature=payload.temp,
        stream=False,
    )
    answer = text_result.get("response", "")
    if not isinstance(answer, str):
        answer = str(answer)

    # Text retrieval (for UI context list)
    text_matches = rag.search_text(
        payload.question,
        top_n=payload.top_k_text,
        chunk_text=True,
    )

    # Image retrieval:
    #  - if image_query_path is provided -> use image-embedding search
    #  - else -> use description-text search (same as /api/query)
    if payload.image_query_path:
        image_matches = rag.search_images_by_image_embedding(
            payload.question,
            image_query_path=payload.image_query_path,
            top_n=payload.top_k_img,
            embedding_size=payload.embedding_size,
        )
    else:
        image_matches = rag.search_images_by_description_text(
            payload.question,
            top_n=payload.top_k_img,
            embedding_size=payload.embedding_size,
        )

    texts_norm = _normalize_text_matches(text_matches)
    images_norm = _normalize_image_matches(image_matches)

    return QueryResponse(
        answer=answer,
        texts=[TextChunk(**t) for t in texts_norm],
        images=[ImageMatch(**img) for img in images_norm],
    )


@app.get("/", response_class=HTMLResponse)
def home():
    html = _render_page(
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
    try:
        rag = _get_rag()
    except RuntimeError as e:
        html = _render_page(
            ran=True,
            q=q,
            top_k_text=top_k_text,
            top_k_img=top_k_img,
            temp=temp,
            answer=f"RAG not available: {e}\n\nSet PROJECT_ID, LOCATION, GOOGLE_APPLICATION_CREDENTIALS in Render and ensure data/cache exist.",
            texts=[],
            images=[],
        )
        return HTMLResponse(html)

    # 1) Retrieve text + images (no need to print citations; we display results directly)
    text_matches = rag.search_text(q, top_n=top_k_text, chunk_text=True)
    image_matches = rag.search_images_by_description_text(q, top_n=top_k_img)

    # 2) Build a multimodal prompt and ask Gemini (NON-streaming for clean UI)
    out = rag.answer_multimodal_query(
        q,
        top_n_text=top_k_text,
        top_n_images=top_k_img,
        temperature=temp,
        stream=False,
        include_step_by_step=False,
    )

    answer = out["response"]
    if not isinstance(answer, str):
        answer = str(answer)

    html = _render_page(
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
