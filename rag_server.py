# rag_server.py
import os
import shutil
import uuid
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

from pipeline import RagConfig, MultimodalRAGPipeline
from utils import get_gemini_response

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
OCR_LANG = _env("OCR_LANG", "mya+eng")

PROJECT_ID = _env("PROJECT_ID", "fortunaii")
LOCATION = _env("LOCATION", "us-central1")

# Allow disabling S3 sync for local/dev (set DISABLE_S3_SYNC=1).
DISABLE_S3_SYNC = _env("DISABLE_S3_SYNC", "0")
INIT_RAG_ON_STARTUP = _env("INIT_RAG_ON_STARTUP", "0")


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
_rag: Optional[MultimodalRAGPipeline] = None
_rag_error: Optional[str] = None


def _sync_from_s3() -> None:
    """Pull PDFs and cache from S3 into local dirs (if S3 configured)."""
    if DISABLE_S3_SYNC == "1":
        return
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
    if DISABLE_S3_SYNC == "1":
        return {"cache": 0, "pdfs": 0}
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


def _get_rag() -> MultimodalRAGPipeline:
    global _rag, _rag_error
    if _rag is not None:
        return _rag
    if _rag_error:
        raise RuntimeError(_rag_error)
    try:
        _sync_from_s3()
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
            ocr_lang=OCR_LANG,
        )
        rag_instance = MultimodalRAGPipeline(cfg)
        # Try to load cache first; if missing, build from PDFs.
        if not rag_instance.load_cache(CACHE_DIR, rebuild_image_objects=False):
            print("Metadata cache not found. Building metadata...")
            rag_instance.build_metadata(
                pdf_folder_path=PDF_FOLDER,
                cache_dir=CACHE_DIR,
                force_rebuild=False,
                generation_config=GenerationConfig(temperature=0.2),
                ocr_fallback=True,
                image_save_dir=IMAGE_DIR,
            )
            _sync_to_s3()
        else:
            print("Metadata cache loaded from disk.")
        _rag = rag_instance
        return _rag
    except Exception as e:
        _rag_error = str(e)
        raise RuntimeError(_rag_error)


@app.on_event("startup")
def _ensure_rag():
    """Optionally init pipeline at startup (fails gracefully)."""
    if INIT_RAG_ON_STARTUP != "1":
        # Keep startup fast; RAG will be initialized lazily on first query/build-cache.
        return
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


def _rewrite_myanmar_to_english_query(
    rag: MultimodalRAGPipeline, myanmar_text: str
) -> str:
    """Rewrite Myanmar user text into English suited for manual embedding search."""
    instruction = (
        "Rewrite the following user question from Myanmar (Burmese) into a short, clear English search query "
        "suited for semantic search over a technical/service manual knowledge base. "
        "Preserve symptoms, part names, procedures, torque values, and numbers. "
        "Output only the English query text, with no explanation, labels, or quotation marks.\n\n"
        f"User question:\n{myanmar_text}\n\nEnglish query:"
    )
    out = get_gemini_response(
        rag.text_model,
        model_input=instruction,
        stream=False,
        generation_config=GenerationConfig(temperature=0.2, max_output_tokens=1024),
    )
    return (out or "").strip()


def _english_answer_to_myanmar(
    rag: MultimodalRAGPipeline, english_answer: str
) -> str:
    """Translate/summarize the English RAG answer into Myanmar for the user."""
    instruction = (
        "Translate the following English technical answer into natural Myanmar (Burmese). "
        "Preserve the full meaning; keep technical terms accurate (use common Roman abbreviations for parts where helpful). "
        "Output only Myanmar (Burmese) script text, with no English preamble or labels.\n\n"
        f"English answer:\n{english_answer}\n\nMyanmar answer:"
    )
    out = get_gemini_response(
        rag.text_model,
        model_input=instruction,
        stream=False,
        generation_config=GenerationConfig(temperature=0.2, max_output_tokens=4096),
    )
    return (out or "").strip()


def _rewrite_japanese_to_english_query(
    rag: MultimodalRAGPipeline, japanese_text: str
) -> str:
    """Rewrite Japanese user text into English suited for manual embedding search."""
    instruction = (
        "Rewrite the following user question from Japanese into a short, clear English search query "
        "suited for semantic search over a technical/service manual knowledge base. "
        "Preserve symptoms, part names, procedures, torque values, and numbers. "
        "Output only the English query text, with no explanation, labels, or quotation marks.\n\n"
        f"User question:\n{japanese_text}\n\nEnglish query:"
    )
    out = get_gemini_response(
        rag.text_model,
        model_input=instruction,
        stream=False,
        generation_config=GenerationConfig(temperature=0.2, max_output_tokens=1024),
    )
    return (out or "").strip()


def _english_answer_to_japanese(
    rag: MultimodalRAGPipeline, english_answer: str
) -> str:
    """Translate/summarize the English RAG answer into Japanese for the user."""
    instruction = (
        "Translate the following English technical answer into natural Japanese. "
        "Preserve the full meaning; keep technical terms accurate. "
        "Output only Japanese text, with no English preamble or labels.\n\n"
        f"English answer:\n{english_answer}\n\nJapanese answer:"
    )
    out = get_gemini_response(
        rag.text_model,
        model_input=instruction,
        stream=False,
        generation_config=GenerationConfig(temperature=0.2, max_output_tokens=4096),
    )
    return (out or "").strip()


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
    # "auto" | "en" | "my"
    answer_language: str = "auto"


class QueryResponse(BaseModel):
    answer: str
    texts: List[TextChunk]
    images: List[ImageMatch]


class MyanmarQueryRequest(BaseModel):
    """Myanmar (Burmese) question; rewritten to English internally for retrieval."""

    question: str
    top_k_text: int = 5
    top_k_img: int = 6
    temp: float = 0.5
    include_intermediate_english: bool = False


class MyanmarQueryResponse(BaseModel):
    answer: str
    texts: List[TextChunk]
    images: List[ImageMatch]
    english_query: Optional[str] = None
    english_answer: Optional[str] = None


class GenerateQuestionRequest(BaseModel):
    model_name: str
    language: str
    count: int = 1


class JapaneseQueryRequest(BaseModel):
    question: str
    top_k_text: int = 5
    top_k_img: int = 6
    temp: float = 0.5
    include_intermediate_english: bool = False


class JapaneseQueryResponse(BaseModel):
    answer: str
    texts: List[TextChunk]
    images: List[ImageMatch]
    english_query: Optional[str] = None
    english_answer: Optional[str] = None


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
    # "auto" | "en" | "my"
    answer_language: str = "auto"


class DiagnosticAction(BaseModel):
    id: str
    title: str
    description: str


class DiagnosticConfidence(BaseModel):
    score: float
    label: Optional[str] = None


class DiagnosticUrgency(BaseModel):
    label: str
    reason: Optional[str] = None


class DiagnosticTorqueSpec(BaseModel):
    label: Optional[str] = None
    value_nm: Optional[float] = None
    value_ft_lbs: Optional[float] = None
    raw: Optional[str] = None


class DiagnosticPayload(BaseModel):
    question: str
    top_k_text: int = 10
    top_k_img: int = 6
    temp: float = 0.4
    # "auto" | "en" | "my"
    answer_language: str = "auto"


class DiagnosticEnvelope(BaseModel):
    summary: str
    confidence: DiagnosticConfidence
    urgency: DiagnosticUrgency
    actions: List[DiagnosticAction]
    torque_spec: Optional[DiagnosticTorqueSpec] = None


class DiagnosticAPIResponse(BaseModel):
    question: str
    diagnostic: DiagnosticEnvelope
    images: List[ImageMatch]


class ChatMessage(BaseModel):
    role: str  # "user" or "model"
    content: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    question: str
    history: List[ChatMessage] = []
    top_k_text: int = 5
    top_k_img: int = 3
    temp: float = 0.4
    answer_language: str = "auto"


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    history: List[ChatMessage]
    texts: List[TextChunk]
    images: List[ImageMatch]


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


@app.get("/app-myanmar", response_class=HTMLResponse)
def app_myanmar_page():
    """Myanmar-input RAG viewer (uses POST /api/query-myanmar)."""
    path = TEMPLATES_DIR / "app-myanmar.html"
    if not path.exists():
        return HTMLResponse("<p>app-myanmar.html not found</p>", status_code=404)
    return FileResponse(path, media_type="text/html")


@app.get("/app/rag-myanmar-app.js")
def app_myanmar_js():
    """Serve the Myanmar-input RAG app script."""
    path = TEMPLATES_DIR / "rag-myanmar-app.js"
    if not path.exists():
        return JSONResponse({"error": "rag-myanmar-app.js not found"}, status_code=404)
    return FileResponse(path, media_type="application/javascript")


@app.get("/v1", response_class=HTMLResponse)
def v1_page():
    """Structured diagnostic UI that calls /api/v1/diagnose."""
    path = TEMPLATES_DIR / "v1.html"
    if not path.exists():
        return HTMLResponse("<p>v1.html not found</p>", status_code=404)
    return FileResponse(path, media_type="text/html")


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
            ocr_lang=OCR_LANG,
        )
        rag_instance = MultimodalRAGPipeline(cfg)
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
        answer_language=payload.answer_language,
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


@app.post("/api/query-myanmar", response_model=MyanmarQueryResponse)
def api_query_myanmar(payload: MyanmarQueryRequest):
    """
    Accept a Myanmar question, rewrite to English for vector search, run the same
    multimodal RAG as /api/query (English answer), then translate the answer to Myanmar.

    Example:

        curl -sS -X POST "http://127.0.0.1:8000/api/query-myanmar" \\
          -H "Content-Type: application/json" \\
          -d '{"question":"<Myanmar text>","top_k_text":5,"top_k_img":6,"temp":0.5,"include_intermediate_english":true}'
    """
    q_raw = (payload.question or "").strip()
    if not q_raw:
        return JSONResponse(
            {"detail": "question must be non-empty."},
            status_code=400,
        )

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

    english_query = _rewrite_myanmar_to_english_query(rag, q_raw)
    if not english_query or english_query == "Exception occurred":
        return JSONResponse(
            {"detail": "Failed to rewrite question to English. Try again or check Vertex AI / Gemini."},
            status_code=503,
        )

    text_matches = rag.search_text(
        english_query,
        top_n=payload.top_k_text,
        chunk_text=True,
    )
    image_matches = rag.search_images_by_description_text(
        english_query,
        top_n=payload.top_k_img,
    )
    out = rag.answer_multimodal_query(
        english_query,
        top_n_text=payload.top_k_text,
        top_n_images=payload.top_k_img,
        temperature=payload.temp,
        stream=False,
        include_step_by_step=False,
        answer_language="en",
    )
    english_answer = out["response"]
    if not isinstance(english_answer, str):
        english_answer = str(english_answer)

    if not english_answer.strip() or english_answer.strip() == "Exception occurred":
        return JSONResponse(
            {"detail": "RAG answer generation failed. Try again or check Vertex AI / Gemini."},
            status_code=503,
        )

    myanmar_answer = _english_answer_to_myanmar(rag, english_answer)
    if not myanmar_answer or myanmar_answer == "Exception occurred":
        return JSONResponse(
            {"detail": "Failed to translate answer to Myanmar. Try again or check Vertex AI / Gemini."},
            status_code=503,
        )

    texts_norm = _normalize_text_matches(text_matches)
    images_norm = _normalize_image_matches(image_matches)

    return MyanmarQueryResponse(
        answer=myanmar_answer,
        texts=[TextChunk(**t) for t in texts_norm],
        images=[ImageMatch(**img) for img in images_norm],
        english_query=english_query if payload.include_intermediate_english else None,
        english_answer=english_answer if payload.include_intermediate_english else None,
    )


@app.post("/api/query-japanese", response_model=JapaneseQueryResponse)
def api_query_japanese(payload: JapaneseQueryRequest):
    """
    Accept a Japanese question, rewrite to English for vector search, run the same
    multimodal RAG as /api/query (English answer), then translate the answer to Japanese.
    """
    q_raw = (payload.question or "").strip()
    if not q_raw:
        return JSONResponse(
            {"detail": "question must be non-empty."},
            status_code=400,
        )

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

    english_query = _rewrite_japanese_to_english_query(rag, q_raw)
    if not english_query or english_query == "Exception occurred":
        return JSONResponse(
            {"detail": "Failed to rewrite question to English. Try again or check Vertex AI / Gemini."},
            status_code=503,
        )

    text_matches = rag.search_text(
        english_query,
        top_n=payload.top_k_text,
        chunk_text=True,
    )
    image_matches = rag.search_images_by_description_text(
        english_query,
        top_n=payload.top_k_img,
    )
    out = rag.answer_multimodal_query(
        english_query,
        top_n_text=payload.top_k_text,
        top_n_images=payload.top_k_img,
        temperature=payload.temp,
        stream=False,
        include_step_by_step=False,
        answer_language="en",
    )
    english_answer = out["response"]
    if not isinstance(english_answer, str):
        english_answer = str(english_answer)

    if not english_answer.strip() or english_answer.strip() == "Exception occurred":
        return JSONResponse(
            {"detail": "RAG answer generation failed. Try again or check Vertex AI / Gemini."},
            status_code=503,
        )

    japanese_answer = _english_answer_to_japanese(rag, english_answer)
    if not japanese_answer or japanese_answer == "Exception occurred":
        return JSONResponse(
            {"detail": "Failed to translate answer to Japanese. Try again or check Vertex AI / Gemini."},
            status_code=503,
        )

    texts_norm = _normalize_text_matches(text_matches)
    images_norm = _normalize_image_matches(image_matches)

    return JapaneseQueryResponse(
        answer=japanese_answer,
        texts=[TextChunk(**t) for t in texts_norm],
        images=[ImageMatch(**img) for img in images_norm],
        english_query=english_query if payload.include_intermediate_english else None,
        english_answer=english_answer if payload.include_intermediate_english else None,
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
        answer_language=payload.answer_language,
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


@app.post("/api/query-upload", response_model=QueryResponse)
async def api_query_upload(
    question: str = Form(...),
    top_k_text: int = Form(5),
    top_k_img: int = Form(1),
    temp: float = Form(0.2),
    answer_language: str = Form("auto"),
    image: Optional[UploadFile] = File(None),
):
    """
    Multipart endpoint: question + optional uploaded image.
    If an image is provided, image retrieval uses image-embedding search.
    Answer is generated from text-context (same behavior as /api/query-with-image).
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

    # Answer from text-side (fast + stable)
    text_result = rag.answer_text_query(
        question,
        top_n=top_k_text,
        temperature=temp,
        stream=False,
        answer_language=answer_language,
    )
    answer = text_result.get("response", "")
    if not isinstance(answer, str):
        answer = str(answer)

    text_matches = rag.search_text(
        question,
        top_n=top_k_text,
        chunk_text=True,
    )

    # Image retrieval
    image_matches: Dict[Any, Dict[str, Any]]
    if image and image.filename:
        # Save to a temp location on disk for embedding extraction
        qimg_dir = Path(CACHE_DIR) / "query_images"
        qimg_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(image.filename).suffix.lower() or ".png"
        dest = qimg_dir / f"query_{uuid.uuid4().hex}{suffix}"
        content = await image.read()
        dest.write_bytes(content)

        image_matches = rag.search_images_by_image_embedding(
            question,
            image_query_path=str(dest),
            top_n=top_k_img,
        )
    else:
        image_matches = rag.search_images_by_description_text(
            question,
            top_n=top_k_img,
            embedding_size=rag.config.embedding_size,
        )

    texts_norm = _normalize_text_matches(text_matches)
    images_norm = _normalize_image_matches(image_matches)

    return QueryResponse(
        answer=answer,
        texts=[TextChunk(**t) for t in texts_norm],
        images=[ImageMatch(**img) for img in images_norm],
    )


@app.post("/api/v1/diagnose", response_model=DiagnosticAPIResponse)
def api_v1_diagnose(payload: DiagnosticPayload):
    """
    v1 diagnostic endpoint.
    Returns structured JSON used by the /v1 dashboard UI.
    """
    try:
        rag = _get_rag()
    except RuntimeError as e:
        raise RuntimeError(
            f"RAG not available: {e}. "
            "Check PROJECT_ID / LOCATION / GOOGLE_APPLICATION_CREDENTIALS and data/cache."
        )

    out = rag.answer_diagnostic_query(
        payload.question,
        top_n_text=payload.top_k_text,
        top_n_images=payload.top_k_img,
        temperature=payload.temp,
        stream=False,
        answer_language=payload.answer_language,
    )

    parsed = out.get("response_parsed") or {}
    image_matches = out.get("image_matches") or {}

    summary = parsed.get("summary") or ""
    conf = parsed.get("confidence") or {}
    urg = parsed.get("urgency") or {}
    actions_raw = parsed.get("actions") or []
    torque_raw = parsed.get("torque_spec")

    # Normalize confidence
    confidence = DiagnosticConfidence(
        score=float(conf.get("score", 0.0) or 0.0),
        label=str(conf.get("label") or "").strip() or None,
    )

    # Normalize urgency
    urgency = DiagnosticUrgency(
        label=str(urg.get("label") or "normal"),
        reason=str(urg.get("reason") or ""),
    )

    # Normalize actions
    actions: List[DiagnosticAction] = []
    if isinstance(actions_raw, list):
        for idx, a in enumerate(actions_raw):
            if not isinstance(a, dict):
                continue
            aid = str(a.get("id") or f"ACTION {idx+1:02d}")
            title = str(a.get("title") or f"Action {idx+1}")
            desc = str(a.get("description") or "")
            actions.append(
                DiagnosticAction(
                    id=aid,
                    title=title,
                    description=desc,
                )
            )

    torque_spec: Optional[DiagnosticTorqueSpec] = None
    if isinstance(torque_raw, dict):
        torque_spec = DiagnosticTorqueSpec(
            label=torque_raw.get("label"),
            value_nm=torque_raw.get("value_nm"),
            value_ft_lbs=torque_raw.get("value_ft_lbs"),
            raw=torque_raw.get("raw"),
        )

    envelope = DiagnosticEnvelope(
        summary=summary,
        confidence=confidence,
        urgency=urgency,
        actions=actions,
        torque_spec=torque_spec,
    )

    images_norm = _normalize_image_matches(image_matches)

    return DiagnosticAPIResponse(
        question=payload.question,
        diagnostic=envelope,
        images=[ImageMatch(**img) for img in images_norm],
    )


@app.post("/api/v1/diagnose-upload", response_model=DiagnosticAPIResponse)
async def api_v1_diagnose_upload(
    question: str = Form(...),
    top_k_text: int = Form(10),
    top_k_img: int = Form(6),
    temp: float = Form(0.4),
    answer_language: str = Form("auto"),
    image: Optional[UploadFile] = File(None),
):
    """
    Multipart variant of v1 diagnostic endpoint that accepts an optional image.
    The image is used for image-embedding-based retrieval; the diagnostic text
    summary is still based on text + image descriptions.
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

    # Compute image matches based on uploaded image (if any)
    if image and image.filename:
        qimg_dir = Path(CACHE_DIR) / "query_images"
        qimg_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(image.filename).suffix.lower() or ".png"
        dest = qimg_dir / f"query_{uuid.uuid4().hex}{suffix}"
        content = await image.read()
        dest.write_bytes(content)

        image_matches = rag.search_images_by_image_embedding(
            question,
            image_query_path=str(dest),
            top_n=top_k_img,
        )
    else:
        image_matches = rag.search_images_by_description_text(
            question,
            top_n=top_k_img,
        )

    out = rag.answer_diagnostic_query(
        question,
        top_n_text=top_k_text,
        top_n_images=top_k_img,
        temperature=temp,
        stream=False,
        answer_language=answer_language,
    )
    parsed = out.get("response_parsed") or {}

    summary = parsed.get("summary") or ""
    conf = parsed.get("confidence") or {}
    urg = parsed.get("urgency") or {}
    actions_raw = parsed.get("actions") or []
    torque_raw = parsed.get("torque_spec")

    confidence = DiagnosticConfidence(
        score=float(conf.get("score", 0.0) or 0.0),
        label=str(conf.get("label") or "").strip() or None,
    )

    urgency = DiagnosticUrgency(
        label=str(urg.get("label") or "normal"),
        reason=str(urg.get("reason") or ""),
    )

    actions: List[DiagnosticAction] = []
    if isinstance(actions_raw, list):
        for idx, a in enumerate(actions_raw):
            if not isinstance(a, dict):
                continue
            aid = str(a.get("id") or f"ACTION {idx+1:02d}")
            title = str(a.get("title") or f"Action {idx+1}")
            desc = str(a.get("description") or "")
            actions.append(
                DiagnosticAction(
                    id=aid,
                    title=title,
                    description=desc,
                )
            )

    torque_spec: Optional[DiagnosticTorqueSpec] = None
    if isinstance(torque_raw, dict):
        torque_spec = DiagnosticTorqueSpec(
            label=torque_raw.get("label"),
            value_nm=torque_raw.get("value_nm"),
            value_ft_lbs=torque_raw.get("value_ft_lbs"),
            raw=torque_raw.get("raw"),
        )

    envelope = DiagnosticEnvelope(
        summary=summary,
        confidence=confidence,
        urgency=urgency,
        actions=actions,
        torque_spec=torque_spec,
    )

    images_norm = _normalize_image_matches(image_matches)

    return DiagnosticAPIResponse(
        question=question,
        diagnostic=envelope,
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
        answer_language="auto",
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
    answer_language: str = Form("auto"),
    image: Optional[UploadFile] = File(None),
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
            answer_language=answer_language,
            answer=f"RAG not available: {e}\n\nSet PROJECT_ID, LOCATION, GOOGLE_APPLICATION_CREDENTIALS in Render and ensure data/cache exist.",
            texts=[],
            images=[],
        )
        return HTMLResponse(html)

    # 1) Retrieve text (always)
    text_matches = rag.search_text(q, top_n=top_k_text, chunk_text=True)

    # 2) Images: if an image is uploaded, use image-embedding retrieval
    if image and image.filename:
        qimg_dir = Path(CACHE_DIR) / "query_images"
        qimg_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(image.filename).suffix.lower() or ".png"
        dest = qimg_dir / f"query_{uuid.uuid4().hex}{suffix}"
        dest.write_bytes(image.file.read())

        image_matches = rag.search_images_by_image_embedding(
            q,
            image_query_path=str(dest),
            top_n=top_k_img,
        )

        # Answer from text-side for consistency with the API path
        text_result = rag.answer_text_query(
            q,
            top_n=top_k_text,
            temperature=temp,
            stream=False,
            answer_language=answer_language,
        )
        answer = text_result.get("response", "")
        if not isinstance(answer, str):
            answer = str(answer)
    else:
        image_matches = rag.search_images_by_description_text(q, top_n=top_k_img)

        # Existing multimodal prompt answer
        out = rag.answer_multimodal_query(
            q,
            top_n_text=top_k_text,
            top_n_images=top_k_img,
            temperature=temp,
            stream=False,
            include_step_by_step=False,
            answer_language=answer_language,
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
        answer_language=answer_language,
        answer=answer,
        texts=_normalize_text_matches(text_matches),
        images=_normalize_image_matches(image_matches),
    )
    return HTMLResponse(html)


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    path = TEMPLATES_DIR / "chat.html"
    if not path.exists():
        return HTMLResponse("<p>chat.html not found</p>", status_code=404)
    return FileResponse(path, media_type="text/html")


def _condense_conversational_query(rag, question: str, history: List[ChatMessage]) -> str:
    """Rewrite follow-up question to a standalone search query containing context."""
    if not history:
        return question

    # Format history into a simple transcript
    history_str = ""
    for msg in history[-4:]:  # Limit to last 4 turns for speed/efficiency
        history_str += f"{msg.role.upper()}: {msg.content}\n"

    instruction = (
        "Given the following conversation history and a follow-up question, "
        "rephrase the follow-up question into a short, standalone English search query "
        "that captures the exact context, machine parts, and troubleshooting intent. "
        "Output ONLY the standalone search query text, without explanations, markdown, or quotation marks.\n\n"
        f"Chat History:\n{history_str}"
        f"Follow-up Question: {question}\n\n"
        "Standalone English Query:"
    )
    
    out = get_gemini_response(
        rag.text_model,
        model_input=instruction,
        stream=False,
        generation_config=GenerationConfig(temperature=0.2, max_output_tokens=256),
    )
    return (out or question).strip()


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(payload: ChatRequest):
    """Multi-turn conversational RAG backend endpoint."""
    try:
        rag = _get_rag()
    except RuntimeError as e:
        return JSONResponse({"detail": f"RAG not ready: {e}"}, status_code=503)

    session_id = payload.session_id or str(uuid.uuid4())
    
    # 1. Condense/Rewrite follow-up query using history
    search_query = _condense_conversational_query(rag, payload.question, payload.history)
    
    # 2. Retrieve resources using condensed query
    text_matches = rag.search_text(search_query, top_n=payload.top_k_text, chunk_text=True)
    image_matches = rag.search_images_by_description_text(search_query, top_n=payload.top_k_img)
    
    # 3. Format conversational prompt context
    context_str = ""
    for idx, t in enumerate(text_matches.values()):
        context_str += f"Manual Clip [{idx+1}]:\n{t.get('chunk_text', '')}\n\n"

    history_str = ""
    for msg in payload.history:
        role_label = "Farmer" if msg.role == "user" else "Tractor Assistant"
        history_str += f"{role_label}: {msg.content}\n"

    system_prompt = (
        "You are an empathetic, expert tractor technician and farmer's advisor.\n"
        "Your goal is to guide the farmer safely and step-by-step through their troubleshooting scenario.\n\n"
        "Guidelines:\n"
        "1. Keep answers concise, extremely practical, and structured as steps or simple recommendations.\n"
        "2. Keep a friendly, helpful tone to support the farmer or mechanic.\n"
        "3. Only use instructions from the provided Operation Manual Clips below. If the manual clips do not contain the answer, "
        "gently instruct the farmer to perform general safety steps and check in with their local dealer.\n\n"
        f"Operation Manual Clips:\n{context_str}\n"
        f"Conversation History:\n{history_str}"
        f"Farmer's Latest Query: {payload.question}\n\n"
        "Tractor Assistant Response:"
    )

    # 4. Generate Answer
    out = get_gemini_response(
        rag.text_model,
        model_input=system_prompt,
        stream=False,
        generation_config=GenerationConfig(temperature=payload.temp, max_output_tokens=1024),
    )
    answer = (out or "").strip()

    # Normalize responses
    texts_norm = _normalize_text_matches(text_matches)
    images_norm = _normalize_image_matches(image_matches)

    # Update history list
    new_history = list(payload.history)
    new_history.append(ChatMessage(role="user", content=payload.question))
    new_history.append(ChatMessage(role="model", content=answer))

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        history=new_history,
        texts=[TextChunk(**t) for t in texts_norm],
        images=[ImageMatch(**img) for img in images_norm]
    )


@app.get("/debug-generator", response_class=HTMLResponse)
def debug_generator_page():
    path = TEMPLATES_DIR / "generator.html"
    if not path.exists():
        return HTMLResponse("<p>generator.html not found</p>", status_code=404)
    return FileResponse(path, media_type="text/html")


@app.get("/api/models")
def get_available_models():
    try:
        rag = _get_rag()
        if rag.text_metadata_df is not None:
            files = list(rag.text_metadata_df["file_name"].unique())
            models = []
            for f in files:
                name_without_ext = os.path.splitext(f)[0]
                models.append({
                    "id": f,
                    "name": name_without_ext.upper()
                })
            return {"models": models}
    except Exception as e:
        print(f"Error fetching models from cache: {e}")
    # Fallback to defaults if cache is not loaded yet
    return {"models": [{"id": "ym358a.pdf", "name": "YM358A"}]}


@app.post("/api/generate-random-question")
def api_generate_random_question(payload: GenerateQuestionRequest):
    import json
    import datetime
    import random
    
    try:
        rag = _get_rag()
    except RuntimeError as e:
        return JSONResponse(
            {"error": f"RAG not available: {e}"},
            status_code=503,
        )

    df = rag.text_metadata_df
    if df is None:
        return JSONResponse(
            {"error": "RAG text metadata cache is not loaded."},
            status_code=500,
        )

    # Filter by model_name (which corresponds to file_name in dataframe, case insensitive)
    matched_df = df[df["file_name"].str.lower() == payload.model_name.lower()]
    if matched_df.empty:
        matched_df = df[df["file_name"].str.lower().str.contains(payload.model_name.lower())]
        
    if matched_df.empty:
        matched_df = df

    if matched_df.empty:
        return JSONResponse(
            {"error": f"No text chunks found in cache for model {payload.model_name}."},
            status_code=404,
        )

    # Helper to check if a chunk is clean
    def is_clean_chunk(text: str) -> bool:
        if not text:
            return False
        clean = text.strip()
        if not clean:
            return False
        if clean.lower() == "no text available in this chunk.":
            return False
        # Remove dots and whitespace and see if we have enough characters
        dots_removed = clean.replace(".", "").strip()
        if len(dots_removed) < 15:
            return False
        return True

    # Filter out bad chunks and keep order
    matched_df = matched_df.reset_index(drop=True)
    clean_indices = [i for i, row in matched_df.iterrows() if is_clean_chunk(row.get("chunk_text") or row.get("text"))]
    
    if not clean_indices:
        return JSONResponse(
            {"error": "No clean/descriptive text chunks found for model question generation."},
            status_code=404,
        )

    count = max(1, min(payload.count, 50))
    
    if count <= len(clean_indices):
        start_indices = random.sample(clean_indices, count)
    else:
        start_indices = random.choices(clean_indices, k=count)

    results = []
    for start_idx in start_indices:
        # Pull up to 5 consecutive clean chunks from matched_df
        chunks_to_use = []
        for offset in range(5):
            candidate_idx = start_idx + offset
            if candidate_idx >= len(matched_df):
                break
            candidate_row = matched_df.iloc[candidate_idx]
            
            # Check if same document
            if candidate_row["file_name"] != matched_df.iloc[start_idx]["file_name"]:
                break
                
            candidate_text = candidate_row.get("chunk_text") or candidate_row.get("text") or ""
            if is_clean_chunk(candidate_text):
                chunks_to_use.append(candidate_row)
            else:
                break

        if not chunks_to_use:
            chunks_to_use = [matched_df.iloc[start_idx]]

        # Merge adjacent chunks' text and page/chunk ranges
        merged_texts = []
        page_nums = []
        chunk_numbers = []
        file_name = chunks_to_use[0]["file_name"]
        
        for c in chunks_to_use:
            c_text = c.get("chunk_text") or c.get("text") or ""
            merged_texts.append(c_text.strip())
            p = int(c.get("page_num", 0))
            chnk = int(c.get("chunk_number", 0))
            if p not in page_nums:
                page_nums.append(p)
            if chnk not in chunk_numbers:
                chunk_numbers.append(chnk)

        merged_chunk_text = "\n\n---\n\n".join(merged_texts)
        page_range_str = ", ".join(map(str, sorted(page_nums)))
        chunk_range_str = ", ".join(map(str, sorted(chunk_numbers)))

        # Build robust prompt for question generation
        if payload.language == "my":
            instruction = (
                "You are an expert tractor mechanic and farmer advisor.\n"
                "Based on the following instruction/manual text chunk(s) from a tractor operation manual, "
                "generate ONE high-quality, natural, and extremely realistic question that either a farmer (end user of the tractor) "
                "or a mechanic would ask in a real-world troubleshooting or maintenance scenario, written in Myanmar (Burmese) language.\n\n"
                "Guidelines for the question:\n"
                "1. The question must be a SINGLE, CONCISE, and SIMPLE sentence focusing on exactly ONE specific practical action or troubleshooting symptom mentioned in the text chunk.\n"
                "2. CRITICAL: Do NOT ask compound questions, do NOT include multiple sub-questions, and do NOT try to cover the entire text chunk. Just select one single, direct topic (e.g., a specific procedure, a single symptom, or a single maintenance task) from the manual text and ask a simple, single-clause query about it.\n"
                "3. Use these exact styles of simple, single-clause queries as reference (translated into natural, colloquial Myanmar/Burmese script):\n"
                "   - 'How to replace the clutch' (ကလပ်ပြား ဘယ်လိုလဲမလဲ)\n"
                "   - 'How to change the engine oil' (အင်ဂျင်ဝိုင် ဘယ်လိုလဲရမလဲ)\n"
                "   - 'What should we do if we cannot wake up our y35a tractor?' (ထရက်တာ စက်နှိုးမရရင် ဘာလုပ်ရမလဲ)\n"
                "   - 'What should we do if the front axle is leaking oil?' (ရှေ့ဝင်ရိုး ဆီယိုနေရင် ဘာလုပ်ရမလဲ)\n"
                "   - 'What should we do if the engine is emitting smoke?' (အင်ဂျင်ကနေ မီးခိုးတွေ ထွက်နေရင် ဘာလုပ်ရမလဲ)\n"
                "   - 'What should we do with tappet clearance?' (တပက် ကစားသံ/တပက်ကလီးယားရင့်စ် ပြဿနာဖြစ်ရင် ဘာလုပ်ရမလဲ)\n"
                "4. Absolutely avoid hyper-specific questions asking for precise figures from a table (e.g., do NOT ask 'What is the maximum allowed pressure for 8-18 6PR front tires?' or 'What is the clearance value in mm?'). "
                "Instead, ask a practical, high-level troubleshooting or maintenance question that would lead a user or mechanic to refer to this manual section.\n"
                "5. Output ONLY the single Myanmar (Burmese) question. Do NOT include any English translation, introductory phrase, explanations, conversational filler, or quotation marks.\n\n"
                f"Manual Chunk(s):\n\"\"\"\n{merged_chunk_text}\n\"\"\"\n\n"
                "Farmer/Mechanic Question (in Myanmar/Burmese script):"
            )
        elif payload.language in ("ja", "jp"):
            instruction = (
                "You are an expert tractor mechanic and farmer advisor.\n"
                "Based on the following instruction/manual text chunk(s) from a tractor operation manual, "
                "generate ONE high-quality, natural, and extremely realistic question that either a farmer (end user of the tractor) "
                "or a mechanic would ask in a real-world troubleshooting or maintenance scenario, written in Japanese.\n\n"
                "Guidelines for the question:\n"
                "1. The question must be a SINGLE, CONCISE, and SIMPLE sentence focusing on exactly ONE specific practical action or troubleshooting symptom mentioned in the text chunk.\n"
                "2. CRITICAL: Do NOT ask compound questions, do NOT include multiple sub-questions, and do NOT try to cover the entire text chunk. Just select one single, direct topic (e.g., a specific procedure, a single symptom, or a single maintenance task) from the manual text and ask a simple, single-clause query about it.\n"
                "3. Use these exact styles of simple, single-clause queries as reference (translated into natural, colloquial Japanese):\n"
                "   - 'How to replace the clutch' (クラッチの交換方法は？)\n"
                "   - 'How to OPERATE THE ENGINE' (エンジンの操作方法は？)\n"
                "   - 'How to change the engine oil' (エンジンオイルの交換方法は？)\n"
                "   - 'What should we do if we cannot wake up our y35a tractor?' (トラクターのエンジンがかからない場合はどうすればよいですか？)\n"
                "   - 'What should we do front axle is leaking oil' (フロントアクスルからオイル漏れしている場合はどうすればよいですか？)\n"
                "   - 'What should we do if the engine is emitting smoke?' (エンジンから煙が出ている場合はどうすればよいですか？)\n"
                "   - 'What should we do with tappet clearance?' (タペットの隙間調整はどうすればよいですか？)\n"
                "4. Absolutely avoid hyper-specific questions asking for precise figures from a table (e.g., do NOT ask 'What is the maximum allowed pressure for 8-18 6PR front tires?' or 'What is the clearance value in mm?'). "
                "Instead, ask a practical, high-level troubleshooting or maintenance question that would lead a user or mechanic to refer to this manual section.\n"
                "5. Output ONLY the single Japanese question. Do NOT include any English translation, introductory phrase, explanations, conversational filler, or quotation marks.\n\n"
                f"Manual Chunk(s):\n\"\"\"\n{merged_chunk_text}\n\"\"\"\n\n"
                "Farmer/Mechanic Question (in Japanese script):"
            )
        else:
            instruction = (
                "You are an expert tractor mechanic and farmer advisor.\n"
                "Based on the following instruction/manual text chunk(s) from a tractor operation manual, "
                "generate ONE high-quality, natural, and extremely realistic question that either a farmer (end user of the tractor) "
                "or a mechanic would ask in a real-world troubleshooting or maintenance scenario, written in English.\n\n"
                "Guidelines for the question:\n"
                "1. The question must be a SINGLE, CONCISE, and SIMPLE sentence focusing on exactly ONE specific practical action or troubleshooting symptom mentioned in the text chunk.\n"
                "2. CRITICAL: Do NOT ask compound questions, do NOT include multiple sub-questions, and do NOT try to cover the entire text chunk. Just select one single, direct topic (e.g., a specific procedure, a single symptom, or a single maintenance task) from the manual text and ask a simple, single-clause query about it.\n"
                "3. Use these exact styles of simple, single-clause queries as reference:\n"
                "   - 'How to replace the clutch'\n"
                "   - 'How to OPERATE THE ENGINE'\n"
                "   - 'How to change the engine oil'\n"
                "   - 'What should we do if we cannot wake up our y35a tractor?'\n"
                "   - 'What should we do front axle is leaking oil'\n"
                "   - 'What should we do if the engine is emitting smoke?'\n"
                "   - 'What should we do with tappet clearance?'\n"
                "4. CRITICAL: Absolutely avoid hyper-specific questions asking for precise figures from a table (e.g., do NOT ask 'What is the maximum allowed pressure for 8-18 6PR front tires?' or 'What is the clearance value in mm?'). "
                "Instead, ask a practical, high-level troubleshooting or maintenance question that would lead a user or mechanic to refer to this manual section.\n"
                "5. Output ONLY the single English question. Do NOT include any introductory phrase, explanations, conversational filler, or quotation marks.\n\n"
                f"Manual Chunk(s):\n\"\"\"\n{merged_chunk_text}\n\"\"\"\n\n"
                "Farmer/Mechanic Question (in English):"
            )

        try:
            generated_question = get_gemini_response(
                rag.text_model,
                model_input=instruction,
                stream=False,
                generation_config=GenerationConfig(temperature=0.7, max_output_tokens=512),
            )
            generated_question = (generated_question or "").strip().strip('"').strip("'")
        except Exception as e:
            return JSONResponse(
                {"error": f"Failed to generate question with Gemini: {str(e)}"},
                status_code=500,
            )

        question_item = {
            "id": str(uuid.uuid4()),
            "file_name": file_name,
            "page_num": page_nums[0],
            "page_range": page_range_str,
            "chunk_number": chunk_numbers[0],
            "chunk_range": chunk_range_str,
            "chunk_text": merged_chunk_text,
            "generated_question": generated_question,
            "language": payload.language,
            "timestamp": datetime.datetime.now().isoformat()
        }
        results.append(question_item)

    # Persistent JSON storage
    questions_json_path = Path(CACHE_DIR) / "generated_questions.json"
    existing_questions = []
    if questions_json_path.exists():
        try:
            with open(questions_json_path, "r", encoding="utf-8") as f:
                existing_questions = json.load(f)
                if not isinstance(existing_questions, list):
                    existing_questions = []
        except Exception as e:
            print(f"Error reading existing questions JSON: {e}")
            
    existing_questions.extend(results)
    
    try:
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        with open(questions_json_path, "w", encoding="utf-8") as f:
            json.dump(existing_questions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error writing to questions JSON: {e}")

    return {
        "success": True,
        "questions": results,
    }


