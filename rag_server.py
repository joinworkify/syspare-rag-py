# rag_server.py
from collections import OrderedDict
import json
import os
import re
import shutil
import threading
import time
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

from pipeline import (
    INSUFFICIENT_CONTEXT_SENTINEL,
    RagConfig,
    MultimodalRAGPipeline,
    has_insufficient_marker,
    strip_insufficient_marker,
)
from syspare_rag.config import (
    ManualConfig,
    ManualRegistry,
    load_manual_registry_from_env,
)
from utils import get_gemini_response, _print_progress

load_dotenv()

try:
    from s3_storage import (
        delete_manual_from_s3,
        download_manual_cache_from_s3,
        download_manual_pdfs_from_s3,
        download_manual_registry_from_s3,
        is_s3_configured,
        upload_image_to_s3,
        upload_manual_cache_to_s3,
        upload_manual_pdf_file_to_s3,
        upload_manual_pdfs_to_s3,
        upload_manual_registry_to_s3,
    )
except ImportError:
    is_s3_configured = lambda: False
    download_manual_cache_from_s3 = lambda *a, **k: 0
    download_manual_pdfs_from_s3 = lambda *a, **k: 0
    download_manual_registry_from_s3 = lambda *a, **k: False
    upload_image_to_s3 = lambda *a, **k: a[1] if len(a) > 1 else ""
    upload_manual_cache_to_s3 = lambda *a, **k: 0
    upload_manual_pdfs_to_s3 = lambda *a, **k: 0
    upload_manual_pdf_file_to_s3 = lambda *a, **k: False
    upload_manual_registry_to_s3 = lambda *a, **k: False
    delete_manual_from_s3 = lambda *a, **k: 0


# -----------------------------
# CONFIG (env for Render; fallback for local)
# -----------------------------
def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    value = _env(key, str(default))
    try:
        return int(value)
    except ValueError:
        print(f"Invalid {key}={value!r}; using {default}.")
        return default


PROJECT_ID = _env("PROJECT_ID", "fortunaii")
LOCATION = _env("LOCATION", "us-central1")

# config for generation
MAX_OUTPUT_TOKENS = 8192       # full answers, translations
MAX_QUERY_TOKENS = 256         # short query rewrites
MAX_CLASSIFY_TOKENS = 128      # yes/no classification
MAX_QUESTION_TOKENS = 512      # single question generation
TEMPERATURE = 0.2

# Allow disabling S3 sync for local/dev (set DISABLE_S3_SYNC=1).
DISABLE_S3_SYNC = _env("DISABLE_S3_SYNC", "0")
INIT_RAG_ON_STARTUP = _env("INIT_RAG_ON_STARTUP", "0")
INIT_ALL_RAG_ON_STARTUP = _env("INIT_ALL_RAG_ON_STARTUP", "0")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.5-flash")
# Shown when the first retrieval pass was too thin and the search was widened.
RETRIEVAL_EXPANDED_MESSAGE = (
    "First pass had insufficient detail, so retrieval was expanded before answering."
)

# Path to the manuals registry JSON (written by add-manual API)
_MANUALS_JSON_PATH: Path = (
    Path(_env("MANUALS_JSON", ""))
    if _env("MANUALS_JSON", "")
    else Path(__file__).resolve().parent / "manuals" / "manuals.json"
)

# Manual registry: list of available manuals + default selection.
manual_registry: ManualRegistry = load_manual_registry_from_env()
DEFAULT_MANUAL = manual_registry.default
DEFAULT_MANUAL_ID = manual_registry.default_id

# Backward-compat env defaults: keep legacy globals pointing at the default manual.
PDF_FOLDER = _env("PDF_FOLDER", DEFAULT_MANUAL.pdf_folder)
CACHE_DIR = _env("CACHE_DIR", DEFAULT_MANUAL.cache_dir)
IMAGE_DIR = _env("IMAGE_DIR", DEFAULT_MANUAL.image_dir)
OCR_LANG = _env("OCR_LANG", DEFAULT_MANUAL.ocr_lang or "mya+eng")

# set for the cache size
RAG_PIPELINE_CACHE_SIZE = max(
    1,
    _env_int("RAG_PIPELINE_CACHE_SIZE", 0)
    or (
        len(manual_registry.list())
        if _env("INIT_ALL_RAG_ON_STARTUP", "0") == "1"
        else 1
    ),
)
# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Syspare RAG Python")

# CORS: allow TSX/viewer on different origin (e.g. localhost:5173 or your frontend)
_cors_origins = _env(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).strip()
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

# Serve each manual's image dir under /static/<manual_id>/...; also keep /static
# pointing at the default manual for backward compatibility.
# Per-manual mounts must be registered BEFORE the broader /static catch-all,
# otherwise Starlette routes /static/<manual_id>/... to the wrong handler.
for _manual in manual_registry.list():
    Path(_manual.image_dir).mkdir(parents=True, exist_ok=True)
    app.mount(
        f"/static/{_manual.manual_id}",
        StaticFiles(directory=_manual.image_dir),
        name=f"static-{_manual.manual_id}",
    )
Path(DEFAULT_MANUAL.image_dir).mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=DEFAULT_MANUAL.image_dir), name="static")


# -----------------------------
# Per-manual pipeline cache (lazy)
# -----------------------------
_pipelines: OrderedDict[str, MultimodalRAGPipeline] = OrderedDict()
_pipeline_errors: Dict[str, str] = {}
_pipeline_cache_lock = threading.RLock()
_pipeline_locks: Dict[str, threading.Lock] = {}
_pipeline_locks_lock = threading.Lock()

# Background training jobs: job_id → {status, progress, message, manual_id}
_training_jobs: Dict[str, dict] = {}
_training_jobs_lock = threading.Lock()


def _get_pipeline_lock(manual_id: str) -> threading.Lock:
    with _pipeline_locks_lock:
        if manual_id not in _pipeline_locks:
            _pipeline_locks[manual_id] = threading.Lock()
        return _pipeline_locks[manual_id]


def _get_cached_pipeline(manual_id: str) -> Optional[MultimodalRAGPipeline]:
    """Return cached pipeline and mark it most recently used."""
    with _pipeline_cache_lock:
        cached = _pipelines.get(manual_id)
        if cached is not None:
            _pipelines.move_to_end(manual_id)
        return cached


def _remember_pipeline(manual_id: str, rag: MultimodalRAGPipeline) -> None:
    """Store a pipeline and evict least recently used manuals over the limit."""
    with _pipeline_cache_lock:
        _pipelines[manual_id] = rag
        _pipelines.move_to_end(manual_id)

        while len(_pipelines) > RAG_PIPELINE_CACHE_SIZE:
            evicted_id, _ = _pipelines.popitem(last=False)
            print(
                f"[{evicted_id}] Evicted RAG pipeline from memory "
                f"(RAG_PIPELINE_CACHE_SIZE={RAG_PIPELINE_CACHE_SIZE})."
            )


def _resolve_manual(manual_id: Optional[str]) -> ManualConfig:
    """Return ManualConfig for an optional manual_id; default if blank."""
    try:
        return manual_registry.get(manual_id)
    except KeyError as exc:
        raise RuntimeError(str(exc))


def _available_model_labels() -> List[str]:
    """Display names of all registered manuals, for cross-model awareness."""
    return [m.display_name for m in manual_registry.list()]


def _sync_from_s3(manual: ManualConfig) -> None:
    """Pull a single manual's PDFs and cache from S3 into its local dirs."""
    if DISABLE_S3_SYNC == "1":
        return
    if not is_s3_configured():
        return
    abs_pdf = _resolve_manual_path(manual.pdf_folder)
    abs_cache = _resolve_manual_path(manual.cache_dir)
    Path(abs_pdf).mkdir(parents=True, exist_ok=True)
    Path(abs_cache).mkdir(parents=True, exist_ok=True)
    n_pdfs = download_manual_pdfs_from_s3(manual.manual_id, abs_pdf)
    n_cache = download_manual_cache_from_s3(manual.manual_id, abs_cache)
    if n_pdfs or n_cache:
        print(
            f"[{manual.manual_id}] S3 sync: downloaded {n_pdfs} PDF(s), "
            f"{n_cache} cache file(s)."
        )


def _resolve_manual_path(p: str) -> str:
    """Resolve a manual path to absolute, anchored at rag_server.py's directory."""
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str(Path(__file__).resolve().parent / path)


def _sync_to_s3(manual: ManualConfig) -> Dict[str, int]:
    """Push a single manual's cache (and PDFs) to S3."""
    if DISABLE_S3_SYNC == "1":
        return {"cache": 0, "pdfs": 0}
    if not is_s3_configured():
        return {"cache": 0, "pdfs": 0}
    n_cache = upload_manual_cache_to_s3(
        manual.manual_id, _resolve_manual_path(manual.cache_dir)
    )
    n_pdfs = upload_manual_pdfs_to_s3(
        manual.manual_id, _resolve_manual_path(manual.pdf_folder)
    )
    if n_cache or n_pdfs:
        print(
            f"[{manual.manual_id}] S3 sync: uploaded {n_cache} cache file(s), "
            f"{n_pdfs} PDF(s)."
        )
    return {"cache": n_cache, "pdfs": n_pdfs}


def _clear_rag_state(manual_id: Optional[str] = None) -> None:
    """Clear in-memory pipeline state for one manual (or all when None)."""
    with _pipeline_cache_lock:
        if manual_id is None:
            _pipelines.clear()
            _pipeline_errors.clear()
            return
        _pipelines.pop(manual_id, None)
        _pipeline_errors.pop(manual_id, None)


def _remap_image_paths(rag: MultimodalRAGPipeline, manual: ManualConfig) -> None:
    """Rewrite cached image_metadata_df.img_path to current image_dir.

    Skips rows that already hold S3 URLs (starts with http).
    Caches built in earlier layouts (e.g. ./cache/images/...) store stale paths.
    We rebase them to manual.image_dir/<basename or relative subpath> so image
    retrieval and URL building work after the per-manual restructure.
    """
    df = rag.image_metadata_df
    if df is None or "img_path" not in df.columns:
        return
    img_root = Path(manual.image_dir).resolve()

    def _rebase(value):
        if not value:
            return value
        if str(value).startswith("http"):
            return value
        original = Path(str(value))
        # If already pointing at a real file under image_dir, keep it.
        candidate = (
            (img_root / original.name) if not original.is_absolute() else original
        )
        try:
            if candidate.exists():
                return str(candidate)
        except Exception:
            pass
        fallback = img_root / Path(str(value)).name
        return str(fallback) if fallback.exists() else str(candidate)

    df["img_path"] = df["img_path"].map(_rebase)


def _upload_images_to_s3_and_rewrite(
    rag: MultimodalRAGPipeline, manual: ManualConfig
) -> int:
    """Upload each extracted image to S3, rewrite img_path to public URL, re-save cache.

    This runs after build_metadata() so local files exist, but before sync_to_s3()
    so the bulk cache upload skips the images/ subdir.
    Returns number of images uploaded.
    """
    if not is_s3_configured():
        return 0
    df = rag.image_metadata_df
    if df is None or "img_path" not in df.columns:
        return 0

    total = int(
        (
            df["img_path"].notna() & ~df["img_path"].astype(str).str.startswith("http")
        ).sum()
    )
    if total == 0:
        return 0
    count = 0
    _upload_start = time.time()
    print(f"\n[{manual.manual_id}] Uploading {total} image(s) to S3...")

    def _upload(value):
        nonlocal count
        if not value or str(value).startswith("http"):
            return value
        try:
            url = upload_image_to_s3(manual.manual_id, str(value))
            count += 1
            _print_progress(
                f"[{manual.manual_id}] S3 upload", count, total, _upload_start
            )
            return url
        except Exception as exc:
            print(f"\nWarning: failed to upload image {value} to S3: {exc}")
            return value

    df["img_path"] = df["img_path"].map(_upload)
    rag.save_cache(manual.cache_dir)
    return count


def _build_rag_config(manual: ManualConfig) -> RagConfig:
    return RagConfig(
        project_id=PROJECT_ID,
        location=LOCATION,
        model_name=GEMINI_MODEL,
        embedding_size=1408,
        embedding_model_name="multimodalembedding@001",
        image_save_dir=manual.image_dir,
        enable_ocr_fallback=True,
        ocr_min_chars=40,
        ocr_dpi=200,
        ocr_lang=manual.ocr_lang or OCR_LANG,
    )


def _get_rag(manual_id: Optional[str] = None) -> MultimodalRAGPipeline:
    """Return a cached pipeline for the requested manual, loading or building lazily."""
    manual = _resolve_manual(manual_id)
    mid = manual.manual_id

    # Fast path: already loaded
    cached = _get_cached_pipeline(mid)
    if cached is not None:
        return cached

    err = _pipeline_errors.get(mid)
    if err:
        raise RuntimeError(err)

    # Slow path: acquire per-manual lock so concurrent requests don't each
    # trigger a full S3 sync + cache load simultaneously.
    with _get_pipeline_lock(mid):
        # Re-check after acquiring lock (another thread may have loaded it)
        cached = _get_cached_pipeline(mid)
        if cached is not None:
            return cached
        err = _pipeline_errors.get(mid)
        if err:
            raise RuntimeError(err)

        try:
            _sync_from_s3(manual)
            cfg = _build_rag_config(manual)
            rag_instance = MultimodalRAGPipeline(cfg)
            # Try to load cache first; if missing, build from PDFs.
            if not rag_instance.load_cache(
                manual.cache_dir, rebuild_image_objects=False
            ):
                print(f"[{mid}] Metadata cache not found. Building metadata...")
                rag_instance.build_metadata(
                    pdf_folder_path=manual.pdf_folder,
                    cache_dir=manual.cache_dir,
                    force_rebuild=False,
                    generation_config=GenerationConfig(temperature=TEMPERATURE),
                    ocr_fallback=True,
                    image_save_dir=manual.image_dir,
                )
                _upload_images_to_s3_and_rewrite(rag_instance, manual)
                print(f"[{mid}] Syncing cache to S3...")
                _sync_to_s3(manual)
                print(f"[{mid}] S3 sync done.")
            else:
                print(f"[{mid}] Metadata cache loaded from disk.")
            print(f"[{mid}] Remapping image paths...")
            _remap_image_paths(rag_instance, manual)
            print(f"[{mid}] Pipeline ready.")
            _remember_pipeline(mid, rag_instance)
            return rag_instance
        except Exception as e:
            _pipeline_errors[mid] = str(e)
            raise RuntimeError(str(e))


def _reload_registry_from_json(path: Path) -> None:
    """Hot-reload manual_registry from a manuals.json file on disk."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        new_manuals = {
            m["manual_id"]: ManualConfig(
                manual_id=str(m["manual_id"]),
                display_name=str(m.get("display_name", m["manual_id"])),
                pdf_folder=str(m["pdf_folder"]),
                cache_dir=str(m["cache_dir"]),
                image_dir=str(m.get("image_dir", str(Path(m["cache_dir"]) / "images"))),
                ocr_lang=str(m.get("ocr_lang", "eng")),
                description=str(m.get("description", "")),
            )
            for m in payload
        }
        manual_registry._manuals = new_manuals
        print(f"[startup] Registry hot-reloaded: {list(new_manuals)}")
    except Exception as exc:
        print(f"[startup] Registry reload failed: {exc}")


@app.on_event("startup")
def _ensure_rag():
    """Warm up pipelines in background so health checks pass immediately."""
    # Pull latest manuals.json from S3 so add/remove changes survive redeployment.
    if is_s3_configured() and DISABLE_S3_SYNC != "1":
        downloaded = download_manual_registry_from_s3(str(_MANUALS_JSON_PATH))
        if downloaded:
            _reload_registry_from_json(_MANUALS_JSON_PATH)

    if INIT_ALL_RAG_ON_STARTUP == "1":

        def _warmup_all():
            for manual in manual_registry.list():
                try:
                    _get_rag(manual.manual_id)
                except Exception as e:
                    print(f"[{manual.manual_id}] RAG warmup failed: {e}")

        threading.Thread(target=_warmup_all, daemon=True).start()
    elif INIT_RAG_ON_STARTUP == "1":

        def _warmup():
            try:
                _get_rag(DEFAULT_MANUAL_ID)
            except Exception as e:
                print(f"[{DEFAULT_MANUAL_ID}] RAG warmup failed: {e}")

        threading.Thread(target=_warmup, daemon=True).start()


def _run_training_job(job_id: str, manual: ManualConfig) -> None:
    """Background thread: run full build_metadata pipeline with phase progress updates."""

    def _set(progress: int, message: str, status: str = "running") -> None:
        with _training_jobs_lock:
            if job_id in _training_jobs:
                _training_jobs[job_id].update(
                    {"progress": progress, "message": message, "status": status}
                )

    try:
        # Phase 1: sync from S3
        _set(0, "Syncing from S3...")
        _sync_from_s3(manual)

        # Phase 2: clear old state, build config + pipeline object
        _set(10, "Clearing previous pipeline state...")
        _clear_rag_state(manual.manual_id)
        cfg = _build_rag_config(manual)
        rag_instance = MultimodalRAGPipeline(cfg)

        # Phase 3: build_metadata (blocking, potentially long)
        # Ticker thread simulates incremental progress 15→72% during extraction
        _set(15, "Extracting text and images from PDFs...")
        _stop_ticker = threading.Event()

        def _ticker():
            step = 0
            max_steps = 57  # 15→72 over ~171s (3s × 57)
            while not _stop_ticker.is_set() and step < max_steps:
                time.sleep(3)
                step += 1
                with _training_jobs_lock:
                    if job_id in _training_jobs:
                        cur = _training_jobs[job_id].get("progress", 15)
                        if cur < 72:
                            _training_jobs[job_id]["progress"] = min(72, 15 + step)

        ticker = threading.Thread(target=_ticker, daemon=True)
        ticker.start()
        try:
            rag_instance.build_metadata(
                pdf_folder_path=manual.pdf_folder,
                cache_dir=manual.cache_dir,
                force_rebuild=True,
                generation_config=GenerationConfig(temperature=TEMPERATURE),
                ocr_fallback=True,
                image_save_dir=manual.image_dir,
                skip_existing_images=False,
            )
        finally:
            _stop_ticker.set()

        # Phase 4: upload images to S3
        _set(75, "Uploading extracted images to S3...")
        n_imgs = _upload_images_to_s3_and_rewrite(rag_instance, manual)

        # Phase 5: sync cache to S3
        _set(85, "Syncing cache to S3...")
        counts = _sync_to_s3(manual)
        counts["images_to_s3"] = n_imgs
        upload_manual_registry_to_s3(str(_MANUALS_JSON_PATH))

        # Phase 6: remap paths + cache pipeline
        _set(95, "Finalizing pipeline...")
        _remap_image_paths(rag_instance, manual)
        _remember_pipeline(manual.manual_id, rag_instance)

        _set(100, "Training complete!", status="done")
    except Exception as exc:
        with _training_jobs_lock:
            if job_id in _training_jobs:
                _training_jobs[job_id].update({"status": "error", "message": str(exc)})


# -----------------------------
# Helpers
# -----------------------------
def _safe_image_url(img_path: str, manual_id: Optional[str] = None) -> str:
    """
    Convert an absolute/relative img_path to a URL under /static/<manual_id>/...

    If img_path is already an S3/HTTP URL, return it directly.
    Handles caches built in older layouts by falling back to basename lookup
    inside the manual's image dir.
    """
    if img_path.startswith("http"):
        return img_path
    manual = _resolve_manual(manual_id)
    p = Path(img_path)
    root = Path(manual.image_dir).resolve()

    rel: Path
    try:
        rel = p.resolve().relative_to(root)
    except Exception:
        rel = Path(p.name)

    return f"/static/{manual.manual_id}/{rel.as_posix()}"


def _normalize_image_matches(
    image_matches: Dict[Any, Dict[str, Any]],
    manual_id: Optional[str] = None,
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
                "img_url": _safe_image_url(str(img_path), manual_id),
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
        generation_config=GenerationConfig(
            temperature=TEMPERATURE, max_output_tokens=MAX_QUERY_TOKENS
        ),
    )
    return (out or "").strip()


def _english_answer_to_myanmar(rag: MultimodalRAGPipeline, english_answer: str) -> str:
    """Translate/summarize the English RAG answer into Myanmar for the user."""
    instruction = (
        "Translate the following English technical answer into natural Myanmar (Burmese). "
        "Preserve the full meaning; keep technical terms accurate (use common Roman abbreviations for parts where helpful). "
        "IMPORTANT: Preserve any [Image X] citation markers (e.g., [Image 1], [Image 2]) EXACTLY as written — do not translate or remove them. "
        "Output only Myanmar (Burmese) script text with the preserved [Image X] markers, with no English preamble or labels.\n\n"
        f"English answer:\n{english_answer}\n\nMyanmar answer:"
    )
    out = get_gemini_response(
        rag.text_model,
        model_input=instruction,
        stream=False,
        generation_config=GenerationConfig(
            temperature=TEMPERATURE, max_output_tokens=MAX_OUTPUT_TOKENS
        ),
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
        generation_config=GenerationConfig(
            temperature=TEMPERATURE, max_output_tokens=MAX_QUERY_TOKENS
        ),
    )
    return (out or "").strip()


def _english_answer_to_japanese(rag: MultimodalRAGPipeline, english_answer: str) -> str:
    """Translate/summarize the English RAG answer into Japanese for the user."""
    instruction = (
        "Translate the following English technical answer into natural Japanese. "
        "Preserve the full meaning; keep technical terms accurate. "
        "IMPORTANT: Preserve any [Image X] citation markers (e.g., [Image 1], [Image 2]) EXACTLY as written — do not translate or remove them. "
        "Output only Japanese text with the preserved [Image X] markers, with no English preamble or labels.\n\n"
        f"English answer:\n{english_answer}\n\nJapanese answer:"
    )
    out = get_gemini_response(
        rag.text_model,
        model_input=instruction,
        stream=False,
        generation_config=GenerationConfig(
            temperature=TEMPERATURE, max_output_tokens=MAX_OUTPUT_TOKENS
        ),
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
    manual_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    texts: List[TextChunk]
    images: List[ImageMatch]
    manual_id: Optional[str] = None
    retrieval_expanded: bool = False
    retrieval_message: Optional[str] = None


class MyanmarQueryRequest(BaseModel):
    """Myanmar (Burmese) question; rewritten to English internally for retrieval."""

    question: str
    top_k_text: int = 5
    top_k_img: int = 6
    temp: float = 0.5
    include_intermediate_english: bool = False
    manual_id: Optional[str] = None


class MyanmarQueryResponse(BaseModel):
    answer: str
    texts: List[TextChunk]
    images: List[ImageMatch]
    english_query: Optional[str] = None
    english_answer: Optional[str] = None
    manual_id: Optional[str] = None
    retrieval_expanded: bool = False
    retrieval_message: Optional[str] = None


class GenerateQuestionRequest(BaseModel):
    model_name: str
    language: str
    count: int = 1
    manual_id: Optional[str] = None


class JapaneseQueryRequest(BaseModel):
    question: str
    top_k_text: int = 5
    top_k_img: int = 6
    temp: float = 0.5
    include_intermediate_english: bool = False
    manual_id: Optional[str] = None


class JapaneseQueryResponse(BaseModel):
    answer: str
    texts: List[TextChunk]
    images: List[ImageMatch]
    english_query: Optional[str] = None
    english_answer: Optional[str] = None
    manual_id: Optional[str] = None
    retrieval_expanded: bool = False
    retrieval_message: Optional[str] = None


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
    manual_id: Optional[str] = None


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
    manual_id: Optional[str] = None


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
    manual_id: Optional[str] = None


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
    manual_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    history: List[ChatMessage]
    texts: List[TextChunk]
    images: List[ImageMatch]
    manual_id: Optional[str] = None
    retrieval_expanded: bool = False


class ManualInfo(BaseModel):
    manual_id: str
    display_name: str
    description: str = ""
    is_default: bool = False
    has_cache: bool = False
    pdf_count: int = 0


class ManualListResponse(BaseModel):
    default_manual_id: str
    manuals: List[ManualInfo]


# -----------------------------
# Template render helper
# -----------------------------
def _render_page(**kwargs: Any) -> str:
    defaults = {
        "manuals": manual_registry.list(),
        "selected_manual_id": DEFAULT_MANUAL_ID,
    }
    merged = {**defaults, **kwargs}
    tpl = jinja_env.get_template("index.html")
    return tpl.render(**merged)


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


@app.get("/api/manuals", response_model=ManualListResponse)
def api_list_manuals():
    """List available manuals, including which has a built cache locally."""
    items: List[ManualInfo] = []
    for m in manual_registry.list():
        cache_pkl = Path(m.cache_dir) / "text_metadata_df.pkl"
        pdf_dir = Path(m.pdf_folder)
        pdf_count = sum(1 for _ in pdf_dir.glob("*.pdf")) if pdf_dir.exists() else 0
        items.append(
            ManualInfo(
                manual_id=m.manual_id,
                display_name=m.display_name,
                description=m.description,
                is_default=(m.manual_id == DEFAULT_MANUAL_ID),
                has_cache=cache_pkl.exists(),
                pdf_count=pdf_count,
            )
        )
    return ManualListResponse(default_manual_id=DEFAULT_MANUAL_ID, manuals=items)


@app.post("/api/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    manual_id: Optional[str] = Form(None),
):
    """Upload a PDF: save under the chosen manual's PDF folder and to S3."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            {"ok": False, "error": "Only PDF files allowed"},
            status_code=400,
        )
    try:
        manual = _resolve_manual(manual_id)
    except RuntimeError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    Path(manual.pdf_folder).mkdir(parents=True, exist_ok=True)
    dest = Path(manual.pdf_folder) / file.filename
    try:
        content = await file.read()
        dest.write_bytes(content)
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": str(e)},
            status_code=500,
        )
    if is_s3_configured():
        upload_manual_pdf_file_to_s3(manual.manual_id, str(dest))
    return JSONResponse(
        {"ok": True, "filename": file.filename, "manual_id": manual.manual_id}
    )


@app.post("/api/clean-cache")
def api_clean_cache(manual_id: Optional[str] = None):
    """Delete local cache for one manual and reset its in-memory pipeline."""
    try:
        manual = _resolve_manual(manual_id)
    except RuntimeError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    cache_path = Path(manual.cache_dir)
    if cache_path.exists():
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    Path(manual.image_dir).mkdir(parents=True, exist_ok=True)
    _clear_rag_state(manual.manual_id)
    return JSONResponse(
        {
            "ok": True,
            "manual_id": manual.manual_id,
            "message": f"Local cache for {manual.manual_id} cleared. "
            "Run a query or Build cache to rebuild.",
        }
    )


@app.post("/api/build-cache")
def api_build_cache(
    manual_id: Optional[str] = None, skip_existing_images: bool = False
):
    """Force rebuild metadata for one manual from PDFs and sync cache + PDFs to S3.

    skip_existing_images=true: skip image extraction for PDFs that already have
    images in the cache (local or S3 URLs). Existing image rows are merged from
    the previous cache and re-uploaded to S3 if still local.
    """
    try:
        manual = _resolve_manual(manual_id)
    except RuntimeError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    _clear_rag_state(manual.manual_id)
    try:
        _sync_from_s3(manual)
        cfg = _build_rag_config(manual)
        rag_instance = MultimodalRAGPipeline(cfg)
        rag_instance.build_metadata(
            pdf_folder_path=manual.pdf_folder,
            cache_dir=manual.cache_dir,
            force_rebuild=True,
            generation_config=GenerationConfig(temperature=TEMPERATURE),
            ocr_fallback=True,
            image_save_dir=manual.image_dir,
            skip_existing_images=skip_existing_images,
        )
        n_imgs = _upload_images_to_s3_and_rewrite(rag_instance, manual)
        counts = _sync_to_s3(manual)
        counts["images_to_s3"] = n_imgs
        _remap_image_paths(rag_instance, manual)
        _remember_pipeline(manual.manual_id, rag_instance)
        return JSONResponse(
            {
                "ok": True,
                "manual_id": manual.manual_id,
                "message": f"Cache for {manual.manual_id} rebuilt and synced to S3.",
                "s3_uploaded": counts,
            }
        )
    except Exception as e:
        return JSONResponse(
            {"ok": False, "manual_id": manual.manual_id, "error": str(e)},
            status_code=500,
        )


@app.post("/api/sync-to-s3")
def api_sync_to_s3(manual_id: Optional[str] = None):
    """Upload one manual's local cache and PDFs to S3."""
    try:
        manual = _resolve_manual(manual_id)
    except RuntimeError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    if not is_s3_configured():
        return JSONResponse(
            {"ok": False, "error": "S3 not configured. Set AWS_* and S3_BUCKET_NAME."},
            status_code=400,
        )
    # Bypass DISABLE_S3_SYNC — this is an explicit user-triggered upload.
    n_cache = upload_manual_cache_to_s3(
        manual.manual_id, _resolve_manual_path(manual.cache_dir)
    )
    n_pdfs = upload_manual_pdfs_to_s3(
        manual.manual_id, _resolve_manual_path(manual.pdf_folder)
    )
    upload_manual_registry_to_s3(str(_MANUALS_JSON_PATH))
    return JSONResponse(
        {
            "ok": True,
            "manual_id": manual.manual_id,
            "message": f"Uploaded {n_cache} cache file(s), "
            f"{n_pdfs} PDF(s) to S3 for {manual.manual_id}.",
            "uploaded": {"cache": n_cache, "pdfs": n_pdfs},
        }
    )


@app.post("/api/pull-from-s3")
def api_pull_from_s3(manual_id: Optional[str] = None):
    """Download one manual's cache and PDFs from S3 into local dirs."""
    try:
        manual = _resolve_manual(manual_id)
    except RuntimeError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    if not is_s3_configured():
        return JSONResponse(
            {"ok": False, "error": "S3 not configured. Set AWS_* and S3_BUCKET_NAME."},
            status_code=400,
        )
    _sync_from_s3(manual)
    _clear_rag_state(manual.manual_id)
    return JSONResponse(
        {
            "ok": True,
            "manual_id": manual.manual_id,
            "message": f"Pulled cache and PDFs from S3 for {manual.manual_id}. RAG state cleared.",
        }
    )


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
        manual = _resolve_manual(payload.manual_id)
        rag = _get_rag(manual.manual_id)
    except RuntimeError as e:
        return JSONResponse(
            {
                "detail": f"RAG not available: {e}. "
                "Check PROJECT_ID / LOCATION / GOOGLE_APPLICATION_CREDENTIALS and data/cache.",
            },
            status_code=503,
        )

    out = rag.answer_multimodal_query(
        payload.question,
        top_n_text=payload.top_k_text,
        top_n_images=payload.top_k_img,
        temperature=payload.temp,
        stream=False,
        include_step_by_step=False,
        answer_language=payload.answer_language,
        manual_label=manual.display_name,
        available_models=_available_model_labels(),
    )
    # Use the matches from whichever retrieval pass actually answered (may be the
    # widened fallback pass) so displayed sources match the answer.
    text_matches = out["text_matches"]
    image_matches = out["image_matches"]
    answer = out["response"]
    if not isinstance(answer, str):
        answer = str(answer)

    # Only return images actually cited; renumber citations to match filtered order
    image_matches, answer = _filter_images_by_citations(image_matches, answer)

    texts_norm = _normalize_text_matches(text_matches)
    images_norm = _normalize_image_matches(image_matches, manual.manual_id)

    retrieval_expanded = bool(out.get("retrieval_expanded"))
    return QueryResponse(
        answer=answer,
        texts=[TextChunk(**t) for t in texts_norm],
        images=[ImageMatch(**img) for img in images_norm],
        manual_id=manual.manual_id,
        retrieval_expanded=retrieval_expanded,
        retrieval_message=RETRIEVAL_EXPANDED_MESSAGE if retrieval_expanded else None,
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
        manual = _resolve_manual(payload.manual_id)
        rag = _get_rag(manual.manual_id)
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
            {
                "detail": "Failed to rewrite question to English. Try again or check Vertex AI / Gemini."
            },
            status_code=503,
        )

    out = rag.answer_multimodal_query(
        english_query,
        top_n_text=payload.top_k_text,
        top_n_images=payload.top_k_img,
        temperature=payload.temp,
        stream=False,
        include_step_by_step=False,
        answer_language="en",
        manual_label=manual.display_name,
        available_models=_available_model_labels(),
    )
    text_matches = out["text_matches"]
    image_matches = out["image_matches"]
    english_answer = out["response"]
    if not isinstance(english_answer, str):
        english_answer = str(english_answer)

    if not english_answer.strip() or english_answer.strip() == "Exception occurred":
        return JSONResponse(
            {
                "detail": "RAG answer generation failed. Try again or check Vertex AI / Gemini."
            },
            status_code=503,
        )

    # Filter images to only those cited in the English answer, then renumber
    image_matches, renumbered_english = _filter_images_by_citations(
        image_matches, english_answer
    )

    myanmar_answer = _english_answer_to_myanmar(rag, renumbered_english)
    if not myanmar_answer or myanmar_answer == "Exception occurred":
        return JSONResponse(
            {
                "detail": "Failed to translate answer to Myanmar. Try again or check Vertex AI / Gemini."
            },
            status_code=503,
        )

    texts_norm = _normalize_text_matches(text_matches)
    images_norm = _normalize_image_matches(image_matches, manual.manual_id)

    retrieval_expanded = bool(out.get("retrieval_expanded"))
    return MyanmarQueryResponse(
        answer=myanmar_answer,
        texts=[TextChunk(**t) for t in texts_norm],
        images=[ImageMatch(**img) for img in images_norm],
        english_query=english_query if payload.include_intermediate_english else None,
        english_answer=(
            renumbered_english if payload.include_intermediate_english else None
        ),
        manual_id=manual.manual_id,
        retrieval_expanded=retrieval_expanded,
        retrieval_message=RETRIEVAL_EXPANDED_MESSAGE if retrieval_expanded else None,
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
        manual = _resolve_manual(payload.manual_id)
        rag = _get_rag(manual.manual_id)
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
            {
                "detail": "Failed to rewrite question to English. Try again or check Vertex AI / Gemini."
            },
            status_code=503,
        )

    out = rag.answer_multimodal_query(
        english_query,
        top_n_text=payload.top_k_text,
        top_n_images=payload.top_k_img,
        temperature=payload.temp,
        stream=False,
        include_step_by_step=False,
        answer_language="en",
        manual_label=manual.display_name,
        available_models=_available_model_labels(),
    )
    text_matches = out["text_matches"]
    image_matches = out["image_matches"]
    english_answer = out["response"]
    if not isinstance(english_answer, str):
        english_answer = str(english_answer)

    if not english_answer.strip() or english_answer.strip() == "Exception occurred":
        return JSONResponse(
            {
                "detail": "RAG answer generation failed. Try again or check Vertex AI / Gemini."
            },
            status_code=503,
        )

    # Filter images to only those cited in the English answer, then renumber
    image_matches, renumbered_english = _filter_images_by_citations(
        image_matches, english_answer
    )

    japanese_answer = _english_answer_to_japanese(rag, renumbered_english)
    if not japanese_answer or japanese_answer == "Exception occurred":
        return JSONResponse(
            {
                "detail": "Failed to translate answer to Japanese. Try again or check Vertex AI / Gemini."
            },
            status_code=503,
        )

    texts_norm = _normalize_text_matches(text_matches)
    images_norm = _normalize_image_matches(image_matches, manual.manual_id)

    retrieval_expanded = bool(out.get("retrieval_expanded"))
    return JapaneseQueryResponse(
        answer=japanese_answer,
        texts=[TextChunk(**t) for t in texts_norm],
        images=[ImageMatch(**img) for img in images_norm],
        english_query=english_query if payload.include_intermediate_english else None,
        english_answer=(
            renumbered_english if payload.include_intermediate_english else None
        ),
        manual_id=manual.manual_id,
        retrieval_expanded=retrieval_expanded,
        retrieval_message=RETRIEVAL_EXPANDED_MESSAGE if retrieval_expanded else None,
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
        manual = _resolve_manual(payload.manual_id)
        rag = _get_rag(manual.manual_id)
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
    images_norm = _normalize_image_matches(image_matches, manual.manual_id)

    return QueryResponse(
        answer=answer,
        texts=[TextChunk(**t) for t in texts_norm],
        images=[ImageMatch(**img) for img in images_norm],
        manual_id=manual.manual_id,
    )


@app.post("/api/query-upload", response_model=QueryResponse)
async def api_query_upload(
    question: str = Form(...),
    top_k_text: int = Form(5),
    top_k_img: int = Form(1),
    temp: float = Form(0.2),
    answer_language: str = Form("auto"),
    image: Optional[UploadFile] = File(None),
    manual_id: Optional[str] = Form(None),
):
    """
    Multipart endpoint: question + optional uploaded image.
    If an image is provided, image retrieval uses image-embedding search.
    Answer is generated from text-context (same behavior as /api/query-with-image).
    """
    try:
        manual = _resolve_manual(manual_id)
        rag = _get_rag(manual.manual_id)
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
        qimg_dir = Path(manual.cache_dir) / "query_images"
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
    images_norm = _normalize_image_matches(image_matches, manual.manual_id)

    return QueryResponse(
        answer=answer,
        texts=[TextChunk(**t) for t in texts_norm],
        images=[ImageMatch(**img) for img in images_norm],
        manual_id=manual.manual_id,
    )


@app.post("/api/v1/diagnose", response_model=DiagnosticAPIResponse)
def api_v1_diagnose(payload: DiagnosticPayload):
    """
    v1 diagnostic endpoint.
    Returns structured JSON used by the /v1 dashboard UI.
    """
    try:
        manual = _resolve_manual(payload.manual_id)
        rag = _get_rag(manual.manual_id)
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

    images_norm = _normalize_image_matches(image_matches, manual.manual_id)

    return DiagnosticAPIResponse(
        question=payload.question,
        diagnostic=envelope,
        images=[ImageMatch(**img) for img in images_norm],
        manual_id=manual.manual_id,
    )


@app.post("/api/v1/diagnose-upload", response_model=DiagnosticAPIResponse)
async def api_v1_diagnose_upload(
    question: str = Form(...),
    top_k_text: int = Form(10),
    top_k_img: int = Form(6),
    temp: float = Form(0.4),
    answer_language: str = Form("auto"),
    image: Optional[UploadFile] = File(None),
    manual_id: Optional[str] = Form(None),
):
    """
    Multipart variant of v1 diagnostic endpoint that accepts an optional image.
    The image is used for image-embedding-based retrieval; the diagnostic text
    summary is still based on text + image descriptions.
    """
    try:
        manual = _resolve_manual(manual_id)
        rag = _get_rag(manual.manual_id)
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
        qimg_dir = Path(manual.cache_dir) / "query_images"
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

    images_norm = _normalize_image_matches(image_matches, manual.manual_id)

    return DiagnosticAPIResponse(
        question=question,
        diagnostic=envelope,
        images=[ImageMatch(**img) for img in images_norm],
        manual_id=manual.manual_id,
    )


@app.get("/", response_class=HTMLResponse)
def home(manual_id: Optional[str] = None):
    manual = _resolve_manual(manual_id)
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
        selected_manual_id=manual.manual_id,
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
    manual_id: Optional[str] = Form(None),
):
    try:
        manual = _resolve_manual(manual_id)
        rag = _get_rag(manual.manual_id)
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
            selected_manual_id=manual_id or DEFAULT_MANUAL_ID,
        )
        return HTMLResponse(html)

    # 1) Retrieve text (always)
    text_matches = rag.search_text(q, top_n=top_k_text, chunk_text=True)

    # 2) Images: if an image is uploaded, use image-embedding retrieval
    if image and image.filename:
        qimg_dir = Path(manual.cache_dir) / "query_images"
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
        # Only show images actually cited; renumber citations to match filtered order
        image_matches, answer = _filter_images_by_citations(image_matches, answer)

    html = _render_page(
        ran=True,
        q=q,
        top_k_text=top_k_text,
        top_k_img=top_k_img,
        temp=temp,
        answer_language=answer_language,
        answer=answer,
        texts=_normalize_text_matches(text_matches),
        images=_normalize_image_matches(image_matches, manual.manual_id),
        selected_manual_id=manual.manual_id,
    )
    return HTMLResponse(html)


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    path = TEMPLATES_DIR / "chat.html"
    if not path.exists():
        return HTMLResponse("<p>chat.html not found</p>", status_code=404)
    return FileResponse(path, media_type="text/html")


def _condense_conversational_query(
    rag, question: str, history: List[ChatMessage]
) -> str:
    """Rewrite follow-up question to a standalone search query containing context."""
    if not history:
        # Detect Japanese characters (Hiragana, Katakana, Kanji)
        if re.search(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]", question):
            return _rewrite_japanese_to_english_query(rag, question)
        # Detect Myanmar characters
        if re.search(r"[\u1000-\u109f]", question):
            return _rewrite_myanmar_to_english_query(rag, question)
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
        generation_config=GenerationConfig(
            temperature=TEMPERATURE, max_output_tokens=MAX_QUERY_TOKENS
        ),
    )
    res = (out or question).strip()
    # Check if the condensed query still contains Japanese or Myanmar characters due to model following errors
    if re.search(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]", res):
        res = _rewrite_japanese_to_english_query(rag, res)
    elif re.search(r"[\u1000-\u109f]", res):
        res = _rewrite_myanmar_to_english_query(rag, res)
    return res


def _needs_retrieval(rag, condensed_query: str, history: List[ChatMessage]) -> bool:
    """Return False when conversation history already contains enough context to answer."""
    if not history:
        return True
    history_str = "\n".join(
        f"{'Farmer' if m.role == 'user' else 'Assistant'}: {m.content}"
        for m in history[-4:]
    )
    prompt = (
        "Given this conversation history, determine if answering the new question "
        "requires looking up NEW information from a technical manual database, "
        "or if the conversation history already contains enough context.\n\n"
        f"Conversation History:\n{history_str}\n\n"
        f"New Question: {condensed_query}\n\n"
        "Answer with ONLY 'yes' (need manual lookup) or 'no' (history is sufficient):"
    )
    out = get_gemini_response(
        rag.text_model,
        model_input=prompt,
        stream=False,
        generation_config=GenerationConfig(
            temperature=0.0, max_output_tokens=MAX_CLASSIFY_TOKENS
        ),
    )
    return "yes" in (out or "yes").strip().lower()


def _filter_images_by_citations(image_matches: dict, answer: str):
    """Return (image_matches, renumbered_answer).

    Keeps all images (does not filter them out), but still rewrites citations
    in the answer text to be sequential based on the cited images.
    """
    cited_positions = {
        int(i) - 1 for i in re.findall(r"\[image[:#\s]*(\d+)\]", answer, re.IGNORECASE)
    }
    if not cited_positions:
        return image_matches, answer

    # Map original 1-based index → new sequential 1-based index
    orig_to_new: dict = {}
    new_idx = 1
    for i in range(len(image_matches)):
        if i in cited_positions:
            orig_to_new[i + 1] = new_idx
            new_idx += 1

    def _replace(m):
        return f"[Image {orig_to_new.get(int(m.group(1)), int(m.group(1)))}]"

    renumbered_answer = re.sub(
        r"\[image[:#\s]*(\d+)\]", _replace, answer, flags=re.IGNORECASE
    )
    return image_matches, renumbered_answer


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(payload: ChatRequest):
    """Multi-turn conversational RAG backend endpoint."""
    try:
        manual = _resolve_manual(payload.manual_id)
        rag = _get_rag(manual.manual_id)
    except RuntimeError as e:
        return JSONResponse({"detail": f"RAG not ready: {e}"}, status_code=503)

    session_id = payload.session_id or str(uuid.uuid4())

    # 1. Condense/Rewrite follow-up query using history
    search_query = _condense_conversational_query(
        rag, payload.question, payload.history
    )

    # 2. Retrieve only when history doesn't already cover the question
    if _needs_retrieval(rag, search_query, payload.history):
        text_matches = rag.search_text(
            search_query, top_n=payload.top_k_text, chunk_text=True
        )
        image_matches = rag.search_images_by_description_text(
            search_query, top_n=payload.top_k_img
        )
    else:
        text_matches = {}
        image_matches = {}

    # 3. Format conversational prompt context
    context_str = ""
    for idx, t in enumerate(text_matches.values()):
        context_str += f"Manual Clip [{idx+1}]:\n{t.get('chunk_text', '')}\n\n"

    needs_retrieval = _needs_retrieval(rag, search_query, payload.history)

    history_str = ""
    for msg in payload.history:
        role_label = "Farmer" if msg.role == "user" else "Tractor Assistant"
        history_str += f"{role_label}: {msg.content}\n"

    lang = (payload.answer_language or "auto").strip().lower()
    if lang in ("en", "english"):
        lang_instruction = "You MUST write your entire response in English."
    elif lang in ("my", "mm", "myanmar", "burmese"):
        lang_instruction = "You MUST write your entire response in Myanmar (Burmese)."
    elif lang in ("ja", "jp", "japanese"):
        lang_instruction = "You MUST write your entire response in Japanese."
    else:
        lang_instruction = "Write your response in the language of the farmer's latest query (e.g. English, Myanmar, or Japanese)."

    available_models = _available_model_labels()
    available_line = (
        f"Manuals/models available in this system: {', '.join(available_models)}.\n"
        if available_models
        else ""
    )

    def _run_chat_pass(tk_text: int, tk_img: int):
        if needs_retrieval:
            tm = rag.search_text(search_query, top_n=tk_text, chunk_text=True)
            im = rag.search_images_by_description_text(search_query, top_n=tk_img)
        else:
            tm, im = {}, {}

        context_str = ""
        for idx, t in enumerate(tm.values()):
            context_str += f"Manual Clip [{idx+1}]:\n{t.get('chunk_text', '')}\n\n"

        context_images_str = ""
        for idx, img in enumerate(im.values()):
            caption = img.get("image_description") or img.get("img_desc") or ""
            context_images_str += f"Image {idx+1}:\nCaption: {caption}\n\n"

        system_prompt = (
            "You are an empathetic, expert tractor technician and farmer's advisor.\n"
            f"You are advising specifically on: {manual.display_name}. "
            f"{available_line}"
            "You can only help with the currently selected manual; you do not have access to other manuals unless the user switches to them.\n"
            "Your goal is to guide the farmer safely and step-by-step through their troubleshooting scenario.\n\n"
            "Guidelines:\n"
            "1. Keep answers concise, extremely practical, and structured as steps or simple recommendations.\n"
            "2. Keep a friendly, helpful tone to support the farmer or mechanic.\n"
            "3. Safety guidance should be relevant and proportional. Do not lead every routine answer with generic safety warnings. If the answer involves a severe or specific hazard (fuel vapor, fire, high pressure fluid, jacking/lifting, blades, electrical shock, or similar), keep the procedure safe and add a short final section titled 'Safety note:' with only the specific hazards that apply. For dangerous or high-risk repair work, advise a qualified technician or dealer instead of giving risky instructions.\n"
            "4. SCOPE: If the farmer asks about a different tractor/machine model than the selected manual, do NOT say information is missing. Instead, tell them this chat only covers the selected manual; if the model they want is in the available list above, ask them to switch the manual selector to it; otherwise say that model is currently unavailable.\n"
            "5. Only use instructions from the provided Operation Manual Clips or details in the Retrieved Images. Do not invent or pad procedures just to fill space.\n"
            "6. Format procedures as numbered steps when order matters. Use short bullet lists for unordered checks. Keep paragraphs short.\n"
            "7. CRITICAL IMAGE CITATION RULE: If you use or refer to details, instructions, or visuals from a retrieved image, you MUST cite it inline using [Image X] where X is the image index (e.g. [Image 1]). Only cite an image if it is relevant.\n"
            "8. CRITICAL MANUAL CLIP CITATION RULE: Do NOT EVER cite, print, or reference any '[Manual Clip X]' or 'Manual Clip' indexes/tags. Speak naturally, using the clip text silently as background knowledge.\n"
            f"9. INSUFFICIENT INFO: If the manual clips do not contain enough actionable information to genuinely help (and the question IS about the selected machine), begin your reply with the token {INSUFFICIENT_CONTEXT_SENTINEL} on its own first line, then say what is missing and advise contacting a local dealer. Include only specific safety notes when relevant. Do NOT use this token for questions about other machine models (handle those per guideline 4).\n"
            f"10. LANGUAGE RULE: {lang_instruction}\n\n"
            f"Operation Manual Clips:\n{context_str}\n"
            f"Retrieved Images Context:\n{context_images_str}\n"
            f"Conversation History:\n{history_str}"
            f"Farmer's Latest Query: {payload.question}\n\n"
            "Tractor Assistant Response:"
        )

        out = get_gemini_response(
            rag.text_model,
            model_input=system_prompt,
            stream=False,
            generation_config=GenerationConfig(
                temperature=payload.temp, max_output_tokens=MAX_OUTPUT_TOKENS
            ),
        )
        return tm, im, (out or "").strip()

    # 2-4. First pass, with a widened dealer-fallback pass if context was too thin.
    text_matches, image_matches, answer = _run_chat_pass(
        payload.top_k_text, payload.top_k_img
    )
    retrieval_expanded = False
    if needs_retrieval and has_insufficient_marker(answer):
        exp_text = min(payload.top_k_text * 2, 40)
        exp_img = min(payload.top_k_img * 2, 24)
        if exp_text > payload.top_k_text or exp_img > payload.top_k_img:
            retrieval_expanded = True
            text_matches, image_matches, answer = _run_chat_pass(exp_text, exp_img)

    answer = strip_insufficient_marker(answer)

    # Only return images actually cited; renumber citations to match filtered order
    image_matches, answer = _filter_images_by_citations(image_matches, answer)

    # Normalize responses
    texts_norm = _normalize_text_matches(text_matches)
    images_norm = _normalize_image_matches(image_matches, manual.manual_id)

    # Update history list
    new_history = list(payload.history)
    new_history.append(ChatMessage(role="user", content=payload.question))
    new_history.append(ChatMessage(role="model", content=answer))

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        history=new_history,
        texts=[TextChunk(**t) for t in texts_norm],
        images=[ImageMatch(**img) for img in images_norm],
        manual_id=manual.manual_id,
        retrieval_expanded=retrieval_expanded,
    )


@app.get("/debug-generator", response_class=HTMLResponse)
def debug_generator_page():
    path = TEMPLATES_DIR / "generator.html"
    if not path.exists():
        return HTMLResponse("<p>generator.html not found</p>", status_code=404)
    return FileResponse(path, media_type="text/html")


@app.get("/api/models")
def get_available_models(manual_id: Optional[str] = None):
    try:
        manual = _resolve_manual(manual_id)
        rag = _get_rag(manual.manual_id)
        df = rag.text_metadata_df
        if df is not None and not df.empty:
            doc_col = next(
                (c for c in ["file_name", "doc_name", "source"] if c in df.columns),
                None,
            )
            if doc_col:
                files = list(df[doc_col].dropna().unique())
                models = [
                    {"id": f, "name": os.path.splitext(f)[0].upper()} for f in files
                ]
                return {"manual_id": manual.manual_id, "models": models}
    except Exception as e:
        print(f"Error fetching models from cache: {e}")
    # Fallback: scan PDF folder for actual files
    try:
        resolved = _resolve_manual(manual_id)
        pdf_files = sorted(Path(resolved.pdf_folder).glob("*.pdf"))
        if pdf_files:
            return {
                "manual_id": resolved.manual_id,
                "models": [{"id": f.name, "name": f.stem.upper()} for f in pdf_files],
            }
    except Exception:
        pass
    return {
        "manual_id": (manual_id or DEFAULT_MANUAL_ID),
        "models": [],
    }


@app.post("/api/generate-random-question")
def api_generate_random_question(payload: GenerateQuestionRequest):
    import datetime
    import random

    try:
        manual = _resolve_manual(payload.manual_id)
        rag = _get_rag(manual.manual_id)
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
        matched_df = df[
            df["file_name"].str.lower().str.contains(payload.model_name.lower())
        ]

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
    clean_indices = [
        i
        for i, row in matched_df.iterrows()
        if is_clean_chunk(row.get("chunk_text") or row.get("text"))
    ]

    if not clean_indices:
        return JSONResponse(
            {
                "error": "No clean/descriptive text chunks found for model question generation."
            },
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
        for offset in range(10):
            candidate_idx = start_idx + offset
            if candidate_idx >= len(matched_df):
                break
            candidate_row = matched_df.iloc[candidate_idx]

            # Check if same document
            if candidate_row["file_name"] != matched_df.iloc[start_idx]["file_name"]:
                break

            candidate_text = (
                candidate_row.get("chunk_text") or candidate_row.get("text") or ""
            )
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
            p = int(c.get("page_num") or 0)
            chnk = int(c.get("chunk_number") or 0)
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
                "   - Tractor Questions:\n"
                "     - The starter motor won't turn when I try to start the tractor. What should I do?\n"
                "\n"
                "     - The starter motor turns, but the engine won't start. How can I fix this?\n"
                "\n"
                "     - As soon as I turn the key, I hear the solenoid clicking/dropping out and the engine won't start. What should I do?\n"
                "\n"
                "     - The battery warning light is showing on the dashboard. How do I fix this?\n"
                "\n"
                "     - The engine oil warning light is on. What should I do?\n"
                "\n"
                "     - The engine overheat light is on. How do I fix this?\n"
                "\n"
                "     - The engine starts, but it stalls out after running for about 5 minutes. What is causing this and how do I fix it?\n"
                "\n"
                "     - The RPM gauge on the dashboard is not moving or working. How do I fix this?\n"
                "\n"
                "     - When I turn the key on, no lights appear on the dashboard at all. What should I do?\n"
                "\n"
                "     - When I drive the tractor in reverse gear, it keeps popping out of gear. How can I fix this?\n"
                "\n"
                "     - I cannot engage the 4-wheel drive (4WD) gear. What should I do?\n"
                "\n"
                "     - The steering wheel is very heavy and difficult to turn. How can I fix this?\n"
                "\n"
                "     - The front loader bucket keeps dropping down on its own. What is causing this and how do I fix it?\n"
                "\n"
                "     - The plow lifting and lowering mechanism feels very heavy and sluggish. What should I do?\n"
                "\n"
                "     - Smoke is coming out of the gearbox and it is getting extremely hot. What is wrong and what should I do?\n"
                "\n"
                "     - The PTO is slipping and cannot handle the load. How can I fix this?\n"
                "\n"
                "     - I found white foam on the gear oil cap. What does this mean and what should I do?\n"
                "\n"
                "     - The plow shakes and bounces when lifting or lowering. How can I fix this?\n"
                "\n"
                "     - When I turn off the engine with the plow lifted, it immediately drops back down to the ground. How do I fix this?\n"
                "\n"
                "     - There is a loud noise coming from the clutch housing whenever I shift between forward and reverse gears. What is causing this?\n"
                "\n"
                "     - Even when I pull the clutch lever while the PTO gear is engaged, the rotary tiller blades at the back keep spinning. How do I fix this?\n"
                "\n"
                "     - Why do the drive chains keep breaking frequently when operating a 6 or 7-disc plow?\n"
                "\n"
                "     - My engine oil level keeps dropping, and I have to top up every day. What is causing this?\n"
                "\n"
                "     - Why do the kingpin seals and wheel flange seals keep blowing out and leaking oil so frequently?\n"
                "\n"
                "     - I only use premium diesel, so why is water constantly getting trapped inside the paper fuel filter bowl?\n"
                "\n"
                "   - Rice Combine Harvester Questions:\n"
                "     - When I turn the key, I can only hear the fuel pump clicking, but no lights appear on the dashboard. What should I do?\n"
                "\n"
                "     - The Stop light is illuminated and the engine won't start. How can I fix this?\n"
                "\n"
                "     - As soon as I engage the threshing drum lever, the engine stalls completely. What should I do?\n"
                "\n"
                "     - The grain unload auger spout cannot rotate left or right. How can I fix this?\n"
                "\n"
                "     - The front cutting header cannot be lifted or lowered. What should I do?\n"
                "\n"
                "     - The frame height adjustment works on one side but won't lift or lower on the other side. How can I fix this?\n"
                "\n"
                "     - The front dust suction fan has stopped spinning. What should I do?\n"
                "\n"
                "     - I have to replace the water pump belts too often because they get worn down very quickly. What is causing this?\n"
                "\n"
                "     - The crawler track support bolts keep breaking frequently. Why is this happening?\n"
                "\n"
                "     - Grease will not go into the grease nipples when I pump it. How often should I grease them, and how do I fix clogged fittings?\n"
                "\n"
                "     - Why does the engine immediately shut off as soon as I disconnect the negative battery terminal?\n"
                "\n"
                "     - When I disconnect the negative battery terminal, the engine doesn't shut off, but the RPM won't increase when I press the throttle. Why is that?\n"
                "\n"
                "     - Why is the coolant expansion tank bubbling and overflowing with water?\n"
                "\n"
                "     - As soon as I turn on the lights, the fuses blow immediately. What is causing this short circuit?\n"
                "\n"
                "     - Why is there a loud knocking sound coming from the engine valves?\n"
                "\n"
                "     - What is causing the cutter bar blades to break so frequently?\n"
                "\n"
                "     - What are the best maintenance practices and operating tips to ensure a long lifespan for the engine and gearbox?\n"
                "\n"
                "     - Why do the pulleys, belts, and chains break so frequently?\n"
                "\n"
                "     - The harvester cannot move forward or backward anymore. What is broken?\n"
                "\n"
                "     - The axle shaft is spinning, but the drive sprocket is not turning. What is causing this failure?\n"
                "\n"
                "     - I just replaced the hydraulic pump with a new one, but it is still not pumping any oil. Why is that?\n"
                "\n"
                "     - The engine lacks power and doesn't pull well, even when I push the throttle all the way down. What is causing this?\n"
                "\n"
                "     - How often or at what intervals should I change the engine coolant?\n"
                "\n"
                "     - The harvester won't move at all when shifted into high or medium gear, but it moves normally when shifted into low gear. What is the cause?\n"
                "\n"
                "     - Why is it critical to use only the manufacturer-specified grade of oil for the engine and gearbox?\n"
                "4. Absolutely avoid hyper-specific questions asking for precise figures from a table (e.g., do NOT ask 'What is the maximum allowed pressure for 8-18 6PR front tires?' or 'What is the clearance value in mm?'). "
                "Instead, ask a practical, high-level troubleshooting or maintenance question that would lead a user or mechanic to refer to this manual section.\n"
                "5. Output ONLY the single Myanmar (Burmese) question. Do NOT include any English translation, introductory phrase, explanations, conversational filler, or quotation marks.\n\n"
                f'Manual Chunk(s):\n"""\n{merged_chunk_text}\n"""\n\n'
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
                "   - Tractor Questions:\n"
                "     - The starter motor won't turn when I try to start the tractor. What should I do?\n"
                "\n"
                "     - The starter motor turns, but the engine won't start. How can I fix this?\n"
                "\n"
                "     - As soon as I turn the key, I hear the solenoid clicking/dropping out and the engine won't start. What should I do?\n"
                "\n"
                "     - The battery warning light is showing on the dashboard. How do I fix this?\n"
                "\n"
                "     - The engine oil warning light is on. What should I do?\n"
                "\n"
                "     - The engine overheat light is on. How do I fix this?\n"
                "\n"
                "     - The engine starts, but it stalls out after running for about 5 minutes. What is causing this and how do I fix it?\n"
                "\n"
                "     - The RPM gauge on the dashboard is not moving or working. How do I fix this?\n"
                "\n"
                "     - When I turn the key on, no lights appear on the dashboard at all. What should I do?\n"
                "\n"
                "     - When I drive the tractor in reverse gear, it keeps popping out of gear. How can I fix this?\n"
                "\n"
                "     - I cannot engage the 4-wheel drive (4WD) gear. What should I do?\n"
                "\n"
                "     - The steering wheel is very heavy and difficult to turn. How can I fix this?\n"
                "\n"
                "     - The front loader bucket keeps dropping down on its own. What is causing this and how do I fix it?\n"
                "\n"
                "     - The plow lifting and lowering mechanism feels very heavy and sluggish. What should I do?\n"
                "\n"
                "     - Smoke is coming out of the gearbox and it is getting extremely hot. What is wrong and what should I do?\n"
                "\n"
                "     - The PTO is slipping and cannot handle the load. How can I fix this?\n"
                "\n"
                "     - I found white foam on the gear oil cap. What does this mean and what should I do?\n"
                "\n"
                "     - The plow shakes and bounces when lifting or lowering. How can I fix this?\n"
                "\n"
                "     - When I turn off the engine with the plow lifted, it immediately drops back down to the ground. How do I fix this?\n"
                "\n"
                "     - There is a loud noise coming from the clutch housing whenever I shift between forward and reverse gears. What is causing this?\n"
                "\n"
                "     - Even when I pull the clutch lever while the PTO gear is engaged, the rotary tiller blades at the back keep spinning. How do I fix this?\n"
                "\n"
                "     - Why do the drive chains keep breaking frequently when operating a 6 or 7-disc plow?\n"
                "\n"
                "     - My engine oil level keeps dropping, and I have to top up every day. What is causing this?\n"
                "\n"
                "     - Why do the kingpin seals and wheel flange seals keep blowing out and leaking oil so frequently?\n"
                "\n"
                "     - I only use premium diesel, so why is water constantly getting trapped inside the paper fuel filter bowl?\n"
                "\n"
                "   - Rice Combine Harvester Questions:\n"
                "     - When I turn the key, I can only hear the fuel pump clicking, but no lights appear on the dashboard. What should I do?\n"
                "\n"
                "     - The Stop light is illuminated and the engine won't start. How can I fix this?\n"
                "\n"
                "     - As soon as I engage the threshing drum lever, the engine stalls completely. What should I do?\n"
                "\n"
                "     - The grain unload auger spout cannot rotate left or right. How can I fix this?\n"
                "\n"
                "     - The front cutting header cannot be lifted or lowered. What should I do?\n"
                "\n"
                "     - The frame height adjustment works on one side but won't lift or lower on the other side. How can I fix this?\n"
                "\n"
                "     - The front dust suction fan has stopped spinning. What should I do?\n"
                "\n"
                "     - I have to replace the water pump belts too often because they get worn down very quickly. What is causing this?\n"
                "\n"
                "     - The crawler track support bolts keep breaking frequently. Why is this happening?\n"
                "\n"
                "     - Grease will not go into the grease nipples when I pump it. How often should I grease them, and how do I fix clogged fittings?\n"
                "\n"
                "     - Why does the engine immediately shut off as soon as I disconnect the negative battery terminal?\n"
                "\n"
                "     - When I disconnect the negative battery terminal, the engine doesn't shut off, but the RPM won't increase when I press the throttle. Why is that?\n"
                "\n"
                "     - Why is the coolant expansion tank bubbling and overflowing with water?\n"
                "\n"
                "     - As soon as I turn on the lights, the fuses blow immediately. What is causing this short circuit?\n"
                "\n"
                "     - Why is there a loud knocking sound coming from the engine valves?\n"
                "\n"
                "     - What is causing the cutter bar blades to break so frequently?\n"
                "\n"
                "     - What are the best maintenance practices and operating tips to ensure a long lifespan for the engine and gearbox?\n"
                "\n"
                "     - Why do the pulleys, belts, and chains break so frequently?\n"
                "\n"
                "     - The harvester cannot move forward or backward anymore. What is broken?\n"
                "\n"
                "     - The axle shaft is spinning, but the drive sprocket is not turning. What is causing this failure?\n"
                "\n"
                "     - I just replaced the hydraulic pump with a new one, but it is still not pumping any oil. Why is that?\n"
                "\n"
                "     - The engine lacks power and doesn't pull well, even when I push the throttle all the way down. What is causing this?\n"
                "\n"
                "     - How often or at what intervals should I change the engine coolant?\n"
                "\n"
                "     - The harvester won't move at all when shifted into high or medium gear, but it moves normally when shifted into low gear. What is the cause?\n"
                "\n"
                "     - Why is it critical to use only the manufacturer-specified grade of oil for the engine and gearbox?\n"
                "4. Absolutely avoid hyper-specific questions asking for precise figures from a table (e.g., do NOT ask 'What is the maximum allowed pressure for 8-18 6PR front tires?' or 'What is the clearance value in mm?'). "
                "Instead, ask a practical, high-level troubleshooting or maintenance question that would lead a user or mechanic to refer to this manual section.\n"
                "5. Output ONLY the single Japanese question. Do NOT include any English translation, introductory phrase, explanations, conversational filler, or quotation marks.\n\n"
                f'Manual Chunk(s):\n"""\n{merged_chunk_text}\n"""\n\n'
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
                "   - Tractor Questions:\n"
                "     - The starter motor won't turn when I try to start the tractor. What should I do?\n"
                "\n"
                "     - The starter motor turns, but the engine won't start. How can I fix this?\n"
                "\n"
                "     - As soon as I turn the key, I hear the solenoid clicking/dropping out and the engine won't start. What should I do?\n"
                "\n"
                "     - The battery warning light is showing on the dashboard. How do I fix this?\n"
                "\n"
                "     - The engine oil warning light is on. What should I do?\n"
                "\n"
                "     - The engine overheat light is on. How do I fix this?\n"
                "\n"
                "     - The engine starts, but it stalls out after running for about 5 minutes. What is causing this and how do I fix it?\n"
                "\n"
                "     - The RPM gauge on the dashboard is not moving or working. How do I fix this?\n"
                "\n"
                "     - When I turn the key on, no lights appear on the dashboard at all. What should I do?\n"
                "\n"
                "     - When I drive the tractor in reverse gear, it keeps popping out of gear. How can I fix this?\n"
                "\n"
                "     - I cannot engage the 4-wheel drive (4WD) gear. What should I do?\n"
                "\n"
                "     - The steering wheel is very heavy and difficult to turn. How can I fix this?\n"
                "\n"
                "     - The front loader bucket keeps dropping down on its own. What is causing this and how do I fix it?\n"
                "\n"
                "     - The plow lifting and lowering mechanism feels very heavy and sluggish. What should I do?\n"
                "\n"
                "     - Smoke is coming out of the gearbox and it is getting extremely hot. What is wrong and what should I do?\n"
                "\n"
                "     - The PTO is slipping and cannot handle the load. How can I fix this?\n"
                "\n"
                "     - I found white foam on the gear oil cap. What does this mean and what should I do?\n"
                "\n"
                "     - The plow shakes and bounces when lifting or lowering. How can I fix this?\n"
                "\n"
                "     - When I turn off the engine with the plow lifted, it immediately drops back down to the ground. How do I fix this?\n"
                "\n"
                "     - There is a loud noise coming from the clutch housing whenever I shift between forward and reverse gears. What is causing this?\n"
                "\n"
                "     - Even when I pull the clutch lever while the PTO gear is engaged, the rotary tiller blades at the back keep spinning. How do I fix this?\n"
                "\n"
                "     - Why do the drive chains keep breaking frequently when operating a 6 or 7-disc plow?\n"
                "\n"
                "     - My engine oil level keeps dropping, and I have to top up every day. What is causing this?\n"
                "\n"
                "     - Why do the kingpin seals and wheel flange seals keep blowing out and leaking oil so frequently?\n"
                "\n"
                "     - I only use premium diesel, so why is water constantly getting trapped inside the paper fuel filter bowl?\n"
                "\n"
                "   - Rice Combine Harvester Questions:\n"
                "     - When I turn the key, I can only hear the fuel pump clicking, but no lights appear on the dashboard. What should I do?\n"
                "\n"
                "     - The Stop light is illuminated and the engine won't start. How can I fix this?\n"
                "\n"
                "     - As soon as I engage the threshing drum lever, the engine stalls completely. What should I do?\n"
                "\n"
                "     - The grain unload auger spout cannot rotate left or right. How can I fix this?\n"
                "\n"
                "     - The front cutting header cannot be lifted or lowered. What should I do?\n"
                "\n"
                "     - The frame height adjustment works on one side but won't lift or lower on the other side. How can I fix this?\n"
                "\n"
                "     - The front dust suction fan has stopped spinning. What should I do?\n"
                "\n"
                "     - I have to replace the water pump belts too often because they get worn down very quickly. What is causing this?\n"
                "\n"
                "     - The crawler track support bolts keep breaking frequently. Why is this happening?\n"
                "\n"
                "     - Grease will not go into the grease nipples when I pump it. How often should I grease them, and how do I fix clogged fittings?\n"
                "\n"
                "     - Why does the engine immediately shut off as soon as I disconnect the negative battery terminal?\n"
                "\n"
                "     - When I disconnect the negative battery terminal, the engine doesn't shut off, but the RPM won't increase when I press the throttle. Why is that?\n"
                "\n"
                "     - Why is the coolant expansion tank bubbling and overflowing with water?\n"
                "\n"
                "     - As soon as I turn on the lights, the fuses blow immediately. What is causing this short circuit?\n"
                "\n"
                "     - Why is there a loud knocking sound coming from the engine valves?\n"
                "\n"
                "     - What is causing the cutter bar blades to break so frequently?\n"
                "\n"
                "     - What are the best maintenance practices and operating tips to ensure a long lifespan for the engine and gearbox?\n"
                "\n"
                "     - Why do the pulleys, belts, and chains break so frequently?\n"
                "\n"
                "     - The harvester cannot move forward or backward anymore. What is broken?\n"
                "\n"
                "     - The axle shaft is spinning, but the drive sprocket is not turning. What is causing this failure?\n"
                "\n"
                "     - I just replaced the hydraulic pump with a new one, but it is still not pumping any oil. Why is that?\n"
                "\n"
                "     - The engine lacks power and doesn't pull well, even when I push the throttle all the way down. What is causing this?\n"
                "\n"
                "     - How often or at what intervals should I change the engine coolant?\n"
                "\n"
                "     - The harvester won't move at all when shifted into high or medium gear, but it moves normally when shifted into low gear. What is the cause?\n"
                "\n"
                "     - Why is it critical to use only the manufacturer-specified grade of oil for the engine and gearbox?\n"
                "4. CRITICAL: Absolutely avoid hyper-specific questions asking for precise figures from a table (e.g., do NOT ask 'What is the maximum allowed pressure for 8-18 6PR front tires?' or 'What is the clearance value in mm?'). "
                "Instead, ask a practical, high-level troubleshooting or maintenance question that would lead a user or mechanic to refer to this manual section.\n"
                "5. Output ONLY the single English question. Do NOT include any introductory phrase, explanations, conversational filler, or quotation marks.\n\n"
                f'Manual Chunk(s):\n"""\n{merged_chunk_text}\n"""\n\n'
                "Farmer/Mechanic Question (in English):"
            )

        try:
            generated_question = get_gemini_response(
                rag.text_model,
                model_input=instruction,
                stream=False,
                generation_config=GenerationConfig(
                    temperature=0.7, max_output_tokens=MAX_QUESTION_TOKENS
                ),
            )
            generated_question = (
                (generated_question or "").strip().strip('"').strip("'")
            )
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
            "timestamp": datetime.datetime.now().isoformat(),
        }
        results.append(question_item)

    # Persistent JSON storage (per-manual file under that manual's cache dir)
    questions_json_path = Path(manual.cache_dir) / "generated_questions.json"
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
        Path(manual.cache_dir).mkdir(parents=True, exist_ok=True)
        with open(questions_json_path, "w", encoding="utf-8") as f:
            json.dump(existing_questions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error writing to questions JSON: {e}")

    return {
        "success": True,
        "questions": results,
    }


# -----------------------------
# Manual Manager Page
# -----------------------------


@app.get("/manage")
def manage_page():
    """Serve the manual manager UI."""
    path = TEMPLATES_DIR / "manage.html"
    if not path.exists():
        return HTMLResponse("<p>manage.html not found</p>", status_code=404)
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.post("/api/add-manual")
async def api_add_manual(
    manual_id: str = Form(...),
    display_name: str = Form(""),
    ocr_lang: str = Form("eng"),
    description: str = Form(""),
    files: List[UploadFile] = File(default=[]),
):
    """Add a new manual to the registry, create directories, and optionally save uploaded PDFs."""
    manual_id = manual_id.strip()
    if not re.fullmatch(r"[a-zA-Z0-9_\-]+", manual_id):
        return JSONResponse(
            {"ok": False, "error": "manual_id must be alphanumeric with _ or - only"},
            status_code=400,
        )
    if manual_id in {m.manual_id for m in manual_registry.list()}:
        return JSONResponse(
            {"ok": False, "error": f"'{manual_id}' already exists in registry"},
            status_code=400,
        )

    resolved_name = display_name.strip() or manual_id.replace("_", " ").title()
    root = Path(__file__).resolve().parent
    pdf_folder = f"manuals/{manual_id}/pdf"
    cache_dir = f"manuals/{manual_id}/cache"
    image_dir = f"manuals/{manual_id}/cache/images"

    # Create directory structure
    for sub in (pdf_folder, cache_dir, image_dir):
        (root / sub).mkdir(parents=True, exist_ok=True)

    # Write to manuals.json registry
    entry = {
        "manual_id": manual_id,
        "display_name": resolved_name,
        "pdf_folder": pdf_folder,
        "cache_dir": cache_dir,
        "image_dir": image_dir,
        "ocr_lang": ocr_lang.strip() or "eng",
        "description": description.strip(),
    }
    if _MANUALS_JSON_PATH.exists():
        existing = json.loads(_MANUALS_JSON_PATH.read_text(encoding="utf-8"))
    else:
        existing = [
            {
                "manual_id": m.manual_id,
                "display_name": m.display_name,
                "pdf_folder": m.pdf_folder,
                "cache_dir": m.cache_dir,
                "image_dir": m.image_dir,
                "ocr_lang": m.ocr_lang,
                "description": m.description,
            }
            for m in manual_registry.list()
        ]
    existing.append(entry)
    _MANUALS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MANUALS_JSON_PATH.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    upload_manual_registry_to_s3(str(_MANUALS_JSON_PATH))

    # Save uploaded PDFs
    pdf_count = 0
    for f in files:
        if f.filename and f.filename.lower().endswith(".pdf"):
            dest = root / pdf_folder / f.filename
            dest.write_bytes(await f.read())
            if is_s3_configured():
                upload_manual_pdf_file_to_s3(manual_id, str(dest))
            pdf_count += 1

    # Hot-reload registry
    new_cfg = ManualConfig(
        manual_id=manual_id,
        display_name=resolved_name,
        pdf_folder=pdf_folder,
        cache_dir=cache_dir,
        image_dir=image_dir,
        ocr_lang=ocr_lang.strip() or "eng",
        description=description.strip(),
    )
    manual_registry._manuals[manual_id] = new_cfg

    # Mount static route for the new manual's images
    try:
        app.mount(
            f"/static/{manual_id}",
            StaticFiles(directory=str(root / image_dir)),
            name=f"static-{manual_id}",
        )
    except Exception:
        pass  # already mounted or not critical

    return JSONResponse({"ok": True, "manual_id": manual_id, "pdf_count": pdf_count})


@app.post("/api/remove-manual")
def api_remove_manual(manual_id: str):
    """Remove a manual: clear pipeline, delete local dirs, remove from manuals.json and S3."""
    manual_id = manual_id.strip()
    try:
        manual = _resolve_manual(manual_id)
    except RuntimeError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    if manual_id == DEFAULT_MANUAL_ID:
        return JSONResponse(
            {"ok": False, "error": "Cannot remove the default manual"},
            status_code=400,
        )

    # Clear in-memory pipeline state
    _clear_rag_state(manual_id)

    # Delete local directories
    root = Path(__file__).resolve().parent
    manual_dir = root / "manuals" / manual_id
    if manual_dir.exists():
        shutil.rmtree(manual_dir)

    # Delete from S3
    s3_deleted = 0
    if is_s3_configured():
        try:
            s3_deleted = delete_manual_from_s3(manual_id)
        except Exception as e:
            print(f"[{manual_id}] S3 delete error (continuing): {e}")

    # Remove from in-memory registry
    manual_registry._manuals.pop(manual_id, None)

    # Remove from manuals.json
    if _MANUALS_JSON_PATH.exists():
        try:
            existing = json.loads(_MANUALS_JSON_PATH.read_text(encoding="utf-8"))
            existing = [m for m in existing if m.get("manual_id") != manual_id]
            _MANUALS_JSON_PATH.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            upload_manual_registry_to_s3(str(_MANUALS_JSON_PATH))
        except Exception as e:
            print(f"[{manual_id}] manuals.json update error: {e}")

    return JSONResponse({"ok": True, "manual_id": manual_id, "s3_deleted": s3_deleted})


@app.post("/api/training/start")
def api_training_start(manual_id: Optional[str] = None):
    """Kick off a background training job for one manual. Returns job_id."""
    try:
        manual = _resolve_manual(manual_id)
    except RuntimeError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    job_id = str(uuid.uuid4())
    with _training_jobs_lock:
        _training_jobs[job_id] = {
            "job_id": job_id,
            "manual_id": manual.manual_id,
            "status": "running",
            "progress": 0,
            "message": "Starting...",
        }
    threading.Thread(
        target=_run_training_job, args=(job_id, manual), daemon=True
    ).start()
    return JSONResponse({"ok": True, "job_id": job_id, "manual_id": manual.manual_id})


@app.get("/api/training/status")
def api_training_status(job_id: str):
    """Poll training job progress."""
    with _training_jobs_lock:
        job = _training_jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "unknown job_id"}, status_code=404)
    return JSONResponse(dict(job))
