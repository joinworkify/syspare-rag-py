# multimodal_rag_pipeline.py
# SAME pipeline you posted, but now includes OCR fallback for pages with little/no extracted text.
#
# How it works:
# - build_metadata() calls your existing utils.get_document_metadata() first (no changes needed in utils).
# - Then it runs OCR fallback on each PDF page where extracted text is too short.
# - It appends OCR text as *extra chunks* into text_metadata_df (and embeds them with Vertex embeddings),
#   so retrieval can find them.
#
# Requirements for OCR fallback:
#   pip install pymupdf pillow pytesseract
#   and install tesseract binary (Colab):
#     apt-get update -qq && apt-get install -y tesseract-ocr

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

import numpy as np
import pandas as pd
import pymupdf as fitz  # pymupdf
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

# Your utilities (downloaded from gs://github-repo/rag/intro_multimodal_rag/...)
from utils import (
    display_images,
    get_document_metadata,
    get_gemini_response,
    get_similar_image_from_query,
    get_similar_text_from_query,
    print_text_to_image_citation,
    print_text_to_text_citation,
)
import re

from syspare_rag.config import RagConfig
from syspare_rag.indexing.embedder import VertexTextEmbedder
from syspare_rag.indexing.validation import (
    inferred_image_pixel_dim,
    validate_image_index,
    validate_text_index,
)

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None

try:
    import pytesseract
except Exception:
    pytesseract = None


# RagConfig lives in syspare_rag.config (re-exported here for backward compatibility).


# -----------------------------
# Pipeline
# -----------------------------
class MultimodalRAGPipeline:
    """
    Multimodal RAG pipeline with caching + OCR fallback.

    Flow:
      1) get_document_metadata() from your utils (original extraction)
      2) OCR fallback for "empty text" pages -> add extra OCR chunks + embeddings
      3) Save/load cache as before
    """

    def __init__(self, config: RagConfig):
        self.config = config

        self.text_model: GenerativeModel = None  # type: ignore
        self.multimodal_model: GenerativeModel = None  # type: ignore
        self.multimodal_model_flash: GenerativeModel = None  # type: ignore

        self._text_embedder: VertexTextEmbedder

        self.text_metadata_df: Optional[pd.DataFrame] = None
        self.image_metadata_df: Optional[pd.DataFrame] = None

        self._init_vertex()

    # -----------------------------
    # Initialization
    # -----------------------------
    def _init_vertex(self) -> None:
        vertexai.init(project=self.config.project_id, location=self.config.location)

        self.text_model = GenerativeModel(self.config.model_name)
        self.multimodal_model = self.text_model
        self.multimodal_model_flash = self.text_model

        self._text_embedder = VertexTextEmbedder(
            model_name=self.config.text_embedding_model
        )

    @property
    def has_metadata(self) -> bool:
        return self.text_metadata_df is not None and self.image_metadata_df is not None

    # -----------------------------
    # Caching helpers
    # -----------------------------
    def _jsonify_cell(self, x: Any) -> Any:
        """CSV-friendly conversion for lists/dicts."""
        if isinstance(x, (list, dict)):
            return json.dumps(x)
        return x

    def save_cache(self, cache_dir: str) -> Dict[str, str]:
        if not self.has_metadata:
            raise RuntimeError("No metadata to cache. Run build_metadata() first.")

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        text_pkl = cache_path / "text_metadata_df.pkl"
        image_pkl = cache_path / "image_metadata_df.pkl"
        text_csv = cache_path / "text_metadata_df.csv"
        image_csv = cache_path / "image_metadata_df.csv"

        self.text_metadata_df.to_pickle(text_pkl)  # type: ignore
        self.image_metadata_df.to_pickle(image_pkl)  # type: ignore

        text_csv_df = self.text_metadata_df.copy()  # type: ignore
        image_csv_df = self.image_metadata_df.copy()  # type: ignore

        for col in text_csv_df.columns:
            text_csv_df[col] = text_csv_df[col].map(self._jsonify_cell)
        for col in image_csv_df.columns:
            image_csv_df[col] = image_csv_df[col].map(self._jsonify_cell)

        if "image_object" in image_csv_df.columns:
            image_csv_df = image_csv_df.drop(columns=["image_object"])

        text_csv_df.to_csv(text_csv, index=False)
        image_csv_df.to_csv(image_csv, index=False)

        return {
            "text_pkl": str(text_pkl),
            "image_pkl": str(image_pkl),
            "text_csv": str(text_csv),
            "image_csv": str(image_csv),
        }

    def load_cache(
        self,
        cache_dir: str,
        *,
        rebuild_image_objects: bool = True,
        validate: bool = True,
    ) -> bool:
        cache_path = Path(cache_dir)
        text_pkl = cache_path / "text_metadata_df.pkl"
        image_pkl = cache_path / "image_metadata_df.pkl"

        if not text_pkl.exists() or not image_pkl.exists():
            return False

        self.text_metadata_df = pd.read_pickle(text_pkl)
        self.image_metadata_df = pd.read_pickle(image_pkl)

        if validate:
            self._validate_loaded_cache()

        if rebuild_image_objects:
            self._rebuild_image_objects_from_paths()

        return True

    def _validate_loaded_cache(self) -> None:
        """Guard against P0-1 (mixed embedding dimensions) on cache load."""
        if self.text_metadata_df is not None:
            validate_text_index(
                self.text_metadata_df,
                text_embedding_dim=self.config.text_embedding_dimension,
            )
        if self.image_metadata_df is not None:
            cached_pixel_dim = inferred_image_pixel_dim(self.image_metadata_df)
            validate_image_index(
                self.image_metadata_df,
                text_embedding_dim=self.config.text_embedding_dimension,
                image_embedding_dim=cached_pixel_dim,
            )
            if (
                cached_pixel_dim is not None
                and cached_pixel_dim != self.config.embedding_size
            ):
                print(
                    f"[pipeline] image pixel embedding dim in cache is "
                    f"{cached_pixel_dim} but RagConfig.embedding_size="
                    f"{self.config.embedding_size}; using cached dim for queries."
                )

    def _rebuild_image_objects_from_paths(self) -> None:
        if PILImage is None or self.image_metadata_df is None:
            return

        path_col = None
        for c in ["img_path", "image_path", "path"]:
            if c in self.image_metadata_df.columns:
                path_col = c
                break
        if path_col is None:
            return

        if "image_object" in self.image_metadata_df.columns:
            return

        def _load_img(p: Any):
            try:
                return PILImage.open(str(p)).convert("RGB")
            except Exception:
                return None

        self.image_metadata_df["image_object"] = self.image_metadata_df[path_col].map(
            _load_img
        )

    # -----------------------------
    # OCR helpers
    # -----------------------------
    def _chunk_text_simple(
        self, text: str, chunk_chars: int, overlap: int
    ) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        chunks: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            j = min(i + chunk_chars, n)
            chunks.append(text[i:j].strip())
            if j >= n:
                break
            i = max(j - overlap, j)
        return [c for c in chunks if c]

    def _render_page(
        self, page: fitz.Page, dpi: int
    ) -> "PILImage.Image":  # pyright: ignore[reportInvalidTypeForm]
        if PILImage is None:
            raise RuntimeError("PIL not installed. pip install pillow")
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return PILImage.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

    def _ocr_page(self, page: fitz.Page) -> str:
        if pytesseract is None:
            raise RuntimeError(
                "pytesseract not installed. Install with: pip install pytesseract "
                "and install tesseract-ocr binary."
            )
        img = self._render_page(page, self.config.ocr_dpi)
        return (
            pytesseract.image_to_string(img, lang=self.config.ocr_lang) or ""
        ).strip()

    def _embed_text(self, text: str) -> List[float]:
        return self._text_embedder.embed_text(text)

    def _pdf_paths(self, pdf_folder_path: str) -> List[Path]:
        p = Path(pdf_folder_path)
        if not p.exists():
            raise FileNotFoundError(f"PDF folder not found: {pdf_folder_path}")
        return sorted([x for x in p.glob("**/*.pdf")])

    def _infer_doc_col(self, df: pd.DataFrame) -> Optional[str]:
        for c in ["doc_name", "source", "file_name", "document_name", "document"]:
            if c in df.columns:
                return c
        return None

    def _infer_page_col(self, df: pd.DataFrame) -> Optional[str]:
        for c in ["page_number", "page", "page_num", "page_idx"]:
            if c in df.columns:
                return c
        return None

    def _infer_chunk_text_col(self, df: pd.DataFrame) -> Optional[str]:
        for c in ["chunk_text", "text_chunk", "text", "chunk"]:
            if c in df.columns:
                return c
        return None

    def _infer_embedding_col(self, df: pd.DataFrame) -> Optional[str]:
        for c in ["text_embedding_chunk", "text_embedding", "chunk_embedding"]:
            if c in df.columns:
                return c
        return None

    def _append_ocr_chunks(
        self,
        text_df: pd.DataFrame,
        pdf_folder_path: str,
    ) -> pd.DataFrame:
        """
        For each PDF page where extracted text looks empty, OCR it and append as new rows.
        We avoid OCR if that page already has enough text in the dataframe.
        """
        if not self.config.enable_ocr_fallback:
            return text_df

        df_empty = text_df.empty or len(text_df.columns) == 0

        doc_col = self._infer_doc_col(text_df) or "file_name"
        page_col = self._infer_page_col(text_df) or "page_num"
        chunk_col = self._infer_chunk_text_col(text_df) or "chunk_text"
        emb_col = self._infer_embedding_col(text_df) or "text_embedding_chunk"

        # Build "how much text do we already have per (doc,page)"
        # Skip groupby when df is empty — all pages need OCR
        if df_empty:
            grp: dict = {}
        else:
            grp = (
                text_df.groupby([doc_col, page_col])[chunk_col]
                .apply(lambda s: sum(len(str(x)) for x in s))
                .to_dict()
            )

        new_rows: List[Dict[str, Any]] = []

        pdf_paths = list(self._pdf_paths(pdf_folder_path))
        total_pages = sum(fitz.open(str(p)).page_count for p in pdf_paths)

        with tqdm(total=total_pages, desc="OCR fallback", unit="pg") as pbar:
            for pdf_path in pdf_paths:
                doc = fitz.open(str(pdf_path))
                doc_name = pdf_path.name

                for i in range(doc.page_count):
                    page_num = i + 1
                    existing_chars = grp.get((doc_name, page_num), 0)

                    if existing_chars >= self.config.ocr_min_chars:
                        pbar.update(1)
                        continue

                    page = doc.load_page(i)
                    base_text = (page.get_text("text") or "").strip()

                    if len(base_text) >= self.config.ocr_min_chars:
                        pbar.update(1)
                        continue

                    ocr_text = self._ocr_page(page)
                    if len(ocr_text) < self.config.ocr_min_chars:
                        pbar.update(1)
                        continue

                    ocr_chunks = self._chunk_text_simple(
                        ocr_text,
                        chunk_chars=self.config.ocr_chunk_chars,
                        overlap=self.config.ocr_chunk_overlap,
                    )

                    for cid, ch in enumerate(ocr_chunks):
                        new_rows.append(
                            {
                                doc_col: doc_name,
                                page_col: page_num,
                                chunk_col: ch,
                                emb_col: self._embed_text(ch),
                                "extraction_method": "ocr",
                                "pdf_path": str(pdf_path),
                                "chunk_id": f"ocr_{page_num}_{cid}",
                            }
                        )

                    pbar.update(1)

                doc.close()

        if not new_rows:
            return text_df

        add_df = pd.DataFrame(new_rows)

        if not df_empty:
            # Align columns to existing df schema
            for c in text_df.columns:
                if c not in add_df.columns:
                    add_df[c] = None
            add_df = add_df[text_df.columns]

        out = pd.concat([text_df, add_df], ignore_index=True)
        return out

    # -----------------------------
    # Build metadata (extract or load cache)
    # -----------------------------
    def build_metadata(
        self,
        pdf_folder_path: str,
        *,
        cache_dir: Optional[str] = None,
        force_rebuild: bool = False,
        image_save_dir: Optional[str] = None,
        image_description_prompt: Optional[str] = None,
        embedding_size: Optional[int] = None,
        # Optional pass-through
        add_sleep_after_page: bool = False,
        sleep_time_after_page: int = 5,
        generation_config: Optional[GenerationConfig] = None,
        safety_settings: Optional[Any] = None,
        # OCR control
        ocr_fallback: Optional[bool] = None,
        # Skip image re-extraction for PDFs that already have images in cache
        skip_existing_images: bool = False,
    ):
        """
        Same as before, but with OCR fallback appended into text_metadata_df.
        """
        if cache_dir and not force_rebuild:
            if self.load_cache(cache_dir, rebuild_image_objects=False):
                return self.text_metadata_df, self.image_metadata_df

        if ocr_fallback is not None:
            self.config.enable_ocr_fallback = bool(ocr_fallback)

        image_save_dir = image_save_dir or self.config.image_save_dir
        image_description_prompt = (
            image_description_prompt or self.config.image_description_prompt
        )
        embedding_size = embedding_size or self.config.embedding_size

        # Determine which PDFs to skip (image-only or entirely)
        skip_image_for_pdfs: set = set()
        skip_pdfs_entirely: set = set()
        existing_image_df: Optional[pd.DataFrame] = None
        existing_text_df: Optional[pd.DataFrame] = None

        if skip_existing_images and cache_dir:
            img_pkl = Path(cache_dir) / "image_metadata_df.pkl"
            txt_pkl = Path(cache_dir) / "text_metadata_df.pkl"

            if img_pkl.exists():
                try:
                    existing_image_df = pd.read_pickle(img_pkl)
                    img_file_col = next(
                        (c for c in ["file_name", "doc_name", "source"] if c in existing_image_df.columns),
                        None,
                    )
                    if img_file_col:
                        skip_image_for_pdfs = set(existing_image_df[img_file_col].dropna().unique().tolist())
                except Exception as exc:
                    print(f"[skip_existing_images] Could not load image cache: {exc}")

            if txt_pkl.exists():
                try:
                    existing_text_df = pd.read_pickle(txt_pkl)
                    txt_file_col = next(
                        (c for c in ["file_name", "doc_name", "source"] if c in existing_text_df.columns),
                        None,
                    )
                    if txt_file_col:
                        cached_text_pdfs = set(existing_text_df[txt_file_col].dropna().unique().tolist())
                        # Skip entirely when BOTH text and image rows exist for a PDF
                        skip_pdfs_entirely = skip_image_for_pdfs & cached_text_pdfs
                        skip_image_for_pdfs -= skip_pdfs_entirely
                except Exception as exc:
                    print(f"[skip_existing_images] Could not load text cache: {exc}")

            if skip_pdfs_entirely:
                print(f"[skip_existing_images] Skipping {len(skip_pdfs_entirely)} fully-cached PDF(s) (text+images)")
            if skip_image_for_pdfs:
                print(f"[skip_existing_images] Skipping image extraction only for {len(skip_image_for_pdfs)} PDF(s)")

        elif skip_existing_images:
            # No cache_dir: fallback to glob for existing image files
            img_dir = Path(image_save_dir)
            if img_dir.exists():
                for pdf_path in self._pdf_paths(pdf_folder_path):
                    fn = pdf_path.name
                    if list(img_dir.glob(f"{fn}_image_*")):
                        skip_image_for_pdfs.add(fn)
                if skip_image_for_pdfs:
                    print(f"[skip_existing_images] Found existing images for: {skip_image_for_pdfs}")

        kwargs: Dict[str, Any] = dict(
            image_save_dir=image_save_dir,
            image_description_prompt=image_description_prompt,
            embedding_size=embedding_size,
        )

        if add_sleep_after_page:
            kwargs["add_sleep_after_page"] = True
            kwargs["sleep_time_after_page"] = sleep_time_after_page
        if generation_config is not None:
            kwargs["generation_config"] = generation_config
        if safety_settings is not None:
            kwargs["safety_settings"] = safety_settings
        if skip_image_for_pdfs:
            kwargs["skip_image_for_pdfs"] = skip_image_for_pdfs
        if skip_pdfs_entirely:
            kwargs["skip_pdfs_entirely"] = skip_pdfs_entirely

        # 1) Original extraction (utils)
        text_df, image_df = get_document_metadata(
            self.multimodal_model,
            pdf_folder_path,
            **kwargs,
        )

        # Merge existing rows for entirely-skipped PDFs
        if skip_existing_images and skip_pdfs_entirely:
            if existing_text_df is not None:
                txt_file_col = next(
                    (c for c in ["file_name", "doc_name", "source"] if c in existing_text_df.columns),
                    None,
                )
                if txt_file_col:
                    skipped_text = existing_text_df[existing_text_df[txt_file_col].isin(skip_pdfs_entirely)]
                    if not skipped_text.empty:
                        text_df = pd.concat([text_df, skipped_text], ignore_index=True)
                        print(f"[skip_existing_images] Merged {len(skipped_text)} existing text row(s).")

            if existing_image_df is not None:
                img_file_col = next(
                    (c for c in ["file_name", "doc_name", "source"] if c in existing_image_df.columns),
                    None,
                )
                if img_file_col:
                    skipped_imgs = existing_image_df[existing_image_df[img_file_col].isin(skip_pdfs_entirely)]
                    if not skipped_imgs.empty:
                        image_df = pd.concat([image_df, skipped_imgs], ignore_index=True)
                        print(f"[skip_existing_images] Merged {len(skipped_imgs)} existing image row(s).")

        # Merge existing image rows for image-only-skipped PDFs
        elif skip_existing_images and existing_image_df is not None and skip_image_for_pdfs:
            img_file_col = next(
                (c for c in ["file_name", "doc_name", "source"] if c in existing_image_df.columns),
                None,
            )
            if img_file_col:
                skipped_imgs = existing_image_df[existing_image_df[img_file_col].isin(skip_image_for_pdfs)]
                if not skipped_imgs.empty:
                    image_df = pd.concat([image_df, skipped_imgs], ignore_index=True)
                    print(f"[skip_existing_images] Merged {len(skipped_imgs)} existing image row(s).")

        # 2) OCR fallback: append OCR chunks + embeddings
        text_df = self._append_ocr_chunks(text_df, pdf_folder_path)

        self.text_metadata_df = text_df
        self.image_metadata_df = image_df

        self._validate_loaded_cache()

        if cache_dir:
            print(f"Saving cache to {cache_dir} ...")
            self.save_cache(cache_dir)
            print("Cache saved.")

        return text_df, image_df

    # -----------------------------
    # Retrieval
    # -----------------------------
    def _expand_query(self, query: str, n_variants: int = 2) -> List[str]:
        """
        Use Gemini to generate alternative phrasings of the query using technical
        synonyms that are more likely to appear in service manuals.
        Falls back to [query] on any error.
        """
        instruction = (
            "You are a technical query expander for agricultural machinery service manuals.\n"
            f"Given the user question, generate exactly {n_variants} alternative phrasings "
            "that use different technical synonyms or related terms that might appear in a service manual.\n"
            "Rules:\n"
            "- Replace colloquial terms with technical equivalents "
            "(e.g. 'water in oil' → 'oil separation at low temperature', 'oil discoloration')\n"
            "- Focus on vocabulary found in maintenance/inspection sections of service manuals\n"
            f"- Output ONLY the {n_variants} alternative questions, one per line, no numbering, no explanation\n\n"
            f"User question: {query}\n"
            "Alternatives:"
        )
        response = get_gemini_response(
            self.text_model,
            model_input=instruction,
            stream=False,
            generation_config=GenerationConfig(temperature=0.3, max_output_tokens=256),
        )
        if not response or response == "Exception occurred":
            return [query]
        variants = [line.strip() for line in response.strip().splitlines() if line.strip()]
        return [query] + variants[:n_variants]

    def search_text(
        self,
        query: str,
        *,
        top_n: int = 3,
        column_name: str = "text_embedding_chunk",
        chunk_text: bool = True,
    ) -> Dict[Any, Dict[str, Any]]:
        if not self.has_metadata:
            raise RuntimeError("Run build_metadata() first.")
        if self.text_metadata_df is None or self.text_metadata_df.empty:
            return {}
        return get_similar_text_from_query(
            query,
            self.text_metadata_df,  # type: ignore
            column_name=column_name,
            top_n=top_n,
            chunk_text=chunk_text,
        )

    def _rerank_chunks(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        """
        Use Gemini to score each candidate chunk against the query (0-10).
        Returns top_n chunks sorted by Gemini relevance score.
        Falls back to cosine-score order if parsing fails.
        """
        if len(candidates) <= top_n:
            return candidates

        snippet_lines = []
        for i, v in enumerate(candidates):
            text = (v.get("chunk_text") or "").strip()[:400]
            snippet_lines.append(f"[{i + 1}] {text}")

        instruction = (
            "You are a relevance ranker for agricultural machinery service manual sections.\n"
            f"User question: {query}\n\n"
            "Rate each numbered section for how directly it answers the question. "
            "Score 0 (irrelevant) to 10 (directly answers).\n"
            "Output ONLY lines in the format: index:score  (e.g. 3:9)\n"
            "One line per section, nothing else.\n\n"
            + "\n\n".join(snippet_lines)
        )
        response = get_gemini_response(
            self.text_model,
            model_input=instruction,
            stream=False,
            generation_config=GenerationConfig(temperature=0.1, max_output_tokens=512),
        )

        scores: Dict[int, float] = {}
        for line in (response or "").strip().splitlines():
            try:
                parts = line.strip().split(":")
                idx = int(parts[0].strip()) - 1  # 0-indexed
                score = float(parts[1].strip())
                scores[idx] = score
            except Exception:
                pass

        def sort_key(i: int) -> float:
            if i in scores:
                return scores[i]
            return candidates[i].get("cosine_score", 0.0)

        sorted_indices = sorted(range(len(candidates)), key=sort_key, reverse=True)
        return [candidates[i] for i in sorted_indices[:top_n]]

    def search_text_multi(
        self,
        query: str,
        *,
        top_n: int = 5,
        column_name: str = "text_embedding_chunk",
        chunk_text: bool = True,
        n_variants: int = 2,
        rerank: bool = True,
        retrieval_multiplier: int = 4,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Multi-query retrieval with Gemini reranking.

        1. Expands query into n_variants alternatives.
        2. For each variant retrieves top_n * retrieval_multiplier candidates.
        3. Merges by chunk identity keeping max cosine score.
        4. Reranks the merged pool with Gemini, returns global top_n.
        """
        if not self.has_metadata:
            raise RuntimeError("Run build_metadata() first.")
        if self.text_metadata_df is None or self.text_metadata_df.empty:
            return {}

        expanded = self._expand_query(query, n_variants=n_variants)
        pool_size = top_n * retrieval_multiplier

        seen: Dict[tuple, Dict[str, Any]] = {}
        for q in expanded:
            matches = get_similar_text_from_query(
                q,
                self.text_metadata_df,  # type: ignore
                column_name=column_name,
                top_n=pool_size,
                chunk_text=chunk_text,
            )
            for value in matches.values():
                key = (
                    value.get("file_name", ""),
                    value.get("page_num", 0),
                    value.get("chunk_number"),
                )
                score = value.get("cosine_score", 0.0)
                if key not in seen or score > seen[key].get("cosine_score", 0.0):
                    seen[key] = value

        candidates = sorted(
            seen.values(), key=lambda x: x.get("cosine_score", 0.0), reverse=True
        )

        if rerank and len(candidates) > top_n:
            candidates = self._rerank_chunks(query, candidates, top_n)
        else:
            candidates = candidates[:top_n]

        return {i: v for i, v in enumerate(candidates)}

    def search_images_by_description_text(
        self,
        query: str,
        *,
        top_n: int = 3,
        column_name: str = "text_embedding_from_image_description",
        embedding_size: Optional[int] = None,
    ) -> Dict[Any, Dict[str, Any]]:
        if not self.has_metadata:
            raise RuntimeError("Run build_metadata() first.")
        if self.image_metadata_df is None or self.image_metadata_df.empty:
            return {}
        embedding_size = embedding_size or self.config.embedding_size
        return get_similar_image_from_query(
            self.text_metadata_df,  # type: ignore
            self.image_metadata_df,  # type: ignore
            query=query,
            column_name=column_name,
            image_emb=False,
            top_n=top_n,
            embedding_size=embedding_size,
        )

    def search_images_by_image_embedding(
        self,
        query: str,
        image_query_path: str,
        *,
        top_n: int = 3,
        column_name: str = "mm_embedding_from_img_only",
        embedding_size: Optional[int] = None,
    ) -> Dict[Any, Dict[str, Any]]:
        if not self.has_metadata:
            raise RuntimeError("Run build_metadata() first.")
        # Prefer the actual embedding dimension stored in image_metadata_df[column_name]
        if embedding_size is None:
            try:
                first_vec = self.image_metadata_df[column_name].iloc[0]  # type: ignore[index]
                if hasattr(first_vec, "__len__"):
                    embedding_size = len(first_vec)  # type: ignore[arg-type]
                else:
                    embedding_size = self.config.embedding_size
            except Exception:
                embedding_size = self.config.embedding_size
        return get_similar_image_from_query(
            self.text_metadata_df,  # type: ignore
            self.image_metadata_df,  # type: ignore
            query=query,
            column_name=column_name,
            image_emb=True,
            image_query_path=image_query_path,
            top_n=top_n,
            embedding_size=embedding_size,
        )

    # -----------------------------
    # Answering
    # -----------------------------
    def _answer_language_line(self, answer_language: str) -> str:
        """
        Returns a short instruction forcing output language.
        Supported values:
          - "auto" (default): no constraint
          - "en": English
          - "my" | "mm": Myanmar (Burmese)
          - "ja" | "jp": Japanese
        """
        lang = (answer_language or "auto").strip().lower()
        if lang in ("en", "english"):
            return "Answer in English.\n"
        if lang in ("my", "mm", "myanmar", "burmese"):
            return "Answer in Myanmar (Burmese).\n"
        if lang in ("ja", "jp", "japanese"):
            return "Answer in Japanese.\n"
        return ""

    def answer_text_query(
        self,
        query: str,
        *,
        top_n: int = 3,
        temperature: float = 0.2,
        stream: bool = True,
        answer_language: str = "auto",
    ) -> Dict[str, Any]:
        matches = self.search_text(query, top_n=top_n, chunk_text=True)
        context = "\n".join([value["chunk_text"] for _, value in matches.items()])

        lang_line = self._answer_language_line(answer_language)
        instruction = (
            "Answer the question with the given context.\n"
            f"{lang_line}"
            'If the information is not available in the context, just return "not available in the context".\n'
            f"Question: {query}\n"
            f"Context: {context}\n"
            "Answer:\n"
        )

        response = get_gemini_response(
            self.text_model,
            model_input=instruction,
            stream=stream,
            generation_config=GenerationConfig(temperature=temperature),
        )

        return {
            "query": query,
            "matches": matches,
            "context": context,
            "prompt": instruction,
            "response": response,
        }

    def answer_image_query_from_description(
        self,
        query: str,
        *,
        top_n: int = 3,
        temperature: float = 1.0,
        stream: bool = True,
        answer_language: str = "auto",
    ) -> Dict[str, Any]:
        matches = self.search_images_by_description_text(query, top_n=top_n)
        top = matches[0]
        context = (
            f"Image: {top['image_object']}\nDescription: {top['image_description']}\n"
        )

        lang_line = self._answer_language_line(answer_language)
        instruction = (
            "Answer the question in JSON format with the given context of Image and its Description. "
            "Only include value.\n"
            f"{lang_line}"
            f"Question: {query}\n"
            f"Context: {context}\n"
            "Answer:\n"
        )

        response = get_gemini_response(
            self.multimodal_model_flash,
            model_input=instruction,
            stream=stream,
            generation_config=GenerationConfig(temperature=temperature),
        )

        return {
            "query": query,
            "matches": matches,
            "top_image": top,
            "context": context,
            "prompt": instruction,
            "response": response,
        }

    def answer_multimodal_query(
        self,
        query: str,
        *,
        top_n_text: int = 10,
        top_n_images: int = 10,
        temperature: float = 1.0,
        stream: bool = True,
        include_step_by_step: bool = True,
        answer_language: str = "auto",
    ) -> Dict[str, Any]:
        text_matches = self.search_text_multi(query, top_n=top_n_text, chunk_text=True)
        image_matches = self.search_images_by_description_text(
            query, top_n=top_n_images
        )

        context_text = [value["chunk_text"] for _, value in text_matches.items()]
        final_context_text = "\n".join(context_text)

        context_images: List[Any] = []
        for idx, (_, value) in enumerate(image_matches.items()):
            context_images.extend(
                [
                    f"Image {idx+1}: ",
                    value["image_object"],
                    "Caption: ",
                    value["image_description"],
                ]
            )

        lang_line = self._answer_language_line(answer_language)
        reasoning_line = (
            "Make sure to think thoroughly before answering the question and put the necessary steps "
            "to arrive at the answer in bullet points for easy explainability.\n"
            if include_step_by_step
            else ""
        )

        prompt = (
            "Instructions: Answer the question using ONLY the information in the Context below. "
            "Do not use external knowledge. "
            "If the answer is not present in the provided context, respond: "
            "\"The information is not available in the provided manual sections.\"\n"
            "CRITICAL CITATION RULE: When answering, if you use or reference information from a specific image to support your explanation, "
            "you MUST cite it inline using the format [Image X] where X is the image index number (e.g. [Image 1], [Image 2]). "
            "Always include these inline references if you rely on details or visual elements from the images.\n\n"
            f"{lang_line}"
            f"{reasoning_line}"
            "Context:\n"
            " - Text Context:\n"
            f"{final_context_text}\n"
            " - Image Context:\n"
            f"{context_images}\n\n"
            f"{query}\n\n"
            "Answer:\n"
        )

        response = get_gemini_response(
            self.multimodal_model,
            model_input=[prompt],
            stream=stream,
            generation_config=GenerationConfig(temperature=temperature),
        )

        return {
            "query": query,
            "text_matches": text_matches,
            "image_matches": image_matches,
            "prompt": prompt,
            "response": response,
        }

    # -----------------------------
    # Structured diagnostic answer
    # -----------------------------
    def _safe_parse_json(self, text: str) -> Any:
        """
        Best-effort JSON parsing helper for model outputs that should be JSON.
        """
        text = (text or "").strip()
        # Direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try to extract first JSON object/array
        m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if m:
            snippet = m.group(1)
            try:
                return json.loads(snippet)
            except Exception:
                pass

        # Fallback: return raw text
        return {"raw": text}

    def answer_diagnostic_query(
        self,
        query: str,
        *,
        top_n_text: int = 10,
        top_n_images: int = 6,
        temperature: float = 0.4,
        stream: bool = False,
        answer_language: str = "auto",
    ) -> Dict[str, Any]:
        """
        Return a structured diagnostic JSON answer suitable for a dashboard UI.

        Shape (high level):
        {
          "summary": "...",
          "confidence": { "score": 0.0-1.0, "label": "direct_match|low|medium|high" },
          "urgency": { "label": "critical|warning|normal|info", "reason": "..." },
          "actions": [
             { "id": "ACTION 01", "title": "...", "description": "..." },
             ...
          ],
          "torque_spec": { "label": "...", "value_nm": 0, "value_ft_lbs": 0, "raw": "..." }
        }
        """
        text_matches = self.search_text(query, top_n=top_n_text, chunk_text=True)
        image_matches = self.search_images_by_description_text(
            query, top_n=top_n_images
        )

        context_text = [value["chunk_text"] for _, value in text_matches.items()]
        final_context_text = "\n".join(context_text)

        context_images: List[Any] = []
        for _, value in image_matches.items():
            context_images.extend(
                [
                    "Image: ",
                    value["image_object"],
                    "Caption: ",
                    value["image_description"],
                ]
            )

        lang_line = self._answer_language_line(answer_language)
        prompt = (
            "You are a service manual assistant. Using ONLY the following manual context "
            "(text + image descriptions), create a structured diagnostic summary in JSON.\n\n"
            f"{lang_line}"
            "JSON schema (keys and nesting):\n"
            "{\n"
            '  "summary": string,                       // one-paragraph diagnostic summary\n'
            '  "confidence": {\n'
            '    "score": number,                      // 0.0 - 1.0\n'
            '    "label": string                       // e.g. "direct_match", "high", "medium", "low"\n'
            "  },\n"
            '  "urgency": {\n'
            '    "label": string,                      // e.g. "critical", "warning", "normal", "info"\n'
            '    "reason": string                      // short explanation of urgency\n'
            "  },\n"
            '  "actions": [                            // ordered field procedures\n'
            "    {\n"
            '      "id": string,                       // e.g. "ACTION 01"\n'
            '      "title": string,                    // short action title\n'
            '      "description": string               // step-by-step description in 1-3 sentences\n'
            "    }\n"
            "  ],\n"
            '  "torque_spec": {                        // optional\n'
            '    "label": string,                      // e.g. "Assembly Mounting Bolt (M10 x 30)"\n'
            '    "value_nm": number|null,              // torque in N·m if available\n'
            '    "value_ft_lbs": number|null,          // torque in ft-lbs if available\n'
            '    "raw": string                         // original torque text from the manual\n'
            "  }\n"
            "}\n\n"
            "If torque information is not present, set torque_spec to null.\n"
            "Do NOT include any additional keys or commentary outside the JSON.\n\n"
            "Context:\n"
            " - Text Context:\n"
            f"{final_context_text}\n"
            " - Image Context:\n"
            f"{context_images}\n\n"
            f"Question: {query}\n\n"
            "Return ONLY the JSON object.\n"
        )

        response = get_gemini_response(
            self.multimodal_model,
            model_input=[prompt],
            stream=stream,
            generation_config=GenerationConfig(temperature=temperature),
        )

        parsed = self._safe_parse_json(response)

        return {
            "query": query,
            "text_matches": text_matches,
            "image_matches": image_matches,
            "prompt": prompt,
            "response_raw": response,
            "response_parsed": parsed,
        }

    # -----------------------------
    # Citations / display helpers
    # -----------------------------
    def print_text_citations(
        self,
        matches: Dict[Any, Dict[str, Any]],
        *,
        print_top: bool = False,
        chunk_text: bool = True,
    ) -> None:
        print_text_to_text_citation(matches, print_top=print_top, chunk_text=chunk_text)

    def print_image_citations(
        self,
        matches: Dict[Any, Dict[str, Any]],
        *,
        print_top: bool = False,
    ) -> None:
        print_text_to_image_citation(matches, print_top=print_top)

    def show_images(self, img_paths: List[str], *, resize_ratio: float = 0.5) -> None:
        display_images(img_paths, resize_ratio=resize_ratio)
