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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import fitz  # pymupdf
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

# Embeddings (Vertex)
# If this import fails, upgrade google-cloud-aiplatform.
from vertexai.vision_models import MultiModalEmbeddingModel

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

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None

try:
    import pytesseract
except Exception:
    pytesseract = None


# -----------------------------
# Config
# -----------------------------
@dataclass
class RagConfig:
    project_id: str
    location: str = "us-central1"

    # Gemini model name
    model_name: str = "gemini-2.0-flash"

    # Embeddings
    embedding_size: int = 1408
    embedding_model_name: str = "multimodalembedding@001"  # common 1408-d model

    # Extraction settings
    image_save_dir: str = "images"
    image_description_prompt: str = (
        "Explain what is going on in the image.\n"
        "If it's a table, extract all elements of the table.\n"
        "If it's a graph, explain the findings in the graph.\n"
        "Do not include any numbers that are not mentioned in the image.\n"
    )

    # OCR settings (fallback)
    enable_ocr_fallback: bool = True
    ocr_min_chars: int = 40  # if extracted text < this, OCR that page
    ocr_dpi: int = 200
    ocr_lang: str = "eng"

    # OCR chunking (simple char-based)
    ocr_chunk_chars: int = 1200
    ocr_chunk_overlap: int = 150


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

        self.embedder: MultiModalEmbeddingModel = None  # type: ignore

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

        # Embedder used for OCR text chunks
        self.embedder = MultiModalEmbeddingModel.from_pretrained(
            self.config.embedding_model_name
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

    def load_cache(self, cache_dir: str, *, rebuild_image_objects: bool = True) -> bool:
        cache_path = Path(cache_dir)
        text_pkl = cache_path / "text_metadata_df.pkl"
        image_pkl = cache_path / "image_metadata_df.pkl"

        if not text_pkl.exists() or not image_pkl.exists():
            return False

        self.text_metadata_df = pd.read_pickle(text_pkl)
        self.image_metadata_df = pd.read_pickle(image_pkl)

        if rebuild_image_objects:
            self._rebuild_image_objects_from_paths()

        return True

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

    def _render_page(self, page: fitz.Page, dpi: int) -> "PILImage.Image": # pyright: ignore[reportInvalidTypeForm]
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
        emb = self.embedder.get_embeddings(contextual_text=text)
        vec = getattr(emb, "text_embedding", None)
        if vec is None:
            raise RuntimeError("Embedding output missing text_embedding.")
        if hasattr(vec, "values"):
            vec = vec.values
        arr = np.array(vec, dtype=np.float32)
        return arr.tolist()

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

        doc_col = self._infer_doc_col(text_df)
        page_col = self._infer_page_col(text_df)
        chunk_col = self._infer_chunk_text_col(text_df)
        emb_col = self._infer_embedding_col(text_df)

        if doc_col is None or page_col is None:
            # Can't reliably map pages -> don't OCR automatically
            return text_df
        if chunk_col is None or emb_col is None:
            return text_df

        # Build "how much text do we already have per (doc,page)"
        grp = (
            text_df.groupby([doc_col, page_col])[chunk_col]
            .apply(lambda s: sum(len(str(x)) for x in s))
            .to_dict()
        )

        new_rows: List[Dict[str, Any]] = []

        for pdf_path in self._pdf_paths(pdf_folder_path):
            doc = fitz.open(str(pdf_path))
            doc_name = pdf_path.name

            for i in range(doc.page_count):
                page_num = i + 1
                existing_chars = grp.get((doc_name, page_num), 0)

                # Only OCR if existing extracted text is "too short"
                if existing_chars >= self.config.ocr_min_chars:
                    continue

                page = doc.load_page(i)
                base_text = (page.get_text("text") or "").strip()

                # if page.get_text already has enough but df didn't capture it, skip OCR
                if len(base_text) >= self.config.ocr_min_chars:
                    continue

                ocr_text = self._ocr_page(page)
                if len(ocr_text) < self.config.ocr_min_chars:
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

            doc.close()

        if not new_rows:
            return text_df

        add_df = pd.DataFrame(new_rows)

        # Ensure all columns exist
        for c in text_df.columns:
            if c not in add_df.columns:
                add_df[c] = None
        add_df = add_df[text_df.columns]  # match column order

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

        # 1) Original extraction (utils)
        text_df, image_df = get_document_metadata(
            self.multimodal_model,
            pdf_folder_path,
            **kwargs,
        )

        # 2) OCR fallback: append OCR chunks + embeddings
        text_df = self._append_ocr_chunks(text_df, pdf_folder_path)

        self.text_metadata_df = text_df
        self.image_metadata_df = image_df

        if cache_dir:
            self.save_cache(cache_dir)

        return text_df, image_df

    # -----------------------------
    # Retrieval
    # -----------------------------
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
        return get_similar_text_from_query(
            query,
            self.text_metadata_df,  # type: ignore
            column_name=column_name,
            top_n=top_n,
            chunk_text=chunk_text,
        )

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
        embedding_size = embedding_size or self.config.embedding_size
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
    def answer_text_query(
        self,
        query: str,
        *,
        top_n: int = 3,
        temperature: float = 0.2,
        stream: bool = True,
    ) -> Dict[str, Any]:
        matches = self.search_text(query, top_n=top_n, chunk_text=True)
        context = "\n".join([value["chunk_text"] for _, value in matches.items()])

        instruction = (
            "Answer the question with the given context.\n"
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
    ) -> Dict[str, Any]:
        matches = self.search_images_by_description_text(query, top_n=top_n)
        top = matches[0]
        context = (
            f"Image: {top['image_object']}\nDescription: {top['image_description']}\n"
        )

        instruction = (
            "Answer the question in JSON format with the given context of Image and its Description. "
            "Only include value.\n"
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
    ) -> Dict[str, Any]:
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

        reasoning_line = (
            "Make sure to think thoroughly before answering the question and put the necessary steps "
            "to arrive at the answer in bullet points for easy explainability.\n"
            if include_step_by_step
            else ""
        )

        prompt = (
            "Instructions: Compare the images and the text provided as Context: to answer multiple Question:\n"
            f"{reasoning_line}"
            'If unsure, respond, "Not enough context to answer".\n\n'
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
