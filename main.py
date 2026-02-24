# example_usage.py
# Updated to use OCR fallback enabled in the updated pipeline.
#
# Requirements (Colab):
#   pip install -q google-cloud-aiplatform pymupdf pillow pytesseract numpy pandas
#   apt-get update -qq && apt-get install -y tesseract-ocr
#
# If you DON'T have tesseract/pytesseract installed, set enable_ocr_fallback=False.

from vertexai.generative_models import GenerationConfig
from pipeline import RagConfig, MultimodalRAGPipeline

cfg = RagConfig(
    project_id="fortunaii",
    location="us-central1",
    model_name="gemini-2.0-flash",
    embedding_size=1408,
    embedding_model_name="multimodalembedding@001",  # used to embed OCR chunks
    image_save_dir="./cache_ym358a/images",  # extracted images stored here
    enable_ocr_fallback=True,  # <- OCR fallback ON
    ocr_min_chars=40,  # <- OCR if extracted page text < 40 chars
    ocr_dpi=200,
    ocr_lang="eng",
)

rag = MultimodalRAGPipeline(cfg)

# 1) First run: extract + OCR fallback + cache dataframes (no second extraction later)
rag.build_metadata(
    pdf_folder_path="./data/",
    cache_dir="./cache_ym358a",
    force_rebuild=False,
    generation_config=GenerationConfig(temperature=0.2),
    ocr_fallback=True,  # <- optional override per run (if you added this arg)
)

# 2) Later runs (new session): just load cache (no extraction, no OCR)
# rag.load_cache("./cache_ym358a", rebuild_image_objects=False)

# -------------------------
# YM358A sample queries
# -------------------------
queries = [
    "YM358A engine model, displacement, rated horsepower and torque",
    "YM358A transmission type and number of forward/reverse gears",
    "YM358A PTO type and PTO speeds",
    "YM358A hydraulic lift capacity and pump flow",
    "YM358A dimensions (LxWxH), wheelbase, ground clearance, and weight",
    "YM358A maintenance schedule: engine oil and filter interval (hours) and oil specification",
    "YM358A troubleshooting: PTO not engaging / not spinning - checks and causes",
]

# Text RAG (now includes OCR text chunks too)
out = rag.answer_text_query(queries[1], top_n=5, temperature=0.2, stream=True)
rag.print_text_citations(out["matches"], print_top=False, chunk_text=True)

# Multimodal RAG (text + images)
multi_q = """Questions about YM358A:
- Summarize key specs (engine, transmission, PTO, hydraulics).
- Extract any spec tables shown in the document images.
- Provide maintenance intervals mentioned in the manual/brochure.
"""
out2 = rag.answer_multimodal_query(
    multi_q,
    top_n_text=10,
    top_n_images=10,
    temperature=1,
    stream=True,
)
rag.print_text_citations(out2["text_matches"], print_top=False, chunk_text=True)
rag.print_image_citations(out2["image_matches"], print_top=False)

# Save cache again if you updated metadata
rag.save_cache("./cache_ym358a")
