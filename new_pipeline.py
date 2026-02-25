from concurrent.futures import ThreadPoolExecutor, as_completed
from IPython.display import display
from langchain_core.documents import Document
from PyPDF2 import PdfReader, PdfWriter
from google.genai import types
from google import genai
from typing import List
from tqdm import tqdm
from json import JSONDecodeError
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import Image as vision_model_Image
from vertexai.vision_models import MultiModalEmbeddingModel
from vertexai.generative_models import Image
from vertexai.generative_models import Image as VxImage
from vertexai.generative_models import GenerationConfig, HarmBlockThreshold, HarmCategory, Image, GenerativeModel

import io, re, json, threading
import numpy as np
import pandas as pd
import fitz, cv2, os, sys
import PIL
from tqdm import tqdm
from pathlib import Path


# if "google.colab" in sys.modules:
#     from google.colab import auth
#     auth.authenticate_user()

class Pipeline:
    def __init__(self):
        self.PROJECT_ID = "fortunaii"
        self.LOCATION = "us-central1" # Your project location
        self.client = genai.Client(vertexai=True, project=self.PROJECT_ID, location=self.LOCATION)
        self.model = "gemini-2.0-flash"
        self.text_model = GenerativeModel(self.model)
        self.generative_multimodal_model = self.text_model
        self.multimodal_model_flash = self.text_model
        self.generation_config = GenerationConfig(temperature=0.2)
        self.markdown_prompt = """You are tasked with converting a PDF document to text in markdown format. Your goal is to accurately represent the content, structure, and layout of the original PDF while using markdown syntax. Follow these instructions carefully: To convert this PDF content to markdown format, follow these steps: 1. Document Structure: - Preserve the overall structure of the document. - Use appropriate markdown syntax for headers, subheaders, and sections. - Maintain the original hierarchy of the document. 2. Text Formatting: - Convert basic text to plain markdown text. - Use markdown syntax for bold (**text**), italic (*text*), and strikethrough (~~text~~) where applicable. - Preserve any special characters or symbols as they appear in the original document. 3. Headers: - Use the appropriate number of hash symbols (#) to represent different header levels. - Example: # for H1, ## for H2, ### for H3, and so on. 4. Paragraphs: - Separate paragraphs with a blank line. - Preserve any indentation or special formatting within paragraphs. 5. Lists: - Use - for unordered lists and 1. 2. 3. for ordered lists. - Maintain the original indentation for nested lists. 6. Tables: - Convert tables to markdown table format. - Use | to separate columns and - to create the header row. - Align columns using : in the header row (e.g., |:---:| for center alignment). 7. Links: - Convert hyperlinks to markdown format: [link text](URL) 8. Images: - For each image/chart in the PDF, insert a placeholder in the following format: [Image Description] - Provide a full description of the image in place of \"Image Description\". - describe the image in detail, like you would describe it to a blind person. 9. Footnotes: - Use markdown footnote syntax: [^1] for the reference and [^1]: Footnote text for the footnote content. - Place all footnotes at the end of the document. 10. Code Blocks: - Use triple backticks (```) to enclose code blocks. - Specify the language after the opening backticks if applicable. 11. Blockquotes: - Use > to indicate blockquotes. - For nested blockquotes, use multiple > symbols. 12. Horizontal Rules: - Use three or more hyphens (---) on a line by themselves to create a horizontal rule. 13. Special Elements: - If there are any special elements in the PDF (e.g., mathematical equations, diagrams), describe them in plain text within square brackets. 14. Preserve Layout: - Maintain the original layout as much as possible, including line breaks and spacing. - Use empty lines and appropriate markdown syntax to recreate the visual structure of the document. Once you have converted the entire PDF content to markdown format. Ensure that all elements of the original document are accurately represented in the markdown version."""
        self.image_description_prompt = str(
            "Explain what is going on in the image.\n"
            "If it's a table, extract all elements of the table.\n"
            "If it's a graph, explain the findings in the graph.\n"
            "Do not include any numbers that are not mentioned in the image.\n"
        )
        self.text_embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        self.multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")

        # Will be populated after processing/building metadata or loading cache
        self.text_metadata_df: Optional[pd.DataFrame] = None
        self.image_metadata_df: Optional[pd.DataFrame] = None


    def get_client(self):
        return self.client

    def get_pdf_reader_and_metadata(self, pdf_path):
        reader = PdfReader(pdf_path)
        metadata = {
            "total_pages": len(reader.pages),
            "title": reader.metadata.get('/Title', ''),
            "author": reader.metadata.get('/Author', ''),
            "creation_date": reader.metadata.get('/CreationDate', '')
        }
        return reader, metadata

    def extract_page(self, reader, page_number):
        total_pages = len(reader.pages)
        if page_number < 0 or page_number >= total_pages:
            print(f"Page {page_number} is out of range. This PDF has {total_pages} pages.")
            return None
        writer = PdfWriter()
        writer.add_page(reader.pages[page_number])
        buffer = io.BytesIO()
        writer.write(buffer)
        return buffer.getvalue()

    def _strip_code_fences(self, text):
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _extract_first_json_object(self, text):
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return m.group(0) if m else None

    def generate(self, page_pdf_bytes: bytes, metadata: dict):
        document_part = types.Part.from_bytes(data=page_pdf_bytes, mime_type="application/pdf")

        contents = [
            types.Content(
                role="user",
                parts=[
                    document_part,
                    types.Part.from_text(text="convert this file")
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            top_p=0.95,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
            response_mime_type="application/json",
            response_schema={
            "type": "OBJECT",
            "properties": {
                "page_content": {"type": "STRING", "description": "The content of the document."}
            },
            "required": ["page_content"],},
            system_instruction=[types.Part.from_text(text=self.markdown_prompt)],
        )

        response = self.get_client().models.generate_content(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        )

        if hasattr(response, "parsed") and response.parsed:
            page_content = response.parsed.get("page_content", "")
            return Document(page_content=page_content, metadata=metadata)

        # 2) Fallback: try to parse response.text
        raw = (getattr(response, "text", "") or "").strip()
        raw = self._strip_code_fences(raw)

        # Sometimes the API returns valid JSON directly
        try:
            obj = json.loads(raw)
            return Document(page_content=obj.get("page_content", raw), metadata=metadata)
        except JSONDecodeError:
            pass

        # 3) Salvage: extract first JSON object from a messy response
        maybe_json = self._extract_first_json_object(raw)
        if maybe_json:
            maybe_json = self._strip_code_fences(maybe_json)
            try:
                obj = json.loads(maybe_json)
                return Document(page_content=obj.get("page_content", raw), metadata=metadata)
            except JSONDecodeError:
                pass

        # 4) Last resort: treat it as plain markdown/text
        return Document(page_content=raw, metadata=metadata)


    def get_text_embedding_from_text_embedding_model(self,text: str, return_array: Optional[bool] = False):
        embeddings = self.text_embedding_model.get_embeddings([text])
        text_embedding = [embedding.values for embedding in embeddings][0]
        if return_array:
            text_embedding = np.fromiter(text_embedding, dtype=float)
        return text_embedding

    def get_image_embedding_from_multimodal_embedding_model(self, image_uri: str, embedding_size: int = 512, text: Optional[str] = None, return_array: Optional[bool] = False):
        image = vision_model_Image.load_from_file(image_uri)
        embeddings = self.multimodal_embedding_model.get_embeddings(image=image, contextual_text=text, dimension=embedding_size)  # 128, 256, 512, 1408
        image_embedding = embeddings.image_embedding
        if return_array:
            image_embedding = np.fromiter(image_embedding, dtype=float)
        return image_embedding


    def get_page_text_embedding(self, text_data: Union[dict, str]) -> dict:
        embeddings_dict = {}
        if not text_data: return embeddings_dict
        if isinstance(text_data, dict):
            for chunk_number, chunk_value in text_data.items():
                text_embed = self.get_text_embedding_from_text_embedding_model(text=chunk_value)
                embeddings_dict[chunk_number] = text_embed
        else:
            text_embed = self.get_text_embedding_from_text_embedding_model(text=text_data)
            embeddings_dict["text_embedding"] = text_embed

        return embeddings_dict

    def get_text_overlapping_chunk(self, text: str, character_limit: int = 1000, overlap: int = 100):
        if overlap > character_limit:
            raise ValueError("Overlap cannot be larger than character limit.")
        chunk_number = 1
        chunked_text_dict = {}
        for i in range(0, len(text), character_limit - overlap):
            end_index = min(i + character_limit, len(text))
            chunk = text[i:end_index]
            chunked_text_dict[chunk_number] = chunk.encode("ascii", "ignore").decode("utf-8", "ignore")
            chunk_number += 1
        return chunked_text_dict


    def get_chunk_text_metadata(self, page_content, overlap: int = 100, character_limit: int = 1000, embedding_size: int = 128) -> dict:
        if overlap > character_limit:
            raise ValueError("Overlap cannot be larger than character limit.")
        page_text_embeddings_dict = self.get_page_text_embedding(page_content)
        chunked_text_dict: dict = self.get_text_overlapping_chunk(page_content, character_limit, overlap)
        chunk_embeddings_dict: dict = self.get_page_text_embedding(chunked_text_dict)
        return page_content, page_text_embeddings_dict, chunked_text_dict, chunk_embeddings_dict


    def find_and_crop_blocks(self, page_img_path, out_dir="./out_images", min_area=0.03, delete_original=True):
        os.makedirs(out_dir, exist_ok=True)
        img = cv2.imread(page_img_path)
        if img is None:
            raise ValueError(f"Could not read {page_img_path}")

        H, W = img.shape[:2]
        page_area = H * W

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        connected = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
        cnts, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crops = []
        base = os.path.splitext(os.path.basename(page_img_path))[0]
        idx = 1

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if (w * h) < (min_area * page_area):
                continue

            crop = img[y:y+h, x:x+w]
            out_path = os.path.join(out_dir, f"{base}_crop_{idx:02d}.png")
            ok = cv2.imwrite(out_path, crop)
            if ok:
                crops.append(out_path)
                idx += 1

        # delete original ONCE (after loop), and don't crash if already deleted
        if delete_original and os.path.exists(page_img_path):
            try:
                os.remove(page_img_path)
            except FileNotFoundError:
                pass

        return crops

    def get_image_for_gemini(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"{image_path} not found")
        image_for_gemini = VxImage.load_from_file(image_path)
        return image_for_gemini, image_path

    def extract_embedded_images(self, pdf_path: str, out_dir: str = "./images"):
        os.makedirs(out_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        saved = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)
            for img_i, img in enumerate(image_list, start=1):
                xref = img[0]
                base = doc.extract_image(xref)
                img_bytes = base["image"]
                ext = base.get("ext", "bin")
                out_path = os.path.join(out_dir, f"page_{page_index+1:03d}_img_{img_i:02d}.{ext}")
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
                self.find_and_crop_blocks(out_path)
                saved.append(out_path)
        return saved


    def extract_page_images_pymupdf(self, doc: fitz.Document, page_number: int, out_dir="./out_images"):
        os.makedirs(out_dir, exist_ok=True)
        page = doc[page_number]
        images = page.get_images(full=True)
        saved = []
        for img_i, img in enumerate(images, start=1):
            xref = img[0]
            base = doc.extract_image(xref)
            ext = base.get("ext", "bin")
            img_bytes = base["image"]
            out_path = os.path.join(out_dir, f"page_{page_number+1:03d}_img_{img_i:02d}.{ext}")
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            cropped_imgs = self.find_and_crop_blocks(out_path, out_dir=out_dir)
            saved.append((cropped_imgs, xref))
        return saved, images


    def get_gemini_response(
        self,
        generative_multimodal_model,
        model_input: List[str],
        stream: bool = True,
        generation_config: Optional[GenerationConfig] = GenerationConfig(
            temperature=0.2, max_output_tokens=2048
        ),
        safety_settings: Optional[dict] = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    ) -> str:
        # Use the Vertex AI GenerativeModel instance passed in, not the google.genai client.
        # This matches the expected signature for generate_content (generation_config, safety_settings, stream).
        response = generative_multimodal_model.generate_content(
            model_input,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=stream,
        )
        if not stream:
            try:
                return (getattr(response, "text", None) or "").strip()
            except Exception as e:
                print(
                    "Exception occurred while calling gemini (non-stream). "
                    "Try lowering safety thresholds [safety_settings: BLOCK_NONE ] if not already done. -----",
                    e,
                )
                return "Exception occurred"

        response_list: List[str] = []
        for chunk in response:
            try:
                if getattr(chunk, "text", None):
                    response_list.append(chunk.text)
            except Exception as e:
                print(
                    "Exception occurred while calling gemini (stream). "
                    "Try lowering safety thresholds [safety_settings: BLOCK_NONE ] if not already done. -----",
                    e,
                )
                response_list.append("Exception occurred")
                continue

        return "".join(response_list).strip()


    def generate_image_description(self,image_for_gemini: VxImage):
        return self.get_gemini_response(
            self.generative_multimodal_model,
            model_input=[self.image_description_prompt, image_for_gemini],
            generation_config=self.generation_config,
            safety_settings=None,
            stream=True,
        )



    def get_text_metadata_df(
        self,
        filename: str, text_metadata: Dict[Union[int, str], Dict]
    ) -> pd.DataFrame:
        final_data_text: List[Dict] = []

        for key, values in text_metadata.items():
            for chunk_number, chunk_text in values["chunked_text_dict"].items():
                data: Dict = {}
                data["file_name"] = filename
                data["page_num"] = int(key) + 1
                data["text"] = values["text"]
                data["text_embedding_page"] = values["page_text_embeddings"][
                    "text_embedding"
                ]
                data["chunk_number"] = chunk_number
                data["chunk_text"] = chunk_text
                data["text_embedding_chunk"] = values["chunk_embeddings_dict"][chunk_number]

                final_data_text.append(data)

        return_df = pd.DataFrame(final_data_text)
        return_df = return_df.reset_index(drop=True)
        return return_df


    def get_image_metadata_df(
        self,
        filename: str, image_metadata: Dict[Union[int, str], Dict]
    ) -> pd.DataFrame:

        final_data_image: List[Dict] = []
        for key, values in image_metadata.items():
            for _, image_values in values.items():
                data: Dict = {}
                data["file_name"] = filename
                data["page_num"] = int(key) + 1
                data["img_num"] = int(image_values["img_num"])
                data["img_path"] = image_values["img_path"]
                data["img_desc"] = image_values["img_desc"]

                data["mm_embedding_from_img_only"] = image_values[
                    "mm_embedding_from_img_only"
                ]
                data["text_embedding_from_image_description"] = image_values[
                    "text_embedding_from_image_description"
                ]
                final_data_image.append(data)

        return_df = pd.DataFrame(final_data_image).dropna()
        return_df = return_df.reset_index(drop=True)
        return return_df


    def process_one_page(self, pdf_path: str, page_number: int, base_metadata: dict, out_root: str):
        # IMPORTANT: open these inside the thread
        reader = PdfReader(pdf_path)
        doc = fitz.open(pdf_path)
        # --- TEXT ---
        page_metadata = base_metadata.copy()
        page_metadata["page"] = page_number + 1

        page_bytes = self.extract_page(reader, page_number)
        document_page = self.generate(page_bytes, page_metadata)

        text, page_text_embeddings_dict, chunked_text_dict, chunk_embeddings_dict = self.get_chunk_text_metadata(
            document_page.page_content
        )

        text_payload = {
            "text": text,
            "page_text_embeddings": page_text_embeddings_dict,
            "chunked_text_dict": chunked_text_dict,
            "chunk_embeddings_dict": chunk_embeddings_dict,
        }

        # --- IMAGES ---
        # Use a unique output folder per page to avoid collisions
        pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
        page_out_dir = os.path.join(out_root, pdf_stem, f"page_{page_number+1:03d}")
        os.makedirs(page_out_dir, exist_ok=True)

        saved_imgs, images = self.extract_page_images_pymupdf(doc, page_number, out_dir=page_out_dir)

        image_payload = {}
        for image_no, _img in enumerate(images):
            image_number = int(image_no + 1)

            # saved_imgs is [(cropped_imgs, xref), ...]
            # cropped_imgs is a list of paths. pick first crop if exists.
            cropped_list = saved_imgs[image_no][0]
            if not cropped_list:
                continue

            image_path = cropped_list[0]
            image_for_gemini, image_name = self.get_image_for_gemini(image_path)
            if image_for_gemini is None:
                continue

            desc = self.generate_image_description(image_for_gemini)
            mm_embed = self.get_image_embedding_from_multimodal_embedding_model(image_uri=image_name, embedding_size=128)
            desc_text_embed = self.get_text_embedding_from_text_embedding_model(text=desc)

            image_payload[image_number] = {
                "img_num": image_number,
                "img_path": image_name,
                "img_desc": desc,
                "mm_embedding_from_img_only": mm_embed,
                "text_embedding_from_image_description": desc_text_embed,
            }

        doc.close()
        return page_number, text_payload, image_payload



    def process_all_pages_parallel(self, pdf_path: str, batch_size: int = 3) -> List[Document]:
        # Get initial metadata and page count
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        metadata = {
            "total_pages": total_pages,
            "title": reader.metadata.get('/Title', ''),
            "author": reader.metadata.get('/Author', ''),
            "creation_date": reader.metadata.get('/CreationDate', '')
        }
        documents = [None] * total_pages  # Pre-allocate list with correct size
        # Create progress bar
        pbar = tqdm(total=total_pages, desc="Processing pages")
        pbar_lock = threading.Lock()

        def process_batch(start_idx: int, end_idx: int):
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                # Create arguments for each page in the batch
                batch_args = [(i, pdf_path, metadata) for i in range(start_idx, min(end_idx, total_pages))]
                # Process the batch in parallel
                for page_number, document in executor.map(self.process_one_page, batch_args):
                    documents[page_number] = document
                    with pbar_lock:
                        pbar.update(1)

        # Process pages in batches
        try:
            for batch_start in range(0, total_pages, batch_size):
                batch_end = batch_start + batch_size
                process_batch(batch_start, batch_end)
        finally:
            pbar.close()

        return documents



    def process_pdf_parallel(self, pdf_path: str, max_workers: int = 6, out_root: str = "./out_images"):
        # base metadata (read once on main thread)
        reader0 = PdfReader(pdf_path)
        total_pages = len(reader0.pages)
        base_metadata = {
            "total_pages": total_pages,
            "title": reader0.metadata.get("/Title", "") if reader0.metadata else "",
            "author": reader0.metadata.get("/Author", "") if reader0.metadata else "",
            "creation_date": reader0.metadata.get("/CreationDate", "") if reader0.metadata else "",
        }

        text_metadata = {}
        image_metadata = {}

        # progress bar + lock for safe updates
        pbar = tqdm(total=total_pages, desc=f"Processing pages (parallel)")

        pbar_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(self.process_one_page, pdf_path, p, base_metadata, out_root)
                for p in range(total_pages)
            ]

            for fut in as_completed(futures):
                page_number, text_payload, image_payload = fut.result()

                text_metadata[page_number] = text_payload
                image_metadata[page_number] = image_payload

                with pbar_lock:
                    pbar.update(1)

        pbar.close()

        # Build your dataframes as before
        file_name = os.path.basename(pdf_path)
        text_df = self.get_text_metadata_df(file_name, text_metadata)
        image_df = self.get_image_metadata_df(file_name, image_metadata)
        return text_df, image_df

    def process(self, pdf_path: str, cache_dir: str):
        """
        Process a single PDF into text/image metadata and store them on the instance.
        """
        text_metadata_df_final, image_metadata_df_final = pd.DataFrame(), pd.DataFrame()
        text_metadata_df, image_metadata_df = self.process_pdf_parallel(
            pdf_path=pdf_path, max_workers=6, out_root=cache_dir
        )

        text_metadata_df_final = pd.concat(
            [text_metadata_df_final, text_metadata_df], axis=0
        )
        image_metadata_df_final = pd.concat(
            [
                image_metadata_df_final,
                image_metadata_df.drop_duplicates(subset=["img_desc"]),
            ],
            axis=0,
        )

        text_metadata_df_final = text_metadata_df_final.reset_index(drop=True)
        image_metadata_df_final = image_metadata_df_final.reset_index(drop=True)

        # Store on the instance so save_cache / search_* / answer_* can use them.
        self.text_metadata_df = text_metadata_df_final
        self.image_metadata_df = image_metadata_df_final

        return text_metadata_df_final, image_metadata_df_final


    def build_metadata(
        self,
        pdf_folder_path: str,
        cache_dir: str,
        *,
        force_rebuild: bool = False,
        generation_config: Optional[GenerationConfig] = None,
        ocr_fallback: bool = True,
        image_save_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Build metadata for all PDFs in `pdf_folder_path`, cache into `cache_dir`,
        and keep aggregated DataFrames on the instance.
        This is an adapter-style API so it can be used by `rag_server.py`.
        """
        cache_path = Path(cache_dir)
        text_pkl = cache_path / "text_metadata_df.pkl"
        image_pkl = cache_path / "image_metadata_df.pkl"

        if not force_rebuild and text_pkl.exists() and image_pkl.exists():
            # Reuse existing cache
            self.load_cache(cache_dir)
            return self.text_metadata_df, self.image_metadata_df

        cache_path.mkdir(parents=True, exist_ok=True)
        # Where to store page images
        image_root = image_save_dir or cache_dir
        Path(image_root).mkdir(parents=True, exist_ok=True)

        text_frames: List[pd.DataFrame] = []
        image_frames: List[pd.DataFrame] = []

        for pdf_path in sorted(Path(pdf_folder_path).glob("*.pdf")):
            if not pdf_path.is_file():
                continue
            t_df, i_df = self.process_pdf_parallel(
                pdf_path=str(pdf_path),
                max_workers=6,
                out_root=image_root,
            )
            text_frames.append(t_df)
            image_frames.append(i_df)

        self.text_metadata_df = (
            pd.concat(text_frames, ignore_index=True) if text_frames else pd.DataFrame()
        )
        self.image_metadata_df = (
            pd.concat(image_frames, ignore_index=True) if image_frames else pd.DataFrame()
        )

        # Persist cache for later fast loads
        self.save_cache(cache_dir)

        return self.text_metadata_df, self.image_metadata_df


    def get_user_query_text_embeddings(self, user_query: str) -> np.ndarray:
        return self.get_text_embedding_from_text_embedding_model(user_query)

    def get_user_query_image_embeddings(self, image_query_path: str, embedding_size: int) -> np.ndarray:
        return self.get_image_embedding_from_multimodal_embedding_model(
            image_uri=image_query_path, embedding_size=embedding_size
        )

    def get_cosine_score(
        self, dataframe: pd.DataFrame, column_name: str, input_text_embed: np.ndarray) -> float:
        text_cosine_score = round(np.dot(dataframe[column_name], input_text_embed), 2)
        return text_cosine_score


    def get_similar_image_from_query(
        self,
        text_metadata_df: pd.DataFrame,
        image_metadata_df: pd.DataFrame,
        query: str = "",
        image_query_path: str = "",
        column_name: str = "",
        image_emb: bool = True,
        top_n: int = 3,
        embedding_size: int = 128,
    ) -> Dict[int, Dict[str, Any]]:
        if image_emb:
            user_query_image_embedding = self.get_user_query_image_embeddings(
                image_query_path, embedding_size
            )
            cosine_scores = image_metadata_df.apply(
                lambda x: self.get_cosine_score(x, column_name, user_query_image_embedding),
                axis=1,
            )
        else:
            user_query_text_embedding = self.get_user_query_text_embeddings(query)
            cosine_scores = image_metadata_df.apply(
                lambda x: self.get_cosine_score(x, column_name, user_query_text_embedding),
                axis=1,
            )

        cosine_scores = cosine_scores[cosine_scores < 1.0]

        top_n_cosine_scores = cosine_scores.nlargest(top_n).index.tolist()
        top_n_cosine_values = cosine_scores.nlargest(top_n).values.tolist()

        final_images: Dict[int, Dict[str, Any]] = {}

        for matched_imageno, indexvalue in enumerate(top_n_cosine_scores):
            final_images[matched_imageno] = {}
            final_images[matched_imageno]["cosine_score"] = top_n_cosine_values[
                matched_imageno
            ]
            final_images[matched_imageno]["image_object"] = Image.load_from_file(
                image_metadata_df.iloc[indexvalue]["img_path"]
            )
            final_images[matched_imageno]["file_name"] = image_metadata_df.iloc[indexvalue][
                "file_name"
            ]
            final_images[matched_imageno]["img_path"] = image_metadata_df.iloc[indexvalue][
                "img_path"
            ]
            final_images[matched_imageno]["page_num"] = image_metadata_df.iloc[indexvalue][
                "page_num"
            ]
            final_images[matched_imageno]["page_text"] = np.unique(
                text_metadata_df[
                    (
                        text_metadata_df["page_num"].isin(
                            [final_images[matched_imageno]["page_num"]]
                        )
                    )
                    & (
                        text_metadata_df["file_name"].isin(
                            [final_images[matched_imageno]["file_name"]]
                        )
                    )
                ]["text"].values
            )
            final_images[matched_imageno]["image_description"] = image_metadata_df.iloc[
                indexvalue
            ]["img_desc"]

        return final_images

    def get_similar_text_from_query(
        self,
        query: str,
        text_metadata_df: pd.DataFrame,
        column_name: str = "",
        top_n: int = 3,
        chunk_text: bool = True,
        print_citation: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        if column_name not in text_metadata_df.columns:
            raise KeyError(f"Column '{column_name}' not found in the 'text_metadata_df'")

        query_vector = self.get_user_query_text_embeddings(query)
        cosine_scores = text_metadata_df.apply(
            lambda row: self.get_cosine_score(
                row,
                column_name,
                query_vector,
            ),
            axis=1,
        )

        top_n_indices = cosine_scores.nlargest(top_n).index.tolist()
        top_n_scores = cosine_scores.nlargest(top_n).values.tolist()

        final_text: Dict[int, Dict[str, Any]] = {}

        for matched_textno, index in enumerate(top_n_indices):
            final_text[matched_textno] = {}
            final_text[matched_textno]["file_name"] = text_metadata_df.iloc[index][
                "file_name"
            ]
            final_text[matched_textno]["page_num"] = text_metadata_df.iloc[index][
                "page_num"
            ]
            final_text[matched_textno]["cosine_score"] = top_n_scores[matched_textno]

            if chunk_text:
                final_text[matched_textno]["chunk_number"] = text_metadata_df.iloc[index][
                    "chunk_number"
                ]
                final_text[matched_textno]["chunk_text"] = text_metadata_df["chunk_text"][
                    index
                ]
            else:
                final_text[matched_textno]["text"] = text_metadata_df["text"][index]

        if print_citation:
            self.print_text_to_text_citation(final_text, chunk_text=chunk_text)

        return final_text


    def display_images(
        self,
        images: Iterable[Union[str, PIL.Image.Image]], resize_ratio: float = 0.5
    ) -> None:
        # Convert paths to PIL images if necessary
        pil_images = []
        for image in images:
            if isinstance(image, str):
                pil_images.append(PIL.Image.open(image))
            else:
                pil_images.append(image)

        # Resize and display each image
        for img in pil_images:
            original_width, original_height = img.size
            new_width = int(original_width * resize_ratio)
            new_height = int(original_height * resize_ratio)
            resized_img = img.resize((new_width, new_height))
            display(resized_img)
            print("\n")

    def search_text(
        self,
        query: str,
        *,
        top_n: int = 3,
        column_name: str = "text_embedding_chunk",
        chunk_text: bool = True,
    ) -> Dict[Any, Dict[str, Any]]:
        if not hasattr(self, "text_metadata_df") or self.text_metadata_df is None:
            raise RuntimeError("No text metadata loaded. Run process() or load_cache() first.")
        return self.get_similar_text_from_query(
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
        if (
            not hasattr(self, "text_metadata_df")
            or self.text_metadata_df is None
            or not hasattr(self, "image_metadata_df")
            or self.image_metadata_df is None
        ):
            raise RuntimeError("No metadata loaded. Run process() or load_cache() first.")
        return self.get_similar_image_from_query(
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
        if (
            not hasattr(self, "text_metadata_df")
            or self.text_metadata_df is None
            or not hasattr(self, "image_metadata_df")
            or self.image_metadata_df is None
        ):
            raise RuntimeError("No metadata loaded. Run process() or load_cache() first.")
        return self.get_similar_image_from_query(
            self.text_metadata_df,  # type: ignore
            self.image_metadata_df,  # type: ignore
            query=query,
            column_name=column_name,
            image_emb=True,
            image_query_path=image_query_path,
            top_n=top_n,
            embedding_size=embedding_size,
        )

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

        response = self.get_gemini_response(
            self.generative_multimodal_model,
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

        response = self.get_gemini_response(
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

        response = self.get_gemini_response(
            self.multimodal_model_flash,
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

    def _jsonify_cell(self, x: Any) -> Any:
        if isinstance(x, (list, dict)):
            return json.dumps(x)
        return x

    def save_cache(self, cache_dir: str):
        """
        Save current text/image metadata to pickle + CSV, mirroring pipeline.save_cache().
        Requires that self.text_metadata_df and self.image_metadata_df are already populated.
        """
        if self.text_metadata_df is None or self.image_metadata_df is None:
            raise RuntimeError("No metadata to cache. Run process() or build_metadata() first.")

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        text_pkl = cache_path / "text_metadata_df.pkl"
        image_pkl = cache_path / "image_metadata_df.pkl"
        text_csv = cache_path / "text_metadata_df.csv"
        image_csv = cache_path / "image_metadata_df.csv"

        self.text_metadata_df.to_pickle(text_pkl)
        self.image_metadata_df.to_pickle(image_pkl)

        text_csv_df = self.text_metadata_df.copy()
        image_csv_df = self.image_metadata_df.copy()

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

    def load_cache(self, cache_dir: str):
        """
        Load cached metadata into the instance and rebuild image objects if needed.
        Returns (text_metadata_df, image_metadata_df) for convenience.
        """
        cache_path = Path(cache_dir)
        text_pkl = cache_path / "text_metadata_df.pkl"
        image_pkl = cache_path / "image_metadata_df.pkl"

        self.text_metadata_df = pd.read_pickle(text_pkl)
        self.image_metadata_df = pd.read_pickle(image_pkl)

        if "image_object" not in self.image_metadata_df.columns:
            self._rebuild_image_objects_from_paths()

        return self.text_metadata_df, self.image_metadata_df



    def _rebuild_image_objects_from_paths(self):
        """
        Rebuild PIL Image objects from stored image path column.
        We are flexible on the column name to match the metadata schema.
        """
        if self.image_metadata_df is None:
            return False

        path_col = None
        for candidate in ["img_path", "image_path", "path"]:
            if candidate in self.image_metadata_df.columns:
                path_col = candidate
                break

        if path_col is None:
            return False

        def _load_img(p: Any):
            try:
                return PIL.Image.open(p)
            except Exception:
                return None

        # Vectorized assignment avoids pandas' boolean indexing broadcasting issues
        self.image_metadata_df["image_object"] = self.image_metadata_df[path_col].map(
            _load_img
        )

        return True


if __name__ == "__main__":
    pipeline = Pipeline()

    # Either build cache once (uncomment next two lines) or assume it already exists.
    # pipeline.process(pdf_path="./test_data/test_pdf.pdf", cache_dir="./cache_test")
    # pipeline.save_cache(cache_dir="./cache_test")

    # Load cached metadata
    text_metadata_df, image_metadata_df = pipeline.load_cache(cache_dir="./cache_test")

    question = "What are the safety instructions for the machine?"

    # 1) Text-only answer
    text_result = pipeline.answer_text_query(
        query=question, top_n=3, temperature=0.2, stream=False
    )
    print("\n===== TEXT-ONLY ANSWER =====\n")
    print(text_result.get("response", ""))

    # 2) Multimodal answer (text + images used as context)
    mm_result = pipeline.answer_multimodal_query(
        query=question,
        top_n_text=3,
        top_n_images=3,
        temperature=0.2,
        stream=False,
        include_step_by_step=True,
    )
    print("\n===== MULTIMODAL ANSWER =====\n")
    print(mm_result.get("response", ""))

    # 3) Show which images were used (paths + short description), so you can open them manually.
    print("\n===== IMAGE CONTEXT (paths + descriptions) =====\n")
    for _, match in mm_result.get("image_matches", {}).items():
        img_path = match.get("img_path")
        desc = match.get("image_description", "")
        page = match.get("page_num") or match.get("page") or "?"
        if not img_path:
            continue
        print(f"- Page {page} | {img_path}")
        if desc:
            snippet = desc.replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            print(f"  desc: {snippet}")
        print()
