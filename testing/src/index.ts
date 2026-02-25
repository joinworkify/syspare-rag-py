import fs from "node:fs/promises";
import path from "node:path";
import pLimit from "p-limit";
import cliProgress from "cli-progress";
import { PDFDocument } from "pdf-lib";
import { GoogleGenAI } from "@google/genai";
import { GoogleAuth } from "google-auth-library";

// pdfjs + canvas (Node)
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";
import { createCanvas } from "canvas";

/* =========================
   Types
========================= */
type Metadata = {
    total_pages: number;
    title?: string;
    author?: string;
    creation_date?: string;
};

type TextRow = {
    file_name: string;
    page_num: number; // 1-based
    text: string;
    chunk_number: number;
    chunk_text: string;
    text_embedding_chunk: number[];
};

type ImageRow = {
    file_name: string;
    page_num: number; // 1-based
    img_num: number;
    img_path: string; // rendered page PNG
    img_desc: string;
    mm_embedding_from_img_only: number[];
    text_embedding_from_image_description: number[];
};

type SearchTextResult = TextRow & { score: number };
type SearchImageResult = ImageRow & { score: number };

/* =========================
   Config
========================= */
const PROJECT_ID = process.env.GOOGLE_CLOUD_PROJECT || "fortunaii";
const LOCATION = process.env.GOOGLE_CLOUD_LOCATION || "us-central1";

const ai = new GoogleGenAI({
    vertexai: true,
    project: PROJECT_ID,
    location: LOCATION,
});

// Paste your full long prompt here if you want exact behavior.
// Keeping it short here to reduce noise; replace with your full prompt.
const PDF_TO_MD_SYSTEM_PROMPT = "You are tasked with converting a PDF document to text in markdown format. Your goal is to accurately represent the content, structure, and layout of the original PDF while using markdown syntax. Follow these instructions carefully: To convert this PDF content to markdown format, follow these steps: 1. Document Structure: - Preserve the overall structure of the document. - Use appropriate markdown syntax for headers, subheaders, and sections. - Maintain the original hierarchy of the document. 2. Text Formatting: - Convert basic text to plain markdown text. - Use markdown syntax for bold (**text**), italic (*text*), and strikethrough (~~text~~) where applicable. - Preserve any special characters or symbols as they appear in the original document. 3. Headers: - Use the appropriate number of hash symbols (#) to represent different header levels. - Example: # for H1, ## for H2, ### for H3, and so on. 4. Paragraphs: - Separate paragraphs with a blank line. - Preserve any indentation or special formatting within paragraphs. 5. Lists: - Use - for unordered lists and 1. 2. 3. for ordered lists. - Maintain the original indentation for nested lists. 6. Tables: - Convert tables to markdown table format. - Use | to separate columns and - to create the header row. - Align columns using : in the header row (e.g., |:---:| for center alignment). 7. Links: - Convert hyperlinks to markdown format: [link text](URL) 8. Images: - For each image/chart in the PDF, insert a placeholder in the following format: [Image Description] - Provide a full description of the image in place of \"Image Description\". - describe the image in detail, like you would describe it to a blind person. 9. Footnotes: - Use markdown footnote syntax: [^1] for the reference and [^1]: Footnote text for the footnote content. - Place all footnotes at the end of the document. 10. Code Blocks: - Use triple backticks (```) to enclose code blocks. - Specify the language after the opening backticks if applicable. 11. Blockquotes: - Use > to indicate blockquotes. - For nested blockquotes, use multiple > symbols. 12. Horizontal Rules: - Use three or more hyphens (---) on a line by themselves to create a horizontal rule. 13. Special Elements: - If there are any special elements in the PDF (e.g., mathematical equations, diagrams), describe them in plain text within square brackets. 14. Preserve Layout: - Maintain the original layout as much as possible, including line breaks and spacing. - Use empty lines and appropriate markdown syntax to recreate the visual structure of the document. Once you have converted the entire PDF content to markdown format. Ensure that all elements of the original document are accurately represented in the markdown version.";

const IMAGE_DESC_PROMPT = [
    "Explain what is going on in the image.",
    "If it's a table, extract all elements of the table.",
    "If it's a graph, explain the findings in the graph.",
    "Do not include any numbers that are not mentioned in the image.",
].join("\n");

/* =========================
   Utils
========================= */
function stripCodeFences(s: string): string {
    const t = (s || "").trim();
    if (!t.startsWith("```")) return t;
    return t.replace(/^```[a-zA-Z0-9_-]*\s*/m, "").replace(/```$/m, "").trim();
}

function extractFirstJsonObject(s: string): string | null {
    const m = s.match(/\{[\s\S]*\}/);
    return m ? m[0] : null;
}

function cosine(a: number[], b: number[]): number {
    let dot = 0,
        na = 0,
        nb = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na === 0 || nb === 0) return 0;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function chunkOverlapping(text: string, characterLimit = 1000, overlap = 100): string[] {
    if (overlap > characterLimit) throw new Error("overlap > characterLimit");
    const out: string[] = [];
    for (let i = 0; i < text.length; i += characterLimit - overlap) {
        const end = Math.min(i + characterLimit, text.length);
        out.push(text.slice(i, end));
    }
    return out;
}

async function fileExists(p: string) {
    try {
        await fs.stat(p);
        return true;
    } catch {
        return false;
    }
}

async function writeJsonl(filePath: string, rows: any[]) {
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    const data = rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
    await fs.writeFile(filePath, data, "utf8");
}

async function readJsonl<T>(filePath: string): Promise<T[]> {
    const txt = await fs.readFile(filePath, "utf8");
    return txt
        .split("\n")
        .map((l) => l.trim())
        .filter(Boolean)
        .map((l) => JSON.parse(l));
}

/* =========================
   Vertex Embeddings (REST)
========================= */
async function getAccessToken(): Promise<string> {
    const auth = new GoogleAuth({ scopes: ["https://www.googleapis.com/auth/cloud-platform"] });
    const client = await auth.getClient();
    const { token } = await client.getAccessToken();
    if (!token) throw new Error("No access token");
    return token;
}

async function embedText(text: string): Promise<number[]> {
    const token = await getAccessToken();
    const url = `https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/text-embedding-004:predict`;

    const body = { instances: [{ content: text }] };

    const res = await fetch(url, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });

    if (!res.ok) throw new Error(`embedText failed: ${res.status} ${await res.text()}`);
    const json: any = await res.json();

    const values: number[] | undefined = json?.predictions?.[0]?.embeddings?.values;
    if (!Array.isArray(values)) throw new Error("Unexpected embedText response format");
    return values;
}

async function embedImageBase64(pngBytesBase64: string, dimension = 128): Promise<number[]> {
    const token = await getAccessToken();
    const url = `https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/multimodalembedding:predict`;

    const body = {
        instances: [{ image: { bytesBase64Encoded: pngBytesBase64 } }],
        parameters: { dimension },
    };

    const res = await fetch(url, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });

    if (!res.ok) throw new Error(`embedImage failed: ${res.status} ${await res.text()}`);
    const json: any = await res.json();

    const values: number[] | undefined = json?.predictions?.[0]?.imageEmbedding;
    if (!Array.isArray(values)) throw new Error("Unexpected embedImage response format");
    return values;
}

/* =========================
   PDF helpers
========================= */
async function getPdfMetadata(pdfPath: string): Promise<Metadata> {
    const bytes = await fs.readFile(pdfPath);
    const doc = await PDFDocument.load(bytes);
    return { total_pages: doc.getPageCount(), title: "", author: "", creation_date: "" };
}

async function extractSinglePagePdfBytes(pdfPath: string, pageIndex0: number): Promise<Uint8Array> {
    const bytes = await fs.readFile(pdfPath);
    const src = await PDFDocument.load(bytes);
    const out = await PDFDocument.create();
    const [copied] = await out.copyPages(src, [pageIndex0]);
    out.addPage(copied);
    return await out.save();
}

async function renderPageToPng(pdfPath: string, pageIndex0: number, outPngPath: string, scale = 2.0): Promise<string> {
    const data = await fs.readFile(pdfPath);
    const loadingTask = pdfjsLib.getDocument({ data });
    const pdf = await loadingTask.promise;
    const page = await pdf.getPage(pageIndex0 + 1);

    const viewport = page.getViewport({ scale });
    const canvas = createCanvas(viewport.width, viewport.height);
    const ctx = canvas.getContext("2d");

    await page.render({ canvasContext: ctx as any, viewport } as any).promise;

    const png = canvas.toBuffer("image/png");
    await fs.mkdir(path.dirname(outPngPath), { recursive: true });
    await fs.writeFile(outPngPath, png);
    return outPngPath;
}

/* =========================
   Gemini calls
========================= */
async function geminiConvertPageToMarkdown(pagePdfBytes: Uint8Array): Promise<string> {
    const response = await ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: [
            {
                role: "user",
                parts: [
                    {
                        inlineData: {
                            data: Buffer.from(pagePdfBytes).toString("base64"),
                            mimeType: "application/pdf",
                        },
                    },
                    { text: "convert this file" },
                ],
            },
        ],
        config: {
            temperature: 0,
            topP: 0.95,
            maxOutputTokens: 8192,
            responseMimeType: "application/json",
            systemInstruction: [{ text: PDF_TO_MD_SYSTEM_PROMPT }],
        },
    });

    const raw = stripCodeFences(response.text || "");
    try {
        const obj = JSON.parse(raw);
        return obj?.page_content ?? raw;
    } catch {
        const maybe = extractFirstJsonObject(raw);
        if (maybe) {
            try {
                const obj = JSON.parse(maybe);
                return obj?.page_content ?? raw;
            } catch {
                return raw;
            }
        }
        return raw;
    }
}

async function geminiDescribeImage(pngBytes: Buffer): Promise<string> {
    const response = await ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: [
            {
                role: "user",
                parts: [
                    { text: IMAGE_DESC_PROMPT },
                    {
                        inlineData: {
                            data: pngBytes.toString("base64"),
                            mimeType: "image/png",
                        },
                    },
                ],
            },
        ],
        config: { temperature: 0.2, maxOutputTokens: 2048 },
    });

    return (response.text || "").trim();
}

/* =========================
   Indexing (per page)
========================= */
async function processOnePage(
    pdfPath: string,
    pageIndex0: number,
    baseMeta: Metadata,
    outRoot: string,
    fileName: string
): Promise<{ textRows: TextRow[]; imageRows: ImageRow[] }> {
    // 1) page PDF bytes -> markdown
    const pagePdfBytes = await extractSinglePagePdfBytes(pdfPath, pageIndex0);
    const mdText = await geminiConvertPageToMarkdown(pagePdfBytes);

    // 2) chunk + text embeddings
    const chunks = chunkOverlapping(mdText, 1000, 100);
    const chunkEmbeddings = await Promise.all(chunks.map((c) => embedText(c)));

    const textRows: TextRow[] = chunks.map((chunkText, idx) => ({
        file_name: fileName,
        page_num: pageIndex0 + 1,
        text: mdText,
        chunk_number: idx + 1,
        chunk_text: chunkText,
        text_embedding_chunk: chunkEmbeddings[idx],
    }));

    // 3) render page -> PNG, describe -> embeddings
    const pageOutDir = path.join(
        outRoot,
        path.parse(fileName).name,
        `page_${String(pageIndex0 + 1).padStart(3, "0")}`
    );
    const pngPath = path.join(pageOutDir, `page_${String(pageIndex0 + 1).padStart(3, "0")}.png`);
    await renderPageToPng(pdfPath, pageIndex0, pngPath, 2.0);

    const pngBytes = await fs.readFile(pngPath);
    const imgDesc = await geminiDescribeImage(Buffer.from(pngBytes));

    const imgEmbedding = await embedImageBase64(Buffer.from(pngBytes).toString("base64"), 128);
    const imgDescTextEmbedding = await embedText(imgDesc);

    const imageRows: ImageRow[] = [
        {
            file_name: fileName,
            page_num: pageIndex0 + 1,
            img_num: 1,
            img_path: pngPath,
            img_desc: imgDesc,
            mm_embedding_from_img_only: imgEmbedding,
            text_embedding_from_image_description: imgDescTextEmbedding,
        },
    ];

    return { textRows, imageRows };
}

/* =========================
   Build index with progress + cache (tqdm-like)
========================= */
async function buildIndexWithProgress(
    pdfPath: string,
    {
        outRoot = "./out_images",
        cacheDir = "./cache",
        maxWorkers = 4,
        useCache = true,
    }: { outRoot?: string; cacheDir?: string; maxWorkers?: number; useCache?: boolean } = {}
): Promise<{ textRows: TextRow[]; imageRows: ImageRow[]; meta: Metadata }> {
    const meta = await getPdfMetadata(pdfPath);
    const fileName = path.basename(pdfPath);

    const textCachePath = path.join(cacheDir, `${fileName}.text.jsonl`);
    const imageCachePath = path.join(cacheDir, `${fileName}.image.jsonl`);

    if (useCache && (await fileExists(textCachePath)) && (await fileExists(imageCachePath))) {
        const textRows = await readJsonl<TextRow>(textCachePath);
        const imageRows = await readJsonl<ImageRow>(imageCachePath);
        console.log(`Loaded cache: text=${textRows.length}, images=${imageRows.length}`);
        return { textRows, imageRows, meta };
    }

    const total = meta.total_pages;
    const limit = pLimit(maxWorkers);

    const bar = new cliProgress.SingleBar(
        { format: "Indexing |{bar}| {value}/{total} pages | ETA: {eta}s" },
        cliProgress.Presets.shades_classic
    );
    bar.start(total, 0);

    const perPage: Array<{ page: number; textRows: TextRow[]; imageRows: ImageRow[] }> = [];

    const tasks = Array.from({ length: total }, (_, p) =>
        limit(async () => {
            const r = await processOnePage(pdfPath, p, meta, outRoot, fileName);
            perPage.push({ page: p, ...r });
            bar.increment();
        })
    );

    await Promise.all(tasks);
    bar.stop();

    perPage.sort((a, b) => a.page - b.page);

    const textRows = perPage.flatMap((r) => r.textRows);
    const imageRows = perPage.flatMap((r) => r.imageRows);

    await writeJsonl(textCachePath, textRows);
    await writeJsonl(imageCachePath, imageRows);
    console.log(`Saved cache: ${cacheDir}`);

    return { textRows, imageRows, meta };
}

/* =========================
   Retrieval
========================= */
async function searchText(query: string, textRows: TextRow[], topN = 3): Promise<SearchTextResult[]> {
    const qEmb = await embedText(query);
    const scored = textRows.map((r) => ({ ...r, score: cosine(r.text_embedding_chunk, qEmb) }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topN);
}

async function searchImagesByDescription(query: string, imageRows: ImageRow[], topN = 3): Promise<SearchImageResult[]> {
    const qEmb = await embedText(query);
    const scored = imageRows.map((r) => ({ ...r, score: cosine(r.text_embedding_from_image_description, qEmb) }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topN);
}

async function searchImagesByImageEmbedding(imagePath: string, imageRows: ImageRow[], topN = 3): Promise<SearchImageResult[]> {
    const pngBytes = await fs.readFile(imagePath);
    const qEmb = await embedImageBase64(Buffer.from(pngBytes).toString("base64"), 128);
    const scored = imageRows.map((r) => ({ ...r, score: cosine(r.mm_embedding_from_img_only, qEmb) }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topN);
}

/* =========================
   ✅ TEST 1B — Ask question + OPTIONAL input image
   Output: answer text + top matched image (paths printed)
========================= */
async function testQuestionWithOptionalInputImage(
    query: string,
    textRows: TextRow[],
    imageRows: ImageRow[],
    opts?: { imageQueryPath?: string; topNText?: number; topNImg?: number }
) {
    const topNText = opts?.topNText ?? 5;
    const topNImg = opts?.topNImg ?? 1;

    const topText = await searchText(query, textRows, topNText);

    const topImages = opts?.imageQueryPath
        ? await searchImagesByImageEmbedding(opts.imageQueryPath, imageRows, topNImg)
        : await searchImagesByDescription(query, imageRows, topNImg);

    const context = topText.map((t) => t.chunk_text).join("\n");

    const instruction = [
        "Answer the question with the given context.",
        'If the information is not available in the context, just return "not available in the context".',
        `Question: ${query}`,
        `Context: ${context}`,
        "Answer:",
    ].join("\n");

    const response = await ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: instruction }] }],
        config: { temperature: 0.2, maxOutputTokens: 1024 },
    });

    console.log("\n=== QUESTION ===\n", query);

    if (opts?.imageQueryPath) {
        console.log("\n=== INPUT IMAGE PATH ===\n", opts.imageQueryPath);
    }

    console.log("\n=== ANSWER (TEXT) ===\n", (response.text || "").trim());

    if (topImages.length > 0) {
        const im = topImages[0];
        console.log("\n=== TOP MATCHED IMAGE ===");
        console.log("score:", im.score);
        console.log("file:", im.file_name, "| page:", im.page_num);
        console.log("path:", im.img_path);
        console.log("desc:", im.img_desc);
    } else {
        console.log("\n(No matched image found.)");
    }

    return { topText, topImages, answer: (response.text || "").trim() };
}

/* =========================
   Main runner
========================= */
async function main() {
    const pdfPath = "../data/ym358a.pdf";

    // Build (index) with progress bar + cache
    const { textRows, imageRows } = await buildIndexWithProgress(pdfPath, {
        maxWorkers: 4,
        useCache: true,
        cacheDir: "./cache_ts/",
        outRoot: "./cache_ts/out_images",
    });

    // --- Run WITHOUT image input ---
    await testQuestionWithOptionalInputImage(
        "What should we do when the tractor gets stuck in a muddy portion in field?",
        textRows,
        imageRows,
        { topNText: 5, topNImg: 1 }
    );

    // --- Run WITH image input ---
    // Use ANY PNG you want (your own query image, or one of the rendered pages)
    const someRenderedPage = "./cache_ts/out_images/ym358a/page_011/page_011.png"; // adjust to real existing path
    if (await fileExists(someRenderedPage)) {
        await testQuestionWithOptionalInputImage(
            "What does this image show and how does it relate to the document?",
            textRows,
            imageRows,
            { imageQueryPath: someRenderedPage, topNText: 5, topNImg: 1 }
        );
    } else {
        console.log("\n(SKIP image-input test: file not found)", someRenderedPage);
    }
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});
