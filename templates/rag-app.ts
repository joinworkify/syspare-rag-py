/**
 * RAG Viewer (API version) — same UI as index.html but uses POST /api/query.
 * Compile: npx tsc (from repo root or templates/)
 */

interface TextChunk {
  chunk_text: string;
  score?: number | null;
  page?: number | null;
  doc?: string | null;
}

interface ImageMatch {
  img_url: string;
  caption: string;
  score?: number | null;
  page?: number | null;
  doc?: string | null;
}

interface QueryRequest {
  question: string;
  top_k_text?: number;
  top_k_img?: number;
  temp?: number;
}

interface QueryResponse {
  answer: string;
  texts: TextChunk[];
  images: ImageMatch[];
}

function el(id: string): HTMLElement | null {
  return document.getElementById(id);
}

function escapeHtml(s: string): string {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function renderPills(doc?: string | null, page?: number | null, score?: number | null): string {
  const parts: string[] = [];
  if (doc) parts.push(`<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-slate-200 text-slate-700">${escapeHtml(doc)}</span>`);
  if (page != null) parts.push(`<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-slate-200 text-slate-700">p${page}</span>`);
  if (score != null) parts.push(`<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800">${score.toFixed(3)}</span>`);
  return parts.join('\n');
}

function renderTextChunk(t: TextChunk): string {
  return `
    <div class="border-b border-slate-100 last:border-0 pb-4 last:pb-0">
      <div class="flex flex-wrap gap-2 mb-2">${renderPills(t.doc, t.page, t.score ?? undefined)}</div>
      <p class="text-sm text-slate-600 leading-relaxed whitespace-pre-wrap break-words">${escapeHtml(t.chunk_text)}</p>
    </div>`;
}

function renderImage(img: ImageMatch, index: number): string {
  const captionShort = img.caption.length > 80 ? img.caption.slice(0, 80) + '…' : img.caption;
  return `
    <button type="button" class="rag-image-thumb text-left rounded-lg border border-slate-200 overflow-hidden bg-slate-50 hover:ring-2 hover:ring-emerald-500/50 hover:border-emerald-400 transition cursor-pointer focus:outline-none focus:ring-2 focus:ring-emerald-500 w-full"
      data-src="${escapeHtml(img.img_url)}"
      data-caption="${escapeHtml(img.caption)}"
      data-doc="${escapeHtml(img.doc ?? '')}"
      data-page="${img.page ?? ''}">
      <img src="${escapeHtml(img.img_url)}" alt="" class="w-full aspect-auto object-cover pointer-events-none" loading="lazy" />
      <div class="p-2 space-y-1">
        <div class="flex flex-wrap gap-1.5">${renderPills(img.doc, img.page, img.score ?? undefined)}</div>
        <p class="text-xs text-slate-500 overflow-hidden text-left line-clamp-2">${escapeHtml(captionShort)}</p>
      </div>
    </button>`;
}

function renderResults(data: QueryResponse): void {
  const answerEl = el('api-answer');
  const imagesEl = el('api-images');
  const textsEl = el('api-texts');
  const resultsEl = el('api-results');
  if (!answerEl || !imagesEl || !textsEl || !resultsEl) return;

  answerEl.innerHTML = '';
  answerEl.appendChild(document.createTextNode(data.answer));

  imagesEl.innerHTML = data.images.map((img, i) => renderImage(img, i)).join('');

  textsEl.innerHTML = data.texts.map(renderTextChunk).join('');

  resultsEl.classList.remove('hidden');
  setupLightbox();
}

function setupLightbox(): void {
  const lightbox = el('image-lightbox');
  const img = el('lightbox-img') as HTMLImageElement | null;
  const captionEl = el('lightbox-caption');
  const closeBtn = el('lightbox-close');
  const backdrop = el('lightbox-backdrop');
  const container = el('api-images');

  if (!lightbox || !img || !captionEl || !closeBtn || !backdrop || !container) return;

  function openLightbox(src: string, caption: string, doc: string, page: string): void {
    img.src = src;
    img.alt = caption || 'Retrieved image';
    const parts: string[] = [];
    if (doc) parts.push(doc);
    if (page) parts.push('p' + page);
    if (caption) parts.push(caption);
    captionEl.textContent = parts.join(' · ');
    captionEl.style.display = parts.length ? 'block' : 'none';
    lightbox.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
  }

  function closeLightbox(): void {
    lightbox.classList.add('hidden');
    document.body.style.overflow = '';
  }

  container.addEventListener('click', (e: Event) => {
    const btn = (e.target as HTMLElement).closest('.rag-image-thumb');
    if (!btn) return;
    e.preventDefault();
    openLightbox(
      btn.getAttribute('data-src') ?? '',
      btn.getAttribute('data-caption') ?? '',
      btn.getAttribute('data-doc') ?? '',
      btn.getAttribute('data-page') ?? ''
    );
  });
  closeBtn.onclick = () => closeLightbox();
  backdrop.onclick = () => closeLightbox();
  document.addEventListener('keydown', (e: KeyboardEvent) => {
    if (e.key === 'Escape' && !lightbox.classList.contains('hidden')) closeLightbox();
  });
}

async function runQuery(): Promise<void> {
  const questionInput = el('api-q') as HTMLInputElement | null;
  const topKTextInput = el('api-top-k-text') as HTMLInputElement | null;
  const topKImgInput = el('api-top-k-img') as HTMLInputElement | null;
  const tempInput = el('api-temp') as HTMLInputElement | null;
  const runBtn = el('api-run-btn');
  const statusEl = el('api-status');
  const resultsEl = el('api-results');

  if (!questionInput || !runBtn || !statusEl) return;

  const question = questionInput.value.trim();
  if (!question) {
    statusEl.textContent = 'Enter a question.';
    statusEl.style.color = '#b91c1c';
    return;
  }

  const payload: QueryRequest = {
    question,
    top_k_text: topKTextInput ? parseInt(topKTextInput.value, 10) || 5 : 5,
    top_k_img: topKImgInput ? parseInt(topKImgInput.value, 10) || 6 : 6,
    temp: tempInput ? parseFloat(tempInput.value) || 0.5 : 0.5,
  };

  runBtn.setAttribute('disabled', 'true');
  runBtn.classList.add('opacity-70');
  statusEl.textContent = 'Running RAG…';
  statusEl.style.color = '#64748b';
  if (resultsEl) resultsEl.classList.add('hidden');

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      statusEl.textContent = data.detail ?? data.error ?? `Error ${res.status}`;
      statusEl.style.color = '#b91c1c';
      return;
    }

    renderResults(data as QueryResponse);
    statusEl.textContent = '';
  } catch (err) {
    statusEl.textContent = err instanceof Error ? err.message : 'Request failed';
    statusEl.style.color = '#b91c1c';
  } finally {
    runBtn.removeAttribute('disabled');
    runBtn.classList.remove('opacity-70');
  }
}

function init(): void {
  const runBtn = el('api-run-btn');
  runBtn?.addEventListener('click', () => runQuery());
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
