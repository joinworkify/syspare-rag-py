/**
 * Myanmar RAG Viewer — uses POST /api/query-myanmar
 * Myanmar input -> Gemini rewrite to English -> RAG -> English -> Gemini to Myanmar
 */
function el(id) {
  return document.getElementById(id);
}
function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}
function renderAnswerHtml(answer) {
  const lines = (answer || '').split(/\r?\n/);
  let html = '';
  let listType = null;
  const closeList = () => {
    if (listType) {
      html += `</${listType}>`;
      listType = null;
    }
  };
  const openList = (type) => {
    if (listType === type) return;
    closeList();
    const cls =
      type === 'ol'
        ? 'list-decimal pl-5 my-2 space-y-1'
        : 'list-disc pl-5 my-2 space-y-1';
    html += `<${type} class="${cls}">`;
    listType = type;
  };
  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      closeList();
      return;
    }
    const ordered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (ordered) {
      openList('ol');
      html += `<li>${escapeHtml(ordered[1])}</li>`;
      return;
    }
    const bullet = trimmed.match(/^[-*•]\s+(.+)$/);
    if (bullet) {
      openList('ul');
      html += `<li>${escapeHtml(bullet[1])}</li>`;
      return;
    }
    closeList();
    const safety = trimmed.match(/^Safety note:\s*(.*)$/i);
    if (safety) {
      html += `<p class="mt-3"><strong>Safety note:</strong> ${escapeHtml(safety[1])}</p>`;
    } else {
      html += `<p class="my-1">${escapeHtml(trimmed)}</p>`;
    }
  });
  closeList();
  return html;
}
function renderPills(doc, page, score) {
  const parts = [];
  if (doc)
    parts.push(
      '<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-slate-200 text-slate-700">' +
        escapeHtml(doc) +
        '</span>'
    );
  if (page != null)
    parts.push(
      '<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-slate-200 text-slate-700">p' +
        page +
        '</span>'
    );
  if (score != null)
    parts.push(
      '<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800">' +
        score.toFixed(3) +
        '</span>'
    );
  return parts.join('\n');
}
function renderTextChunk(t) {
  return (
    '\n<div class="border-b border-slate-100 last:border-0 pb-4 last:pb-0">' +
    '\n  <div class="flex flex-wrap gap-2 mb-2">' +
    renderPills(t.doc, t.page, t.score ?? undefined) +
    '\n  </div>' +
    '\n  <p class="text-sm text-slate-600 leading-relaxed whitespace-pre-wrap break-words">' +
    escapeHtml(t.chunk_text) +
    '\n  </p>' +
    '\n</div>'
  );
}
function renderImage(img) {
  const captionShort =
    img.caption && img.caption.length > 80 ? img.caption.slice(0, 80) + '…' : img.caption;
  return (
    '\n<button type="button" class="rag-image-thumb text-left rounded-lg border border-slate-200 overflow-hidden bg-slate-50 hover:ring-2 hover:ring-emerald-500/50 hover:border-emerald-400 transition cursor-pointer focus:outline-none focus:ring-2 focus:ring-emerald-500 w-full"' +
    '\n  data-src="' +
    escapeHtml(img.img_url) +
    '"' +
    '\n  data-caption="' +
    escapeHtml(img.caption) +
    '"' +
    '\n  data-doc="' +
    escapeHtml(img.doc ?? '') +
    '"' +
    '\n  data-page="' +
    (img.page ?? '') +
    '"' +
    '\n>' +
    '\n  <img src="' +
    escapeHtml(img.img_url) +
    '" alt="" class="w-full aspect-auto object-cover pointer-events-none" loading="lazy" />' +
    '\n  <div class="p-2 space-y-1">' +
    '\n    <div class="flex flex-wrap gap-1.5">' +
    renderPills(img.doc, img.page, img.score ?? undefined) +
    '\n    </div>' +
    '\n    <p class="text-xs text-slate-500 overflow-hidden text-left line-clamp-2">' +
    escapeHtml(captionShort || '') +
    '\n    </p>' +
    '\n  </div>' +
    '\n</button>'
  );
}
function renderResults(data) {
  const answerEl = el('api-answer');
  const imagesEl = el('api-images');
  const textsEl = el('api-texts');
  const resultsEl = el('api-results');
  const debugEl = el('api-english-debug');
  const debugQueryEl = el('api-english-query');
  const debugAnswerEl = el('api-english-answer');
  if (!answerEl || !imagesEl || !textsEl || !resultsEl) return;

  answerEl.innerHTML = '';
  answerEl.innerHTML = renderAnswerHtml(data.answer ?? '');
  imagesEl.innerHTML = (data.images || []).map((img) => renderImage(img)).join('');
  textsEl.innerHTML = (data.texts || []).map(renderTextChunk).join('');

  if (debugEl && debugQueryEl && debugAnswerEl) {
    const show = !!data.english_query || !!data.english_answer;
    debugEl.classList.toggle('hidden', !show);
    debugQueryEl.textContent = data.english_query ?? '';
    debugAnswerEl.textContent = data.english_answer ?? '';
  }

  resultsEl.classList.remove('hidden');
  setupLightbox();
}
function setupLightbox() {
  const lightbox = el('image-lightbox');
  const img = el('lightbox-img');
  const captionEl = el('lightbox-caption');
  const closeBtn = el('lightbox-close');
  const backdrop = el('lightbox-backdrop');
  const container = el('api-images');
  if (!lightbox || !img || !captionEl || !closeBtn || !backdrop || !container) return;
  function openLightbox(src, caption, doc, page) {
    img.src = src;
    img.alt = caption || 'Retrieved image';
    const parts = [];
    if (doc) parts.push(doc);
    if (page) parts.push('p' + page);
    if (caption) parts.push(caption);
    captionEl.textContent = parts.join(' · ');
    lightbox.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
  }
  function closeLightbox() {
    lightbox.classList.add('hidden');
    document.body.style.overflow = '';
  }
  container.addEventListener('click', function (e) {
    const btn = e.target.closest('.rag-image-thumb');
    if (!btn) return;
    e.preventDefault();
    openLightbox(
      btn.getAttribute('data-src') || '',
      btn.getAttribute('data-caption') || '',
      btn.getAttribute('data-doc') || '',
      btn.getAttribute('data-page') || ''
    );
  });
  closeBtn.onclick = () => closeLightbox();
  backdrop.onclick = () => closeLightbox();
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && !lightbox.classList.contains('hidden')) closeLightbox();
  });
}
async function runQuery() {
  const questionInput = el('api-q');
  const topKTextInput = el('api-top-k-text');
  const topKImgInput = el('api-top-k-img');
  const tempInput = el('api-temp');
  const includeIntermediateInput = el('api-include-intermediate-english');
  const runBtn = el('api-run-btn');
  const statusEl = el('api-status');
  const resultsEl = el('api-results');
  if (!questionInput || !runBtn || !statusEl) return;

  const question = questionInput.value.trim();
  if (!question) {
    statusEl.textContent = 'Enter a Myanmar question.';
    statusEl.style.color = '#b91c1c';
    return;
  }

  const payload = {
    question,
    top_k_text: topKTextInput ? parseInt(topKTextInput.value, 10) || 5 : 5,
    top_k_img: topKImgInput ? parseInt(topKImgInput.value, 10) || 6 : 6,
    temp: tempInput ? parseFloat(tempInput.value) || 0.5 : 0.5,
    include_intermediate_english:
      includeIntermediateInput && 'checked' in includeIntermediateInput
        ? !!includeIntermediateInput.checked
        : false,
  };

  runBtn.setAttribute('disabled', 'true');
  runBtn.classList.add('opacity-70');
  statusEl.textContent = 'Searching the selected manual…';
  statusEl.style.color = '#64748b';
  const statusSteps = [
    'Searching the selected manual…',
    'Checking retrieved clips and diagrams…',
    'If the first pass is dealer-only, searching wider…',
  ];
  let statusStepIndex = 0;
  const statusTimer = setInterval(() => {
    statusStepIndex = (statusStepIndex + 1) % statusSteps.length;
    statusEl.textContent = statusSteps[statusStepIndex];
  }, 1800);
  if (resultsEl) resultsEl.classList.add('hidden');

  try {
    const res = await fetch('/api/query-myanmar', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      statusEl.textContent = data.detail ?? data.error ?? 'Error ' + res.status;
      statusEl.style.color = '#b91c1c';
      return;
    }
    if (data.retrieval_expanded) {
      statusEl.textContent =
        data.retrieval_message ??
        'First pass had insufficient detail, so retrieval was expanded before answering.';
      await new Promise((resolve) => setTimeout(resolve, 900));
    }
    renderResults(data);
    statusEl.textContent = '';
  } catch (err) {
    statusEl.textContent = err instanceof Error ? err.message : 'Request failed';
    statusEl.style.color = '#b91c1c';
  } finally {
    clearInterval(statusTimer);
    runBtn.removeAttribute('disabled');
    runBtn.classList.remove('opacity-70');
  }
}
function init() {
  const runBtn = el('api-run-btn');
  if (runBtn) runBtn.addEventListener('click', () => runQuery());
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

