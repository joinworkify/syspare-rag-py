import { useState, useCallback } from 'react'

// API types (match rag_server.py QueryRequest / QueryResponse)
interface TextChunk {
  chunk_text: string
  score?: number | null
  page?: number | null
  doc?: string | null
}

interface ImageMatch {
  img_url: string
  caption: string
  score?: number | null
  page?: number | null
  doc?: string | null
}

interface QueryResponse {
  answer: string
  texts: TextChunk[]
  images: ImageMatch[]
}

const DEFAULT_QUESTION =
  'Every 2 years what should we do for the safety precautions of YM358A tractor?'

function Pill({
  doc,
  page,
  score,
}: {
  doc?: string | null
  page?: number | null
  score?: number | null
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {doc != null && doc !== '' && (
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-slate-200 text-slate-700">
          {doc}
        </span>
      )}
      {page != null && (
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-slate-200 text-slate-700">
          p{page}
        </span>
      )}
      {score != null && (
        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800">
          {score.toFixed(3)}
        </span>
      )}
    </div>
  )
}

export interface RagViewerProps {
  /** RAG API base URL (e.g. https://your-rag.onrender.com). Default: same origin or VITE_RAG_API_URL */
  apiBaseUrl?: string
}

export default function RagViewer({ apiBaseUrl }: RagViewerProps) {
  const [question, setQuestion] = useState(DEFAULT_QUESTION)
  const [topKText, setTopKText] = useState(5)
  const [topKImg, setTopKImg] = useState(6)
  const [temp, setTemp] = useState(0.5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [lightbox, setLightbox] = useState<{
    src: string
    caption: string
    doc?: string
    page?: string
  } | null>(null)

  const baseUrl =
    apiBaseUrl ??
    (import.meta.env.VITE_RAG_API_URL as string | undefined) ??
    ''

  const runQuery = useCallback(async () => {
    const q = question.trim()
    if (!q) {
      setError('Enter a question.')
      return
    }
    setError(null)
    setResult(null)
    setLoading(true)
    try {
      const res = await fetch(`${baseUrl}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({
          question: q,
          top_k_text: topKText,
          top_k_img: topKImg,
          temp,
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        setError(data.detail ?? data.error ?? `Error ${res.status}`)
        return
      }
      setResult(data as QueryResponse)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }, [question, topKText, topKImg, temp, baseUrl])

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8">
      {/* Header */}
      <header className="mb-8">
        <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">
          Multimodal RAG Viewer
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Query → retrieve text & images → Gemini answer (API)
        </p>
      </header>

      {/* Search card */}
      <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 mb-8">
        <div className="space-y-4">
          <div>
            <label htmlFor="rag-q" className="block text-sm font-medium text-slate-700 mb-1.5">
              Your question
            </label>
            <input
              id="rag-q"
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask something about the documents..."
              className="w-full px-4 py-3 rounded-lg border border-slate-300 focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20 outline-none transition text-slate-900 placeholder-slate-400"
            />
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label htmlFor="rag-top-k-text" className="block text-sm font-medium text-slate-600 mb-1">
                Top-K text
              </label>
              <input
                id="rag-top-k-text"
                type="number"
                value={topKText}
                onChange={(e) => setTopKText(parseInt(e.target.value, 10) || 5)}
                min={1}
                max={20}
                className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20 outline-none transition text-slate-900"
              />
            </div>
            <div>
              <label htmlFor="rag-top-k-img" className="block text-sm font-medium text-slate-600 mb-1">
                Top-K images
              </label>
              <input
                id="rag-top-k-img"
                type="number"
                value={topKImg}
                onChange={(e) => setTopKImg(parseInt(e.target.value, 10) || 6)}
                min={1}
                max={20}
                className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20 outline-none transition text-slate-900"
              />
            </div>
            <div>
              <label htmlFor="rag-temp" className="block text-sm font-medium text-slate-600 mb-1">
                Temperature
              </label>
              <input
                id="rag-temp"
                type="number"
                value={temp}
                onChange={(e) => setTemp(parseFloat(e.target.value) || 0.5)}
                min={0}
                max={2}
                step={0.1}
                className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20 outline-none transition text-slate-900"
              />
            </div>
          </div>
          <div className="pt-1 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={runQuery}
              disabled={loading}
              className="inline-flex items-center px-5 py-2.5 rounded-lg bg-emerald-600 text-white font-medium hover:bg-emerald-700 focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 transition disabled:opacity-70 disabled:pointer-events-none"
            >
              {loading ? 'Running…' : 'Run RAG'}
            </button>
          </div>
          {error && (
            <p className="text-sm text-red-600">{error}</p>
          )}
        </div>
      </section>

      {/* Results */}
      {result && (
        <>
          <div className="grid grid-cols-1 xl:grid-cols-5 gap-6 mb-8">
            <div className="xl:col-span-3 bg-white rounded-xl border border-slate-200 shadow-sm p-6">
              <h2 className="text-lg font-semibold text-slate-900 mb-3">Gemini Answer</h2>
              <div className="prose prose-slate max-w-none text-slate-700 leading-relaxed whitespace-pre-wrap break-words">
                {result.answer}
              </div>
            </div>
            <div className="xl:col-span-2 bg-white rounded-xl border border-slate-200 shadow-sm p-6">
              <h2 className="text-lg font-semibold text-slate-900 mb-3">Retrieved Images</h2>
              <p className="text-xs text-slate-500 mb-2">Click an image to view larger</p>
              <div className="grid grid-cols-2 gap-3">
                {result.images.map((img, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() =>
                      setLightbox({
                        src: img.img_url.startsWith('http') ? img.img_url : `${baseUrl}${img.img_url}`,
                        caption: img.caption,
                        doc: img.doc ?? undefined,
                        page: img.page != null ? String(img.page) : undefined,
                      })
                    }
                    className="text-left rounded-lg border border-slate-200 overflow-hidden bg-slate-50 hover:ring-2 hover:ring-emerald-500/50 hover:border-emerald-400 transition cursor-pointer focus:outline-none focus:ring-2 focus:ring-emerald-500 w-full"
                  >
                    <img
                      src={img.img_url.startsWith('http') ? img.img_url : `${baseUrl}${img.img_url}`}
                      alt=""
                      className="w-full aspect-auto object-cover pointer-events-none"
                      loading="lazy"
                    />
                    <div className="p-2 space-y-1">
                      <Pill doc={img.doc} page={img.page} score={img.score} />
                      <p
                        className="text-xs text-slate-500 overflow-hidden text-left line-clamp-2"
                        style={{
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                        }}
                      >
                        {img.caption}
                      </p>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
            <h2 className="text-lg font-semibold text-slate-900 mb-4">Retrieved Text Chunks</h2>
            <div className="space-y-4">
              {result.texts.map((t, i) => (
                <div
                  key={i}
                  className="border-b border-slate-100 last:border-0 pb-4 last:pb-0"
                >
                  <div className="mb-2">
                    <Pill doc={t.doc} page={t.page} score={t.score} />
                  </div>
                  <p className="text-sm text-slate-600 leading-relaxed whitespace-pre-wrap break-words">
                    {t.chunk_text}
                  </p>
                </div>
              ))}
            </div>
          </section>
        </>
      )}

      {/* Lightbox */}
      {lightbox && (
        <div
          className="fixed inset-0 z-50"
          role="dialog"
          aria-modal="true"
          aria-label="View image"
        >
          <div
            className="absolute inset-0 bg-black/80 backdrop-blur-sm"
            onClick={() => setLightbox(null)}
            aria-hidden
          />
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <div className="relative max-w-[90vw] max-h-[90vh] flex flex-col items-center">
              <button
                type="button"
                onClick={() => setLightbox(null)}
                className="absolute -top-10 right-0 flex items-center justify-center w-9 h-9 rounded-full bg-white/10 hover:bg-white/20 text-white transition z-10"
                aria-label="Close"
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
              <img
                src={lightbox.src}
                alt={lightbox.caption}
                className="max-w-full max-h-[85vh] w-auto h-auto object-contain rounded-lg shadow-2xl"
              />
              <div className="mt-3 px-4 py-2 max-w-2xl rounded-lg bg-white/95 text-slate-700 text-sm text-center">
                {[lightbox.doc, lightbox.page ? `p${lightbox.page}` : null, lightbox.caption]
                  .filter(Boolean)
                  .join(' · ')}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
