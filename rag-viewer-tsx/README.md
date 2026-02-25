# RAG Viewer (TSX)

Same UI as the main RAG viewer (`index.html`), built with React + TypeScript. Uses the RAG server **API** (`POST /api/query`) so you can run this app in another repo and point it at your deployed RAG server.

## In this repo

```bash
cd rag-viewer-tsx
npm install
npm run dev
```

Open http://localhost:5173. By default it calls the same origin; if the RAG server runs on another port, set `VITE_RAG_API_URL` (see below).

Build for production:

```bash
npm run build
```

Output is in `dist/`. You can deploy `dist/` to any static host or serve it from your RAG server.

---

## Use in another repo

1. **Copy the viewer**  
   Copy the whole `rag-viewer-tsx` folder into your other repo (e.g. `src/rag-viewer/` or `apps/rag-viewer/`).

2. **Install dependencies**  
   In the copied folder:
   ```bash
   npm install
   ```

3. **Set the RAG API URL**  
   Create `.env` (or set in your app’s env):
   ```bash
   VITE_RAG_API_URL=https://your-rag-server.onrender.com
   ```
   Omit or leave empty if the RAG server is on the same origin.

4. **Run or embed**  
   - **Standalone:** `npm run dev` then open http://localhost:5173.  
   - **Embed in your app:** Import the component:
     ```tsx
     import RagViewer from './rag-viewer-tsx/src/RagViewer'
     // ...
     <RagViewer apiBaseUrl="https://your-rag-server.onrender.com" />
     ```
     Or build and deploy the built files from `dist/` and load them in your app.

## API contract

The viewer calls:

- **POST** `{apiBaseUrl}/api/query`
- **Body:** `{ question: string, top_k_text?: number, top_k_img?: number, temp?: number }`
- **Response:** `{ answer: string, texts: TextChunk[], images: ImageMatch[] }`

Image URLs in the response may be relative (e.g. `/static/foo.jpeg`). The viewer resolves them against `apiBaseUrl` when the app is on a different origin.

## Props

- **`apiBaseUrl`** (optional): Base URL of the RAG server. Overrides `VITE_RAG_API_URL`. Use when embedding in another app.
