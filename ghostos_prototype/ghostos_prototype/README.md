# GhostOS — Frontend wired to the real backend

This step connects the exact-mockup UI to the actual local RAG backend, so the
Files & Data, Timeline, and stats you see are real — not placeholder numbers.

## What changed

**Backend (`backend/`)**
- `vectorstore.py` — added two new tables:
  - `files`: one row per real file on disk (path, category, collection, size, modified time) — the source for Categories, AI Collections, Recent Files, and the Storage donut.
  - `events`: a flat activity log — the source for the Timeline screen. Populated by the indexer (file cataloguing) and `browser_connector.py` (page visits).
- `indexer.py` — now catalogues **every** file it walks (not just the text-extractable ones). Text-extractable types (`.txt .md .pdf .docx .py .js .json .csv`) still get chunked + embedded for RAG search; everything else (images, videos, zips, etc.) is still recorded for the Files & Data view, just without embeddings. Added a simple keyword-based `classify_collection()` for "AI Collections" (Work / College / Finance / Personal / Projects) — a placeholder heuristic you can later swap for real LLM-based clustering.
- `browser_connector.py` — now also logs an `events` row per visited page, so browsing shows up on the Timeline.
- `app.py` — new endpoints:
  - `GET /api/categories`
  - `GET /api/collections`
  - `GET /api/recent-files?limit=N`
  - `GET /api/storage`
  - `GET /api/timeline?date=YYYY-MM-DD&limit=N`

All of the above were smoke-tested against a mixed sample folder (txt/csv/png/py files, plus a `.env` to confirm the sensitive blacklist still works) — cataloguing, categorization, and the timeline log all populate correctly even without Ollama running.

**Frontend (`frontend/index.html`)**
- On load, it now calls the backend (`http://localhost:5000`) for categories, collections, recent files, storage, and timeline data.
- If the backend isn't reachable, it silently falls back to the same demo data as before, and shows a **"Demo mode · backend not connected"** badge in the top bar so you always know which you're looking at. When connected, the badge reads **"Connected · live data"**.
- The **Connect to System** button now actually prompts for a folder path and calls `POST /api/index`, then refreshes the screen with the real results.

## How to run it

```
cd backend
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen2.5:14b
python app.py
```
Then open `frontend/index.html` directly in your browser.

Click **Connect to System**, enter a folder path, and watch the Files & Data
and Timeline screens fill in with your real files.

## Honest limitations (still true from the original prototype)

- Timeline is built from *what this backend actually observes*: files being
  indexed + browser history. It is not yet the live OCR/window-tracking
  "ambient capture" from the original manifesto — that's a separate future
  layer (Screen OCR / Audio / Clipboard services in the v2.0 architecture).
- "AI Collections" uses a simple keyword-in-path heuristic, not real semantic
  clustering — good enough to demo the UI, but should be revisited before
  relying on it.
- Brute-force vector search, single-user/single-machine — same caveats as
  before.