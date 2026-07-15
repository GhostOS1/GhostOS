# GhostOS — Local AI Verification

This document states plainly what runs on-device, what (if anything) touches the internet, and whether any user data ever leaves the machine — and shows how each claim can be independently checked against the source.

---

## 1. What Runs Fully On-Device

| Component | Runs where |
|---|---|
| Flask backend (`app.py`) | Local process |
| Frontend (`index.html`) | Loaded from local disk into the browser, no CDN dependency |
| Embedding generation (`nomic-embed-text`) | Local Ollama instance |
| Chat generation (`gemma4:e2b`) | Local Ollama instance |
| Vector search (dense + BM25 + RRF + rerank) | Local SQLite + in-process Python (`vectorstore.py`) |
| File indexing (`indexer.py`) | Local process, reads local disk only |
| File watching (`watcher.py`) | Local process (`watchdog`) |
| Browser history/bookmarks/downloads (`browser_connector.py`) | Reads local Chrome/Edge profile files directly from disk |
| Activity tracking (`activity_tracker.py`) | Local OS APIs (foreground window polling) |
| OCR (`ocr_service.py`, optional) | Local Tesseract binary |
| Speech-to-text (`voice_service.py`, optional) | Local `faster-whisper`, runs fully offline |
| System monitoring (`system_agent.py`) | Local `psutil` |

**All indexed content, embeddings, conversation history, and model inference happen on the machine GhostOS is installed on.**

---

## 2. What Requires Internet

Internet access is needed at exactly two points, both one-time/setup-only, never during normal use:

1. **Installing Ollama itself** — a one-time download of the Ollama runtime.
2. **Pulling the models** (`ollama pull gemma4:e2b`, `ollama pull nomic-embed-text`) — a one-time download so the model weights exist locally. After this, Ollama serves them entirely offline.
3. *(Optional)* Installing the Tesseract OCR binary, if OCR support is enabled — also a one-time download, and GhostOS never triggers this download itself (the README makes clear it must be installed separately).

**After setup, GhostOS runs with no internet connection required.** You can disconnect from the network entirely and it continues to index, search, and answer questions.

---

## 3. Does Any User Data Leave the Device?

**No.** This was verified by searching the entire backend source for outbound network calls:

```bash
grep -rn "requests\.\(get\|post\)" --include="*.py" .
```

This returns exactly three real outbound calls in the whole codebase, and all three target the local Ollama server:

| File | Call | Destination |
|---|---|---|
| `embeddings.py` | `requests.post(...)` | `http://localhost:11434/api/embeddings` (local Ollama) |
| `ai_agent.py` | `requests.post(...)` | `http://localhost:11434/api/chat` (local Ollama) |
| `diagnostics.py` | `requests.get(...)` | `http://localhost:11434/api/tags` (local Ollama health check) |

Every other `https://` string that shows up in the codebase is either:
- Inside `test_*.py` files (dummy/example URLs used in unit tests, e.g. `example.test`, `example.invalid` — never actually contacted), or
- Part of **URL parsing/handling logic** (e.g. validating a URL before the `open_url` action asks the OS to open it in the user's own browser, or reading `path_or_url` out of the user's own local browser history database) — GhostOS reads or validates these strings, it does not send data to them.

**No telemetry, no analytics, no crash reporting, no account/login system exists in the codebase.**

---

## 4. How to Verify This Yourself

You don't have to take this document's word for it — two independent checks:

1. **Source-level check** (what this document is based on):
   ```bash
   grep -rn "requests\." --include="*.py" .
   ```
   Confirm every hit resolves to `localhost`/`127.0.0.1`, is inside a `test_*.py` file, or is string-handling logic rather than an actual request.

2. **Runtime check** (empirical, no code reading required):
   - Disconnect the machine from the internet (or use a network monitor like Wireshark / Windows Resource Monitor's network tab).
   - Use GhostOS normally — ask it questions, index files, browse the timeline.
   - Confirm no outbound traffic appears other than loopback traffic to `localhost:11434` (Ollama).

---

## 5. Known Caveats (Stated Plainly)

- If a future version of GhostOS adds an optional cloud model fallback or update-checker, that would need to be a clearly-labeled opt-in and this document would need to be revised — it currently reflects the codebase as of this writing, not a permanent guarantee.
- Browser history reading is **local-only by design**, but it does mean GhostOS has access to the same locally-stored browsing data any local application with disk access could read (see [`PRIVACY_AND_SAFETY.md`](PRIVACY_AND_SAFETY.md) for the full discussion of that tradeoff).
