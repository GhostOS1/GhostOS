# GhostOS — Attribution

This document lists every pretrained model, dataset, library, API, and piece of pre-existing work GhostOS's implementation relies on, based directly on `requirements.txt`, `requirements-ocr.txt`, `requirements-voice.txt`, `config.py`, and the frontend source.

---

## 1. Pretrained Models

| Model | Role | Source |
|---|---|---|
| `gemma4:e2b` | Chat / answer generation | Served locally via [Ollama](https://ollama.com); default configurable in `ghostos_settings.json` |
| `nomic-embed-text` | Text embeddings for semantic search | Served locally via Ollama; default configurable in `ghostos_settings.json` |

Both models are used purely for **inference** — GhostOS does not fine-tune, retrain, or modify either model's weights. All model-specific licensing terms are set by their respective publishers, not by GhostOS; check Ollama's model listing for each model's license before redistribution.

---

## 2. Datasets

**None.** GhostOS does not train or fine-tune any model, and does not ship, bundle, or depend on any third-party dataset. The only "data" GhostOS works with is the end user's own local files, browser history, and activity — generated at runtime, not a pre-existing dataset.

---

## 3. Runtime / Inference Engine

| Component | Role |
|---|---|
| [Ollama](https://ollama.com) | Local LLM/embedding serving runtime — GhostOS talks to it over `localhost` via its REST API (`/api/chat`, `/api/embeddings`, `/api/tags`) |

---

## 4. Backend Libraries (Python)

From `requirements.txt`:

| Library | Version pinned | Purpose |
|---|---|---|
| Flask | 3.0.3 | Local HTTP backend/API server |
| requests | 2.32.3 | HTTP client — used only to talk to local Ollama |
| pypdf | 4.3.1 | PDF text extraction for indexing |
| python-docx | 1.1.2 | Word document text extraction for indexing |
| numpy | ≥1.26 | Vector math for cosine similarity search |
| watchdog | 4.0.2 | Local filesystem change monitoring |
| psutil | 6.0.0 | CPU/RAM/disk/battery telemetry for System Agent |

From `requirements-ocr.txt` (optional):

| Library | Version pinned | Purpose |
|---|---|---|
| pytesseract | 0.3.13 | Python wrapper for the Tesseract OCR engine |
| Pillow | ≥10.4, <13 | Image handling for OCR |
| PyMuPDF | ≥1.24, <2 | PDF rasterization for scanned-document OCR |

From `requirements-voice.txt` (optional):

| Library | Version pinned | Purpose |
|---|---|---|
| faster-whisper | 1.0.3 | Local, offline speech-to-text |

**External binary dependency (not a Python package):** [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — the actual OCR engine `pytesseract` wraps. Installed separately by the user; GhostOS never downloads it automatically (documented in `README.md`).

---

## 5. Frontend

- **No external JS/CSS frameworks are loaded.** `index.html` is a single, dependency-free file — no CDN script tags, no bundler output.
- **Icon system:** the frontend uses a hand-rolled, dependency-free SVG icon renderer, using the `data-lucide="..."` attribute naming convention popularized by the [Lucide](https://lucide.dev) icon set for developer familiarity — but it does **not** import the Lucide package or its icon SVGs; icons are implemented locally (see the comment in `index.html`: *"Dependency-free local line icons"*).
- **Utility CSS classes** (Tailwind-style naming, e.g. `flex`, `gap-2`) are hand-written in this project rather than loaded from the Tailwind CDN, specifically so the UI works fully offline.

---

## 6. APIs

**None external.** The only API GhostOS's backend calls is its own local Ollama instance (`http://localhost:11434`) — see [`LOCAL_AI_VERIFICATION.md`](LOCAL_AI_VERIFICATION.md) for the full network audit. No third-party cloud APIs (search, translation, analytics, etc.) are integrated.

---

## 7. Pre-Existing Work / Prior Art Referenced

- **Reciprocal Rank Fusion (RRF)** — the fusion technique used in `vectorstore.py`'s `hybrid_search()` to combine dense and BM25 rankings is drawn from the original RRF paper (Cormack, Clarke, and Büttcher, *"Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods,"* SIGIR 2009) — cited directly in the code's own comments.
- **BM25** — the keyword-ranking algorithm implemented in `bm25_search()` is a standard, well-established information-retrieval ranking function (Robertson & Zaragoza et al.); GhostOS's implementation is a from-scratch pure-Python version, not a wrapped third-party BM25 library.
- **Conceptual comparison point:** GhostOS's problem framing (ambient local memory of screen/file/browser activity) is explicitly positioned in relation to existing tools like Windows Recall and Rewind — referenced for context in `README.md`, not used as source code or a dependency.

---

## 8. Summary

Everything GhostOS runs on is either (a) a pretrained model served locally through Ollama, (b) a standard, permissively-licensed open-source Python/library dependency, or (c) code written for this project. No proprietary third-party API, no training dataset, and no cloud service is part of the system.
