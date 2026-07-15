# GhostOS — Architecture

This document covers the system diagram, model pipeline, data flow, local/cloud component split, and the key design decisions behind GhostOS.

---

## 1. System Diagram

```mermaid
flowchart TB
    subgraph UI["Frontend - index.html (single file, offline)"]
        Chat[Chat panel]
        Files[Files browser]
        Timeline[Timeline view]
        Connect[Connect to System modal]
    end

    subgraph Backend["Flask backend - app.py"]
        Router[router.py - Intent Router]
        FilesAgent[Files Agent]
        TimelineAgent[Timeline Agent]
        BrowserAgent[Browser Agent]
        MemoryAgent[Memory Agent]
        SystemAgent[System Agent]
        AIAgent[AI Agent]
        ActionAgent[Action Agent + permissions]
    end

    subgraph Local["Local data + perception"]
        Indexer[indexer.py]
        Watcher[watcher.py - watchdog]
        VectorStore[(vectorstore.py - SQLite + numpy)]
        BrowserConn[browser_connector.py]
        Activity[activity_tracker.py]
        OCR[ocr_service.py - Tesseract, optional]
        Voice[voice_service.py - faster-whisper, optional]
    end

    subgraph Ollama["Local Ollama runtime"]
        Embed[nomic-embed-text]
        ChatModel[gemma4:e2b]
    end

    UI <--> Backend
    Router --> FilesAgent & TimelineAgent & BrowserAgent & MemoryAgent & SystemAgent & AIAgent
    Connect --> Indexer
    Indexer --> VectorStore
    Indexer --> Embed
    Watcher --> Indexer
    BrowserConn --> VectorStore
    Activity --> VectorStore
    MemoryAgent --> VectorStore
    MemoryAgent --> Embed
    AIAgent --> ChatModel
    ActionAgent -.validated actions.-> UI
    OCR -.optional text extraction.-> Indexer
    Voice -.optional transcription.-> Indexer

    style Ollama fill:#e8e0ff
    style Local fill:#e0f0ff
```

Everything below the frontend runs as local processes on the user's machine. There is no cloud tier in this diagram — see [§4](#4-localcloud-component-split) for why that's a hard boundary, not just a default.

---

## 2. Model Pipeline

GhostOS uses two local models via Ollama, plus a non-neural reranking step:

```mermaid
flowchart LR
    A[Raw content\nfiles / browser records / activity] --> B[Chunking]
    B --> C["nomic-embed-text\n(Ollama /api/embeddings)"]
    C --> D[(Vector stored in SQLite)]

    Q[User query] --> QE["nomic-embed-text\n(embed the query)"]
    QE --> Dense[Dense cosine similarity search]
    Q --> BM25[BM25 keyword search\npure-Python, brute-force over chunks]
    Dense --> RRF[Reciprocal Rank Fusion\nfuses dense + BM25 ranks]
    BM25 --> RRF
    RRF --> Rerank["Lexical rerank()\nvector score + BM25 score + content/filename/phrase signals"]
    Rerank --> Context[Top-k context chunks]
    Context --> Prompt[System prompt + context + rolling history]
    Prompt --> Gen["gemma4:e2b\n(Ollama /api/chat, streamed)"]
    Gen --> Answer[Answer streamed token-by-token to UI]
```

**Why RRF instead of averaging scores directly:** cosine similarity (roughly 0–1) and BM25 (unbounded) live on incompatible scales, so averaging them would let whichever score happens to be numerically larger dominate. RRF sidesteps that by only caring about each result's *rank* in each list, not its raw score.

**Why a lexical reranker instead of a neural one:** GhostOS's only two models are the embedder and the chat model — there's no dedicated reranking model in the stack. Instead, `rerank()` combines calibrated vector score, BM25 score, and content/filename/phrase-match signals into a final ranking. This keeps the pipeline to two Ollama calls per query (one embed, one chat) instead of three.

---

## 3. Data Flow

### 3.1 Ingestion (Connect to System → indexed)

```mermaid
flowchart TD
    A[Connect to System] --> B[Discover standard folders\n+ VS Code / git projects]
    B --> C[indexer.py walks each folder]
    C --> D{Blacklisted path\nor filename?}
    D -->|Yes| E[Skip - never read, never stored]
    D -->|No| F[Extract text: pypdf / python-docx / plain text\noptionally OCR for images/scans]
    F --> G[Chunk text]
    G --> H[Embed each chunk - nomic-embed-text]
    H --> I[(Store: chunk + embedding + source path + hash\nin SQLite via vectorstore.py)]
    I --> J[watcher.py starts watching the same folders]
```

### 3.2 Query (question → answer)

```mermaid
flowchart TD
    A[User message] --> B[router.py classify_intent]
    B --> C{Intent}
    C -->|Filename/folder lookup| D[Files Agent: direct SQLite lookup]
    C -->|"today / recent / yesterday"| E[Timeline Agent: reads events table]
    C -->|Browser-related| F[Browser Agent: filters browser_history_* rows]
    C -->|Semantic/general| G[Memory Agent: hybrid_search + rerank]
    C -->|System status| H[System Agent: psutil snapshot]
    D & E & F & G & H --> I[AI Agent builds prompt]
    I --> J[Ollama /api/chat - streamed]
    J --> K[Answer shown in UI]
```

### 3.3 Continuous background flow

- `watcher.py` (watchdog `Observer`) fires on file create/modify/delete inside indexed folders → re-runs the relevant slice of the ingestion pipeline.
- `activity_tracker.py` polls the OS foreground window and writes app-usage events straight into the timeline table.
- `browser_connector.py` periodically re-reads local Chrome/Edge history/bookmarks/downloads files and re-indexes new records.

None of these three loops make a network call at any point.

---

## 4. Local/Cloud Component Split

| Component | Where it runs | Network calls |
|---|---|---|
| Flask backend (`app.py`) | Local process | None |
| Frontend (`index.html`) | Local browser, loaded from disk | None (utility CSS hand-rolled, no CDN dependency) |
| Embedding (`nomic-embed-text`) | Local Ollama | `localhost` only (`/api/embeddings`) |
| Chat generation (`gemma4:e2b`) | Local Ollama | `localhost` only (`/api/chat`) |
| Vector store (SQLite + numpy) | Local disk | None |
| File indexer / watcher | Local process | None |
| Browser connector | Reads local Chrome/Edge profile files directly | None — no live tab access, no browser API calls |
| OCR (Tesseract, optional) | Local binary | None |
| Voice/STT (`faster-whisper`, optional) | Local, fully offline | None |
| System monitor (`psutil`) | Local process | None |

**There is no cloud tier.** Every component in GhostOS's current implementation runs on-device, including model inference — the only network traffic in the whole system is loopback traffic to the local Ollama server (`localhost:11434`). If a future feature ever needed a third-party API call (e.g. optional cloud model fallback), that would be a deliberate, clearly-labeled opt-in — not a default.

---

## 5. Key Design Decisions

**SQLite + numpy instead of a dedicated vector database.**
Keeps the dependency footprint minimal and the whole store inspectable with any SQLite browser. Explicitly scoped as "good for tens of thousands of chunks" — the codebase isolates vector operations behind `vectorstore.py` so swapping in sqlite-vec, LanceDB, or Chroma later doesn't require touching the agents.

**Hybrid search (dense + BM25) over dense-only retrieval.**
Dense embeddings are good at semantic similarity but can miss exact terms (a filename, an error code, a proper noun). BM25 catches those directly. Fusing with RRF avoids having to hand-tune a weighting between two differently-scaled signals.

**Pattern-matching Intent Router instead of an LLM-based router.**
`router.py` classifies intent with plain string/keyword matching, not a model call. This keeps routing instant and deterministic, and means a routing decision never depends on Ollama being warm or available — only the final generation step does.

**In-process file watcher instead of a separate watcher process.**
Earlier prototypes reportedly ran the watcher as a separate process synchronized via a JSON file, which could fall out of sync with the indexer. Running `watchdog` in-process inside the same Flask app removes that entire class of bug.

**Allowlisted actions instead of arbitrary command execution.**
`action_agent.py` only exposes a fixed set of pre-approved actions (open file/folder/URL/app, create note/folder) through `action_registry.py`, checked against `action_permissions.py` before anything touches the OS. The model can request an action, but it cannot execute arbitrary shell commands.

**Hardcoded sensitive-path blacklist, not an opt-in filter.**
The indexer blacklist (password managers, credential stores, known sensitive paths) runs underneath every indexing pass regardless of which folder the user selects — it's a floor, not a setting someone could accidentally leave off.

**OCR and voice are optional and degrade gracefully.**
Both `ocr_service.py` and `voice_service.py` check for their local dependency (Tesseract binary, `faster-whisper` package) at runtime and fall back cleanly if absent, rather than requiring them for GhostOS to run at all.

**System Agent is monitoring-only for now.**
`system_agent.py` currently only reads CPU/RAM/disk/battery via `psutil`. OS-level actions (shutdown, toggling Bluetooth, etc.) are deliberately not implemented yet, pending a dedicated permissions/safety pass — this was a scope decision, not an oversight.

---

## 6. Related Documents

- [`README.md`](README.md) — overview, setup, usage, screenshots
- Technical Report *(pending — see model/runtime, latency, and resource-usage details there)*
