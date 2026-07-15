# GhostOS — Technical Report

> **Note on the numbers in this report:** I can read your codebase, but I can't run your actual GhostOS instance or your hardware from here — so the fields below split into two kinds: **known facts** (pulled directly from `config.py` / `ghostos_settings.json`) and **measurements you need to fill in** by running the commands provided. Submitting fabricated latency/memory numbers is worse than submitting real, modest ones — judges generally check.

---

## 1. Model and Runtime Used

| Role | Model | Runtime |
|---|---|---|
| Chat / generation | `gemma4:e2b` (configurable via `ghostos_settings.json`) | Ollama, local, via `/api/chat` (streamed) |
| Embeddings | `nomic-embed-text` (configurable) | Ollama, local, via `/api/embeddings` |
| OCR (optional) | Tesseract | Local binary, via `pytesseract` |
| Speech-to-text (optional) | `faster-whisper` | Local, fully offline |

Both core models are pulled and served through a local **Ollama** instance (`localhost:11434` by default) — no model weights or inference calls leave the machine.

---

## 2. Quantization / Optimization Techniques

**What's confirmed from the codebase:**
- Chat responses are **streamed token-by-token** (`stream_reply()` in `ai_agent.py`, `"stream": True` in the Ollama payload) rather than waiting for a full generation — this reduces perceived latency even though total generation time is unchanged.
- Retrieval avoids a third model call: instead of a neural reranker, `rerank()` in `vectorstore.py` uses a lexical/calibrated scoring function (vector score + BM25 + content/filename/phrase signals) — cheaper than running a cross-encoder reranker per query.
- The BM25 keyword search is pure-Python and brute-force over stored chunks — fine at prototype scale, but is the piece most likely to need optimization (an actual inverted index) as the corpus grows past the "tens of thousands of chunks" range the vector store is scoped for.

**What needs to be filled in — quantization level:**
Ollama models are typically distributed pre-quantized (commonly GGUF, various bit-widths). To report the actual quantization GhostOS is running:
```bash
ollama show gemma4:e2b
ollama show nomic-embed-text
```
This prints the quantization level (e.g. `Q4_K_M`) and parameter count directly — paste those values in here.

---

## 3. Model Size

Fill in from:
```bash
ollama list
```
This lists every pulled model with its on-disk size (e.g. `gemma4:e2b   5.4 GB`). Report both models' sizes here, plus total disk footprint of the Ollama models directory if relevant to your submission.

---

## 4. Inference Latency

Not yet measured. Suggested method — GhostOS already logs enough to time this, or you can measure directly against Ollama:

```bash
# Time a single chat completion end-to-end
curl -s -w "\n%{time_total}s\n" -X POST http://localhost:11434/api/chat \
  -d '{"model":"gemma4:e2b","messages":[{"role":"user","content":"What files did I open today?"}],"stream":false}' \
  -o /dev/null
```

Report, ideally averaged over 5–10 runs:

| Metric | Value |
|---|---|
| Time to first token | *(fill in)* |
| Total response time (short answer, ~50 tokens) | *(fill in)* |
| Tokens/sec (generation) | *(fill in)* |
| Embedding latency (single chunk) | *(fill in)* |
| End-to-end query latency (router → retrieval → generation) | *(fill in)* |

---

## 5. CPU / GPU / NPU Usage

Not yet measured. GhostOS's own `system_agent.py` already reads CPU/RAM via `psutil` — the simplest approach is to watch that panel in the UI while a query runs, or use OS tools directly:

- **Windows:** Task Manager → Performance tab, or `Get-Counter` in PowerShell, while a query is in flight.
- **GPU (if Ollama is using one):** `nvidia-smi` (NVIDIA) polled during inference, or check Ollama's own logs — Ollama prints whether it loaded a model onto GPU or fell back to CPU at startup.

Report:

| Resource | Idle | During inference |
|---|---|---|
| CPU usage | *(fill in)* | *(fill in)* |
| GPU usage (if applicable) | *(fill in)* | *(fill in)* |
| NPU usage (if applicable) | N/A unless explicitly targeting NPU-accelerated Ollama build | |

---

## 6. Peak Memory Usage

Not yet measured. Two things matter separately:

- **Ollama's own memory** (model weights loaded + KV cache) — visible via `ollama ps` while a model is loaded, or Task Manager for the `ollama` process.
- **GhostOS backend memory** (Flask process + in-memory state) — Task Manager for the `python app.py` process.

```bash
ollama ps
```
This shows currently loaded models and their memory footprint directly.

| Process | Peak memory |
|---|---|
| Ollama (model loaded) | *(fill in from `ollama ps`)* |
| GhostOS Flask backend | *(fill in from Task Manager)* |
| Combined peak (worst case, indexing + querying at once) | *(fill in)* |

---

## 7. Tested Device Specifications

Not yet filled in — this needs to describe the actual machine(s) GhostOS was tested on. Suggested fields:

| Spec | Value |
|---|---|
| OS | Windows *(version)* |
| CPU | *(model, cores/threads)* |
| RAM | *(total GB)* |
| GPU | *(model, VRAM, or "none / CPU-only")* |
| Storage | *(SSD/HDD, free space at test time)* |
| Ollama version | `ollama --version` |

If you tested on more than one machine (e.g. a lower-spec laptop to demonstrate it still runs), list both — that's a stronger submission than a single high-end result, since it speaks to real-world usability.

---

## 8. Summary

*(Fill in once the numbers above are in: a 2–3 sentence honest summary — e.g. "GhostOS runs entirely on CPU on a mid-range laptop, with response times of roughly Xs for typical queries and a peak memory footprint of around Y GB, dominated by the loaded chat model." Judges weigh an honest, modest number far more than an unverified impressive one.)*
