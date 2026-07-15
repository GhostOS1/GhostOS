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

Measured via `ollama list` on the test device:

| Model | Size |
|---|---|
| `gemma4:e2b` (chat) | 7.2 GB |
| `nomic-embed-text:latest` (embeddings) | 274 MB |

**Total footprint for the two models GhostOS actually uses: ~7.47 GB.**

*(Other models present on this machine — `qwen2.5:14b`, `qwen2.5:3b`, `phi3`, `llama3` — are not used by GhostOS's default config and aren't counted toward its footprint; they're leftover from other local experimentation.)*

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

Confirmed via `ollama ps` while `gemma4:e2b` was loaded and running:

```
NAME          ID              SIZE      PROCESSOR    CONTEXT    UNTIL
gemma4:e2b    7fbdbf8f5e45    6.8 GB    100% CPU     4096       4 minutes from now
```

The `PROCESSOR` column shows **100% CPU** — meaning Ollama allocated the entire model to CPU, with **0% offloaded to GPU**. This confirms the earlier prediction in §7: the integrated AMD Radeon 610M is not being used for inference on this device. Context window is 4096 tokens.

| Resource | Value |
|---|---|
| Processor allocation | 100% CPU, 0% GPU (confirmed via `ollama ps`) |
| GPU usage | None — integrated AMD 610M not utilized by Ollama on this device |
| CPU usage (%) during inference | *(fill in — check Task Manager → Performance tab while asking GhostOS a question)* |
| NPU usage | N/A — not targeted |

---

## 6. Peak Memory Usage

Confirmed via `ollama ps` (see §5): **`gemma4:e2b` occupies 6.8 GB while loaded**, running entirely on CPU with a 4096-token context window.

| Process | Peak memory |
|---|---|
| Ollama (model loaded) | **6.8 GB** (confirmed via `ollama ps`) |
| GhostOS Flask backend | *(fill in — check Task Manager for the `python.exe` running `app.py`)* |
| Combined peak (worst case, indexing + querying at once) | *(fill in — add Flask backend's figure to the 6.8 GB above)* |

With 16 GB total RAM (13.8 GB usable) on the test device, the 6.8 GB model load leaves roughly 7 GB free for the OS, browser, and the GhostOS backend itself — workable, but worth noting as a real constraint for lower-RAM machines.

---

## 7. Tested Device Specifications

| Spec | Value |
|---|---|
| Device name | DESKTOP-Q7R670I |
| OS | Windows, 64-bit, x64-based processor |
| CPU | AMD Ryzen 5 7520U with Radeon Graphics, 2.80 GHz |
| RAM | 16.0 GB installed (13.8 GB usable) |
| GPU | AMD Radeon(TM) 610M (integrated, 2 GB) |
| Storage | 477 GB total, 326 GB used at test time |
| Ollama version | 0.31.1 |

**Note on GPU usage — confirmed, not just predicted:** `ollama ps` (see §5) shows `gemma4:e2b` running at **100% CPU / 0% GPU**, confirming the integrated AMD Radeon 610M is not used for inference on this device.

If you tested on more than one machine (e.g. a lower-spec laptop to demonstrate it still runs), list both — that's a stronger submission than a single high-end result, since it speaks to real-world usability.

---

## 8. Estimated Performance (Remaining Unmeasured Items)

> ⚠️ Memory and processor allocation are now **confirmed real values** (§5, §6) via `ollama ps`. The values below are still *typical ballpark ranges* — not measured — only for latency and CPU load, which need a timed query and Task Manager reading to confirm. Replace before final submission.

| Metric | Estimated range |
|---|---|
| Time to first token | ~1–3s |
| Tokens/sec (generation, CPU-only, 7.2GB model) | ~3–10 tok/s |
| Total response time (short ~50-token answer) | ~5–15s |
| Embedding latency (single chunk, 274MB model) | <1s |
| CPU usage % during inference | 60–100% (likely pegs several cores heavily — confirm via Task Manager) |

## 9. Summary

*(Fill in once the numbers above are in: a 2–3 sentence honest summary — e.g. "GhostOS runs entirely on CPU on a mid-range laptop, with response times of roughly Xs for typical queries and a peak memory footprint of around Y GB, dominated by the loaded chat model." Judges weigh an honest, modest number far more than an unverified impressive one.)*
