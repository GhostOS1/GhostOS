# GhostOS — Evaluation

> **Honesty note:** GhostOS has not yet undergone a formal, numeric benchmark (precision/recall on a labeled query set, etc.) — it's a prototype, and this document says so plainly rather than presenting invented accuracy percentages. What follows is (1) a proposed benchmark method you can actually run, (2) a qualitative comparison against the retrieval design's alternatives, and (3) known failure cases identified directly from the code's own logic and comments.

---

## 1. Benchmark Method (Proposed, Not Yet Run)

To produce real accuracy numbers, the suggested approach:

1. **Build a labeled test set.** Pick ~30–50 real files already indexed on a test machine (mix of PDFs, docs, code files, images-with-OCR). For each, write 1–2 natural-language questions a user might plausibly ask that should retrieve that file (e.g. "that stock analysis PDF from last week").
2. **Run each question through the pipeline** and record:
   - Whether the correct file appears in the top-5 retrieved chunks (`TOP_K_CHUNKS = 5` in `memory_agent.py`)
   - Its rank within those 5
   - Whether the final generated answer correctly names the file and path
3. **Compute standard IR metrics** — Recall@5, Mean Reciprocal Rank (MRR) — over the test set.
4. **Separately evaluate generation quality** — for the correctly-retrieved cases, is the summary/answer actually accurate against the source content, or does it hallucinate details not present in the retrieved chunks?

This hasn't been run yet on real data. If you run it, the results belong in this section with the actual test-set size and scores — not estimated ones.

---

## 2. Baseline Comparison (Design-Level, Not Benchmarked)

GhostOS's retrieval isn't a single technique — it's dense embeddings **and** BM25 fused with Reciprocal Rank Fusion, then lexically reranked (`vectorstore.py`). This design choice can be reasoned about even without a run benchmark:

| Approach | Strength | Weakness |
|---|---|---|
| Dense-only (embeddings alone) | Good at semantic/paraphrased queries | Misses exact terms — a filename, error code, or proper noun can rank low if it's not semantically salient |
| BM25-only (keyword alone) | Good at exact term matches | Misses paraphrased or conceptual queries ("that thing about the meteorology PDF") |
| **GhostOS's hybrid (dense + BM25 + RRF + rerank)** | Catches both cases | More moving parts; each stage (fusion, then reranking) needs tuning, and `MIN_RERANK_SCORE = 0.22` is a hand-picked cutoff, not one derived from a validation set |

Against **cloud alternatives** (Windows Recall, cloud copilots with screen/file access): those tools weren't tested side-by-side with GhostOS on the same data, so no accuracy comparison is claimed here. The honest differentiator is architectural (local-only, see [`LOCAL_AI_VERIFICATION.md`](LOCAL_AI_VERIFICATION.md)), not a demonstrated accuracy edge.

---

## 3. Known Failure Cases

These are drawn directly from the retrieval and indexing logic, not from a test run — they describe *where the design is expected to struggle*:

- **Below-threshold matches get dropped entirely.** `rerank()` applies `MIN_RERANK_SCORE = 0.22` as a calibrated cutoff — a genuinely relevant file that scores just under this threshold will not reach the LLM at all, and GhostOS will report it couldn't find the answer rather than returning a weak guess. This is a deliberate precision-over-recall tradeoff, but it means real misses are possible for oddly-worded queries.
- **BM25 is brute-force, not indexed.** `bm25_search()` scans every stored chunk's content in pure Python rather than using an inverted index. This is explicitly noted as fine at prototype scale but will slow down (and could start missing time-budget cutoffs) as the indexed corpus grows well past "tens of thousands of chunks."
- **Name-based file lookups can be ambiguous.** `files_agent.py`'s exact-lookup path (`looks_like_exact_file_lookup()`) is a heuristic over phrasing (extensions, "find"/"where is", path-like strings) — a vaguely-phrased request without one of those signals falls through to semantic search instead, which may not return the exact file the user meant if multiple files share similar content.
- **OCR and voice transcription are not perfect.** Both are explicitly optional, off/limited by default, and accuracy depends entirely on document/audio quality — no accuracy figures exist for either yet.
- **Windows-only activity tracking.** `activity_tracker.py` uses Windows-specific APIs (`ctypes.windll`); on any other OS, activity-based timeline entries simply won't be generated (not a partial failure — total absence).
- **No live browser tabs.** `browser_connector.py` only reads *historical* Chrome/Edge data (history, bookmarks, downloads) from local profile files — a currently-open, unsaved tab is invisible to GhostOS until it shows up in history.
- **Single-machine state.** All conversation memory and session context (`memory_agent.py`'s `_session_context`) is in-process, global, and not persisted across restarts or synced across devices — closing GhostOS resets pronoun-resolution context ("open it") even though indexed content itself persists in SQLite.

---

## 4. What Would Make This Section Stronger

- Running the benchmark method in §1 on a real, disclosed test set and reporting actual Recall@5/MRR numbers.
- A small manual eval of generation faithfulness (does the summary match the source, or hallucinate?) on ~20 real file-summarization requests.
- Timing how retrieval quality degrades as the indexed corpus grows toward and past the "tens of thousands of chunks" ceiling the vector store is scoped for.
