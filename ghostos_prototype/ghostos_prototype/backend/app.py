"""
app.py
GhostOS prototype backend.

This file is now deliberately thin - it's the Flask/HTTP layer only. All
the actual work happens in router.py (Intent Router) and agents/*.py
(Files, Timeline, Browser, Memory, System, AI agents). Previously this one
file held the classifier, every retrieval function, session state, AND
the Flask routes - the "one giant app.py" the agent-router refactor was
meant to fix. See router.py and agents/ for the logic that used to live
here.

Endpoints:
  POST /api/connect-system -> the one-click "Connect to System" flow.
                               Discovers the user's standard folders itself
                               (no path is ever sent by the client), indexes
                               them, and starts the watcher. Streams
                               newline-delimited JSON progress events,
                               ending with one {"type":"done", ...} summary
                               line. See core/connect_system.py.
  POST /api/chat            -> ask a question, get a RAG-grounded answer
                                (body: {"message": "..."})
  POST /api/chat/reset      -> clears conversation history + session memory
  GET  /api/stats            -> how much has been indexed so far

Requires Ollama running locally with:
  ollama pull nomic-embed-text
  ollama pull gemma4:e2b
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import os
import queue
import threading
import random
import tempfile
import getpass
from pathlib import Path
from urllib.parse import urlparse
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor

from embeddings import get_embedding
from vectorstore import (
    init_db, get_stats, get_categories, get_collections,
    get_recent_files, get_storage_breakdown, get_timeline,
    search_files_by_name, hybrid_search, rerank,
    is_catalogued_path,
)
from indexer import extract_text
from connect_system import connect_system, get_background_status

from router import (
    classify_intent, GREETING_REPLIES, THANKS_REPLIES, FAREWELL_REPLIES,
)
from files_agent import folder_agent, file_agent
from timeline_agent import timeline_agent
from browser_agent import browser_agent
from system_agent import system_agent
from memory_agent import (
    search_agent, wants_reference_resolution, build_reference_context,
    snapshot_session, get_history, record_turn, update_session_from_turn,
    reset as reset_memory, RETRIEVAL_POOL_SIZE, MIN_RERANK_SCORE,
)
from ai_agent import stream_reply
from config import HOST, PORT, CHAT_MODEL, EMBED_MODEL, OLLAMA_BASE_URL
import requests

app = Flask(__name__)
CORS(app)  # allows the simple HTML frontend to call this API from a file:// or localhost origin

SYSTEM_PROMPT = """You are Ghost, the AI assistant inside GhostOS — a private,
offline-first assistant that answers questions using the user's own indexed
files, running entirely on their machine.

First, decide what kind of message this is:

1. Casual messages (greetings, small talk, thanks, introductions, or questions about what
   you can do) — respond naturally and briefly, like a normal assistant.
   - When the user starts a new conversation with greetings such as "Hi", "Hello", "Hey",
     "Good morning", "Good afternoon", or "Good evening", greet them warmly in return
     before continuing the conversation.
   - If the user's name is available, you may naturally include it in the greeting
     (e.g., "Hello, John!"). If no name is available, use a generic greeting.
   - Keep greetings friendly, concise, and varied rather than repeating the same wording
     every time.
   - Never say "I couldn't find that in your indexed files" for greetings, small talk,
     introductions, or other casual conversation; that rule only applies to case 2 below.

2. Questions about the user's own files or content (documents, notes,
   anything that should live in their indexed memory) — answer using ONLY
   the context provided below.
   - If the answer isn't in the provided context, say so plainly: "I
     couldn't find that in your indexed files." Never guess or invent
     file names, dates, or facts not present in the context.
   - When useful, mention which file the information came from.
   - Be concise and direct — don't pad answers with unnecessary caveats.

Rules that always apply:
- Only claim to have found something if it's actually present in the
  context below. Never fabricate filenames, dates, or content.
- You can search and summarize the user's indexed files. You cannot open
  apps, send emails, modify files, or take any action on the system —
  if asked to do something outside searching/answering, say that's not
  supported yet rather than pretending it happened.
- You have access to recent conversation turns and, when relevant, a
  "Conversation memory" context block that resolves pronouns like "it",
  "that", or "this" to whichever file/folder/page was most recently
  discussed. Use it to answer naturally (e.g. "That's report.pdf, in your
  Documents folder"). The action limitation above still applies even when
  the reference is known — you can say what "it" is, not open/move/delete it.
- Some context entries are live system stats (CPU/RAM/disk usage) rather
  than files — treat those as current readings to report, not files to
  reference or claim are "indexed."
- Mention that everything runs locally only if the user asks about privacy
  or where their data goes — don't bring it up unprompted.

***IMPORTANT:***
Every incoming message contains two sections:

1. User Message
2. Indexed Context

Your first task is to determine whether the User Message requires the Indexed Context.

- If the User Message is a greeting, introduction, farewell, thanks, small talk, or a general question (such as "Hi", "Hello", "Good morning", "How are you?", "Who are you?", "What can you do?"), completely ignore the Indexed Context and reply naturally.

- Only consult the Indexed Context when the user is asking about information that could exist in their own documents, notes, emails, PDFs, or other indexed files.

- The Indexed Context may be empty or unrelated. Never assume it contains the answer.

- If the answer is not present in the Indexed Context for a file-related question, reply:
"I couldn't find that in your indexed files."

- Never mention missing indexed files for greetings, casual conversation, or general knowledge questions.
"""

init_db()


@app.route("/api/health", methods=["GET"])
def api_health():
    """Readiness snapshot used by the UI and installers."""
    ollama = {"available": False, "chat_model": False, "embedding_model": False, "error": None}
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        response.raise_for_status()
        names = {m.get("name") for m in response.json().get("models", []) if m.get("name")}

        def model_is_installed(configured_name: str) -> bool:
            # Ollama may report an implicit latest tag explicitly:
            # "nomic-embed-text" and "nomic-embed-text:latest" are the
            # same model reference and should both satisfy readiness.
            if configured_name in names:
                return True
            if ":" not in configured_name and f"{configured_name}:latest" in names:
                return True
            return False

        ollama.update({
            "available": True,
            "chat_model": model_is_installed(CHAT_MODEL),
            "embedding_model": model_is_installed(EMBED_MODEL),
        })
    except Exception as exc:
        ollama["error"] = str(exc)
    ready = ollama["available"] and ollama["chat_model"] and ollama["embedding_model"]
    return jsonify({"status": "ready" if ready else "degraded", "ollama": ollama, "database": get_stats()}), (200 if ready else 503)


@app.route("/api/connect-system", methods=["POST"])
def api_connect_system():
    """
    The one-click "Connect to System" flow. Body is OPTIONAL and, if
    present, may only contain {"scan_entire_drives": true} - an explicit
    opt-in for the slower whole-drive scan. No folder path is ever
    supplied by the client for the standard scan, by design.
    core.connect_system detects the OS, discovers the user's standard
    folders (plus VS Code projects and Git repos - see connect_system.py),
    indexes them, and starts the watcher; this route just runs that on a
    background thread and streams its progress events back as they happen.

    Response is newline-delimited JSON (one JSON object per line), same
    "generator" streaming approach as /api/chat below, just NDJSON instead
    of raw text tokens:
      {"type": "progress", "message": "..."}   (zero or more)
      {"type": "done", "status": "connected", "folders": 6, "background_indexing": true}

    NOTE: connect_system() now only runs its fast phase synchronously
    (OS detect + standard-folder discovery + starting the watcher) - this
    whole request/stream typically finishes in well under a second. The
    slow part (VS Code/Git/drive discovery, indexing, embeddings) keeps
    running in a background thread *after* this response has already
    closed. Poll GET /api/index-status to track that.
    """
    body = request.get_json(silent=True) or {}
    scan_entire_drives = bool(body.get("scan_entire_drives", False))

    progress_q: "queue.Queue" = queue.Queue()

    def on_progress(event: dict):
        progress_q.put(event)

    def run():
        try:
            result = connect_system(progress_callback=on_progress, scan_entire_drives=scan_entire_drives)
            progress_q.put({"type": "done", **result})
        except Exception as e:
            progress_q.put({"type": "error", "message": str(e)})
        finally:
            progress_q.put(None)  # sentinel: stream is finished

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while True:
            event = progress_q.get()
            if event is None:
                break
            yield json.dumps(event) + "\n"

    return Response(stream(), mimetype="application/x-ndjson")


@app.route("/api/index-status", methods=["GET"])
def api_index_status():
    """
    Poll target for the background indexing job started by
    /api/connect-system's fast phase. Returns a snapshot like:
      {"active": true, "done": false, "phase": "indexing",
       "current_folder": "Documents", "folders_total": 9,
       "folders_completed": 3, "files_processed": 412,
       "chunks_added": 1889, "vscode_projects_found": 2,
       "git_repos_found": 5, "drive_folders_found": 0,
       "error": null, "stats": {...}}
    "phase": "idle" and "done": true (with no job ever started) is the
    state before the user has ever clicked Connect. The frontend polls
    this every couple seconds after showing "Connected" so it can show a
    quiet "still organizing your files..." indicator without blocking on
    it, per connect_system.py's Phase 1/Phase 2 split.
    """
    return jsonify(get_background_status())


@app.route("/api/stats", methods=["GET"])
def api_stats():
    return jsonify(get_stats())


@app.route("/api/profile", methods=["GET"])
def api_profile():
    username = getpass.getuser() or os.environ.get("USERNAME") or "User"
    display_name = username.replace("_", " ").replace(".", " ").strip().title()
    initials = "".join(part[0] for part in display_name.split()[:2]).upper() or "U"
    return jsonify({
        "username": username,
        "display_name": display_name,
        "initials": initials,
        "computer_name": os.environ.get("COMPUTERNAME", "This PC"),
    })


@app.route("/api/open", methods=["POST"])
def api_open():
    """Open a catalogued file/folder or a recorded HTTP(S) URL on Windows."""
    target = str((request.get_json(silent=True) or {}).get("target", "")).strip()
    if not target:
        return jsonify({"error": "Missing target"}), 400
    parsed = urlparse(target)
    is_web = parsed.scheme in ("http", "https") and bool(parsed.netloc)
    if not is_web and not is_catalogued_path(target):
        return jsonify({"error": "Only indexed files can be opened"}), 403
    try:
        os.startfile(target)  # Windows-only by design; GhostOS currently targets Windows.
        return jsonify({"status": "opened", "target": target})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/categories", methods=["GET"])
def api_categories():
    """Powers the 'Categories' grid on the Files & Data screen."""
    return jsonify(get_categories())


@app.route("/api/collections", methods=["GET"])
def api_collections():
    """Powers the 'AI Collections' grid on the Files & Data screen."""
    return jsonify(get_collections())


@app.route("/api/recent-files", methods=["GET"])
def api_recent_files():
    """Powers the 'Recent Files' table on the Files & Data screen. Also
    powers the Categories / AI Collections tiles when clicked - pass
    ?category=<n> or ?collection=<n> to filter to just that bucket."""
    limit = request.args.get("limit", default=20, type=int)
    category = request.args.get("category")
    collection = request.args.get("collection")
    return jsonify(get_recent_files(limit=limit, category=category, collection=collection))


@app.route("/api/storage", methods=["GET"])
def api_storage():
    """Powers the Storage Overview donut chart."""
    return jsonify(get_storage_breakdown())


@app.route("/api/timeline", methods=["GET"])
def api_timeline():
    """Powers the Timeline screen. Optional ?date=YYYY-MM-DD, ?limit=N."""
    date = request.args.get("date")
    limit = request.args.get("limit", default=200, type=int)
    return jsonify(get_timeline(date_prefix=date, limit=limit))


@app.route("/api/search", methods=["GET"])
def api_search():
    """
    Powers the topbar (Ctrl+K) and Files & Data search bars. Returns two
    result lists so the frontend can show them as separate groups:
      - "files": instant filename/path substring matches (SQLite LIKE,
        no embeddings involved - works even if Ollama isn't running)
      - "content": hybrid (vector + BM25) search over indexed chunk
        content, fused and re-ranked the same way Memory Agent's
        search_agent() does for chat. Silently empty if embedding fails,
        so the search bar still works without Ollama - it just won't
        surface content matches.
    """
    query = request.args.get("q", "").strip()
    limit = request.args.get("limit", default=8, type=int)
    if not query:
        return jsonify({"files": [], "content": []})

    file_matches = search_files_by_name(query, limit=limit)

    content_matches = []
    try:
        query_embedding = get_embedding(query)
        fused = hybrid_search(query_embedding, query, top_k=limit, pool_size=max(limit * 4, RETRIEVAL_POOL_SIZE))
        reranked = rerank(query, fused, top_k=limit)
        content_matches = [
            {
                "source_path": m["source_path"],
                "snippet": m["content"][:220],
                "score": round(m["score"], 3),
            }
            for m in reranked if m["score"] >= MIN_RERANK_SCORE
        ]
    except Exception as e:
        print(f"[search] semantic search skipped (embedding failed): {e}")

    return jsonify({"files": file_matches, "content": content_matches})


def instant_reply_stream(text: str):
    """Same NDJSON/plain-text contract as the normal generate() streamer
    below (frontend doesn't need to change) but skips straight to the
    canned reply - no Ollama round trip at all."""
    yield text
    yield f"\n\n[[SOURCES:{json.dumps([])}]]"


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    The full pipeline, matching the v2.0 architecture's request flow:
      User -> Intent Router (router.classify_intent) -> the relevant
      agent(s) (Files / Timeline / Browser / Memory / System) -> Context
      Builder (below) -> AI Agent (Gemma) -> streamed response.
    """
    attachment_context = ""
    if request.content_type and request.content_type.startswith("multipart/form-data"):
        user_message = request.form.get("message", "").strip()
        uploaded = request.files.get("attachment")
        if uploaded and uploaded.filename:
            suffix = Path(secure_filename(uploaded.filename)).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                uploaded.save(tmp.name)
                tmp_path = Path(tmp.name)
            try:
                text = extract_text(tmp_path)
                if text.strip():
                    attachment_context = f"[Attached file: {uploaded.filename}]\n{text[:20000]}"
                else:
                    attachment_context = f"[Attached file: {uploaded.filename}; text extraction is unavailable for this format]"
            finally:
                tmp_path.unlink(missing_ok=True)
    else:
        data = request.get_json(silent=True) or {}
        user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Missing 'message' in request body"}), 400

    intent = classify_intent(user_message)

    # Snapshot session memory once per request.
    session_snapshot = snapshot_session()

    # Structured intents are decided by classify_intent() alone (it stays a
    # pure function, easy to test in isolation) - but "open it" / "show
    # that" only make sense with state classify_intent has no access to,
    # so the stateful override happens here instead: a short pronoun-y
    # message gets rerouted to reference_query only if there's actually
    # something in session memory (Memory Agent) to resolve it against.
    if intent in ("semantic_query", "general") and wants_reference_resolution(user_message) \
            and (session_snapshot["last_file"] or session_snapshot["last_folder"] or session_snapshot["last_browser_tab"]):
        intent = "reference_query"

    print(f"[intent] {user_message!r} -> {intent}")

    # Fast path: greetings/thanks/farewells never touch embeddings or the
    # LLM - this is the #1 latency win, since previously every message
    # (even "Hi") paid for an embedding call + vectorstore scan + a full
    # Gemma generation over the entire SYSTEM_PROMPT.
    if intent in ("greeting", "thanks", "farewell"):
        reply = random.choice({"greeting": GREETING_REPLIES, "thanks": THANKS_REPLIES,
                                "farewell": FAREWELL_REPLIES}[intent])
        record_turn(user_message, reply)
        return Response(instant_reply_stream(reply), mimetype="text/plain")

    # ---- Router: dispatch to exactly the agent(s) this intent needs ----
    # Structured intents (folder/exact-file/timeline/system) never touch
    # embeddings; only semantic_query pays for an embedding call, and
    # reference_query pays for nothing at all - it's pure session-memory
    # lookup via Memory Agent.
    matches, filename_matches, timeline_matches, folder_matches = [], [], [], []
    folder_name = None
    reference_context = None
    system_context = None

    if intent == "folder_query":
        folder_matches, folder_name = folder_agent(user_message)

    elif intent == "exact_file_query":
        filename_matches = file_agent(user_message)

    elif intent == "timeline_query":
        timeline_matches = timeline_agent(user_message)

    elif intent == "system_query":
        system_context = system_agent(user_message)

    elif intent == "reference_query":
        # "open it" / "show that" - no new retrieval at all, just resolve
        # the pronoun against Memory Agent's session state. Running a
        # hybrid search on the literal word "it" would only return noise.
        reference_context = build_reference_context(session_snapshot)

    elif intent == "semantic_query":
        # The only path that pays for an embedding call. Memory Agent
        # (vector+BM25 hybrid search) and Files Agent (filename keyword,
        # as a cheap fallback for content that was never embedded) still
        # run concurrently since they're independent I/O calls.
        with ThreadPoolExecutor(max_workers=2) as pool:
            search_future = pool.submit(search_agent, user_message)
            file_future = pool.submit(file_agent, user_message)
            matches = search_future.result()
            filename_matches = file_future.result()

    # intent == "general" -> no retrieval at all, straight to Gemma.

    # Browser Agent doesn't hit the network/DB itself - it just splits
    # Memory Agent's results into "regular files" vs "visited web pages".
    browser_matches = browser_agent(matches)
    file_content_matches = [m for m in matches if m not in browser_matches]

    # ---- Context Builder ----
    context_blocks = []
    sources = []
    if attachment_context:
        context_blocks.append(attachment_context)

    if folder_matches:
        file_list = "\n".join(
            f"- {f['name']}  ({f['size_bytes']} bytes, modified {f['modified_at']})"
            for f in folder_matches
        )
        context_blocks.append(
            f"[Files found directly in your {folder_name} folder - this is a "
            f"complete structured listing, not a content search]\n{file_list}"
        )
        for f in folder_matches:
            sources.append({"path": f["path"], "score": None})

    for m in file_content_matches:
        context_blocks.append(f"[Source: {m['source_path']}]\n{m['content']}")
        sources.append({"path": m["source_path"], "score": round(m["score"], 3)})

    if filename_matches:
        file_list = "\n".join(f"- {f['name']}  (located at: {f['path']})" for f in filename_matches)
        context_blocks.append(
            f"[Files whose name/location match your question - their content may "
            f"not be indexed, only their location is known]\n{file_list}"
        )
        for f in filename_matches:
            sources.append({"path": f["path"], "score": None})

    if browser_matches:
        visit_list = "\n".join(f"- {m['content']}" for m in browser_matches)
        context_blocks.append(f"[Recently visited web pages matching your question]\n{visit_list}")
        for m in browser_matches:
            sources.append({"path": m["source_path"], "score": round(m["score"], 3)})

    if timeline_matches:
        activity_list = "\n".join(f"- {e['timestamp']}: {e['title']}" for e in timeline_matches)
        context_blocks.append(f"[Recent activity timeline]\n{activity_list}")
        for e in timeline_matches:
            sources.append({"path": e.get("path_or_url"), "score": None})

    if system_context:
        context_blocks.append(system_context)

    if reference_context:
        context_blocks.append(reference_context)

    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no relevant memory found)"

    # ---- Update session memory (Memory Agent) for the next turn ----
    update_session_from_turn(intent, folder_matches, folder_name, filename_matches,
                              file_content_matches, browser_matches, user_message)

    # ---- Gemma (final generation, via AI Agent) ----
    full_prompt = f"""User Message:
                    {user_message}

                    Indexed Context (only use this if the user's message requires information from their indexed files):
                    "Always look for user message then decide u have to use the context or not. If the user message is a greeting, small talk, thanks, or general question, ignore the context and respond naturally. If the user message is a question about their own files or content, use ONLY the context provided below to answer. Some context entries are file NAME/LOCATION matches only (their content wasn't indexed) - for those, you can tell the user where the file is located, but do not claim to know what's inside it. If the answer isn't in the context, say "I couldn't find that in your indexed files." Do not invent file names, dates, or facts not present in the context.
                    {context_text}
                    """

    history_snapshot = get_history()

    def generate():
        response_text = ""
        try:
            for token in stream_reply(SYSTEM_PROMPT, history_snapshot, full_prompt):
                response_text += token
                yield token
        except Exception as exc:
            response_text = "The local AI model is unavailable. Check that Ollama is running and gemma4:e2b is installed."
            yield response_text
            print(f"[chat] generation failed: {exc}")

        # Store the clean (un-templated) turn for future requests.
        record_turn(user_message, response_text)

        # After the stream ends, send sources as a final marker the frontend can parse
        yield f"\n\n[[SOURCES:{json.dumps(sources)}]]"

    return Response(generate(), mimetype="text/plain")


@app.route("/api/chat/reset", methods=["POST"])
def api_chat_reset():
    """Clears conversation history and session memory (Memory Agent) -
    the backend equivalent of starting a new chat. Not wired to a
    frontend button yet, but useful for testing and for a future 'New
    Chat' action."""
    reset_memory()
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    print("GhostOS backend starting on http://localhost:5000")
    print(f"Make sure Ollama is running with {CHAT_MODEL} and {EMBED_MODEL} pulled.")
    # threaded=True is critical: Flask's dev server otherwise handles ONE
    # request at a time. Without it, a long-running Gemma stream, a
    # connect-system scan, or even the frontend's 20s background polling
    # (loadRealData) would block every other request behind it - including
    # instant replies like "Hi" that don't touch Ollama at all.
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
