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

from flask import Flask, request, jsonify, Response, send_file
import json
import os
import queue
import threading
import random
import tempfile
import getpass
from pathlib import Path
from urllib.parse import unquote, urlparse
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor

from embeddings import get_embedding
from vectorstore import (
    init_db, get_stats, get_categories, get_collections,
    get_recent_files, get_storage_breakdown, get_timeline,
    normalize_timeline_event_kind, TIMELINE_EVENT_KINDS,
    search_files_by_name, hybrid_search, rerank,
    is_catalogued_path, clear_file_index, clear_all_local_data,
)
from indexer import SUPPORTED_EXTENSIONS, extract_text, refresh_catalog_collections
from ocr_service import IMAGE_EXTENSIONS
from voice_service import AUDIO_EXTENSIONS, get_voice_status, transcribe_audio
from connect_system import (
    connect_system, get_background_status, cancel_background_indexing,
    reset_background_status,
)

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
from diagnostics import get_diagnostics
from timeline_sessions import group_events, summarize_day
from settings_store import get_settings, update_settings
from insights import build_insights, clear_insight_state, dismiss_insight
from action_agent import execute_action
from action_registry import list_actions
from watcher import pause_collectors
from browser_connector import reset_browser_sync_state

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
_lifecycle_lock = threading.Lock()

_LOCAL_UI_ORIGINS = {
    f"http://127.0.0.1:{PORT}",
    f"http://localhost:{PORT}",
    f"http://[::1]:{PORT}",
}


@app.before_request
def reject_cross_site_mutations():
    """Block websites from driving private local mutation/action endpoints."""
    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return None
    origin = (request.headers.get("Origin") or "").rstrip("/")
    fetch_site = (request.headers.get("Sec-Fetch-Site") or "").casefold()
    if fetch_site == "cross-site" or (origin and origin not in _LOCAL_UI_ORIGINS):
        return jsonify({"error": "Cross-site requests are not allowed."}), 403
    return None


def execute_permitted_action(action: str, arguments: dict | None) -> dict:
    """Apply the local settings allowlist before dispatching every action."""
    if not get_settings()["action_permissions"].get(action, False):
        message = "This action is disabled in GhostOS settings."
        return {
            "success": False,
            "action": action,
            "target": None,
            "message": message,
            "error": {"code": "permission_disabled", "message": message},
        }
    return execute_action(action, arguments)

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
- Safe local actions are executed only by GhostOS's validated action layer
  before model generation. You must never execute or invent commands, and
  must never claim that a file, app, URL, or folder was opened unless an
  explicit action result is supplied by the backend.
- You have access to recent conversation turns and, when relevant, a
  "Conversation memory" context block that resolves pronouns like "it",
  "that", or "this" to whichever file/folder/page was most recently
  discussed. Use it to answer naturally (e.g. "That's report.pdf, in your
  Documents folder"). Never infer that an action succeeded merely because
  the reference is known.
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
refresh_catalog_collections()


@app.route("/", methods=["GET"])
def frontend_index():
    """Serve the bundled frontend so API calls can use the configured origin."""
    frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    return send_file(frontend_path)


@app.route("/api/health", methods=["GET"])
def api_health():
    """Detailed local readiness snapshot used by the UI and setup flow."""
    report = get_diagnostics()
    return jsonify(report), (200 if report["status"] == "ready" else 503)


@app.route("/api/diagnostics", methods=["GET"])
def api_diagnostics():
    return jsonify(get_diagnostics())


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
    settings = get_settings()
    scan_entire_drives = bool(
        body.get("scan_entire_drives", settings["scan_entire_drives"])
    )

    # Serialize connect/rebuild/clear transitions.  Acquiring before the
    # worker starts closes the small window where a clear request could run
    # after this route returned but before connect_system marked its job active.
    if not _lifecycle_lock.acquire(blocking=False):
        return jsonify({"error": "Another GhostOS lifecycle operation is already starting."}), 409

    progress_q: "queue.Queue" = queue.Queue()

    def on_progress(event: dict):
        progress_q.put(event)

    def run():
        try:
            result = connect_system(
                progress_callback=on_progress,
                scan_entire_drives=scan_entire_drives,
                additional_folders=settings["indexed_folders"],
                excluded_folders=settings["excluded_folders"],
            )
            progress_q.put({"type": "done", **result})
        except Exception as e:
            progress_q.put({"type": "error", "message": str(e)})
        finally:
            progress_q.put(None)  # sentinel: stream is finished
            _lifecycle_lock.release()

    try:
        threading.Thread(target=run, daemon=True).start()
    except Exception:
        _lifecycle_lock.release()
        raise

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


@app.route("/api/index-cancel", methods=["POST"])
def api_index_cancel():
    """Request graceful cancellation of the active background index job."""
    result = cancel_background_indexing()
    return jsonify(result), (202 if result.get("accepted") else 409)


@app.route("/api/index/rebuild", methods=["POST"])
def api_index_rebuild():
    """Clear derived file/chunk data and start a fresh background index."""
    body = request.get_json(silent=True) or {}
    if body.get("confirm") != "rebuild-index":
        return jsonify({"error": "Explicit rebuild confirmation is required."}), 400
    with _lifecycle_lock:
        if get_background_status().get("active"):
            return jsonify({"error": "Cancel or wait for the current indexing job first."}), 409
        pause_collectors()
        reset_browser_sync_state()
        removed = clear_file_index()
        settings = get_settings()
        result = connect_system(
            scan_entire_drives=bool(settings["scan_entire_drives"]),
            additional_folders=settings["indexed_folders"],
            excluded_folders=settings["excluded_folders"],
        )
    return jsonify({"status": "rebuilding", "removed": removed, "connect": result}), 202


@app.route("/api/data/clear", methods=["POST"])
def api_clear_local_data():
    """Clear user-derived GhostOS data after an explicit confirmation token."""
    body = request.get_json(silent=True) or {}
    if body.get("confirm") != "clear-local-data":
        return jsonify({"error": "Explicit clear confirmation is required."}), 400
    with _lifecycle_lock:
        if get_background_status().get("active"):
            return jsonify({"error": "Cancel or wait for indexing before clearing data."}), 409
        pause_collectors()
        removed = clear_all_local_data()
        reset_background_status()
        reset_browser_sync_state()
        clear_insight_state()
        reset_memory()
    return jsonify({"status": "cleared", "removed": removed})


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


@app.route("/api/settings", methods=["GET", "PUT"])
def api_settings():
    if request.method == "GET":
        return jsonify({
            "settings": get_settings(),
            "privacy": "All indexed content, activity, settings, and AI processing stay on this device.",
        })
    try:
        settings = update_settings(request.get_json(silent=True) or {})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "saved", "settings": settings, "restart_required": True})


@app.route("/api/insights", methods=["GET"])
def api_insights():
    return jsonify({"insights": build_insights()})


@app.route("/api/insights/<insight_id>/dismiss", methods=["POST"])
def api_dismiss_insight(insight_id: str):
    if not insight_id.isalnum() or len(insight_id) > 64:
        return jsonify({"error": "Invalid insight id"}), 400
    dismiss_insight(insight_id)
    return jsonify({"status": "dismissed", "id": insight_id})


@app.route("/api/open", methods=["POST"])
def api_open():
    """Backward-compatible safe open endpoint used by the existing UI."""
    target = str((request.get_json(silent=True) or {}).get("target", "")).strip()
    if not target:
        return jsonify({"error": "Missing target"}), 400
    if target.lower().startswith("file://"):
        parsed = urlparse(target)
        if parsed.netloc not in {"", "localhost"}:
            return jsonify({"error": "Remote file URLs are not allowed"}), 400
        local_path = unquote(parsed.path)
        if os.name == "nt" and len(local_path) >= 3 and local_path[0] == "/" and local_path[2] == ":":
            local_path = local_path[1:]
        target = str(Path(local_path.replace("/", os.sep)))
    is_web = target.lower().startswith(("http://", "https://"))
    if not is_web and not is_catalogued_path(target):
        return jsonify({"error": "Only indexed files can be opened"}), 403
    action = "open_url" if is_web else ("open_folder" if Path(target).is_dir() else "open_file")
    key = "url" if is_web else "path"
    result = execute_permitted_action(action, {key: target})
    status = 200 if result["success"] else (403 if result.get("error", {}).get("code") == "permission_disabled" else 400)
    return jsonify(result), status


@app.route("/api/actions", methods=["GET"])
def api_actions():
    return jsonify({"actions": list_actions(), "permissions": get_settings()["action_permissions"]})


@app.route("/api/actions/execute", methods=["POST"])
def api_execute_action():
    body = request.get_json(silent=True) or {}
    action = str(body.get("action", ""))
    result = execute_permitted_action(action, body.get("arguments"))
    status = 200 if result["success"] else (403 if result.get("error", {}).get("code") == "permission_disabled" else 400)
    return jsonify(result), status


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
    offset = request.args.get("offset", default=0, type=int)
    category = request.args.get("category")
    collection = request.args.get("collection")
    return jsonify(get_recent_files(
        limit=min(max(limit, 1), 5000),
        offset=max(offset, 0),
        category=category,
        collection=collection,
    ))


@app.route("/api/storage", methods=["GET"])
def api_storage():
    """Powers the Storage Overview donut chart."""
    return jsonify(get_storage_breakdown())


@app.route("/api/timeline", methods=["GET"])
def api_timeline():
    """Timeline events with optional date, limit and source-category filters."""
    date = request.args.get("date")
    limit = request.args.get("limit", default=200, type=int)
    try:
        event_kind = normalize_timeline_event_kind(
            request.args.get("event_kind"), kind=request.args.get("kind")
        )
    except ValueError as exc:
        return jsonify({
            "error": str(exc),
            "allowed_kinds": sorted(TIMELINE_EVENT_KINDS),
        }), 400
    return jsonify(get_timeline(
        date_prefix=date,
        limit=min(max(limit, 1), 5000),
        event_kind=event_kind,
    ))


@app.route("/api/timeline/sessions", methods=["GET"])
def api_timeline_sessions():
    date = request.args.get("date")
    limit = request.args.get("limit", default=1000, type=int)
    return jsonify(group_events(get_timeline(date_prefix=date, limit=min(max(limit, 1), 5000))))


@app.route("/api/timeline/summary", methods=["GET"])
def api_timeline_summary():
    date = request.args.get("date")
    events = get_timeline(date_prefix=date, limit=5000)
    return jsonify(summarize_day(events, date=date))


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
    query = request.args.get("q", "").strip()[:500]
    limit = min(max(request.args.get("limit", default=8, type=int), 1), 100)
    if not query:
        return jsonify({"files": [], "content": []})

    file_matches = search_files_by_name(query, limit=limit)

    content_matches = [
        {
            "source_path": match["source_path"],
            "snippet": match["content"][:220],
            "score": round(match["score"], 3),
        }
        for match in search_agent(query)[:limit]
    ]

    return jsonify({"files": file_matches, "content": content_matches})


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    """
    Transcribes a short recorded audio clip fully locally (faster-whisper,
    CPU, int8). Nothing is sent off-device. Mirrors the OCR feature: off by
    default, requires voice_enabled=true in settings, and requires the
    optional requirements-voice.txt package to actually be installed.
    """
    if not get_settings().get("voice_enabled"):
        return jsonify({"error": "Voice input is turned off in Settings."}), 400

    status = get_voice_status()
    if not status["available"]:
        return jsonify({
            "error": (
                "Local speech engine isn't installed. Run "
                ".\\setup_backend.ps1 -WithVoice, then restart the backend."
            )
        }), 503

    uploaded = request.files.get("audio")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "No audio was uploaded."}), 400

    suffix = Path(secure_filename(uploaded.filename)).suffix.lower() or ".webm"
    if suffix not in AUDIO_EXTENSIONS:
        return jsonify({"error": f"Unsupported audio format: {suffix}"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        uploaded.save(tmp.name)
        tmp_path = Path(tmp.name)
    try:
        if tmp_path.stat().st_size == 0:
            return jsonify({"error": "Recording was empty."}), 400
        text = transcribe_audio(tmp_path)
    except Exception as exc:
        return jsonify({"error": f"Transcription failed: {exc}"}), 500
    finally:
        tmp_path.unlink(missing_ok=True)

    return jsonify({"text": text})


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
            allowed_extensions = set(SUPPORTED_EXTENSIONS)
            if get_settings().get("ocr_enabled"):
                allowed_extensions.update(IMAGE_EXTENSIONS)
            if suffix not in allowed_extensions:
                attachment_context = (
                    f"[Attached file: {uploaded.filename}; this format is not supported for "
                    "local text extraction. Audio/video transcription is not implemented.]"
                )
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    uploaded.save(tmp.name)
                    tmp_path = Path(tmp.name)
                try:
                    text = extract_text(tmp_path)
                    if text.strip():
                        attachment_context = f"[Attached file: {uploaded.filename}]\n{text[:20000]}"
                    else:
                        attachment_context = f"[Attached file: {uploaded.filename}; no local text could be extracted]"
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
    if intent in ("semantic_query", "general", "exact_file_query") and wants_reference_resolution(user_message) \
            and (session_snapshot["last_file"] or session_snapshot["last_folder"] or session_snapshot["last_browser_tab"]):
        intent = "reference_query"

    print(f"[intent] {user_message!r} -> {intent}")

    # The model never executes actions. A narrow deterministic path handles
    # reference actions such as "open it" using already-resolved local state.
    normalized_message = user_message.casefold().strip(" .!?\t\r\n")
    if intent == "reference_query" and normalized_message.startswith(("open ", "launch ", "show ")):
        remembered_file = session_snapshot.get("last_file")
        remembered_url = session_snapshot.get("last_browser_tab")
        remembered_folder = session_snapshot.get("last_folder_path")
        if remembered_file:
            result = execute_permitted_action("open_file", {"path": remembered_file["path"]})
        elif remembered_url:
            result = execute_permitted_action("open_url", {"url": remembered_url["url"]})
        elif remembered_folder and Path(remembered_folder).is_absolute():
            result = execute_permitted_action("open_folder", {"path": remembered_folder})
        else:
            result = {"success": False, "message": "I don't have a specific file, folder, or page to open yet."}
        reply = result["message"]
        record_turn(user_message, reply)
        return Response(instant_reply_stream(reply), mimetype="text/plain")

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
            f"structured listing of up to 50 recently modified matches, not a content search]\n{file_list}"
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
    """Clear conversation history/session memory for the UI's New Chat action."""
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