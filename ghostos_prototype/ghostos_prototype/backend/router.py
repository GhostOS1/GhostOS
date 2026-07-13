"""
router.py
Intent Router - the single place that decides which agent(s) a message
needs, matching the "agent router" box in the v2.0 architecture diagram:

    User -> Intent Router -> {Files, Timeline, Browser, Memory, System, AI}

Kept deliberately dependency-light: classify_intent() below never touches
a network, a database, or an embedding call - it's pure string matching,
which is what makes it cost microseconds instead of seconds and stay easy
to unit-test on its own (see the "why" comment in app.py's old
classify_intent - unchanged reasoning, just relocated here now that it's
not sharing a file with six agents' worth of other logic).

One thing deliberately does NOT live here: the 'reference_query' override
for pronoun follow-ups ("open it"). That needs session memory
(agents/memory_agent.py's state), which this module has no access to by
design - classify_intent() stays a pure function of the message text
alone. app.py applies that stateful override after calling classify_intent.
"""

import re

from files_agent import detect_folder, looks_like_exact_file_lookup
from timeline_agent import wants_timeline
from system_agent import wants_system_info

GREETING_WORDS = {"hi", "hii", "hiii", "hello", "hey", "heya", "yo", "sup",
                   "good morning", "good afternoon", "good evening", "morning", "evening",
                   "namaste", "hola"}
THANKS_WORDS = {"thanks", "thank you", "thanks a lot", "thx", "ty", "tysm",
                 "cool", "great", "nice", "awesome", "perfect", "ok", "okay", "got it", "cool thanks"}
FAREWELL_WORDS = {"bye", "goodbye", "see you", "cya", "gtg", "good night", "night"}

# Any of these appearing is a strong signal the message is about the
# user's own indexed files/memory at all (decides semantic_query vs
# general below).
FILE_INTENT_KEYWORDS = {
    "file", "files", "document", "documents", "doc", "docs", "pdf", "pdfs",
    "find", "search", "where", "locate", "open", "download", "downloads",
    "recent", "indexed", "invoice", "note", "notes", "meeting", "email", "browser",
    "history", "visited", "syllabus", "resume", "project", "photo", "photos", "image",
    "images", "video", "videos", "spreadsheet", "excel", "presentation", "slides",
    "summarize", "summary", "folder", "directory", "page", "website",
    "site", "link", "url",
}

PERSONAL_RECALL_PATTERNS = (
    re.compile(r"\bwhat did i (?:write|read|see|visit|open|work on)\b"),
    re.compile(r"\bdid i (?:write|read|see|visit|open|work on)\b"),
    re.compile(r"\b(?:remind me|do you remember|remember when|we discussed|i said|i wrote|i read|i saw)\b"),
    re.compile(r"\b(?:that|the|my) (?:blog|article|topic|thing|page) about\b"),
    re.compile(r"\bwhat was that (?:blog|article|topic|thing|page)\b"),
)

GREETING_REPLIES = [
    "Hey! How can I help you today?",
    "Hello! What can I do for you?",
    "Hi there — what do you need?",
    "Hey, good to see you. What are we working on?",
]
THANKS_REPLIES = ["You're welcome!", "Anytime!", "Glad I could help.", "No problem at all."]


def _contains_term(text: str, term: str) -> bool:
    """Match intent terms as words/phrases, not arbitrary substrings."""
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


def _matches_small_talk(text: str, phrases: set[str], *, allow_suffix: bool = False) -> bool:
    """Recognize short standalone small talk without swallowing requests.

    ``startswith`` previously classified ``hey find report.pdf`` as a
    greeting.  The two benign suffixes cover natural greetings while a
    message containing an actual task continues through the router.
    """
    if text in phrases:
        return True
    if not allow_suffix:
        return False
    return any(text in {f"{phrase} there", f"{phrase} ghostos"} for phrase in phrases)
FAREWELL_REPLIES = ["See you later!", "Goodbye — reach out anytime.", "Take care!"]


def classify_intent(message: str) -> str:
    """
    Returns one of: 'greeting', 'thanks', 'farewell', 'folder_query',
    'exact_file_query', 'timeline_query', 'system_query', 'semantic_query',
    'general'. Order matters: more specific/structured intents (which map
    to a cheap direct SQLite lookup or, for system_query, a stdlib/psutil
    read) are checked before the catch-all semantic bucket, so a
    structured request never falls through to an unnecessary embedding
    call.
    """
    text = " ".join(message.casefold().strip().strip("!.,?").split())
    if not text:
        return "greeting"
    word_count = len(text.split())

    if word_count <= 4 and _matches_small_talk(text, GREETING_WORDS, allow_suffix=True):
        return "greeting"
    if word_count <= 6 and _matches_small_talk(text, THANKS_WORDS):
        return "thanks"
    if word_count <= 3 and _matches_small_talk(text, FAREWELL_WORDS):
        return "farewell"

    # Retrieval precedence mirrors the real pipeline: explicit filename /
    # path, then structured folder, then broader activity/content search.
    if looks_like_exact_file_lookup(text):
        return "exact_file_query"
    if detect_folder(text):
        return "folder_query"
    if wants_timeline(text):
        return "timeline_query"
    if wants_system_info(text):
        return "system_query"
    if any(pattern.search(text) for pattern in PERSONAL_RECALL_PATTERNS):
        return "semantic_query"
    if any(_contains_term(text, kw) for kw in FILE_INTENT_KEYWORDS):
        return "semantic_query"
    return "general"
