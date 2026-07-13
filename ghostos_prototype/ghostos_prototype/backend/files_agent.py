"""
agents/files_agent.py
Files Agent - structured (non-embedding) file lookups: "what's in my
Downloads/Desktop/..." folder listings, and exact filename/path matches
("find resume.pdf", "where's the invoice"). Split out of app.py's old
folder_agent()/file_agent() so file-lookup logic lives in one importable
place instead of being two of many functions in one large module.

Neither function here ever touches embeddings or Ollama - both are plain
SQLite lookups (see vectorstore.py), which is exactly why the Intent
Router (router.py) checks for these before falling through to the more
expensive semantic_query path.
"""

import re
from difflib import SequenceMatcher
from pathlib import Path

from vectorstore import search_files_keywords, search_files_by_name, get_files_by_folder

STANDARD_FOLDERS = ["Desktop", "Documents", "Downloads", "Pictures", "Videos", "Music"]
FOLDER_ALIASES = {
    "downloads": "Downloads", "download": "Downloads", "downloads folder": "Downloads",
    "desktop": "Desktop",
    "documents": "Documents", "docs folder": "Documents",
    "pictures": "Pictures", "photos folder": "Pictures",
    "videos": "Videos", "videos folder": "Videos",
    "music": "Music",
}

EXACT_FILE_EXT_RE = re.compile(r"\.[a-z0-9]{1,10}(?:\b|$)", re.IGNORECASE)
EXACT_FILE_TRIGGER_PHRASES = ("find ", "where is", "where's", "locate ")
_PATH_RE = re.compile(r"(?:\b[a-z]:[\\/]|[/\\][^\s]+[/\\])", re.IGNORECASE)
_CONTENT_HINT_RE = re.compile(
    r"\b(?:about|containing|contains|mentioning|mentions|related to|content|inside|says?)\b",
    re.IGNORECASE,
)
_REFERENCE_HINT_RE = re.compile(
    r"\b(?:it|that|this|previous|earlier|today|yesterday|recently|last week|"
    r"this morning|we discussed|i mentioned)\b",
    re.IGNORECASE,
)
_FILLER_WORDS = {
    "a", "an", "the", "my", "me", "please", "file", "files", "document",
    "documents", "named", "called", "name", "where", "what", "which", "is",
    "are", "find", "locate", "show", "open", "search", "for", "of", "do",
    "you", "have", "could", "can", "would", "like",
}


def detect_folder(text: str) -> str | None:
    """Returns the canonical standard-folder name (e.g. 'Downloads') if the
    message is asking to browse/list one of them, else None."""
    normalized = " ".join((text or "").casefold().split())
    for alias, folder in sorted(FOLDER_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized):
            return folder

    # Also support user-created folder names ("files in college folder")
    # using the path metadata already stored by GhostOS.  Referential
    # phrases belong to Memory Agent, not a literal folder named
    # "we discussed".
    if _REFERENCE_HINT_RE.search(normalized):
        return None
    patterns = (
        r"\b(?:in|from|under|inside)\s+(?:my\s+|the\s+)?[\"']?(?P<name>[\w .-]{2,60}?)[\"']?\s+(?:folder|directory)\b",
        r"\b(?:folder|directory)\s+(?:named|called)\s+[\"']?(?P<name>[\w .-]{2,60})[\"']?",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            candidate = match.group("name").strip(" .-_\"'")
            if candidate and candidate not in {"this", "that", "previous", "a", "the"}:
                return candidate
    return None


def looks_like_exact_file_lookup(text: str) -> bool:
    """A filename with an extension, or an explicit find/locate/where-is
    phrasing, means the user already knows (part of) the name - that's a
    direct SQLite name/path match, never a semantic search."""
    normalized = " ".join((text or "").casefold().split())
    if EXACT_FILE_EXT_RE.search(normalized) or _PATH_RE.search(normalized):
        return True
    if detect_folder(normalized):
        return False
    if _CONTENT_HINT_RE.search(normalized) or _REFERENCE_HINT_RE.search(normalized):
        return False
    if re.search(r"\b(?:file|document)\s+(?:named|called)\b", normalized):
        return True
    return any(normalized.startswith(phrase) for phrase in EXACT_FILE_TRIGGER_PHRASES)


def _normal_key(value: str) -> str:
    return "".join(re.findall(r"[^\W_]+", (value or "").casefold()))


def _query_terms(query: str) -> tuple[str, list[str]]:
    """Return a filename-like phrase and its useful tokens."""
    raw_tokens = re.findall(r"[^\W_]+(?:\.[a-z0-9]{1,10})?", (query or "").casefold())
    terms = [token for token in raw_tokens if token not in _FILLER_WORDS and len(_normal_key(token)) > 1]
    return " ".join(terms), terms


def _candidate_score(candidate: dict, phrase: str, terms: list[str]) -> tuple[float, str]:
    """Calibrate exact, partial and typo matches onto a 0-1 scale."""
    name = str(candidate.get("name") or Path(str(candidate.get("path") or "")).name)
    stem = Path(name).stem
    path = str(candidate.get("path") or "")
    name_key, stem_key, path_key = _normal_key(name), _normal_key(stem), _normal_key(path)
    phrase_key = _normal_key(phrase)
    term_keys = [_normal_key(term) for term in terms if _normal_key(term)]

    if phrase_key and phrase_key == name_key:
        return 1.0, "exact"
    if phrase_key and phrase_key == stem_key:
        return 0.99, "exact"
    if any(key == name_key or key == stem_key for key in term_keys):
        return 0.97, "exact"

    partial_keys = [key for key in term_keys if len(key) >= 3 and (key in name_key or key in stem_key)]
    coverage = len(partial_keys) / len(term_keys) if term_keys else 0.0
    if phrase_key and len(phrase_key) >= 3 and phrase_key in name_key:
        return min(0.95, 0.84 + 0.11 * len(phrase_key) / max(len(name_key), 1)), "partial"
    if coverage:
        partial_score = 0.62 + 0.28 * coverage
    else:
        partial_score = 0.0

    fuzzy_values = []
    if phrase_key:
        fuzzy_values.extend([
            SequenceMatcher(None, phrase_key, name_key).ratio(),
            SequenceMatcher(None, phrase_key, stem_key).ratio(),
        ])
    for key in term_keys:
        fuzzy_values.append(SequenceMatcher(None, key, stem_key).ratio())
    fuzzy_score = max(fuzzy_values, default=0.0)

    path_bonus = 0.04 if any(len(key) >= 3 and key in path_key for key in term_keys) else 0.0
    score = min(0.96, max(partial_score, fuzzy_score) + path_bonus)
    return score, "partial" if partial_score >= fuzzy_score else "typo"


def _deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for item in candidates:
        key = str(item.get("path") or "").replace("/", "\\").casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def folder_agent(query: str) -> tuple[list, str | None]:
    """Direct structured lookup for 'what's in my Downloads/Desktop/...'
    requests - a plain SQLite path match, never touches embeddings or the
    vector store. Returns (files, folder_name)."""
    folder_name = detect_folder(query.lower())
    if not folder_name:
        return [], None
    return get_files_by_folder(folder_name), folder_name


def file_agent(query: str) -> list:
    """Filename/path keyword matches - catches files whose content was
    never embedded (installers, unsupported extensions, etc.), and still
    works even if Ollama's embedding model is down."""
    phrase, terms = _query_terms(query)
    if not terms:
        return []

    # Gather a deliberately wider metadata pool, then rank it here.  This
    # prevents SQLite's "most recently modified" ordering from putting a
    # partial result above an exact filename while retaining typo fallback.
    candidates = list(search_files_keywords(query.casefold(), limit=30))
    probes = list(dict.fromkeys([phrase, *terms[:6]]))
    for probe in probes:
        if probe:
            candidates.extend(search_files_by_name(probe, limit=20))

    ranked = []
    for item in _deduplicate_candidates(candidates):
        score, match_type = _candidate_score(item, phrase, terms)
        # Fuzzy coincidences are common in a large filesystem.  A genuine
        # one/two-character typo normally scores well above 0.72; weaker
        # guesses should produce an honest not-found result instead.  Exact
        # and substring/partial matches use their own calibrated floor.
        if match_type == "typo":
            min_score = 0.78 if max((len(_normal_key(term)) for term in terms), default=0) <= 3 else 0.72
        elif match_type == "partial":
            min_score = 0.62
        else:
            min_score = 0.0
        if score < min_score:
            continue
        enriched = dict(item)
        enriched["match_score"] = round(score, 4)
        enriched["match_type"] = match_type
        ranked.append(enriched)

    # When the complete requested name/stem exists, do not pad the answer
    # with weaker substring coincidences such as ``zipapp.pyc`` for
    # ``app.py``.  Multiple exact paths are retained because duplicate
    # filenames in different folders are legitimate results.
    if any(item["match_type"] == "exact" and item["match_score"] >= 0.99 for item in ranked):
        ranked = [
            item for item in ranked
            if item["match_type"] == "exact" and item["match_score"] >= 0.99
        ]

    ranked.sort(key=lambda item: (item["match_score"], str(item.get("modified_at") or "")), reverse=True)
    return ranked[:5]
