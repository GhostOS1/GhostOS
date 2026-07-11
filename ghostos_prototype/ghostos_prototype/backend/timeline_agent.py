"""
agents/timeline_agent.py
Timeline Agent - recent activity from the events table (populated by
indexer.py and browser_connector.py, see vectorstore.py's `events` table).
Only pulled when the question actually sounds like it's asking about
*when*/*recently* something happened, not on every request. Split out of
app.py; logic unchanged, just relocated into its own agent module.
"""

from vectorstore import get_timeline

TIME_INTENT_KEYWORDS = {
    "today", "yesterday", "recent", "recently", "earlier", "this morning",
    "last night", "just now", "my day", "my activity", "what did i do",
}


def wants_timeline(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in TIME_INTENT_KEYWORDS)


def timeline_agent(query: str, limit: int = 8) -> list:
    if not wants_timeline(query):
        return []
    return get_timeline(limit=limit)
