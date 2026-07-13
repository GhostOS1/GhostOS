"""Select historical browser results from Memory Agent retrieval.

``browser_connector.py`` indexes Chrome/Edge history, bookmarks, and
downloads under the established ``browser_history_`` routing namespace;
their structured content/source-type suffix keeps the real record kind clear.
This lightweight agent only separates those results from local-file results.

Live tabs are intentionally not claimed.  They still require an explicitly
configured browser extension or local CDP provider implementing the interface
in ``browser_intelligence.py``.
"""


def browser_agent(search_results: list) -> list:
    return [m for m in search_results if str(m.get("source_type", "")).startswith("browser_history_")]
