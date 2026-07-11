"""
agents/browser_agent.py
Browser Agent - doesn't hit the network or the DB itself. It pulls the
browser-history hits back out of Memory Agent's search results so they
can be labeled as "recently visited" in the context instead of looking
like a regular indexed file. browser_connector.py stores visited pages as
chunks in the same vectorstore table as files (see vectorstore.py),
tagged with source_type='browser_history_<browser>' - this just filters
for that tag.

Note (see the honest audit): this agent still only sees browsing
*history*. Tabs, bookmarks, downloads, form data, and sessions - the
richer "Better Browser Intelligence" scope from the roadmap - aren't
implemented; that would require a live browser extension/CDP connection,
not just reading the History sqlite file.
"""


def browser_agent(search_results: list) -> list:
    return [m for m in search_results if str(m.get("source_type", "")).startswith("browser_history_")]
