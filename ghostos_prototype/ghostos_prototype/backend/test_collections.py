from pathlib import Path

import pytest

import vectorstore
from indexer import classify_collection, refresh_catalog_collections


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (r"C:\Users\A\Projects\GhostOS\app.py", "Projects"),
        (r"C:\Users\A\ghostos_prototype\README.md", "Projects"),
        (r"C:\Users\A\Work\Client\meeting-notes.docx", "Work"),
        (r"C:\Users\A\College\semester-project\paper.pdf", "College"),
        (r"C:\Users\A\Personal\Family\photo.jpg", "Personal"),
        (r"C:\Users\A\Documents\Tax\invoice.pdf", "Finance"),
        (r"C:\Users\A\Downloads\weather-data.csv", "Other"),
        # Keyword matching uses path tokens, not accidental substrings.
        (r"C:\Users\A\Documents\homework.py", "Other"),
        (r"C:\USERS\A\WORK\REPORT.PDF", "Work"),
    ],
)
def test_classify_collection_has_meaningful_case_insensitive_buckets(path, expected):
    assert classify_collection(Path(path)) == expected


def test_refresh_catalog_collections_reclassifies_existing_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(vectorstore, "DB_PATH", tmp_path / "collections.db")
    vectorstore.init_db()

    fixtures = [
        ("one", r"C:\Users\A\Projects\GhostOS\app.py", "Projects"),
        ("two", r"C:\Users\A\Work\Client\brief.pdf", "Work"),
        ("three", r"C:\Users\A\Downloads\weather.csv", "Other"),
    ]
    for file_hash, path, _expected in fixtures:
        vectorstore.upsert_file(
            file_hash=file_hash,
            path=path,
            name=Path(path).name,
            extension=Path(path).suffix,
            category="Others",
            collection="Projects",  # legacy classifier's catch-all value
            size_bytes=1,
            modified_at="2026-07-13T10:00:00+05:30",
            embedded=False,
        )

    result = refresh_catalog_collections()

    assert result == {"scanned": 3, "updated": 2}
    counts = {item["name"]: item["count"] for item in vectorstore.get_collections()}
    assert counts == {"Projects": 1, "Work": 1, "Other": 1}
    assert vectorstore.get_recent_files(collection="Other")[0]["path"].endswith(
        "weather.csv"
    )

    # The repair is idempotent, so running it at each safe backend startup is cheap.
    assert refresh_catalog_collections() == {"scanned": 3, "updated": 0}
