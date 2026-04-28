"""Global content-addressed cache for Shazam segment results.

Unlike the per-file SegmentCache (keyed by audio_file_hash + start_ms + duration_ms),
this cache is keyed by the SHA256 of the actual segment MP3 bytes. This means identical
audio segments produce cache hits even when extracted from different source files
(e.g., a trimmed recording vs. the full recording).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING

from libsync.id.shazam.models import SegmentResult

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger("libsync")


class GlobalSegmentCache:
    """Content-addressed global cache for Shazam results.

    Keyed by SHA256 hash of segment MP3 file bytes.
    Results stored WITHOUT start_ms (same audio can appear at different positions).
    Single SQLite DB at ~/.libsync/data/shazam_cache/shazam_global_cache.db
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_segment_results (
                    content_hash TEXT PRIMARY KEY,
                    result_json TEXT,
                    track_id TEXT,
                    title TEXT,
                    artist TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, content_hash: str, start_ms: int = 0) -> SegmentResult | None:
        """Look up a cached result by content hash.

        The cache stores results without position; callers pass the desired
        start_ms so the returned SegmentResult is positioned correctly.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM global_segment_results WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()

            if row is None:
                return None

            raw_response = json.loads(row["result_json"]) if row["result_json"] else None
            return SegmentResult(
                start_ms=start_ms,
                raw_response=raw_response,
                track_id=row["track_id"],
                title=row["title"],
                artist=row["artist"],
            )

    def set(self, content_hash: str, result: SegmentResult) -> None:
        """Store a result keyed by content hash (start_ms is NOT stored)."""
        result_json = json.dumps(result.raw_response) if result.raw_response else None

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO global_segment_results
                (content_hash, result_json, track_id, title, artist)
                VALUES (?, ?, ?, ?, ?)
                """,
                (content_hash, result_json, result.track_id, result.title, result.artist),
            )

    def get_stats(self) -> dict[str, int]:
        """Return cache statistics."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM global_segment_results").fetchone()[0]
            with_match = conn.execute(
                "SELECT COUNT(*) FROM global_segment_results WHERE track_id IS NOT NULL"
            ).fetchone()[0]
            return {"total_entries": total, "entries_with_match": with_match}
