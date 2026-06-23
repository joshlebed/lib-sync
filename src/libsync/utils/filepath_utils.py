"""Centralized filepath utilities for libsync.

This module contains all path-related constants and functions used across the libsync codebase.
All data is stored under ~/.libsync/data by default, with organized subdirectories for different
purposes. The base directory can be overridden with the LIBSYNC_DATA_DIR environment variable
(useful for relocating data, or for keeping test runs out of your real home directory).
"""

import os
import time
from datetime import datetime
from pathlib import Path

# Default base directory for all libsync storage.
DEFAULT_LIBSYNC_DATA_DIR = Path.home() / ".libsync" / "data"


def get_data_dir() -> Path:
    """Return the base libsync data directory, creating it if needed.

    Resolved at call time from the LIBSYNC_DATA_DIR environment variable so the
    location can be redirected per-process (e.g. tests point it at a tmp dir).
    Falls back to ~/.libsync/data.
    """
    data_dir = Path(os.environ.get("LIBSYNC_DATA_DIR") or DEFAULT_LIBSYNC_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _get_subdir(name: str) -> Path:
    """Return (and create) a named subdirectory under the data dir."""
    subdir = get_data_dir() / name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def get_logs_dir() -> Path:
    return _get_subdir("logs")


def get_spotify_playlist_backups_dir() -> Path:
    return _get_subdir("spotify_playlist_backups")


def get_rekordbox_xml_backups_dir() -> Path:
    return _get_subdir("rekordbox_xml_backups")


def get_shazam_cache_dir() -> Path:
    return _get_subdir("shazam_cache")


# Backwards-compatible module-level constants. These are snapshots taken at
# import time; prefer the get_*() helpers above, which honor LIBSYNC_DATA_DIR at
# call time. Kept because the id/ pipeline imports them directly.
LIBSYNC_DATA_DIR = get_data_dir()
LIBSYNC_LOGS_DIR = get_logs_dir()
SPOTIFY_PLAYLIST_BACKUPS_DIR = get_spotify_playlist_backups_dir()
REKORDBOX_XML_BACKUPS_DIR = get_rekordbox_xml_backups_dir()


# Spotify-related paths
def get_spotify_playlist_mapping_db_path(rekordbox_xml_path: str, user_id: str) -> str:
    """Get path for Spotify playlist mapping cache."""
    return str(
        get_data_dir()
        / f"libsync_spotify_playlist_mapping_cache_{get_sanitized_xml_path(rekordbox_xml_path)}_{user_id}.csv"
    )


def get_user_spotify_playlists_list_db_path(user_id: str) -> str:
    """Get path for user's Spotify playlists cache."""
    return str(get_data_dir() / f"libsync_spotify_playlists_cache_{user_id}.txt")


def get_spotify_search_cache_path(rekordbox_xml_path: str) -> str:
    """Get path for Spotify search results cache."""
    return str(
        get_data_dir()
        / f"libsync_search_results_cache_{get_sanitized_xml_path(rekordbox_xml_path)}.db"
    )


def get_spotify_playlist_cache_path() -> str:
    """Get the path for the primary spotify playlist cache file."""
    return str(get_spotify_playlist_backups_dir() / "playlists.pickle")


def get_spotify_playlist_backup_path() -> str:
    """Get the path for a timestamped spotify playlist backup file."""
    return str(
        get_spotify_playlist_backups_dir()
        / f"playlists_{time.strftime('%Y.%m.%d_%H.%M.%S')}.pickle"
    )


# Rekordbox-related paths
def get_libsync_song_mapping_csv_path(rekordbox_xml_path: str) -> str:
    """Get path for libsync song mapping cache."""
    return str(
        get_data_dir()
        / f"libsync_song_mapping_cache_{get_sanitized_xml_path(rekordbox_xml_path)}.csv"
    )


def get_libsync_pending_tracks_spotify_to_rekordbox_db_path(
    rekordbox_xml_path: str,
) -> str:
    """Get path for pending tracks from Spotify to Rekordbox."""
    return str(
        get_data_dir()
        / f"libsync_pending_tracks_spotify_to_rekordbox_cache_{get_sanitized_xml_path(rekordbox_xml_path)}.csv"
    )


def get_rekordbox_xml_backup_path(xml_path: str, mtime: float) -> Path:
    """Get backup path for a Rekordbox XML file using its modification time."""
    mtime_datetime = datetime.fromtimestamp(mtime)
    timestamp_str = mtime_datetime.strftime("%Y.%m.%d_%H.%M.%S")
    xml_filename = Path(xml_path).name
    backup_filename = f"{Path(xml_filename).stem}_{timestamp_str}.xml"
    return get_rekordbox_xml_backups_dir() / backup_filename


# Log file paths
def get_log_file_path() -> Path:
    """Get path for current log file with timestamp."""
    log_filename = f"libsync_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    return get_logs_dir() / log_filename


# Failed matches export
def get_failed_matches_export_path() -> Path:
    """Get path for failed matches export file."""
    filename = f"failed_matches_{datetime.now()}.txt".replace(" ", "_")
    return get_data_dir() / filename


# YouTube download paths
def get_youtube_download_output_template() -> str:
    """Get output template for YouTube downloads."""
    return str(get_data_dir() / "%(id)s_audio_download")


# Shazam cache paths (snapshot constants; see get_shazam_cache_dir() for call-time resolution)
SHAZAM_CACHE_DIR = get_shazam_cache_dir()
SHAZAM_GLOBAL_CACHE_PATH = str(SHAZAM_CACHE_DIR / "shazam_global_cache.db")


def get_shazam_segment_cache_path(audio_file_path: str) -> str:
    """Get path for Shazam segment-level cache database.

    The cache is named using a hash of the audio file path to keep
    the filename short while remaining unique per audio file.

    Args:
        audio_file_path: Path to the audio file being processed

    Returns:
        Path to the SQLite cache database
    """
    import hashlib

    # Create a short hash of the audio path for the filename
    path_hash = hashlib.sha256(audio_file_path.encode()).hexdigest()[:12]
    return str(get_shazam_cache_dir() / f"shazam_cache_{path_hash}.db")


# Utility functions
def get_sanitized_xml_path(xml_path: str) -> str:
    """Sanitize XML path for use in filenames."""
    return xml_path.replace("/", "_")
