"""Shared pytest fixtures.

The two important jobs here:

1. Keep every test hermetic. libsync stores caches/CSVs/logs under a base dir
   resolved from the LIBSYNC_DATA_DIR env var (see libsync.utils.filepath_utils).
   We point that at a throwaway temp dir for the whole session *before any test
   module imports libsync*, and again at a fresh per-test dir, so nothing ever
   touches the developer's real ~/.libsync.

2. Reset module-level singletons (SpotifyAuthManager) between tests so state
   never leaks from one test to the next.
"""

import os
import tempfile

import pytest

# Set a session-wide default BEFORE libsync is imported anywhere. conftest.py is
# imported by pytest ahead of test modules, so the import-time snapshot constants
# in filepath_utils also land in this temp dir rather than in $HOME.
os.environ.setdefault("LIBSYNC_DATA_DIR", tempfile.mkdtemp(prefix="libsync-test-"))


@pytest.fixture(autouse=True)
def isolate_data_dir(tmp_path, monkeypatch):
    """Redirect all libsync on-disk storage to a per-test temp directory."""
    data_dir = tmp_path / "libsync_data"
    data_dir.mkdir()
    monkeypatch.setenv("LIBSYNC_DATA_DIR", str(data_dir))
    return data_dir


@pytest.fixture(autouse=True)
def reset_spotify_auth_singleton():
    """Clear SpotifyAuthManager class-level cache before and after each test."""
    from libsync.spotify.spotify_auth import SpotifyAuthManager

    def _clear():
        SpotifyAuthManager._auth_manager = None
        SpotifyAuthManager._user_id = None
        SpotifyAuthManager._spotify_client = None
        SpotifyAuthManager._access_token = None

    _clear()
    yield
    _clear()


@pytest.fixture(autouse=True)
def reset_log_file_handle():
    """Reset string_utils' lazily-opened log handle around each test.

    The handle is cached at module level, so without this the first test to log
    anything would open a file in *its* tmp dir and every later test would keep
    writing to that (now-deleted) directory. Resetting per-test means each test's
    log lands in its own isolated LIBSYNC_DATA_DIR, and closing it avoids leaking
    an unclosed file handle (ResourceWarning).
    """
    from libsync.utils import string_utils

    def _close():
        if string_utils._output_file is not None:
            string_utils._output_file.close()
            string_utils._output_file = None

    _close()
    yield
    _close()


@pytest.fixture
def fake_spotify_auth(monkeypatch):
    """Stub out Spotify auth so no real credentials or network are needed.

    Returns a small record of what was requested so tests can assert on e.g.
    token refreshes.
    """
    from libsync.spotify.spotify_auth import SpotifyAuthManager

    state = {"access_token_calls": [], "user_id": "test-user"}

    def fake_get_access_token(cls, force_refresh: bool = False):
        state["access_token_calls"].append(force_refresh)
        return "refreshed-token" if force_refresh else "test-token"

    monkeypatch.setattr(SpotifyAuthManager, "get_access_token", classmethod(fake_get_access_token))
    monkeypatch.setattr(
        SpotifyAuthManager, "get_user_id", classmethod(lambda cls: state["user_id"])
    )
    return state
