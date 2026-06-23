"""Integration tests for the Spotify aiohttp retry helpers.

These drive the real asyncio + aiohttp code paths in spotify_api_utils, but the
HTTP transport is faked with aioresponses — no network, no credentials. This is
where the 429-backoff / 401-refresh / give-up-after-N-retries logic is pinned
down.
"""

import asyncio

import aiohttp
import pytest
from aioresponses import aioresponses

from libsync.spotify import spotify_api_utils
from libsync.utils import constants

URL = "https://api.spotify.com/v1/tracks?ids=abc"


@pytest.fixture
def no_backoff(monkeypatch):
    """Skip the real backoff sleeps so retry tests run instantly."""

    async def _instant_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)


async def test_returns_json_on_success():
    with aioresponses() as mocked:
        mocked.get(URL, status=200, payload={"tracks": []})
        async with aiohttp.ClientSession() as session:
            result = await spotify_api_utils.get_from_spotify_with_retry(
                session, URL, {}, None, "fetching tracks"
            )
    assert result == {"tracks": []}


async def test_retries_after_429_then_succeeds(no_backoff):
    with aioresponses() as mocked:
        mocked.get(URL, status=429, headers={"Retry-After": "0"})
        mocked.get(URL, status=200, payload={"ok": True})
        async with aiohttp.ClientSession() as session:
            result = await spotify_api_utils.get_from_spotify_with_retry(
                session, URL, {}, None, "fetching tracks"
            )
    assert result == {"ok": True}


async def test_refreshes_token_on_401(no_backoff, fake_spotify_auth):
    with aioresponses() as mocked:
        mocked.get(URL, status=401)
        mocked.get(URL, status=200, payload={"ok": True})
        async with aiohttp.ClientSession() as session:
            result = await spotify_api_utils.get_from_spotify_with_retry(
                session, URL, {"Authorization": "Bearer stale"}, None, "fetching tracks"
            )
    assert result == {"ok": True}
    # a forced refresh (force_refresh=True) must have happened
    assert True in fake_spotify_auth["access_token_calls"]


async def test_gives_up_after_max_retries(no_backoff):
    call_count = 0

    def _count_callback(url, **kwargs):
        nonlocal call_count
        call_count += 1

    with aioresponses() as mocked:
        mocked.get(URL, status=500, body="server error", repeat=True, callback=_count_callback)
        async with aiohttp.ClientSession() as session:
            with pytest.raises(ConnectionError, match="failed after"):
                await spotify_api_utils.get_from_spotify_with_retry(
                    session, URL, {}, None, "fetching tracks"
                )
    # every retry attempt actually hit the (mocked) endpoint
    assert call_count == constants.MAX_RETRIES


async def test_modify_put_returns_json_on_success():
    put_url = "https://api.spotify.com/v1/playlists/p1/tracks"
    with aioresponses() as mocked:
        mocked.put(put_url, status=200, payload={"snapshot_id": "s1"})
        async with aiohttp.ClientSession() as session:
            result = await spotify_api_utils.modify_spotify_with_retry(
                session, put_url, {}, {"uris": []}, "PUT", "overwriting playlist"
            )
    assert result == {"snapshot_id": "s1"}


async def test_modify_rejects_unsupported_method():
    with aioresponses():
        async with aiohttp.ClientSession() as session:
            with pytest.raises(ValueError, match="Unsupported method"):
                await spotify_api_utils.modify_spotify_with_retry(
                    session, "https://api.spotify.com/x", {}, {}, "DELETE", "bad"
                )
