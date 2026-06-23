"""Integration tests for sync_spotify_playlists.

The spotipy client and the spotify_api_utils driver functions are faked and
record what they were asked to do, so we can assert on the *side effects*
(playlists created, write jobs issued) without any network. The key behaviors
pinned here: dry runs never write, and a declined confirmation never writes.
"""

from types import SimpleNamespace

import pytest
from factories import make_rb_track

from libsync.spotify import spotify_api_utils
from libsync.spotify import sync_spotify_playlists as ssp
from libsync.spotify.spotify_auth import SpotifyAuthManager
from libsync.utils.rekordbox_library import RekordboxPlaylist


class FakeSpotifyClient:
    """Records the mutating spotipy calls sync makes."""

    def __init__(self):
        self.created = []
        self.unfollowed = []
        self._counter = 0

    def user_playlist_create(self, user, name, public, description):
        self._counter += 1
        playlist_id = f"created-{self._counter}"
        self.created.append({"user": user, "name": name, "public": public, "id": playlist_id})
        return {"name": name, "id": playlist_id}

    def current_user_unfollow_playlist(self, playlist_id):
        self.unfollowed.append(playlist_id)


@pytest.fixture
def spotify_world(monkeypatch, fake_spotify_auth):
    """Fake every external Spotify surface sync_spotify_playlists touches."""
    client = FakeSpotifyClient()
    state = SimpleNamespace(
        client=client,
        owned={},  # libsync_owned_spotify_playlists: playlist_id -> [track_id, ...]
        all_playlists=set(),  # all of the user's spotify playlist ids
        overwrite_calls=[],  # each entry is the jobs list passed to overwrite_playlists
        song_details_calls=[],
    )

    monkeypatch.setattr(SpotifyAuthManager, "get_spotify_client", classmethod(lambda cls: client))
    monkeypatch.setattr(
        spotify_api_utils, "get_user_playlists_details", lambda ids: dict(state.owned)
    )
    monkeypatch.setattr(
        spotify_api_utils, "get_all_user_playlists_set", lambda: set(state.all_playlists)
    )
    monkeypatch.setattr(
        spotify_api_utils,
        "overwrite_playlists",
        lambda jobs: state.overwrite_calls.append(jobs) or [],
    )
    monkeypatch.setattr(
        spotify_api_utils,
        "get_spotify_song_details",
        lambda uris: state.song_details_calls.append(list(uris)) or {},
    )
    return state


def call_sync(monkeypatch, *, rekordbox_playlists, mapping, collection, dry_run, overwrite):
    return ssp.sync_spotify_playlists(
        rekordbox_xml_path="lib.xml",
        rekordbox_playlists=rekordbox_playlists,
        rekordbox_to_spotify_map=mapping,
        make_playlists_public=False,
        dry_run=dry_run,
        use_cached_spotify_playlist_data=False,
        collection=collection,
        overwrite_spotify_playlists=overwrite,
    )


def test_creates_playlist_and_issues_write_job_when_confirmed(monkeypatch, spotify_world):
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    track = make_rb_track(id="t1", name="Hold On", artist="Taiki Nulight")
    playlists = [RekordboxPlaylist(name="p1", tracks=["t1"])]
    mapping = {"t1": "spotify:track:a"}

    result = call_sync(
        monkeypatch,
        rekordbox_playlists=playlists,
        mapping=mapping,
        collection={"t1": track},
        dry_run=False,
        overwrite=True,
    )

    # a new spotify playlist was created for p1
    assert len(spotify_world.client.created) == 1
    assert spotify_world.client.created[0]["name"] == "[ls] p1"
    created_id = spotify_world.client.created[0]["id"]
    assert result["p1"] == created_id

    # and the tracks were written to it
    assert spotify_world.overwrite_calls == [[[created_id, ["spotify:track:a"]]]]


def test_dry_run_makes_no_writes(monkeypatch, spotify_world):
    # an existing libsync-owned playlist whose spotify contents differ from rekordbox
    monkeypatch.setattr(ssp.db_read_operations, "get_playlist_id_map", lambda _xml: {"p1": "pid1"})
    spotify_world.owned = {"pid1": ["b"]}  # spotify has track b
    spotify_world.all_playlists = {"pid1"}

    # if input() were called in a dry run, this would raise and fail the test
    def _no_input(_prompt):
        raise AssertionError("dry run should not prompt for confirmation")

    monkeypatch.setattr("builtins.input", _no_input)

    track = make_rb_track(id="t1", name="Hold On", artist="Taiki Nulight")
    playlists = [RekordboxPlaylist(name="p1", tracks=["t1"])]
    mapping = {"t1": "spotify:track:a"}  # rekordbox has track a

    call_sync(
        monkeypatch,
        rekordbox_playlists=playlists,
        mapping=mapping,
        collection={"t1": track},
        dry_run=True,
        overwrite=True,
    )

    # nothing was written or mutated despite there being a real diff
    assert spotify_world.overwrite_calls == []
    assert spotify_world.client.created == []
    assert spotify_world.client.unfollowed == []


def test_declined_confirmation_makes_no_writes(monkeypatch, spotify_world):
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    track = make_rb_track(id="t1", name="Hold On", artist="Taiki Nulight")
    playlists = [RekordboxPlaylist(name="p1", tracks=["t1"])]
    mapping = {"t1": "spotify:track:a"}

    call_sync(
        monkeypatch,
        rekordbox_playlists=playlists,
        mapping=mapping,
        collection={"t1": track},
        dry_run=False,
        overwrite=True,
    )

    # the playlist is still created (that's not gated by the confirmation),
    # but no track-write job is issued when the user declines
    assert spotify_world.overwrite_calls == []


def test_deletes_playlist_removed_from_rekordbox(monkeypatch, spotify_world):
    # mapping has p1 -> pid1, but rekordbox no longer has a p1 playlist
    monkeypatch.setattr(ssp.db_read_operations, "get_playlist_id_map", lambda _xml: {"p1": "pid1"})
    spotify_world.owned = {}
    spotify_world.all_playlists = {"pid1"}  # still exists on spotify -> eligible for delete
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    call_sync(
        monkeypatch,
        rekordbox_playlists=[],  # no rekordbox playlists at all
        mapping={},
        collection={},
        dry_run=False,
        overwrite=True,
    )

    assert spotify_world.client.unfollowed == ["pid1"]
