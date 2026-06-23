"""Unit tests for the pure diff helpers in sync_spotify_playlists.

This is the logic behind what gets written to Spotify — including the
overwrite-vs-append behavior that has regressed before — so it's worth pinning
down precisely. All pure: no network, no spotipy client.
"""

from factories import make_rb_track

from libsync.spotify import sync_spotify_playlists as ssp
from libsync.utils.constants import SpotifyMappingDbFlags


def uri(track_id: str) -> str:
    return f"spotify:track:{track_id}"


def playlist(name: str, track_ids: list[str]):
    from libsync.utils.rekordbox_library import RekordboxPlaylist

    return RekordboxPlaylist(name=name, tracks=track_ids)


class TestGetFilteredSpotifyUris:
    def test_keeps_only_valid_uris_in_order(self):
        rb_playlist = playlist("p1", ["1", "2", "3", "4"])
        mapping = {
            "1": uri("a"),
            "2": SpotifyMappingDbFlags.NOT_ON_SPOTIFY,
            "3": SpotifyMappingDbFlags.SKIP_TRACK,
            # "4" intentionally unmapped
        }
        result = ssp.get_filtered_spotify_uris_from_rekordbox_playlist(rb_playlist, mapping)
        assert result == [uri("a")]


class TestGetPlaylistDiffs:
    def test_fresh_playlist_gets_full_add_job(self):
        rb_playlists = [playlist("p1", ["1"])]
        mapping = {"1": uri("a")}
        playlist_id_map = {"p1": "spid1"}
        libsync_owned = {}  # playlist not yet known to libsync

        jobs, additions = ssp.get_playlist_diffs(
            rb_playlists, mapping, playlist_id_map, libsync_owned, overwrite_spotify_playlists=False
        )
        assert jobs == [["spid1", [uri("a")]]]
        assert additions == {}

    def test_overwrite_removes_spotify_only_tracks(self):
        # rekordbox has only track a; spotify playlist currently has a and b.
        rb_playlists = [playlist("p1", ["1"])]
        mapping = {"1": uri("a")}
        playlist_id_map = {"p1": "spid1"}
        libsync_owned = {"spid1": ["a", "b"]}  # stored as bare track ids

        jobs, additions = ssp.get_playlist_diffs(
            rb_playlists, mapping, playlist_id_map, libsync_owned, overwrite_spotify_playlists=True
        )
        # b is dropped — the target is exactly the rekordbox set
        assert jobs == [["spid1", [uri("a")]]]
        # b is still surfaced as a spotify-only addition (for the rekordbox report)
        assert additions == {"p1": [uri("b")]}

    def test_non_overwrite_appends_spotify_only_tracks_to_end(self):
        # rekordbox has a, c; spotify has a, b. Without overwrite, b is appended.
        rb_playlists = [playlist("p1", ["1", "2"])]
        mapping = {"1": uri("a"), "2": uri("c")}
        playlist_id_map = {"p1": "spid1"}
        libsync_owned = {"spid1": ["a", "b"]}

        jobs, additions = ssp.get_playlist_diffs(
            rb_playlists, mapping, playlist_id_map, libsync_owned, overwrite_spotify_playlists=False
        )
        assert jobs == [["spid1", [uri("a"), uri("c"), uri("b")]]]
        assert additions == {"p1": [uri("b")]}

    def test_no_change_produces_no_job(self):
        rb_playlists = [playlist("p1", ["1"])]
        mapping = {"1": uri("a")}
        playlist_id_map = {"p1": "spid1"}
        libsync_owned = {"spid1": ["a"]}

        jobs, additions = ssp.get_playlist_diffs(
            rb_playlists, mapping, playlist_id_map, libsync_owned, overwrite_spotify_playlists=True
        )
        assert jobs == []
        assert additions == {}

    def test_playlist_not_in_id_map_is_skipped(self):
        rb_playlists = [playlist("unmapped", ["1"])]
        mapping = {"1": uri("a")}
        jobs, additions = ssp.get_playlist_diffs(
            rb_playlists, mapping, {}, {}, overwrite_spotify_playlists=True
        )
        assert jobs == []
        assert additions == {}


class TestCalculateDiff:
    def test_identifies_new_downloads(self):
        # b is on spotify but not mapped back to any rekordbox track => needs download
        new_spotify_additions = {"p1": [uri("b")]}
        rekordbox_to_spotify_map = {"1": uri("a")}

        downloads, songs_to_playlists, playlists_to_songs = ssp.calculate_diff(
            new_spotify_additions, rekordbox_to_spotify_map
        )
        assert downloads == {uri("b")}
        assert songs_to_playlists == {uri("b"): ["p1"]}
        assert playlists_to_songs == {"p1": [uri("b")]}

    def test_track_already_in_collection_is_not_a_download(self):
        new_spotify_additions = {"p1": [uri("a")]}
        rekordbox_to_spotify_map = {"1": uri("a")}
        downloads, _, _ = ssp.calculate_diff(new_spotify_additions, rekordbox_to_spotify_map)
        assert downloads == set()


class TestDescribeSpotifyUri:
    def test_resolves_track_in_collection(self):
        collection = {"1": make_rb_track(id="1", name="Hold On", artist="Taiki Nulight")}
        reverse_map = {uri("a"): "1"}
        assert (
            ssp.describe_spotify_uri(uri("a"), reverse_map, collection) == "Taiki Nulight - Hold On"
        )

    def test_falls_back_when_unmapped(self):
        out = ssp.describe_spotify_uri(uri("a"), {}, {})
        assert out == f"{uri('a')} (track not in rekordbox collection)"

    def test_falls_back_when_mapped_but_missing_from_collection(self):
        reverse_map = {uri("a"): "1"}  # maps to a track id absent from the collection
        out = ssp.describe_spotify_uri(uri("a"), reverse_map, {})
        assert out == f"{uri('a')} (track not in rekordbox collection)"
