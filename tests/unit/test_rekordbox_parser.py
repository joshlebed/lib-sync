"""Unit test for the Rekordbox XML parser against a real sample export.

Reads a checked-in fixture file (not the network), so it's a fast, hermetic
end-to-end test of get_rekordbox_library. The backup-copy side effect lands in
the per-test LIBSYNC_DATA_DIR (see conftest).
"""

from pathlib import Path

from libsync.analyze.get_rekordbox_library import (
    get_rekordbox_library,
    should_keep_track_in_collection,
)

SAMPLE_XML = str(
    Path(__file__).resolve().parents[2] / "sample_data" / "small_example_rekordbox_export.xml"
)


def test_parses_collection_tracks():
    library = get_rekordbox_library(SAMPLE_XML, skip_collection_playlist=True)

    assert set(library.collection.keys()) == {"214856896", "44216344"}
    hold_on = library.collection["214856896"]
    assert hold_on.name == "Hold On (Original Mix)"
    assert hold_on.artist == "Taiki Nulight"
    assert hold_on.tonality == "7A"


def test_parses_playlists_and_filters_to_collection():
    library = get_rekordbox_library(SAMPLE_XML, skip_collection_playlist=True)

    playlists = {p.name: p.tracks for p in library.playlists}
    assert "playlist 1 (root)" in playlists
    # the playlist references one track key that exists in the collection
    assert playlists["playlist 1 (root)"] == ["44216344"]


def test_collection_playlist_added_when_not_skipped():
    library = get_rekordbox_library(SAMPLE_XML, skip_collection_playlist=False)
    playlists = {p.name: p.tracks for p in library.playlists}
    assert "Collection" in playlists
    assert set(playlists["Collection"]) == {"214856896", "44216344"}


def test_xml_path_is_recorded():
    library = get_rekordbox_library(SAMPLE_XML, skip_collection_playlist=True)
    assert library.xml_path == SAMPLE_XML


class _FakeTrack:
    def __init__(self, kind):
        self._kind = kind

    def get(self, key):
        return self._kind if key == "Kind" else None


def test_should_keep_track_excludes_unknown_format():
    assert should_keep_track_in_collection(_FakeTrack("MP3 File")) is True
    assert should_keep_track_in_collection(_FakeTrack("Unknown Format")) is False
