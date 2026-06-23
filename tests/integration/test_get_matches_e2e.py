"""End-to-end test of the matching pipeline (get_spotify_matches).

Drives the whole search -> score -> auto-match -> persist flow with the Spotify
search call faked at the spotify_api_utils seam and interactive mode disabled,
so there's no network and no stdin. On-disk caches/CSVs land in the per-test
LIBSYNC_DATA_DIR (see conftest).
"""

import csv

from factories import make_rb_library, make_rb_track, make_spotify_collection, make_spotify_track

from libsync.spotify import get_spotify_matches as gsm
from libsync.spotify import spotify_api_utils
from libsync.utils import filepath_utils


def _fake_search_returning(candidates):
    """Build a fake get_spotify_search_results that returns the same candidates
    for every query (the matcher then picks by similarity)."""

    def _search(queries):
        return {query: list(candidates) for query in queries}

    return _search


def run_matches(monkeypatch, library, candidates, *, mapping=None, pending=None):
    monkeypatch.setattr(
        spotify_api_utils, "get_spotify_search_results", _fake_search_returning(candidates)
    )
    return gsm.get_spotify_matches(
        rekordbox_to_spotify_map=mapping if mapping is not None else {},
        rekordbox_library=library,
        rb_track_ids_flagged_for_rematch=set(),
        pending_tracks_spotify_to_rekordbox=pending if pending is not None else {},
        ignore_spotify_search_cache=False,
        skip_interactive_mode=True,
        skip_interactive_mode_pending_tracks=True,
    )


def test_auto_matches_exact_track(monkeypatch):
    track = make_rb_track(id="t1", name="Hold On", artist="Taiki Nulight")
    library = make_rb_library(tracks=[track], xml_path="lib.xml")
    match = make_spotify_track(uri="spotify:track:good", name="Hold On", artists=["Taiki Nulight"])

    result = run_matches(monkeypatch, library, [match])

    assert result == {"t1": "spotify:track:good"}


def test_no_match_below_threshold_leaves_track_unmapped(monkeypatch):
    track = make_rb_track(id="t1", name="Hold On", artist="Taiki Nulight")
    library = make_rb_library(tracks=[track], xml_path="lib.xml")
    unrelated = make_spotify_track(
        uri="spotify:track:x", name="Totally Different", artists=["Someone Else"]
    )

    result = run_matches(monkeypatch, library, [unrelated])

    # auto-match failed and interactive mode is skipped -> no entry persisted
    assert result == {}


def test_writes_song_mapping_csv(monkeypatch):
    track = make_rb_track(id="t1", name="Hold On", artist="Taiki Nulight")
    library = make_rb_library(tracks=[track], xml_path="lib.xml")
    match = make_spotify_track(uri="spotify:track:good", name="Hold On", artists=["Taiki Nulight"])

    run_matches(monkeypatch, library, [match])

    csv_path = filepath_utils.get_libsync_song_mapping_csv_path("lib.xml")
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0][0] == "Rekordbox id"  # header present
    data_rows = rows[1:]
    assert any(r[0] == "t1" and r[3] == "spotify:track:good" for r in data_rows)


def test_matches_against_pending_tracks_first(monkeypatch):
    track = make_rb_track(id="t1", name="Hold On", artist="Taiki Nulight")
    library = make_rb_library(tracks=[track], xml_path="lib.xml")
    pending = make_spotify_collection(
        make_spotify_track(uri="spotify:track:pending", name="Hold On", artists=["Taiki Nulight"])
    )
    # search returns nothing useful; the match should come from pending tracks
    result = run_matches(monkeypatch, library, [], pending=pending)

    assert result["t1"] == "spotify:track:pending"


def test_already_matched_track_is_left_untouched(monkeypatch):
    track = make_rb_track(id="t1", name="Hold On", artist="Taiki Nulight")
    library = make_rb_library(tracks=[track], xml_path="lib.xml")
    # a different candidate is offered, but the existing mapping should win
    other = make_spotify_track(uri="spotify:track:new", name="Hold On", artists=["Taiki Nulight"])

    result = run_matches(monkeypatch, library, [other], mapping={"t1": "spotify:track:existing"})

    assert result == {"t1": "spotify:track:existing"}
