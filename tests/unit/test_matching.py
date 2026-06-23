"""Unit tests for the pure matching helpers in get_spotify_matches.

These pick/score/select functions are the heart of sync correctness and need no
network — they operate on already-fetched search results.
"""

from factories import make_rb_track, make_spotify_collection, make_spotify_track

from libsync.spotify import get_spotify_matches as gsm
from libsync.utils.constants import SpotifyMappingDbFlags


def test_get_sorted_list_tracks_with_similarity_sorts_descending():
    rb_track = make_rb_track(name="Hold On", artist="Taiki Nulight")
    collection = make_spotify_collection(
        make_spotify_track(uri="spotify:track:bad", name="Nope", artists=["Nobody"]),
        make_spotify_track(uri="spotify:track:good", name="Hold On", artists=["Taiki Nulight"]),
    )
    ranked = gsm.get_sorted_list_tracks_with_similarity(rb_track, collection)
    scores = [score for _track, score in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0][0]["uri"] == "spotify:track:good"


def test_pick_matching_track_automatically_returns_uri_above_threshold():
    rb_track = make_rb_track(name="Hold On", artist="Taiki Nulight")
    collection = make_spotify_collection(
        make_spotify_track(uri="spotify:track:good", name="Hold On", artists=["Taiki Nulight"])
    )
    assert gsm.pick_matching_track_automatically(rb_track, collection) == "spotify:track:good"


def test_pick_matching_track_automatically_returns_none_below_threshold():
    rb_track = make_rb_track(name="Hold On", artist="Taiki Nulight")
    collection = make_spotify_collection(
        make_spotify_track(
            uri="spotify:track:bad", name="Totally Unrelated", artists=["Someone Else"]
        )
    )
    assert gsm.pick_matching_track_automatically(rb_track, collection) is None


def test_pick_matching_track_automatically_empty_results_is_none():
    rb_track = make_rb_track()
    assert gsm.pick_matching_track_automatically(rb_track, {}) is None


def test_pick_matching_track_respects_custom_threshold():
    rb_track = make_rb_track(name="Hold On", artist="Taiki Nulight")
    # a near-but-imperfect artist match scores below 0.95 but above a loose 0.5
    collection = make_spotify_collection(
        make_spotify_track(uri="spotify:track:x", name="Hold On", artists=["Taiki"])
    )
    assert gsm.pick_matching_track_automatically(rb_track, collection) is None
    assert (
        gsm.pick_matching_track_automatically(rb_track, collection, min_similarity_threshold=0.4)
        == "spotify:track:x"
    )


def test_get_spotify_queries_from_rb_track_are_deduped_and_encoded():
    rb_track = make_rb_track(name="Hold On", artist="Taiki Nulight")
    queries = gsm.get_spotify_queries_from_rb_track(rb_track)

    assert len(queries) == len(set(queries)), "queries should be deduplicated"
    assert all(q == q.lower() for q in queries), "queries should be lowercased"
    assert all(" " not in q for q in queries), "spaces should be url-encoded to +"
    # both the title and artist should appear among the generated queries
    assert any("hold" in q for q in queries)
    assert any("taiki" in q for q in queries)


def test_get_cached_results_for_track_collects_tracks_for_its_queries():
    rb_track = make_rb_track(name="Hold On", artist="Taiki Nulight")
    queries = gsm.get_spotify_queries_from_rb_track(rb_track)
    matched_track = make_spotify_track(uri="spotify:track:good", name="Hold On")
    # park the result under one of the queries this track generates
    search_results = {queries[0]: [matched_track], "unrelated query": []}

    found = gsm.get_cached_results_for_track(rb_track, search_results)
    assert found == {"spotify:track:good": matched_track}


class TestGetRbTrackIdsToMatch:
    def test_unmatched_tracks_are_included(self):
        result = gsm.get_rb_track_ids_to_match(["1", "2"], {}, set())
        assert result == ["1", "2"]

    def test_already_matched_tracks_are_skipped(self):
        result = gsm.get_rb_track_ids_to_match(["1", "2"], {"2": "spotify:track:x"}, set())
        assert result == ["1"]

    def test_tracks_flagged_for_rematch_are_included(self):
        result = gsm.get_rb_track_ids_to_match(
            ["1", "2"], {"1": "spotify:track:x", "2": "spotify:track:y"}, {"1"}
        )
        assert result == ["1"]

    def test_previously_skipped_tracks_are_retried(self):
        result = gsm.get_rb_track_ids_to_match(
            ["1"], {"1": SpotifyMappingDbFlags.SKIP_TRACK}, set()
        )
        assert result == ["1"]
