"""Unit tests for libsync.utils.similarity_utils — pure scoring, no mocks."""

from factories import make_rb_track, make_spotify_collection, make_spotify_track

from libsync.utils import similarity_utils


def test_get_string_similarity_identical_is_one():
    assert similarity_utils.get_string_similarity("hold on", "hold on") == 1.0


def test_get_string_similarity_is_case_insensitive():
    assert similarity_utils.get_string_similarity("Hold On", "hold on") == 1.0


def test_get_string_similarity_disjoint_is_low():
    assert similarity_utils.get_string_similarity("abc", "xyz") < 0.2


def test_calculate_similarity_metric_is_product():
    metric = similarity_utils.calculate_similarity_metric(
        {"name_similarity": 0.5, "artist_similarity": 0.4}
    )
    assert metric == 0.2


def test_remove_accents_normalizes():
    # NFKD decomposition splits the accent off the base letter
    assert "e" in similarity_utils.remove_accents("café")


def test_calculate_similarities_exact_match_scores_one():
    rb_track = make_rb_track(name="Hold On", artist="Taiki Nulight")
    collection = make_spotify_collection(
        make_spotify_track(uri="spotify:track:a", name="Hold On", artists=["Taiki Nulight"])
    )
    scores = similarity_utils.calculate_similarities(rb_track, collection)
    assert scores["spotify:track:a"] == 1.0


def test_calculate_similarities_ranks_better_match_higher():
    rb_track = make_rb_track(name="Hold On", artist="Taiki Nulight")
    collection = make_spotify_collection(
        make_spotify_track(uri="spotify:track:good", name="Hold On", artists=["Taiki Nulight"]),
        make_spotify_track(
            uri="spotify:track:bad", name="Completely Different", artists=["Other Person"]
        ),
    )
    scores = similarity_utils.calculate_similarities(rb_track, collection)
    assert scores["spotify:track:good"] > scores["spotify:track:bad"]


def test_calculate_similarities_ignores_suffix_on_spotify_name():
    # remove_suffixes is applied to the spotify name, so "(Original Mix)" shouldn't hurt
    rb_track = make_rb_track(name="Hold On", artist="Taiki Nulight")
    collection = make_spotify_collection(
        make_spotify_track(
            uri="spotify:track:a", name="Hold On (Original Mix)", artists=["Taiki Nulight"]
        )
    )
    scores = similarity_utils.calculate_similarities(rb_track, collection)
    assert scores["spotify:track:a"] == 1.0
