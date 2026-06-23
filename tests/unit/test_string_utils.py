"""Unit tests for libsync.utils.string_utils — pure string helpers, no mocks."""

import pytest
from factories import make_rb_track, make_spotify_track

from libsync.utils import string_utils
from libsync.utils.constants import SPOTIFY_TRACK_URI_PREFIX


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Hold On (Original Mix)", "Hold On "),
        ("Track [Original Version]", "Track "),
        ("Song (ORIGINAL)", "Song "),
        ("no suffix here", "no suffix here"),
    ],
)
def test_remove_original_mix(title, expected):
    assert string_utils.remove_original_mix(title) == expected


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Banger (Extended Mix)", "Banger "),
        ("Banger (Extended Version)", "Banger "),
        ("Banger extended", "Banger "),
    ],
)
def test_remove_extended_mix(title, expected):
    assert string_utils.remove_extended_mix(title) == expected


def test_remove_radio_and_bootleg():
    assert string_utils.remove_radio_mix("Tune (Radio Edit)") == "Tune "
    assert string_utils.remove_bootleg("Tune (Bootleg)") == "Tune "


def test_remove_suffixes_is_case_insensitive_and_chained():
    # remove_suffixes applies all of the individual removers
    assert string_utils.remove_suffixes("Hold On (original mix)").strip() == "Hold On"
    assert string_utils.remove_suffixes("Hold On (EXTENDED MIX)").strip() == "Hold On"


def test_strip_punctuation():
    assert string_utils.strip_punctuation("a.b,c!d?") == "abcd"
    assert string_utils.strip_punctuation("hello world") == "hello world"


def test_get_name_varieties_dedupes():
    # a name with no suffix yields a single variety; one with a suffix yields two
    assert string_utils.get_name_varieties_from_track_name("Plain Title") == ["Plain Title"]
    varieties = set(string_utils.get_name_varieties_from_track_name("Title (Original Mix)"))
    assert "Title" in varieties
    assert "Title (Original Mix)" in varieties


@pytest.mark.parametrize(
    "artist_field, expected",
    [
        ("Solo Artist", ["Solo Artist"]),
        ("A & B", ["A", "B"]),
        ("A, B", ["A", "B"]),
        ("A feat. B", ["A", "B"]),
        ("A ft. B", ["A", "B"]),
        ("A / B", ["A", "B"]),
    ],
)
def test_get_artists_from_rb_track_splits_on_delimiters(artist_field, expected):
    track = make_rb_track(artist=artist_field)
    assert string_utils.get_artists_from_rb_track(track) == expected


def test_spotify_uri_id_roundtrip():
    track_id = "4iV5W9uYEdYUVa79Axb7Rh"
    uri = string_utils.get_spotify_uri_from_id(track_id)
    assert uri == SPOTIFY_TRACK_URI_PREFIX + track_id
    assert string_utils.is_spotify_uri(uri)
    assert string_utils.get_spotify_id_from_uri(uri) == track_id


def test_is_spotify_uri_false_for_other_values():
    assert not string_utils.is_spotify_uri("libsync:NOT_ON_SPOTIFY")
    assert not string_utils.is_spotify_uri("https://open.spotify.com/track/x")


def test_get_spotify_uri_from_url_offline():
    # spotipy parses the URL locally (no network) into a track URI
    track_id = "4iV5W9uYEdYUVa79Axb7Rh"
    url = f"https://open.spotify.com/track/{track_id}"
    assert string_utils.get_spotify_uri_from_url(url) == f"spotify:track:{track_id}"


def test_generate_spotify_playlist_name():
    assert string_utils.generate_spotify_playlist_name("Techno") == "[ls] Techno"


def test_pretty_print_spotify_track():
    track = make_spotify_track(name="Hold On", artists=["Taiki Nulight", "Friend"])
    assert string_utils.pretty_print_spotify_track(track) == "Taiki Nulight, Friend - Hold On"


def test_pretty_print_spotify_track_with_url():
    track = make_spotify_track(name="Hold On", artists=["Taiki Nulight"], track_id="xyz")
    out = string_utils.pretty_print_spotify_track(track, include_url=True)
    # external_urls value has the leading "https://" (8 chars) stripped
    assert out == "open.spotify.com/track/xyz  Taiki Nulight - Hold On"


def test_pretty_print_spotify_track_handles_invalid():
    assert string_utils.pretty_print_spotify_track({"artists": None, "name": None}) == (
        "invalid spotify track"
    )
