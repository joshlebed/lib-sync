"""Tiny builders for test data.

One place to construct RekordboxTrack/Library objects and Spotify-shaped JSON so
tests don't hand-roll (and drift on) the dict shapes the production code expects.
"""

from __future__ import annotations

from libsync.utils.rekordbox_library import (
    RekordboxLibrary,
    RekordboxPlaylist,
    RekordboxTrack,
)


def make_rb_track(
    id: str = "1",
    name: str = "Hold On",
    artist: str = "Taiki Nulight",
    album: str | None = None,
    tonality: str | None = None,
) -> RekordboxTrack:
    return RekordboxTrack(id=id, name=name, artist=artist, album=album, tonality=tonality)


def make_rb_library(
    tracks: list[RekordboxTrack] | None = None,
    playlists: list[RekordboxPlaylist] | None = None,
    xml_path: str = "test_library.xml",
) -> RekordboxLibrary:
    tracks = tracks if tracks is not None else [make_rb_track()]
    return RekordboxLibrary(
        xml_path=xml_path,
        collection={t.id: t for t in tracks},
        playlists=playlists if playlists is not None else [],
    )


def make_spotify_track(
    uri: str = "spotify:track:abc123",
    name: str = "Hold On",
    artists: list[str] | None = None,
    track_id: str | None = None,
) -> dict:
    """A Spotify track object shaped like the fields libsync actually reads."""
    artists = artists if artists is not None else ["Taiki Nulight"]
    if track_id is None:
        track_id = uri.split(":")[-1]
    return {
        "uri": uri,
        "id": track_id,
        "name": name,
        "artists": [{"name": a} for a in artists],
        "external_urls": {"spotify": f"https://open.spotify.com/track/{track_id}"},
    }


def make_spotify_collection(*tracks: dict) -> dict[str, dict]:
    """A SpotifySongCollection: uri -> track object."""
    return {t["uri"]: t for t in tracks}
