"""Unit tests for libsync.id.shazam.models — pure parsing and scoring."""

from libsync.id.shazam.models import (
    SegmentResult,
    SegmentSpec,
    TrackMatch,
)


def test_segment_spec_end_ms():
    assert SegmentSpec(start_ms=1000, duration_ms=15000).end_ms == 16000


class TestSegmentResultFromResponse:
    def test_none_response_has_no_match(self):
        result = SegmentResult.from_shazam_response(5000, None)
        assert result.start_ms == 5000
        assert result.has_match is False
        assert result.track_id is None

    def test_response_without_track_has_no_match(self):
        result = SegmentResult.from_shazam_response(5000, {"matches": []})
        assert result.has_match is False

    def test_response_with_track_extracts_fields(self):
        response = {"track": {"key": "track-123", "title": "Hold On", "subtitle": "Taiki Nulight"}}
        result = SegmentResult.from_shazam_response(5000, response)
        assert result.has_match is True
        assert result.track_id == "track-123"
        assert result.title == "Hold On"
        assert result.artist == "Taiki Nulight"
        assert result.raw_response is response


class TestTrackMatch:
    def _match(self, **kwargs):
        defaults = dict(
            shazam_id="t1",
            title="Hold On",
            artist="Taiki Nulight",
            first_seen_ms=0,
            last_seen_ms=0,
            match_timestamps=[0],
        )
        defaults.update(kwargs)
        return TrackMatch(**defaults)

    def test_match_count(self):
        match = self._match(match_timestamps=[0, 1000, 2000])
        assert match.match_count == 3

    def test_add_match_updates_bounds(self):
        match = self._match(first_seen_ms=1000, last_seen_ms=1000, match_timestamps=[1000])
        match.add_match(5000)
        match.add_match(500)
        assert match.first_seen_ms == 500
        assert match.last_seen_ms == 5000
        assert match.match_count == 3

    def test_calculate_confidence_full_score(self):
        # 3+ matches (full count score) spread over >=90s (full spread score) => 1.0
        match = self._match(
            first_seen_ms=0,
            last_seen_ms=90000,
            match_timestamps=[0, 45000, 90000],
        )
        assert match.calculate_confidence() == 1.0

    def test_calculate_confidence_single_match_is_low(self):
        match = self._match(first_seen_ms=0, last_seen_ms=0, match_timestamps=[0])
        # count_score = 1/3, spread_score = 0 => 0.2
        assert match.calculate_confidence() == (1 / 3) * 0.6
