"""Integration tests for ShazamRecognizer.recognize_segment.

The Shazam client is faked via the _get_shazam() seam, so these exercise the
real cache / metrics / error-handling logic with no network and no audio files
(the segment path is never opened on these paths).
"""

from libsync.id.shazam.models import SegmentCacheKey, SegmentResult
from libsync.id.shazam.recognizer import ShazamRecognizer


class FakeShazam:
    def __init__(self, response=None, raise_exc=None):
        self.calls = 0
        self._response = response
        self._raise = raise_exc

    async def recognize(self, _segment_path):
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return self._response


class FakeCache:
    """Minimal stand-in for SegmentCache (dict-backed, records sets)."""

    def __init__(self, initial=None):
        self.store = dict(initial or {})
        self.sets = []

    def get(self, key):
        return self.store.get(str(key))

    def set(self, key, result):
        self.sets.append((str(key), result))
        self.store[str(key)] = result


def _recognizer_with(monkeypatch, fake):
    recognizer = ShazamRecognizer(max_concurrent=2)
    monkeypatch.setattr(recognizer, "_get_shazam", lambda: fake)
    return recognizer


async def test_per_file_cache_hit_skips_api(monkeypatch):
    fake = FakeShazam(response={"track": {"key": "should-not-be-used"}})
    recognizer = _recognizer_with(monkeypatch, fake)

    cached = SegmentResult(start_ms=1000, track_id="cached-id", title="T", artist="A")
    key = SegmentCacheKey("audiohash", 1000, 15000)
    cache = FakeCache({str(key): cached})

    result = await recognizer.recognize_segment("seg.mp3", 1000, "audiohash", 15000, cache=cache)

    assert result is cached
    assert fake.calls == 0
    assert recognizer.get_metrics()["total_api_calls"] == 0


async def test_cache_miss_calls_api_and_stores_result(monkeypatch):
    fake = FakeShazam(
        response={"track": {"key": "track-1", "title": "Hold On", "subtitle": "Taiki Nulight"}}
    )
    recognizer = _recognizer_with(monkeypatch, fake)
    cache = FakeCache()

    result = await recognizer.recognize_segment("seg.mp3", 2000, "audiohash", 15000, cache=cache)

    assert fake.calls == 1
    assert result.has_match
    assert result.track_id == "track-1"
    assert result.artist == "Taiki Nulight"
    # the fresh result was written back to the cache
    assert cache.sets and cache.sets[-1][1] is result
    assert recognizer.get_metrics()["total_api_calls"] == 1


async def test_no_match_response_is_recorded_without_match(monkeypatch):
    fake = FakeShazam(response={})  # Shazam found nothing
    recognizer = _recognizer_with(monkeypatch, fake)

    result = await recognizer.recognize_segment("seg.mp3", 3000, "audiohash", 15000)

    assert fake.calls == 1
    assert result.has_match is False
    assert result.start_ms == 3000
    assert recognizer.get_metrics()["errors"] == 0


async def test_api_error_returns_empty_result_and_counts_error(monkeypatch):
    fake = FakeShazam(raise_exc=RuntimeError("shazam exploded"))
    recognizer = _recognizer_with(monkeypatch, fake)

    result = await recognizer.recognize_segment("seg.mp3", 4000, "audiohash", 15000)

    assert isinstance(result, SegmentResult)
    assert result.has_match is False
    assert result.start_ms == 4000
    metrics = recognizer.get_metrics()
    assert metrics["errors"] == 1
    assert metrics["total_api_calls"] == 1
