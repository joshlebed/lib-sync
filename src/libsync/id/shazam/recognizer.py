"""Parallel Shazam recognition with rate limiting and caching."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING

from aiohttp_retry import ExponentialRetry
from shazamio import HTTPClient, Shazam

from libsync.id.shazam.cache import SegmentCache
from libsync.id.shazam.global_cache import GlobalSegmentCache
from libsync.id.shazam.models import SegmentCacheKey, SegmentResult, compute_segment_content_hash

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("libsync")

# Threshold (seconds) above which a single recognize() call is considered slow,
# almost certainly due to 429-retry backoff inside shazamio/aiohttp_retry.
_SLOW_CALL_THRESHOLD = 5.0


class ShazamRecognizer:
    """Parallel Shazam API requests with semaphore-based rate limiting.

    Uses asyncio.Semaphore to limit concurrent requests and prevent
    overwhelming the Shazam API with too many parallel requests.
    """

    def __init__(self, max_concurrent: int = 10, request_delay: float = 0.0):
        """Initialize the recognizer.

        Args:
            max_concurrent: Maximum number of concurrent in-flight Shazam requests
            request_delay: Minimum seconds between consecutive API calls (global rate limit)
        """
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_lock = asyncio.Lock()
        self._shazam: Shazam | None = None

        # Metrics for tracking rate limiting behaviour
        self._total_api_calls = 0
        self._slow_calls = 0
        self._total_api_time = 0.0
        self._max_call_time = 0.0
        self._global_cache_hits = 0
        self._errors = 0

    def _get_shazam(self) -> Shazam:
        """Get or create the Shazam client with retry configuration."""
        if self._shazam is None:
            self._shazam = Shazam(
                http_client=HTTPClient(
                    retry_options=ExponentialRetry(
                        attempts=3,
                        max_timeout=10.0,
                        statuses={500, 502, 503, 504},
                    ),
                ),
            )
        return self._shazam

    def get_metrics(self) -> dict[str, float | int]:
        """Return current recognition metrics."""
        avg_time = self._total_api_time / self._total_api_calls if self._total_api_calls else 0.0
        return {
            "total_api_calls": self._total_api_calls,
            "slow_calls_likely_retries": self._slow_calls,
            "errors": self._errors,
            "total_api_time_s": round(self._total_api_time, 1),
            "avg_call_time_s": round(avg_time, 2),
            "max_call_time_s": round(self._max_call_time, 2),
            "global_cache_hits": self._global_cache_hits,
        }

    async def recognize_segment(
        self,
        segment_path: str,
        start_ms: int,
        audio_hash: str,
        duration_ms: int,
        cache: SegmentCache | None = None,
        global_cache: GlobalSegmentCache | None = None,
    ) -> SegmentResult:
        """Recognize a single segment with semaphore limiting.

        Args:
            segment_path: Path to the audio segment file
            start_ms: Segment start time in milliseconds
            audio_hash: Hash of the source audio file
            duration_ms: Segment duration in milliseconds
            cache: Optional per-file cache to check/store results
            global_cache: Optional global content-addressed cache

        Returns:
            SegmentResult with recognition data
        """
        cache_key = SegmentCacheKey(audio_hash, start_ms, duration_ms)

        # 1. Check per-file cache first (outside semaphore, no file I/O)
        if cache:
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for segment at {start_ms}ms")
                return cached_result

        # 2. Check global content-addressed cache (outside semaphore)
        content_hash: str | None = None
        if global_cache:
            content_hash = compute_segment_content_hash(segment_path)
            global_result = global_cache.get(content_hash, start_ms=start_ms)
            if global_result is not None:
                self._global_cache_hits += 1
                logger.debug(f"Global cache hit for segment at {start_ms}ms (hash={content_hash})")
                # Populate per-file cache so future runs skip the hash too
                if cache:
                    cache.set(cache_key, global_result)
                return global_result

        # 3. Acquire semaphore for API call (limits in-flight requests)
        # Build the client outside any lock — _get_shazam() is idempotent after
        # first call and we don't want construction failures to leave the
        # rate lock held without metrics accounting.
        shazam = self._get_shazam()
        async with self._semaphore:
            # Rate limit: serialize request SENDING with minimum delay between calls.
            # The lock is held only during sleep + request launch, then released
            # so the next request can start while we wait for the response.
            async with self._rate_lock:
                if self.request_delay > 0:
                    await asyncio.sleep(self.request_delay)
                t0 = time.monotonic()
                response_future = asyncio.ensure_future(shazam.recognize(segment_path))

            # Wait for response OUTSIDE the rate lock (allows next request to send)
            try:
                response = await response_future
                elapsed = time.monotonic() - t0

                self._total_api_calls += 1
                self._total_api_time += elapsed
                self._max_call_time = max(self._max_call_time, elapsed)

                if elapsed > _SLOW_CALL_THRESHOLD:
                    self._slow_calls += 1
                    logger.warning(
                        f"Slow recognition at {start_ms}ms: {elapsed:.1f}s "
                        f"(likely 429 retries, {self._slow_calls} slow so far)"
                    )

                result = SegmentResult.from_shazam_response(start_ms, response)

                # Cache the result in both per-file and global caches
                if cache:
                    cache.set(cache_key, result)
                if global_cache and content_hash:
                    global_cache.set(content_hash, result)

                if result.has_match:
                    logger.debug(f"Match at {start_ms}ms: {result.artist} - {result.title}")
                else:
                    logger.debug(f"No match at {start_ms}ms")

                return result

            except Exception as e:
                elapsed = time.monotonic() - t0
                self._total_api_calls += 1
                self._total_api_time += elapsed
                self._max_call_time = max(self._max_call_time, elapsed)
                self._errors += 1
                logger.error(f"Shazam error at {start_ms}ms after {elapsed:.1f}s: {e}")
                return SegmentResult(start_ms=start_ms)

    async def recognize_batch(
        self,
        segment_paths: list[tuple[str, int]],
        audio_hash: str,
        duration_ms: int,
        cache: SegmentCache | None = None,
        global_cache: GlobalSegmentCache | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        cleanup_segments: bool = True,
    ) -> list[SegmentResult]:
        """Recognize multiple segments in parallel.

        Args:
            segment_paths: List of (segment_path, start_ms) tuples
            audio_hash: Hash of the source audio file
            duration_ms: Segment duration in milliseconds
            cache: Optional per-file cache for results
            global_cache: Optional global content-addressed cache
            progress_callback: Optional callback(completed, total) for progress
            cleanup_segments: Whether to delete segment files after recognition

        Returns:
            List of SegmentResults
        """
        batch_start = time.monotonic()
        calls_before = self._total_api_calls

        tasks = [
            self.recognize_segment(path, start_ms, audio_hash, duration_ms, cache, global_cache)
            for path, start_ms in segment_paths
        ]

        results: list[SegmentResult] = []

        # Process with progress tracking using as_completed
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(tasks))

        batch_elapsed = time.monotonic() - batch_start
        batch_api_calls = self._total_api_calls - calls_before

        if batch_api_calls > 0:
            throughput = batch_api_calls / batch_elapsed if batch_elapsed > 0 else 0
            logger.info(
                f"Batch done: {batch_api_calls} API calls in {batch_elapsed:.1f}s "
                f"({throughput:.1f} req/s, concurrency={self.max_concurrent}, "
                f"slow={self._slow_calls})"
            )

        # Cleanup segment files if requested
        if cleanup_segments:
            for path, _ in segment_paths:
                try:
                    os.remove(path)
                except OSError as e:
                    logger.warning(f"Failed to remove segment file {path}: {e}")

        return results

    async def recognize_batch_ordered(
        self,
        segment_paths: list[tuple[str, int]],
        audio_hash: str,
        duration_ms: int,
        cache: SegmentCache | None = None,
        global_cache: GlobalSegmentCache | None = None,
        cleanup_segments: bool = True,
    ) -> list[SegmentResult]:
        """Recognize multiple segments, returning results in input order.

        Args:
            segment_paths: List of (segment_path, start_ms) tuples
            audio_hash: Hash of the source audio file
            duration_ms: Segment duration in milliseconds
            cache: Optional per-file cache for results
            global_cache: Optional global content-addressed cache
            cleanup_segments: Whether to delete segment files after recognition

        Returns:
            List of SegmentResults in same order as input
        """
        tasks = [
            self.recognize_segment(path, start_ms, audio_hash, duration_ms, cache, global_cache)
            for path, start_ms in segment_paths
        ]

        results = await asyncio.gather(*tasks)

        # Cleanup segment files if requested
        if cleanup_segments:
            for path, _ in segment_paths:
                try:
                    os.remove(path)
                except OSError as e:
                    logger.warning(f"Failed to remove segment file {path}: {e}")

        return list(results)


async def extract_and_recognize_parallel(
    audio_path: str,
    temp_dir: str,
    segments_to_process: list[int],
    audio_hash: str,
    segment_duration_ms: int,
    cache: SegmentCache,
    max_concurrent_shazam: int = 10,
    max_ffmpeg_workers: int = 4,
    progress_callback: Callable[[int, int, str], None] | None = None,
    recognizer: ShazamRecognizer | None = None,
    global_cache: GlobalSegmentCache | None = None,
) -> list[SegmentResult]:
    """Combined extraction and recognition pipeline.

    Extracts ALL segments first (CPU-bound, parallel FFmpeg), then
    recognizes them (network-bound, parallel Shazam API calls).

    Args:
        audio_path: Path to source audio file
        temp_dir: Directory for temporary segment files
        segments_to_process: List of start_ms values to process
        audio_hash: Hash of the source audio file
        segment_duration_ms: Duration of each segment in milliseconds
        cache: Cache for storing results
        max_concurrent_shazam: Max parallel Shazam requests
        max_ffmpeg_workers: Max parallel FFmpeg processes
        progress_callback: Optional callback(completed, total, phase)
        recognizer: Optional shared ShazamRecognizer instance (reused across passes)
        global_cache: Optional global content-addressed cache

    Returns:
        List of SegmentResults for all processed segments
    """
    from libsync.id.shazam.extractor import SegmentExtractor
    from libsync.id.shazam.models import SegmentSpec

    # Create segment specs
    specs = [SegmentSpec(start_ms=s, duration_ms=segment_duration_ms) for s in segments_to_process]

    # Phase 1: Extract ALL segments (CPU-bound, no rate limit concerns)
    t0 = time.monotonic()
    extractor = SegmentExtractor(max_workers=max_ffmpeg_workers)

    def extraction_progress(done: int, total: int) -> None:
        if progress_callback:
            progress_callback(done, total, "extracting")

    segment_paths = await extractor.extract_batch(
        audio_path, temp_dir, specs, progress_callback=extraction_progress
    )

    extraction_time = time.monotonic() - t0
    logger.info(
        f"Extraction complete: {len(segment_paths)} segments in {extraction_time:.1f}s "
        f"({max_ffmpeg_workers} FFmpeg workers)"
    )

    # Phase 2: Recognize ALL segments (network-bound)
    t0 = time.monotonic()
    if recognizer is None:
        recognizer = ShazamRecognizer(max_concurrent=max_concurrent_shazam)

    def recognition_progress(done: int, total: int) -> None:
        if progress_callback:
            progress_callback(done, total, "recognizing")

    results = await recognizer.recognize_batch(
        segment_paths,
        audio_hash,
        segment_duration_ms,
        cache=cache,
        global_cache=global_cache,
        progress_callback=recognition_progress,
        cleanup_segments=True,
    )

    recognition_time = time.monotonic() - t0
    logger.info(
        f"Recognition complete: {len(results)} segments in {recognition_time:.1f}s "
        f"(concurrency={recognizer.max_concurrent})"
    )

    # Cleanup extractor
    extractor.shutdown()

    return results
