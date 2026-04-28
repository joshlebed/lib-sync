"""Module to get track IDs from a recording using parallel Shazam recognition.

This module implements a two-pass recognition strategy:
1. Discovery pass: Wide spacing (30s steps) to quickly find tracks
2. Gap-filling pass: Dense spacing (7.5s steps) in unidentified regions

The system uses parallel processing for both FFmpeg segment extraction
and Shazam API calls, with SQLite-backed segment-level caching.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import time
from datetime import timedelta
from typing import TYPE_CHECKING

import ffmpeg
from tqdm import tqdm

from libsync.id.download_audio import download_mp3_from_youtube_url
from libsync.id.shazam.cache import SegmentCache
from libsync.id.shazam.global_cache import GlobalSegmentCache
from libsync.id.shazam.models import SegmentCacheKey, SegmentResult, TrackMatch
from libsync.id.shazam.recognizer import ShazamRecognizer, extract_and_recognize_parallel
from libsync.id.youtube_dl_utils import (
    get_mp3_output_path,
    get_youtube_video_id_from_url,
)
from libsync.utils.constants import (
    FORCE_REDO_SHAZAM,
    SHAZAM_FFMPEG_WORKERS,
    SHAZAM_MAX_CONCURRENT,
    SHAZAM_MIN_CONFIDENCE,
    SHAZAM_MIN_GAP_MS,
    SHAZAM_MIN_MATCHES,
    SHAZAM_PASS1_STEP_MS,
    SHAZAM_PASS2_STEP_MS,
    SHAZAM_REQUEST_DELAY,
    SHAZAM_SEGMENT_LENGTH_MS,
    SHOW_URL_IN_SHAZAM_OUTPUT,
)
from libsync.utils.filepath_utils import (
    LIBSYNC_DATA_DIR,
    SHAZAM_GLOBAL_CACHE_PATH,
    get_shazam_segment_cache_path,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger("libsync")


def get_results_output_path(audio_path: str) -> str:
    """Get path for human-readable results output file."""
    import hashlib

    path_hash = hashlib.sha256(audio_path.encode()).hexdigest()[:12]
    return str(LIBSYNC_DATA_DIR / f"shazam_results_{path_hash}.txt")


def write_results_file(
    output_path: str,
    audio_path: str,
    matches: dict[str, TrackMatch],
    cache_stats: dict[str, int],
    phase: str = "in_progress",
) -> None:
    """Write current results to a human-readable text file.

    This file is updated incrementally so results are preserved even if the process crashes.
    """
    from datetime import datetime

    sorted_matches = sorted(matches.values(), key=lambda m: m.first_seen_ms)

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SHAZAM TRACK IDENTIFICATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Audio file: {audio_path}\n")
        f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Status: {phase}\n\n")

        f.write(f"Progress: {cache_stats['total_segments']} segments processed\n")
        f.write(f"Segments with matches: {cache_stats['segments_with_match']}\n")
        f.write(f"Unique tracks found: {cache_stats['unique_tracks']}\n\n")

        f.write("-" * 80 + "\n")
        f.write("IDENTIFIED TRACKS (sorted by timestamp)\n")
        f.write("-" * 80 + "\n\n")

        for match in sorted_matches:
            timestamp = str(timedelta(milliseconds=match.first_seen_ms))
            confidence = match.calculate_confidence()
            f.write(f"  {timestamp}  [{match.match_count}x] {match.artist} - {match.title}\n")
            f.write(f"             Confidence: {confidence:.2f}, Shazam ID: {match.shazam_id}\n\n")

        f.write("-" * 80 + "\n")
        f.write(f"Total: {len(sorted_matches)} unique tracks\n")
        f.write("=" * 80 + "\n")

    logger.info(f"Results written to: {output_path}")


def get_track_ids_from_youtube_link(youtube_url: str) -> None:
    """Analyze audio file to find track IDs from a YouTube video.

    Args:
        youtube_url: URL of YouTube video to analyze
    """
    logger.info(f"get_track_ids_from_youtube_link with youtube_url: {youtube_url}")

    youtube_video_id = get_youtube_video_id_from_url(youtube_url)
    mp3_output_path = get_mp3_output_path(youtube_video_id)
    logger.info(f"using youtube_video_id: {youtube_video_id}, mp3_output_path: {mp3_output_path}")

    if not os.path.isfile(mp3_output_path):
        logger.info("couldn't find file, downloading from youtube")
        download_mp3_from_youtube_url(youtube_url)
    else:
        logger.info("found file, skipping download")

    get_track_ids_from_audio_file(mp3_output_path)


def get_audio_duration_ms(audio_path: str) -> int:
    """Get the duration of an audio file in milliseconds.

    Args:
        audio_path: Path to the audio file

    Returns:
        Duration in milliseconds

    Raises:
        ValueError: If no audio stream found in the file
    """
    probe = ffmpeg.probe(audio_path)
    audio_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "audio"), None
    )
    if not audio_stream:
        raise ValueError(f"No audio stream found in {audio_path}")

    return int(float(audio_stream["duration"]) * 1000)


def generate_pass1_segments(total_duration_ms: int) -> list[int]:
    """Generate segment start times for discovery pass (wide spacing).

    Args:
        total_duration_ms: Total audio duration in milliseconds

    Returns:
        List of start times in milliseconds
    """
    segments = []
    for start_ms in range(0, total_duration_ms, SHAZAM_PASS1_STEP_MS):
        if start_ms + SHAZAM_SEGMENT_LENGTH_MS <= total_duration_ms + 5000:
            segments.append(start_ms)
    return segments


def find_gaps(
    matches: dict[str, TrackMatch], total_duration_ms: int, min_gap_ms: int = SHAZAM_MIN_GAP_MS
) -> list[tuple[int, int]]:
    """Find gaps in identified track regions.

    Args:
        matches: Dictionary of track matches
        total_duration_ms: Total audio duration in milliseconds
        min_gap_ms: Minimum gap size to consider for filling

    Returns:
        List of (start_ms, end_ms) tuples for gaps
    """
    if not matches:
        # No matches at all - the entire file is a gap
        return [(0, total_duration_ms)]

    # Get all identified regions (first_seen to last_seen + segment length)
    regions = []
    for match in matches.values():
        # Add buffer around the match region
        region_start = max(0, match.first_seen_ms - SHAZAM_SEGMENT_LENGTH_MS)
        region_end = min(total_duration_ms, match.last_seen_ms + SHAZAM_SEGMENT_LENGTH_MS * 2)
        regions.append((region_start, region_end))

    # Sort by start time and merge overlapping regions
    regions.sort()
    merged: list[tuple[int, int]] = []
    for start, end in regions:
        if merged and start <= merged[-1][1]:
            # Overlapping - extend the previous region
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Find gaps between merged regions
    gaps = []

    # Gap at the beginning
    if merged and merged[0][0] > min_gap_ms:
        gaps.append((0, merged[0][0]))

    # Gaps between regions
    for i in range(len(merged) - 1):
        gap_start = merged[i][1]
        gap_end = merged[i + 1][0]
        if gap_end - gap_start >= min_gap_ms:
            gaps.append((gap_start, gap_end))

    # Gap at the end
    if merged and total_duration_ms - merged[-1][1] >= min_gap_ms:
        gaps.append((merged[-1][1], total_duration_ms))

    return gaps


def generate_pass2_segments(gaps: list[tuple[int, int]]) -> list[int]:
    """Generate segment start times for gap-filling pass (dense spacing).

    Args:
        gaps: List of (start_ms, end_ms) tuples for gaps

    Returns:
        List of start times in milliseconds
    """
    segments = []
    for gap_start, gap_end in gaps:
        for start_ms in range(gap_start, gap_end, SHAZAM_PASS2_STEP_MS):
            if start_ms + SHAZAM_SEGMENT_LENGTH_MS <= gap_end + 5000:
                segments.append(start_ms)
    return segments


def aggregate_matches(results: list[SegmentResult]) -> dict[str, TrackMatch]:
    """Aggregate segment results into track matches.

    Args:
        results: List of SegmentResults from recognition

    Returns:
        Dictionary mapping track_id to TrackMatch objects
    """
    matches: dict[str, TrackMatch] = {}

    for result in results:
        if not result.has_match:
            continue

        track_id = result.track_id
        if track_id is None:
            continue

        if track_id not in matches:
            matches[track_id] = TrackMatch(
                shazam_id=track_id,
                title=result.title or "",
                artist=result.artist or "",
                first_seen_ms=result.start_ms,
                last_seen_ms=result.start_ms,
                match_timestamps=[result.start_ms],
            )
        else:
            matches[track_id].add_match(result.start_ms)

    return matches


async def recognize_segments_two_pass(
    audio_path: str,
    force_redo: bool = False,
) -> dict[str, TrackMatch]:
    """Recognize songs using two-pass strategy with parallel processing.

    Pass 1 (Discovery): Wide spacing to quickly identify tracks
    Pass 2 (Gap-filling): Dense spacing in unidentified regions

    Args:
        audio_path: Path to the audio file
        force_redo: If True, ignore cache and reprocess all segments

    Returns:
        Dictionary mapping track_id to TrackMatch objects
    """
    t_start = time.monotonic()

    # Get audio duration
    total_duration_ms = get_audio_duration_ms(audio_path)
    total_duration_str = str(timedelta(milliseconds=total_duration_ms))
    logger.info(f"Audio duration: {total_duration_str}")

    # Compute audio hash for cache
    audio_hash = SegmentCacheKey.compute_file_hash(audio_path)
    logger.info(f"Audio hash: {audio_hash}")

    # Initialize caches
    cache_path = get_shazam_segment_cache_path(audio_path)
    cache = SegmentCache(cache_path)
    logger.info(f"Using per-file cache at: {cache_path}")

    global_cache = GlobalSegmentCache(SHAZAM_GLOBAL_CACHE_PATH)
    global_stats = global_cache.get_stats()
    logger.info(
        f"Using global cache at: {SHAZAM_GLOBAL_CACHE_PATH} "
        f"({global_stats['total_entries']} entries)"
    )

    # Setup results output file
    results_path = get_results_output_path(audio_path)
    logger.info(f"Results will be written to: {results_path}")
    print(f"Results file: {results_path}")

    # Clear cache if force_redo
    if force_redo:
        cache.clear_for_audio(audio_hash)
        logger.info("Cleared existing cache due to force_redo")

    # Get already cached segments
    cached_segments = cache.get_cached_segments(audio_hash)

    # Create temp directory and shared recognizer (so metrics accumulate across passes)
    temp_dir = tempfile.mkdtemp(prefix="libsync_shazam_")
    recognizer = ShazamRecognizer(
        max_concurrent=SHAZAM_MAX_CONCURRENT,
        request_delay=SHAZAM_REQUEST_DELAY,
    )

    print(
        f"Shazam concurrency: {SHAZAM_MAX_CONCURRENT}, "
        f"request delay: {SHAZAM_REQUEST_DELAY}s "
        f"(env vars: SHAZAM_MAX_CONCURRENT, SHAZAM_REQUEST_DELAY)"
    )

    try:
        # === PASS 1: Discovery ===
        pass1_all = generate_pass1_segments(total_duration_ms)
        pass1_uncached = [s for s in pass1_all if s not in cached_segments]

        logger.info(
            f"Pass 1 (Discovery): {len(pass1_all)} segments, {len(pass1_uncached)} uncached"
        )

        if pass1_uncached:
            print(f"Pass 1: Identifying tracks ({len(pass1_uncached)} segments)...")
            t_pass1 = time.monotonic()
            with tqdm(total=len(pass1_uncached), desc="Pass 1", unit="seg") as pbar:

                def pass1_progress(done: int, total: int, phase: str) -> None:
                    if phase == "recognizing":
                        pbar.update(1)

                await extract_and_recognize_parallel(
                    audio_path=audio_path,
                    temp_dir=temp_dir,
                    segments_to_process=pass1_uncached,
                    audio_hash=audio_hash,
                    segment_duration_ms=SHAZAM_SEGMENT_LENGTH_MS,
                    cache=cache,
                    max_concurrent_shazam=SHAZAM_MAX_CONCURRENT,
                    max_ffmpeg_workers=SHAZAM_FFMPEG_WORKERS,
                    progress_callback=pass1_progress,
                    recognizer=recognizer,
                    global_cache=global_cache,
                )

            pass1_elapsed = time.monotonic() - t_pass1
            metrics = recognizer.get_metrics()
            print(
                f"Pass 1 done in {pass1_elapsed:.1f}s — "
                f"{metrics['total_api_calls']} API calls, "
                f"{metrics['errors']} errors, "
                f"{metrics['slow_calls_likely_retries']} slow, "
                f"avg {metrics['avg_call_time_s']}s/call, "
                f"max {metrics['max_call_time_s']}s, "
                f"global cache hits: {metrics['global_cache_hits']}"
            )

        # Aggregate pass 1 results
        pass1_results = cache.get_all_results(audio_hash)
        matches = aggregate_matches(pass1_results)

        # Show intermediate results
        logger.info(f"Pass 1 found {len(matches)} unique tracks")

        # Write intermediate results to file
        stats = cache.get_cache_stats(audio_hash)
        write_results_file(results_path, audio_path, matches, stats, phase="Pass 1 complete")

        # === PASS 2: Gap-filling ===
        gaps = find_gaps(matches, total_duration_ms)
        pass2_all = generate_pass2_segments(gaps)

        # Update cached_segments with pass 1 results
        cached_segments = cache.get_cached_segments(audio_hash)
        pass2_uncached = [s for s in pass2_all if s not in cached_segments]

        logger.info(
            f"Pass 2 (Gap-filling): {len(gaps)} gaps, {len(pass2_all)} segments, {len(pass2_uncached)} uncached"
        )

        if pass2_uncached:
            print(f"Pass 2: Filling gaps ({len(pass2_uncached)} segments)...")
            t_pass2 = time.monotonic()
            with tqdm(total=len(pass2_uncached), desc="Pass 2", unit="seg") as pbar:

                def pass2_progress(done: int, total: int, phase: str) -> None:
                    if phase == "recognizing":
                        pbar.update(1)

                await extract_and_recognize_parallel(
                    audio_path=audio_path,
                    temp_dir=temp_dir,
                    segments_to_process=pass2_uncached,
                    audio_hash=audio_hash,
                    segment_duration_ms=SHAZAM_SEGMENT_LENGTH_MS,
                    cache=cache,
                    max_concurrent_shazam=SHAZAM_MAX_CONCURRENT,
                    max_ffmpeg_workers=SHAZAM_FFMPEG_WORKERS,
                    progress_callback=pass2_progress,
                    recognizer=recognizer,
                    global_cache=global_cache,
                )

            pass2_elapsed = time.monotonic() - t_pass2
            metrics = recognizer.get_metrics()
            print(
                f"Pass 2 done in {pass2_elapsed:.1f}s — "
                f"{metrics['total_api_calls']} API calls total, "
                f"{metrics['errors']} errors, "
                f"{metrics['slow_calls_likely_retries']} slow, "
                f"avg {metrics['avg_call_time_s']}s/call, "
                f"max {metrics['max_call_time_s']}s, "
                f"global cache hits: {metrics['global_cache_hits']}"
            )

        # === RETRY: Re-process any failed segments ===
        all_expected = set(pass1_all) | set(pass2_all)
        cached_segments = cache.get_cached_segments(audio_hash)
        failed_segments = sorted(all_expected - cached_segments)
        retry_cooldowns = [120, 300, 600]  # 2min, 5min, 10min — rate limits last 15-20min

        for retry_attempt, cooldown in enumerate(retry_cooldowns):
            if not failed_segments:
                break
            print(
                f"\n{len(failed_segments)} segments failed — "
                f"retrying after {cooldown}s cooldown (attempt {retry_attempt + 1}/{len(retry_cooldowns)})..."
            )
            await asyncio.sleep(cooldown)

            # Use a fresh recognizer for retries so error metrics are clean.
            # max_concurrent is forced to 1 (regardless of env) because retries
            # mean we just hit the rate limiter — be maximally conservative.
            retry_recognizer = ShazamRecognizer(
                max_concurrent=1,
                request_delay=SHAZAM_REQUEST_DELAY,
            )

            with tqdm(total=len(failed_segments), desc="Retry", unit="seg") as pbar:

                def retry_progress(done: int, total: int, phase: str) -> None:
                    if phase == "recognizing":
                        pbar.update(1)

                await extract_and_recognize_parallel(
                    audio_path=audio_path,
                    temp_dir=temp_dir,
                    segments_to_process=failed_segments,
                    audio_hash=audio_hash,
                    segment_duration_ms=SHAZAM_SEGMENT_LENGTH_MS,
                    cache=cache,
                    max_concurrent_shazam=1,
                    max_ffmpeg_workers=SHAZAM_FFMPEG_WORKERS,
                    progress_callback=retry_progress,
                    recognizer=retry_recognizer,
                    global_cache=global_cache,
                )

            retry_metrics = retry_recognizer.get_metrics()
            cached_segments = cache.get_cached_segments(audio_hash)
            still_failed = sorted(all_expected - cached_segments)
            recovered = len(failed_segments) - len(still_failed)
            error_rate = (
                retry_metrics["errors"] / retry_metrics["total_api_calls"]
                if retry_metrics["total_api_calls"]
                else 0
            )
            print(
                f"Retry recovered {recovered}/{len(failed_segments)} segments "
                f"(error rate: {error_rate:.0%})"
            )
            failed_segments = still_failed

        if failed_segments:
            print(f"\n{len(failed_segments)} segments still failed after retries")

        # Final aggregation
        all_results = cache.get_all_results(audio_hash)
        final_matches = aggregate_matches(all_results)

        # Log stats and write final results
        stats = cache.get_cache_stats(audio_hash)
        logger.info(
            f"Final: {stats['unique_tracks']} tracks from "
            f"{stats['segments_with_match']}/{stats['total_segments']} segments with matches"
        )

        # Write final results to file
        write_results_file(results_path, audio_path, final_matches, stats, phase="Complete")
        print(f"\nResults saved to: {results_path}")

        # Log recognition metrics for tuning
        _write_recognition_metrics_log(
            audio_path=audio_path,
            concurrency=SHAZAM_MAX_CONCURRENT,
            metrics=recognizer.get_metrics(),
            wall_clock_s=round(time.monotonic() - t_start, 1),
            pass1_segments=len(pass1_all),
            pass2_segments=len(pass2_all),
        )

        return final_matches

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


RECOGNITION_METRICS_LOG_PATH = str(LIBSYNC_DATA_DIR / "shazam_recognition_metrics.jsonl")


def _write_recognition_metrics_log(
    audio_path: str,
    concurrency: int,
    metrics: dict[str, float | int],
    wall_clock_s: float,
    pass1_segments: int,
    pass2_segments: int,
) -> None:
    """Append a JSON line of per-run recognition metrics for concurrency tuning."""
    import json
    from datetime import datetime

    total_api = metrics["total_api_calls"]
    entry = {
        "timestamp": datetime.now().isoformat(),
        "audio_file": os.path.basename(audio_path),
        "concurrency": concurrency,
        "total_api_calls": total_api,
        "errors": metrics["errors"],
        "slow_calls": metrics["slow_calls_likely_retries"],
        "slow_call_rate": round(metrics["slow_calls_likely_retries"] / total_api, 3)
        if total_api
        else 0,
        "avg_call_time_s": metrics["avg_call_time_s"],
        "max_call_time_s": metrics["max_call_time_s"],
        "throughput_req_s": round(total_api / wall_clock_s, 1) if wall_clock_s else 0,
        "wall_clock_s": wall_clock_s,
        "global_cache_hits": metrics["global_cache_hits"],
        "pass1_segments": pass1_segments,
        "pass2_segments": pass2_segments,
    }

    with open(RECOGNITION_METRICS_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info(f"Recognition metrics entry written to: {RECOGNITION_METRICS_LOG_PATH}")
    print(f"Recognition metrics log: {RECOGNITION_METRICS_LOG_PATH}")


def get_track_ids_from_audio_file(recording_audio_file_path: str) -> None:
    """Analyze audio file using Shazam to find track IDs.

    Uses two-pass recognition strategy with parallel processing and
    segment-level SQLite caching.

    Args:
        recording_audio_file_path: Path to audio file to analyze
    """
    logger.info(f"get_track_ids_from_audio_file: {recording_audio_file_path}")

    # Run two-pass recognition
    matches = asyncio.run(
        recognize_segments_two_pass(recording_audio_file_path, force_redo=FORCE_REDO_SHAZAM)
    )

    # Sort matches by first seen timestamp
    sorted_matches = sorted(matches.values(), key=lambda m: m.first_seen_ms)

    # Print results
    print("\n" + "=" * 80)
    print("IDENTIFIED TRACKS")
    print("=" * 80)

    filtered_count = 0
    for match in sorted_matches:
        confidence = match.calculate_confidence()

        # Filter by confidence and match count
        if match.match_count >= SHAZAM_MIN_MATCHES and confidence >= SHAZAM_MIN_CONFIDENCE:
            timestamp = str(timedelta(milliseconds=match.first_seen_ms))
            url_component = f"  [{match.shazam_id}]" if SHOW_URL_IN_SHAZAM_OUTPUT else ""
            print(
                f"{match.match_count:3}x  {timestamp}  {match.artist:30} - {match.title:40}{url_component}"
            )
            filtered_count += 1
        else:
            logger.debug(
                f"Filtered out: {match.artist} - {match.title} "
                f"(count={match.match_count}, confidence={confidence:.2f})"
            )

    print("=" * 80)
    print(f"Total: {filtered_count} tracks (filtered from {len(matches)} detected)")
    print()
