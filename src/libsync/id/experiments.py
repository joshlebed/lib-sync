"""Experiment framework for testing Shazam recognition strategies.

Runs 4 experiments on a DJ set recording to improve track identification:
1. Dense probing: 15s segments every 5s across entire audio
2. Multi-duration: 10s and 25s segments at baseline positions
3. Reinforcement: Dense probing around weak (1x) matches
4. Contextual confidence: Re-score existing matches using position context
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta

from tqdm import tqdm

from libsync.id.get_ids_from_recording import (
    aggregate_matches,
    get_audio_duration_ms,
    write_results_file,
)
from libsync.id.shazam.cache import SegmentCache
from libsync.id.shazam.global_cache import GlobalSegmentCache
from libsync.id.shazam.models import SegmentCacheKey, TrackMatch
from libsync.id.shazam.recognizer import ShazamRecognizer, extract_and_recognize_parallel
from libsync.utils.constants import (
    SHAZAM_FFMPEG_WORKERS,
    SHAZAM_MAX_CONCURRENT,
    SHAZAM_MIN_CONFIDENCE,
    SHAZAM_MIN_MATCHES,
    SHAZAM_REQUEST_DELAY,
    SHAZAM_SEGMENT_LENGTH_MS,
)
from libsync.utils.filepath_utils import (
    LIBSYNC_DATA_DIR,
    SHAZAM_GLOBAL_CACHE_PATH,
    get_shazam_segment_cache_path,
)

logger = logging.getLogger("libsync")

EXPERIMENT_RESULTS_PATH = str(LIBSYNC_DATA_DIR / "shazam_experiment_results.jsonl")


# ---------------------------------------------------------------------------
# Segment generators
# ---------------------------------------------------------------------------


def generate_dense_segments(
    total_duration_ms: int, step_ms: int = 5000, duration_ms: int = 15000
) -> list[tuple[int, int]]:
    """Exp 1: 15s segments every 5s across entire audio.

    The `+ 5000` tolerance lets the final segment overrun the audio end by
    up to 5s; ffmpeg silently truncates and Shazam still gets enough audio.
    """
    segments = []
    for start_ms in range(0, total_duration_ms, step_ms):
        if start_ms + duration_ms <= total_duration_ms + 5000:
            segments.append((start_ms, duration_ms))
    return segments


def generate_multi_duration_segments(
    total_duration_ms: int,
    step_ms: int = 30000,
    durations: list[int] | None = None,
) -> list[tuple[int, int]]:
    """Exp 2: 10s and 25s segments at Pass 1 positions (30s steps).

    Skips 15s duration since that's already cached from baseline.
    """
    if durations is None:
        durations = [10000, 25000]
    segments = []
    for start_ms in range(0, total_duration_ms, step_ms):
        for dur in durations:
            if start_ms + dur <= total_duration_ms + 5000:
                segments.append((start_ms, dur))
    return segments


def generate_reinforcement_segments(
    weak_matches: list[TrackMatch],
    total_duration_ms: int,
    radius_ms: int = 45000,
    step_ms: int = 3000,
    duration_ms: int = 15000,
) -> list[tuple[int, int]]:
    """Exp 3: 15s segments every 3s within +/-45s of each weak (1x) match."""
    segments_set: set[tuple[int, int]] = set()
    for match in weak_matches:
        center_ms = match.first_seen_ms
        probe_start = max(0, center_ms - radius_ms)
        probe_end = min(total_duration_ms, center_ms + radius_ms)
        for start_ms in range(probe_start, probe_end, step_ms):
            if start_ms + duration_ms <= total_duration_ms + 5000:
                segments_set.add((start_ms, duration_ms))
    return sorted(segments_set)


# ---------------------------------------------------------------------------
# Contextual confidence scorer (Exp 4)
# ---------------------------------------------------------------------------


def calculate_contextual_confidence(
    match: TrackMatch,
    all_matches_sorted: list[TrackMatch],
    total_duration_ms: int,
) -> float:
    """Re-score a match using position context.

    Base: existing calculate_confidence()
    Bonuses:
    - +0.2 if match falls in a gap between two strong (>=2x) matches
    - +0.15 if match timestamp is plausible (2-7 min from neighbors)
    - +0.1 if at recording start/end (first/last 5 min)
    Cap at 1.0.
    """
    base = match.calculate_confidence()
    bonus = 0.0

    match_ms = match.first_seen_ms
    strong_matches = [
        m for m in all_matches_sorted if m.match_count >= 2 and m.shazam_id != match.shazam_id
    ]

    # Find neighbors
    prev_strong = None
    next_strong = None
    for m in strong_matches:
        if m.first_seen_ms < match_ms:
            prev_strong = m
        elif m.first_seen_ms > match_ms and next_strong is None:
            next_strong = m

    # Bonus: in a gap between two strong matches
    if prev_strong is not None and next_strong is not None:
        bonus += 0.2

    # Bonus: plausible track duration from neighbors (2-7 min)
    min_plausible_ms = 2 * 60 * 1000  # 2 min
    max_plausible_ms = 7 * 60 * 1000  # 7 min
    if prev_strong is not None:
        gap = match_ms - prev_strong.last_seen_ms
        if min_plausible_ms <= gap <= max_plausible_ms:
            bonus += 0.15
    if next_strong is not None:
        gap = next_strong.first_seen_ms - match_ms
        if min_plausible_ms <= gap <= max_plausible_ms:
            bonus += 0.15

    # Bonus: at recording start/end (first/last 5 min)
    boundary_ms = 5 * 60 * 1000
    if match_ms <= boundary_ms or match_ms >= total_duration_ms - boundary_ms:
        bonus += 0.1

    return min(base + bonus, 1.0)


# ---------------------------------------------------------------------------
# Unified experiment runner
# ---------------------------------------------------------------------------


def _get_baseline_matches(cache: SegmentCache, audio_hash: str) -> dict[str, TrackMatch]:
    """Load all cached results and aggregate into baseline matches."""
    all_results = cache.get_all_results(audio_hash)
    return aggregate_matches(all_results)


def _format_timestamp(ms: int) -> str:
    """Format milliseconds as H:MM:SS."""
    return str(timedelta(milliseconds=ms)).split(".")[0]


def _print_experiment_header(label: str, description: str) -> None:
    print()
    print("\u2550" * 70)
    print(f"EXPERIMENT: {label} -- {description}")
    print("\u2550" * 70)


def _print_experiment_results(
    label: str,
    parameters: dict,
    segments_total: int,
    segments_new: int,
    segments_cached: int,
    api_errors: int,
    tracks: list[dict],
    baseline_matches: dict[str, TrackMatch],
    wall_clock_s: float,
) -> None:
    """Print formatted experiment results to console."""
    passing = [t for t in tracks if t["passes_filter"]]
    raw_count = len(tracks)
    filtered_count = len(passing)
    baseline_filtered = sum(
        1
        for m in baseline_matches.values()
        if m.match_count >= SHAZAM_MIN_MATCHES and m.calculate_confidence() >= SHAZAM_MIN_CONFIDENCE
    )

    avg_call_s = f"{wall_clock_s / segments_new:.2f}" if segments_new > 0 else "N/A"

    print(f"Parameters: {parameters}")
    print(
        f"Segments: {segments_total} total, {segments_new} new API calls, {segments_cached} cache hits"
    )
    print(f"API: {api_errors} errors, avg {avg_call_s}s/call, wall clock {wall_clock_s:.1f}s")
    print()
    print(f"TRACKS FOUND ({raw_count} raw, {filtered_count} passing filter):")

    for t in sorted(tracks, key=lambda x: -x["match_count"]):
        marker = ""
        baseline_match = baseline_matches.get(t["track_id"])
        if t["passes_filter"]:
            if baseline_match and baseline_match.match_count >= SHAZAM_MIN_MATCHES:
                marker = f" (baseline: {baseline_match.match_count}x)"
            else:
                marker = " NEW"
            status = "\u2713"
        else:
            status = " "
        print(
            f"  {t['match_count']:2}x  {_format_timestamp(t['first_seen_ms'])}  "
            f"{t['artist']:25} - {t['title']:35} {status}{marker}"
        )

    new_vs_baseline = filtered_count - baseline_filtered
    print()
    print(
        f"vs BASELINE: {'+' if new_vs_baseline >= 0 else ''}{new_vs_baseline} new tracks, {filtered_count} total (was {baseline_filtered})"
    )
    print("\u2550" * 70)


async def run_experiment(
    audio_path: str,
    audio_hash: str,
    cache: SegmentCache,
    global_cache: GlobalSegmentCache,
    label: str,
    description: str,
    segments: list[tuple[int, int]],
    parameters: dict,
    baseline_matches: dict[str, TrackMatch],
    total_duration_ms: int,
    contextual_rescoring: bool = False,
) -> dict:
    """Run one experiment.

    Args:
        audio_path: Path to the audio file
        audio_hash: Hash of the audio file
        cache: Per-file segment cache
        global_cache: Global content-addressed cache
        label: Experiment label (e.g., "dense_5s")
        description: Human-readable description
        segments: List of (start_ms, duration_ms) tuples to process
        parameters: Dict of experiment parameters for logging
        baseline_matches: Baseline matches for comparison
        total_duration_ms: Total audio duration in ms
        contextual_rescoring: If True, skip API calls and just re-score

    Returns:
        JSONL entry dict
    """
    _print_experiment_header(label, description)

    t_start = time.monotonic()
    api_errors = 0
    segments_new = 0
    segments_cached = 0

    if not contextual_rescoring and segments:
        # Group segments by duration
        by_duration: dict[int, list[int]] = {}
        for start_ms, duration_ms in segments:
            by_duration.setdefault(duration_ms, []).append(start_ms)

        temp_dir = tempfile.mkdtemp(prefix="libsync_exp_")
        recognizer = ShazamRecognizer(
            max_concurrent=SHAZAM_MAX_CONCURRENT,
            request_delay=SHAZAM_REQUEST_DELAY,
        )

        try:
            for duration_ms, start_list in sorted(by_duration.items()):
                # Filter out already-cached segments for this duration
                cached_at_dur = cache.get_cached_segments_for_duration(audio_hash, duration_ms)
                uncached = [s for s in start_list if s not in cached_at_dur]
                segments_cached += len(start_list) - len(uncached)
                segments_new += len(uncached)

                if not uncached:
                    print(f"  Duration {duration_ms}ms: all {len(start_list)} segments cached")
                    continue

                print(
                    f"  Duration {duration_ms}ms: {len(uncached)} new, {len(start_list) - len(uncached)} cached"
                )

                with tqdm(
                    total=len(uncached), desc=f"{label} ({duration_ms}ms)", unit="seg"
                ) as pbar:

                    def progress(done: int, total: int, phase: str) -> None:
                        if phase == "recognizing":
                            pbar.update(1)

                    await extract_and_recognize_parallel(
                        audio_path=audio_path,
                        temp_dir=temp_dir,
                        segments_to_process=uncached,
                        audio_hash=audio_hash,
                        segment_duration_ms=duration_ms,
                        cache=cache,
                        max_concurrent_shazam=SHAZAM_MAX_CONCURRENT,
                        max_ffmpeg_workers=SHAZAM_FFMPEG_WORKERS,
                        progress_callback=progress,
                        recognizer=recognizer,
                        global_cache=global_cache,
                    )

                metrics = recognizer.get_metrics()
                api_errors = metrics["errors"]
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        # Contextual rescoring or no segments — no API calls
        segments_cached = len(segments)

    wall_clock_s = round(time.monotonic() - t_start, 1)

    # Load ALL cached results and aggregate
    all_results = cache.get_all_results(audio_hash)
    matches = aggregate_matches(all_results)

    # Apply contextual rescoring if requested
    if contextual_rescoring:
        sorted_matches = sorted(matches.values(), key=lambda m: m.first_seen_ms)

    # Build tracks list
    tracks = []
    for m in matches.values():
        if contextual_rescoring:
            confidence = calculate_contextual_confidence(m, sorted_matches, total_duration_ms)
        else:
            confidence = m.calculate_confidence()

        passes = m.match_count >= SHAZAM_MIN_MATCHES and confidence >= SHAZAM_MIN_CONFIDENCE
        tracks.append(
            {
                "track_id": m.shazam_id,
                "artist": m.artist,
                "title": m.title,
                "match_count": m.match_count,
                "confidence": round(confidence, 2),
                "first_seen_ms": m.first_seen_ms,
                "last_seen_ms": m.last_seen_ms,
                "passes_filter": passes,
            }
        )

    tracks.sort(key=lambda t: -t["match_count"])

    # Summary stats
    tracks_passing = sum(1 for t in tracks if t["passes_filter"])
    baseline_filtered = sum(
        1
        for m in baseline_matches.values()
        if m.match_count >= SHAZAM_MIN_MATCHES and m.calculate_confidence() >= SHAZAM_MIN_CONFIDENCE
    )

    segments_total = len(segments)
    total_in_cache = cache.get_cache_stats(audio_hash)["total_segments"]

    _print_experiment_results(
        label=label,
        parameters=parameters,
        segments_total=segments_total,
        segments_new=segments_new,
        segments_cached=segments_cached,
        api_errors=api_errors,
        tracks=tracks,
        baseline_matches=baseline_matches,
        wall_clock_s=wall_clock_s,
    )

    # Build JSONL entry
    entry = {
        "experiment_label": label,
        "timestamp": datetime.now().isoformat(),
        "audio_file": os.path.basename(audio_path),
        "parameters": parameters,
        "segments_total": segments_total,
        "segments_new_api_calls": segments_new,
        "segments_cache_hits": segments_cached,
        "api_errors": api_errors,
        "wall_clock_s": wall_clock_s,
        "total_segments_in_cache": total_in_cache,
        "tracks": tracks,
        "summary": {
            "tracks_detected_raw": len(tracks),
            "tracks_passing_filter": tracks_passing,
            "baseline_tracks_filtered": baseline_filtered,
            "new_tracks_vs_baseline": tracks_passing - baseline_filtered,
            "match_rate_pct": round(
                sum(1 for t in tracks if t["match_count"] > 0) / segments_total * 100, 1
            )
            if segments_total > 0
            else 0,
        },
    }

    # Write JSONL
    with open(EXPERIMENT_RESULTS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"JSONL entry written to: {EXPERIMENT_RESULTS_PATH}")

    # Write human-readable results file
    path_hash = hashlib.sha256(audio_path.encode()).hexdigest()[:12]
    results_path = str(LIBSYNC_DATA_DIR / f"shazam_experiment_{label}_{path_hash}.txt")
    cache_stats = cache.get_cache_stats(audio_hash)
    write_results_file(results_path, audio_path, matches, cache_stats, phase=f"Experiment: {label}")
    print(f"Results file: {results_path}")

    return entry


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def run_all_experiments(
    audio_path: str,
    experiments: list[str] | None = None,
) -> None:
    """Run experiments in optimal order.

    Order: contextual -> reinforce -> dense -> multi_duration
    Each experiment re-loads cache state so it benefits from prior results.

    Args:
        audio_path: Path to the audio file
        experiments: List of experiment names to run, or None for all
    """
    all_experiment_names = ["contextual", "reinforce", "dense", "multi_duration"]
    to_run = experiments if experiments is not None else all_experiment_names

    # Validate experiment names
    invalid = set(to_run) - set(all_experiment_names)
    if invalid:
        print(f"Unknown experiments: {invalid}. Valid: {all_experiment_names}")
        return

    print(f"Running experiments: {to_run}")
    print()

    # Setup shared state
    total_duration_ms = get_audio_duration_ms(audio_path)
    audio_hash = SegmentCacheKey.compute_file_hash(audio_path)
    cache_path = get_shazam_segment_cache_path(audio_path)
    cache = SegmentCache(cache_path)
    global_cache = GlobalSegmentCache(SHAZAM_GLOBAL_CACHE_PATH)

    logger.info(f"Audio: {audio_path}")
    logger.info(f"Duration: {timedelta(milliseconds=total_duration_ms)}")
    logger.info(f"Audio hash: {audio_hash}")
    logger.info(f"Cache: {cache_path}")

    # Get baseline matches (from whatever is already cached)
    baseline_matches = _get_baseline_matches(cache, audio_hash)
    baseline_filtered = sum(
        1
        for m in baseline_matches.values()
        if m.match_count >= SHAZAM_MIN_MATCHES and m.calculate_confidence() >= SHAZAM_MIN_CONFIDENCE
    )
    print(f"Baseline: {len(baseline_matches)} tracks detected, {baseline_filtered} passing filter")
    print(f"Audio duration: {timedelta(milliseconds=total_duration_ms)}")
    print(f"Cache: {cache.get_cache_stats(audio_hash)}")

    # --- Exp 4: Contextual Confidence (0 API calls) ---
    if "contextual" in to_run:
        # Use all currently-cached segments as "segments" for accounting
        cached_count = cache.get_cache_stats(audio_hash)["total_segments"]
        dummy_segments = [(i, SHAZAM_SEGMENT_LENGTH_MS) for i in range(cached_count)]

        await run_experiment(
            audio_path=audio_path,
            audio_hash=audio_hash,
            cache=cache,
            global_cache=global_cache,
            label="contextual",
            description="Contextual confidence re-scoring (0 API calls)",
            segments=dummy_segments,
            parameters={"method": "contextual_rescoring", "api_calls": 0},
            baseline_matches=baseline_matches,
            total_duration_ms=total_duration_ms,
            contextual_rescoring=True,
        )

    # --- Exp 3: Reinforcement (~200 API calls) ---
    if "reinforce" in to_run:
        # Re-load matches to benefit from any prior experiments
        current_matches = _get_baseline_matches(cache, audio_hash)
        weak_matches = [m for m in current_matches.values() if m.match_count == 1]
        print(f"\nReinforcement: {len(weak_matches)} weak (1x) matches to probe")

        segments = generate_reinforcement_segments(weak_matches, total_duration_ms)

        await run_experiment(
            audio_path=audio_path,
            audio_hash=audio_hash,
            cache=cache,
            global_cache=global_cache,
            label="reinforce",
            description=f"Reinforcement probing around {len(weak_matches)} weak matches",
            segments=segments,
            parameters={
                "weak_matches": len(weak_matches),
                "radius_ms": 45000,
                "step_ms": 3000,
                "segment_duration_ms": 15000,
            },
            baseline_matches=baseline_matches,
            total_duration_ms=total_duration_ms,
        )

    # --- Exp 1: Dense Probing (~224 new API calls) ---
    if "dense" in to_run:
        segments = generate_dense_segments(total_duration_ms)

        await run_experiment(
            audio_path=audio_path,
            audio_hash=audio_hash,
            cache=cache,
            global_cache=global_cache,
            label="dense_5s",
            description="Dense probing (15s segments every 5s)",
            segments=segments,
            parameters={"step_ms": 5000, "segment_duration_ms": 15000},
            baseline_matches=baseline_matches,
            total_duration_ms=total_duration_ms,
        )

    # --- Exp 2: Multi-Duration (~160 API calls) ---
    if "multi_duration" in to_run:
        segments = generate_multi_duration_segments(total_duration_ms)

        await run_experiment(
            audio_path=audio_path,
            audio_hash=audio_hash,
            cache=cache,
            global_cache=global_cache,
            label="multi_duration",
            description="Multi-duration (10s + 25s at baseline positions)",
            segments=segments,
            parameters={
                "step_ms": 30000,
                "durations_ms": [10000, 25000],
            },
            baseline_matches=baseline_matches,
            total_duration_ms=total_duration_ms,
        )

    print("\nAll experiments complete!")
    print(f"Results: {EXPERIMENT_RESULTS_PATH}")
