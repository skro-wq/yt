#!/workspaces/youtube-transcript-batch/venv/bin/python3
"""
Batch YouTube Transcript Downloader

Downloads transcripts for YouTube videos listed in CSV files and saves
them as structured JSON batch files with incremental progress tracking.

Usage:
    python process_all_batches.py input.csv
    python process_all_batches.py ./csv_directory/
    python process_all_batches.py input.csv --workers 8 --batch-size 100
    python process_all_batches.py ./csvs/ --output-dir /data/transcripts
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import (
    AgeRestricted,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flush_print(*args, **kwargs):
    """Print with immediate flush so output appears in real-time."""
    print(*args, **kwargs)
    sys.stdout.flush()


def load_csv(csv_path: Path) -> list[dict]:
    """
    Load a CSV file with columns video_id, title, url.

    Returns a list of dicts with those three keys.
    """
    videos = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalise header names (strip whitespace, lowercase)
        if reader.fieldnames is None:
            return videos
        normalised = {h: h.strip().lower().replace(" ", "_") for h in reader.fieldnames}
        for row in reader:
            normed = {normalised[k]: v.strip() for k, v in row.items() if k in normalised}
            vid = normed.get("video_id", "")
            title = normed.get("title", "")
            url = normed.get("url", "")
            if vid:
                videos.append({"video_id": vid, "title": title, "url": url})
    return videos


def discover_csvs(input_path: Path) -> list[Path]:
    """Return a sorted list of CSV paths from *input_path*.

    If *input_path* is a single file, return it in a list.
    If it is a directory, return every *.csv inside it (non-recursive).
    """
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        csvs = sorted(input_path.glob("*.csv"))
        if not csvs:
            flush_print(f"[WARN] No CSV files found in {input_path}")
        return csvs
    flush_print(f"[ERROR] Input path does not exist: {input_path}")
    sys.exit(1)


def show_name_from_csv(csv_path: Path) -> str:
    """Derive a show/dataset name from the CSV filename.

    Examples:
        pawn_stars_videos.csv  -> pawn_stars_videos
        /data/hardcore_pawn.csv -> hardcore_pawn
    """
    return csv_path.stem


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Thread-safe tracker that persists completed video IDs to disk."""

    def __init__(self, progress_path: Path):
        self._path = progress_path
        self._lock = Lock()
        self._data: dict = self._load()

    # -- persistence --

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                flush_print(f"[WARN] Corrupt progress file, starting fresh: {self._path}")
        return {
            "completed_video_ids": [],
            "failed_video_ids": [],
            "last_updated": None,
        }

    def save(self):
        with self._lock:
            self._data["last_updated"] = datetime.now(timezone.utc).isoformat()
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            tmp.replace(self._path)

    # -- query --

    def is_completed(self, video_id: str) -> bool:
        with self._lock:
            return video_id in self._data["completed_video_ids"]

    # -- mutate --

    def mark_completed(self, video_id: str):
        with self._lock:
            if video_id not in self._data["completed_video_ids"]:
                self._data["completed_video_ids"].append(video_id)

    def mark_failed(self, video_id: str):
        with self._lock:
            if video_id not in self._data["failed_video_ids"]:
                self._data["failed_video_ids"].append(video_id)

    @property
    def completed_count(self) -> int:
        with self._lock:
            return len(self._data["completed_video_ids"])

    @property
    def failed_count(self) -> int:
        with self._lock:
            return len(self._data["failed_video_ids"])


# ---------------------------------------------------------------------------
# Transcript fetching
# ---------------------------------------------------------------------------

# Module-level API instance (thread-safe for read-only config)
_api = YouTubeTranscriptApi()

# Exceptions that signal rate-limiting / IP blocking — consecutive hits
# of these should trigger an automatic stop.
_RATE_LIMIT_EXCEPTIONS = (IpBlocked, RequestBlocked)


def fetch_transcript(video: dict) -> dict:
    """Fetch the transcript for a single video.

    Returns a dict ready for inclusion in the batch JSON:
        {video_id, title, url, transcript, error}
    """
    video_id = video["video_id"]
    result = {
        "video_id": video_id,
        "title": video.get("title", ""),
        "url": video.get("url", ""),
        "transcript": None,
        "error": None,
    }
    try:
        fetched = _api.fetch(video_id)
        result["transcript"] = [
            {"text": s.text, "start": s.start, "duration": s.duration}
            for s in fetched
        ]
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable,
            VideoUnplayable, AgeRestricted) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    except _RATE_LIMIT_EXCEPTIONS as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        # Re-raise so the caller can detect consecutive rate-limit hits.
        raise
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# ---------------------------------------------------------------------------
# Batch writing
# ---------------------------------------------------------------------------

def write_batch(batch_videos: list[dict], batch_num: int, output_dir: Path):
    """Write a list of video results to a numbered batch JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_path = output_dir / f"batch_{batch_num:03d}.json"
    payload = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "batch_num": batch_num,
            "video_count": len(batch_videos),
            "success_count": sum(1 for v in batch_videos if v.get("transcript")),
            "error_count": sum(1 for v in batch_videos if v.get("error")),
        },
        "videos": batch_videos,
    }
    tmp = batch_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(batch_path)
    flush_print(f"  [BATCH] Wrote {batch_path}  "
                f"({payload['metadata']['success_count']} ok, "
                f"{payload['metadata']['error_count']} err)")


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_csv(csv_path: Path, output_dir: Path, workers: int,
                batch_size: int) -> bool:
    """Process a single CSV file. Returns True if stopped early due to
    rate-limiting, False otherwise."""

    show_name = show_name_from_csv(csv_path)
    show_output_dir = output_dir / show_name
    progress_path = show_output_dir / "progress.json"
    show_output_dir.mkdir(parents=True, exist_ok=True)

    flush_print(f"\n{'='*60}")
    flush_print(f"Processing: {csv_path}")
    flush_print(f"Output dir: {show_output_dir}")
    flush_print(f"{'='*60}")

    videos = load_csv(csv_path)
    if not videos:
        flush_print("  No videos found in CSV — skipping.")
        return False

    tracker = ProgressTracker(progress_path)

    # Filter out already-completed videos.
    pending = [v for v in videos if not tracker.is_completed(v["video_id"])]
    flush_print(f"  Total videos: {len(videos)}  |  "
                f"Already done: {len(videos) - len(pending)}  |  "
                f"Pending: {len(pending)}")

    if not pending:
        flush_print("  All videos already processed — nothing to do.")
        return False

    # We will accumulate results for the current batch.
    current_batch: list[dict] = []
    # Determine the starting batch number from existing files.
    existing_batches = sorted(show_output_dir.glob("batch_*.json"))
    if existing_batches:
        # Parse the highest batch number and continue from the next.
        last_name = existing_batches[-1].stem  # e.g. "batch_003"
        batch_num = int(last_name.split("_")[1]) + 1
    else:
        batch_num = 1

    consecutive_failures = 0
    max_consecutive_failures = 5
    rate_limited = False
    processed_since_save = 0

    def _handle_result(result: dict):
        """Book-keeping after one video completes."""
        nonlocal consecutive_failures, rate_limited, processed_since_save
        nonlocal current_batch, batch_num

        video_id = result["video_id"]

        if result["transcript"] is not None:
            tracker.mark_completed(video_id)
            consecutive_failures = 0
            flush_print(f"    [OK]   {video_id}  {result['title'][:50]}")
        else:
            tracker.mark_failed(video_id)
            tracker.mark_completed(video_id)
            error_str = result.get("error", "")
            # Only count rate-limit errors toward consecutive failures
            is_rate_limit = any(e in error_str for e in ("IpBlocked", "RequestBlocked"))
            if is_rate_limit:
                consecutive_failures += 1
            else:
                # Non-rate-limit failure (TranscriptsDisabled, etc) - reset counter
                consecutive_failures = 0
            flush_print(f"    [FAIL] {video_id}  {error_str[:80]}")

        current_batch.append(result)
        processed_since_save += 1

        # Flush batch to disk when full.
        if len(current_batch) >= batch_size:
            write_batch(current_batch, batch_num, show_output_dir)
            current_batch = []
            batch_num += 1

        # Save progress every 5 videos.
        if processed_since_save >= 5:
            tracker.save()
            processed_since_save = 0

        # Auto-stop on consecutive failures (likely rate-limited).
        if consecutive_failures >= max_consecutive_failures:
            rate_limited = True

    # ----- parallel download -----
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_video = {}
        # Submit work in chunks to allow early termination.
        for video in pending:
            if rate_limited:
                break
            fut = pool.submit(fetch_transcript, video)
            future_to_video[fut] = video

        for fut in as_completed(future_to_video):
            if rate_limited:
                # Cancel remaining futures.
                for f in future_to_video:
                    f.cancel()
                break
            video = future_to_video[fut]
            try:
                result = fut.result()
            except _RATE_LIMIT_EXCEPTIONS as exc:
                # Build a result dict for the failed video.
                result = {
                    "video_id": video["video_id"],
                    "title": video.get("title", ""),
                    "url": video.get("url", ""),
                    "transcript": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            except Exception as exc:
                result = {
                    "video_id": video["video_id"],
                    "title": video.get("title", ""),
                    "url": video.get("url", ""),
                    "transcript": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            _handle_result(result)

    # Flush any remaining videos in the current batch.
    if current_batch:
        write_batch(current_batch, batch_num, show_output_dir)

    # Final progress save.
    tracker.save()

    flush_print(f"\n  Completed: {tracker.completed_count}  |  "
                f"Failed: {tracker.failed_count}")

    if rate_limited:
        flush_print("  [STOP] Auto-stopped after "
                     f"{max_consecutive_failures} consecutive failures "
                     "(possible rate limit). Re-run later to resume.")
        return True

    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-download YouTube transcripts from CSV file(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a CSV file (video_id,title,url) or a directory of CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Root output directory (default: results/).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of videos per output JSON batch file (default: 50).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    flush_print(f"YouTube Transcript Batch Downloader")
    flush_print(f"  Input:      {args.input_path}")
    flush_print(f"  Output dir: {args.output_dir}")
    flush_print(f"  Workers:    {args.workers}")
    flush_print(f"  Batch size: {args.batch_size}")

    csv_files = discover_csvs(args.input_path)
    if not csv_files:
        flush_print("[ERROR] No CSV files to process.")
        sys.exit(1)

    flush_print(f"  CSV files:  {len(csv_files)}")

    start = time.monotonic()
    stopped_early = False

    for csv_path in csv_files:
        hit_limit = process_csv(
            csv_path,
            output_dir=args.output_dir,
            workers=args.workers,
            batch_size=args.batch_size,
        )
        if hit_limit:
            stopped_early = True
            break

    elapsed = time.monotonic() - start
    flush_print(f"\nDone in {elapsed:.1f}s.")
    if stopped_early:
        flush_print("Note: Processing was interrupted by rate limiting. "
                     "Run again to resume from where it left off.")
        sys.exit(2)


if __name__ == "__main__":
    main()
