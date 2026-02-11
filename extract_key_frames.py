#!/usr/bin/env python3
"""
extract_key_frames.py - Extract video frames at key negotiation moments.

Reads negotiations_analysis.json for timestamps.
Uses ThreadPoolExecutor with configurable workers.
For each video: gets stream URL via yt-dlp, extracts frames via ffmpeg.
Saves frames as JPG to key_frames/{video_id}/{moment_type}_{timestamp}s.jpg.
Maintains key_frames/manifest.json with frame metadata.
Incremental: skips already-processed videos based on manifest.
Auto-stops after 5 consecutive failures.
Saves manifest every 5 videos.

Outputs are stored via symlinks to the persistent path:
  /root/.claude/projects/-workspaces-youtube-transcript-batch/data/
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

BASE_DIR = "/workspaces/youtube-transcript-batch"
PERSISTENT_DIR = "/root/.claude/projects/-workspaces-youtube-transcript-batch/data"
KEY_FRAMES_DIR = os.path.join(BASE_DIR, "key_frames")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

YTDLP_PATH = "/workspaces/youtube-transcript-batch/venv/bin/yt-dlp"
YTDLP_TIMEOUT = 30  # seconds


def log(msg):
    print(msg)
    sys.stdout.flush()


def ensure_symlink(workspace_path, persistent_path):
    """Ensure workspace_path is a symlink to persistent_path."""
    os.makedirs(persistent_path, exist_ok=True)
    if os.path.islink(workspace_path):
        return
    if os.path.isdir(workspace_path) and not os.listdir(workspace_path):
        os.rmdir(workspace_path)
    if not os.path.exists(workspace_path):
        os.symlink(persistent_path, workspace_path)


def load_manifest(manifest_path):
    """Load existing manifest or create new one."""
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    return {
        "source": "",
        "total_moments": 0,
        "total_frames": 0,
        "window_seconds": 1.0,
        "frames": [],
    }


def save_manifest(manifest, manifest_path):
    """Save manifest to disk."""
    manifest["total_frames"] = len(manifest["frames"])
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def get_processed_videos(manifest):
    """Get set of video_ids already processed."""
    processed = set()
    for frame in manifest.get("frames", []):
        processed.add(frame.get("video_id", ""))
    return processed


def get_stream_url(video_id):
    """Get the direct stream URL for a video using yt-dlp."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        YTDLP_PATH,
        "--get-url",
        "-f", "best[ext=mp4]/best",
        "--no-warnings",
        "--no-playlist",
        url,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=YTDLP_TIMEOUT,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
        else:
            return None
    except subprocess.TimeoutExpired:
        log(f"    TIMEOUT: yt-dlp timed out for {video_id}")
        return None
    except Exception as e:
        log(f"    ERROR: yt-dlp failed for {video_id}: {e}")
        return None


def extract_frame(stream_url, timestamp, output_path):
    """Extract a single frame from a video stream at the given timestamp."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", stream_url,
        "-frames:v", "1",
        "-q:v", "2",
        "-y",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0 and os.path.isfile(output_path)
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def process_video(video_id, moments, key_frames_dir):
    """Process a single video: get stream URL and extract frames."""
    frames = []
    errors = []

    # Get stream URL
    stream_url = get_stream_url(video_id)
    if not stream_url:
        return frames, [f"Could not get stream URL for {video_id}"]

    video_dir = os.path.join(key_frames_dir, video_id)
    os.makedirs(video_dir, exist_ok=True)

    for moment in moments:
        timestamp = moment["timestamp"]
        moment_type = moment["moment_type"]
        ts_str = f"{timestamp:.1f}".replace(".", ".")

        filename = f"{moment_type}_{ts_str}s.jpg"
        output_path = os.path.join(video_dir, filename)

        # Skip if frame already exists
        if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            frame_info = {
                "file": f"{video_id}/{filename}",
                "video_id": video_id,
                "timestamp": timestamp,
                "moment_type": moment_type,
                "amount": moment.get("amount"),
                "speaker": moment.get("speaker", "unknown"),
                "subject": moment.get("subject", ""),
            }
            frames.append(frame_info)
            continue

        success = extract_frame(stream_url, timestamp, output_path)
        if success:
            frame_info = {
                "file": f"{video_id}/{filename}",
                "video_id": video_id,
                "timestamp": timestamp,
                "moment_type": moment_type,
                "amount": moment.get("amount"),
                "speaker": moment.get("speaker", "unknown"),
                "subject": moment.get("subject", ""),
            }
            frames.append(frame_info)
        else:
            errors.append(f"Failed to extract frame at {timestamp}s for {video_id}")

    return frames, errors


def build_moments_from_segments(segments):
    """Convert negotiation segments to frame extraction moments."""
    video_moments = defaultdict(list)

    for seg in segments:
        video_id = seg.get("video_id", "")
        if not video_id:
            continue

        start_time = seg.get("start_time", 0)
        end_time = seg.get("end_time", start_time)
        segment_type = seg.get("segment_type", "unknown")
        dollar_amounts = seg.get("dollar_amounts", [])
        speaker = seg.get("speaker", "unknown")
        text = seg.get("text", "")

        # Create a moment at the start of each segment
        moment = {
            "timestamp": start_time,
            "moment_type": segment_type,
            "amount": dollar_amounts[0] if dollar_amounts else None,
            "speaker": speaker,
            "subject": text[:100],
        }
        video_moments[video_id].append(moment)

        # If segment is long enough, also add a moment at midpoint
        duration = end_time - start_time
        if duration > 10:
            mid_moment = dict(moment)
            mid_moment["timestamp"] = (start_time + end_time) / 2
            mid_moment["moment_type"] = f"{segment_type}_mid"
            video_moments[video_id].append(mid_moment)

    return video_moments


def main():
    parser = argparse.ArgumentParser(
        description="Extract video frames at key negotiation moments."
    )
    parser.add_argument(
        "--input",
        default=os.path.join(RESULTS_DIR, "negotiations_analysis.json"),
        help="Input negotiations analysis JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default=KEY_FRAMES_DIR,
        help=f"Output directory for frames (default: {KEY_FRAMES_DIR})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=5,
        help="Auto-stop after N consecutive failures (default: 5)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually extracting frames"
    )
    args = parser.parse_args()

    log("=" * 60)
    log("KEY FRAME EXTRACTION - Pawn Shop Negotiations")
    log("=" * 60)

    # Ensure output directory symlink
    persistent_kf = os.path.join(PERSISTENT_DIR, "key_frames")
    ensure_symlink(args.output_dir, persistent_kf)

    # Read negotiations analysis
    log(f"\nReading input: {args.input}")
    if not os.path.isfile(args.input):
        log(f"ERROR: Input file not found: {args.input}")
        log("Run analyze_negotiations.py first.")
        sys.exit(1)

    with open(args.input, "r") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    log(f"  Loaded {len(segments)} negotiation segments")

    # Build moments from segments
    log("\nBuilding frame extraction moments...")
    video_moments = build_moments_from_segments(segments)
    total_moments = sum(len(m) for m in video_moments.values())
    log(f"  Videos: {len(video_moments)}")
    log(f"  Total moments: {total_moments}")

    # Load manifest for incremental processing
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    manifest = load_manifest(manifest_path)
    manifest["source"] = args.input
    manifest["total_moments"] = total_moments

    processed_videos = get_processed_videos(manifest)
    log(f"  Already processed videos: {len(processed_videos)}")

    # Filter to unprocessed videos
    to_process = {
        vid: moments for vid, moments in video_moments.items()
        if vid not in processed_videos
    }
    log(f"  Videos to process: {len(to_process)}")

    if not to_process:
        log("\nAll videos already processed. Nothing to do.")
        save_manifest(manifest, manifest_path)
        log(f"Manifest saved: {manifest_path}")
        return

    if args.dry_run:
        log("\n[DRY RUN] Would process:")
        for vid, moments in list(to_process.items())[:10]:
            log(f"  {vid}: {len(moments)} moments")
        if len(to_process) > 10:
            log(f"  ... and {len(to_process) - 10} more videos")
        return

    # Check yt-dlp availability
    if not os.path.isfile(YTDLP_PATH):
        log(f"\nERROR: yt-dlp not found at {YTDLP_PATH}")
        sys.exit(1)

    # Process videos
    log(f"\nProcessing with {args.workers} worker(s)...")
    consecutive_failures = 0
    videos_processed = 0
    total_frames_extracted = 0
    manifest_lock = Lock()

    video_list = list(to_process.items())

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_video = {}

        for video_id, moments in video_list:
            future = executor.submit(
                process_video, video_id, moments, args.output_dir
            )
            future_to_video[future] = video_id

        for future in as_completed(future_to_video):
            video_id = future_to_video[future]

            try:
                frames, errors = future.result()

                if frames:
                    with manifest_lock:
                        manifest["frames"].extend(frames)
                    total_frames_extracted += len(frames)
                    consecutive_failures = 0
                    log(f"  [{videos_processed + 1}/{len(video_list)}] "
                        f"{video_id}: {len(frames)} frame(s) extracted")
                else:
                    consecutive_failures += 1
                    log(f"  [{videos_processed + 1}/{len(video_list)}] "
                        f"{video_id}: FAILED - {errors[0] if errors else 'unknown error'}")

                if errors:
                    for err in errors:
                        log(f"    WARNING: {err}")

            except Exception as e:
                consecutive_failures += 1
                log(f"  [{videos_processed + 1}/{len(video_list)}] "
                    f"{video_id}: EXCEPTION - {e}")

            videos_processed += 1

            # Save manifest periodically
            if videos_processed % 5 == 0:
                with manifest_lock:
                    save_manifest(manifest, manifest_path)
                log(f"  [Manifest saved at {videos_processed} videos]")

            # Auto-stop on consecutive failures
            if consecutive_failures >= args.max_failures:
                log(f"\n  AUTO-STOP: {consecutive_failures} consecutive failures. Stopping.")
                # Cancel remaining futures
                for f in future_to_video:
                    f.cancel()
                break

    # Final manifest save
    save_manifest(manifest, manifest_path)
    log(f"\nManifest saved: {manifest_path}")

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  Videos processed:       {videos_processed}")
    log(f"  Frames extracted:       {total_frames_extracted}")
    log(f"  Total frames in manifest: {len(manifest['frames'])}")
    log(f"  Consecutive failures:   {consecutive_failures}")
    log("\n" + "=" * 60)
    log("Done!")


if __name__ == "__main__":
    main()
