#!/usr/bin/env python3
"""
merge_results.py - Merge all batch JSON result files into a single master dataset.

Scans results/pawn_stars/, results/cajun_pawn_stars/, results/hardcore_pawn/ for JSON files.
Combines all videos into master_dataset/all_transcripts.json.
Deduplicates by video_id.
Prints summary stats.

Outputs are stored via symlinks to the persistent path:
  /root/.claude/projects/-workspaces-youtube-transcript-batch/data/
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

BASE_DIR = "/workspaces/youtube-transcript-batch"
PERSISTENT_DIR = "/root/.claude/projects/-workspaces-youtube-transcript-batch/data"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MASTER_DIR = os.path.join(BASE_DIR, "master_dataset")

SHOW_DIRS = ["pawn_stars", "cajun_pawn_stars", "hardcore_pawn"]


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


def find_batch_json_files(results_dir):
    """Find all JSON files in results subdirectories and root batch files."""
    json_files = []

    # Scan show-specific result directories
    for show in SHOW_DIRS:
        show_dir = os.path.join(results_dir, show)
        if os.path.isdir(show_dir):
            pattern = os.path.join(show_dir, "*.json")
            found = glob.glob(pattern)
            for f in found:
                json_files.append((f, show))
            log(f"  Found {len(found)} JSON file(s) in results/{show}/")

    # Also scan for batch result files in the base directory
    root_pattern = os.path.join(BASE_DIR, "batch_results_*.json")
    root_files = glob.glob(root_pattern)
    for f in root_files:
        json_files.append((f, "unknown"))
    if root_files:
        log(f"  Found {len(root_files)} batch result file(s) in project root")

    return json_files


def extract_videos_from_file(filepath, default_show):
    """Extract video records from a batch JSON file."""
    videos = []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        # Handle batch format: {"metadata": ..., "videos": [...]}
        if isinstance(data, dict) and "videos" in data:
            for video in data["videos"]:
                video["_source_file"] = os.path.basename(filepath)
                if "show" not in video:
                    video["show"] = default_show
                videos.append(video)

        # Handle list format: [{video}, {video}, ...]
        elif isinstance(data, list):
            for video in data:
                if isinstance(video, dict):
                    video["_source_file"] = os.path.basename(filepath)
                    if "show" not in video:
                        video["show"] = default_show
                    videos.append(video)

        # Handle single video format
        elif isinstance(data, dict) and "video_id" in data:
            data["_source_file"] = os.path.basename(filepath)
            if "show" not in data:
                data["show"] = default_show
            videos.append(data)

    except (json.JSONDecodeError, IOError) as e:
        log(f"  WARNING: Could not read {filepath}: {e}")

    return videos


def deduplicate_videos(all_videos):
    """Deduplicate videos by video_id, keeping the most complete record."""
    seen = {}
    duplicates = 0

    for video in all_videos:
        vid = video.get("video_id")
        if not vid:
            continue

        if vid in seen:
            duplicates += 1
            # Keep the record with more data (more keys or longer transcript)
            existing = seen[vid]
            existing_size = len(json.dumps(existing))
            new_size = len(json.dumps(video))
            if new_size > existing_size:
                seen[vid] = video
        else:
            seen[vid] = video

    return list(seen.values()), duplicates


def compute_stats(videos):
    """Compute summary statistics."""
    stats = {
        "total_videos": len(videos),
        "by_show": defaultdict(int),
        "by_status": defaultdict(int),
        "total_transcript_entries": 0,
        "total_sentences": 0,
        "total_duration_seconds": 0,
        "videos_with_transcripts": 0,
    }

    for video in videos:
        show = video.get("show", "unknown")
        status = video.get("status", "unknown")
        stats["by_show"][show] += 1
        stats["by_status"][status] += 1

        if "transcript" in video and video["transcript"]:
            stats["videos_with_transcripts"] += 1
            t = video["transcript"]
            if isinstance(t, list):
                stats["total_transcript_entries"] += len(t)
            elif isinstance(t, dict):
                raw = t.get("raw_entries", [])
                stats["total_transcript_entries"] += len(raw)

        if "analysis_fields" in video and video["analysis_fields"]:
            sentences = video["analysis_fields"].get("sentence_boundaries", [])
            stats["total_sentences"] += len(sentences)

        if "video_metadata" in video and video["video_metadata"]:
            dur = video["video_metadata"].get("duration_seconds", 0)
            stats["total_duration_seconds"] += dur or 0

    # Convert defaultdicts to regular dicts for JSON serialization
    stats["by_show"] = dict(stats["by_show"])
    stats["by_status"] = dict(stats["by_status"])

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Merge all batch JSON result files into a single master dataset."
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help=f"Directory containing show result subdirectories (default: {RESULTS_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        default=MASTER_DIR,
        help=f"Output directory for merged dataset (default: {MASTER_DIR})"
    )
    parser.add_argument(
        "--output-file",
        default="all_transcripts.json",
        help="Output filename (default: all_transcripts.json)"
    )
    args = parser.parse_args()

    log("=" * 60)
    log("MERGE RESULTS - Pawn Shop Negotiation Dataset")
    log("=" * 60)

    # Ensure output directory exists with symlink
    persistent_master = os.path.join(PERSISTENT_DIR, "master_dataset")
    ensure_symlink(args.output_dir, persistent_master)

    log(f"\nScanning for JSON files in: {args.results_dir}")
    json_files = find_batch_json_files(args.results_dir)

    if not json_files:
        log("\nNo JSON files found. Nothing to merge.")
        log("Expected files in: results/pawn_stars/, results/cajun_pawn_stars/, results/hardcore_pawn/")
        log("Also checked for batch_results_*.json in project root.")
        sys.exit(0)

    log(f"\nTotal JSON files found: {len(json_files)}")

    # Extract all videos
    log("\nExtracting videos from files...")
    all_videos = []
    for filepath, show in json_files:
        videos = extract_videos_from_file(filepath, show)
        log(f"  {os.path.basename(filepath)}: {len(videos)} video(s)")
        all_videos.extend(videos)

    log(f"\nTotal videos extracted: {len(all_videos)}")

    # Deduplicate
    log("\nDeduplicating by video_id...")
    unique_videos, dup_count = deduplicate_videos(all_videos)
    log(f"  Removed {dup_count} duplicate(s)")
    log(f"  Unique videos: {len(unique_videos)}")

    # Compute stats
    stats = compute_stats(unique_videos)

    # Build master dataset
    master = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "source_files": len(json_files),
            "duplicates_removed": dup_count,
            "stats": stats,
        },
        "videos": unique_videos,
    }

    # Write output
    output_path = os.path.join(args.output_dir, args.output_file)
    log(f"\nWriting master dataset to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(master, f, indent=2)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log(f"  File size: {file_size_mb:.2f} MB")

    # Print summary
    log("\n" + "=" * 60)
    log("SUMMARY STATISTICS")
    log("=" * 60)
    log(f"  Total unique videos:        {stats['total_videos']}")
    log(f"  Videos with transcripts:     {stats['videos_with_transcripts']}")
    log(f"  Total transcript entries:    {stats['total_transcript_entries']}")
    log(f"  Total sentences:             {stats['total_sentences']}")
    log(f"  Total duration:              {stats['total_duration_seconds']:.1f}s "
        f"({stats['total_duration_seconds']/3600:.1f}h)")
    log(f"\n  By show:")
    for show, count in sorted(stats["by_show"].items()):
        log(f"    {show}: {count}")
    log(f"\n  By status:")
    for status, count in sorted(stats["by_status"].items()):
        log(f"    {status}: {count}")
    log("\n" + "=" * 60)
    log("Done!")


if __name__ == "__main__":
    main()
