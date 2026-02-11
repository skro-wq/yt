#!/usr/bin/env python3
"""
collect_urls.py - Collect video URLs from YouTube for pawn shop shows.

Uses yt-dlp to search for and list videos from pawn shop TV show channels,
outputting deduplicated CSV files of video IDs, titles, and URLs.

Usage:
    python collect_urls.py [--output-dir OUTPUT_DIR]
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Path to the yt-dlp binary inside the project venv
YT_DLP_BIN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "yt-dlp"
)

# Persistent storage path for results
PERSISTENT_DIR = Path(
    "/root/.claude/projects/-workspaces-youtube-transcript-batch/data/pawn_urls"
)

# Default output directory (relative to cwd)
DEFAULT_OUTPUT_DIR = "pawn_urls"

# ──────────────────────────────────────────────
#  Source definitions
# ──────────────────────────────────────────────
#
# Each source is a dict with:
#   name      - human-readable label and CSV filename stem
#   urls      - list of yt-dlp URLs to query (searches + channel pages)

SOURCES = [
    {
        "name": "pawn_stars",
        "label": "Pawn Stars",
        "urls": [
            "ytsearch500:Pawn Stars full episode",
            "ytsearch500:Pawn Stars History Channel",
            "https://www.youtube.com/@PawnStars/videos",
            "https://www.youtube.com/@HistoryChannel/search?query=Pawn+Stars",
        ],
    },
    {
        "name": "cajun_pawn_stars",
        "label": "Cajun Pawn Stars",
        "urls": [
            "ytsearch500:Cajun Pawn Stars full episode",
            "ytsearch500:Cajun Pawn Stars",
        ],
    },
    {
        "name": "hardcore_pawn",
        "label": "Hardcore Pawn",
        "urls": [
            "ytsearch500:Hardcore Pawn full episode",
            "ytsearch500:Hardcore Pawn truTV",
            "https://www.youtube.com/@HardcorePawn/videos",
        ],
    },
]


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────


def run_ytdlp(url: str, timeout: int = 600) -> list[dict]:
    """
    Call yt-dlp with --flat-playlist to extract video ids and titles
    from *url*.  Returns a list of dicts with keys: video_id, title, url.

    Uses the ``--print`` flag twice so each video produces exactly two
    lines of output: the video id followed by the title.
    """
    cmd = [
        YT_DLP_BIN,
        "--flat-playlist",
        "--print", "id",
        "--print", "title",
        "--no-warnings",
        "--ignore-errors",
        "--extractor-retries", "3",
        "--socket-timeout", "30",
        url,
    ]

    results = []
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        lines = proc.stdout.strip().splitlines()

        # Lines come in pairs: id, title, id, title, ...
        if len(lines) % 2 != 0:
            # If output is malformed, drop the trailing unpaired line
            lines = lines[: len(lines) - (len(lines) % 2)]

        for i in range(0, len(lines), 2):
            video_id = lines[i].strip()
            title = lines[i + 1].strip()
            if video_id and title:
                results.append(
                    {
                        "video_id": video_id,
                        "title": title,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                    }
                )

        if proc.returncode != 0 and proc.stderr:
            # Print a short note but do not fail -- partial results are fine
            stderr_brief = proc.stderr.strip().splitlines()[-1]
            print(f"  [warn] yt-dlp returned code {proc.returncode} for {url}")
            print(f"         {stderr_brief}")

    except subprocess.TimeoutExpired:
        print(f"  [error] Timed out after {timeout}s for: {url}")
    except FileNotFoundError:
        print(f"  [error] yt-dlp binary not found at {YT_DLP_BIN}")
        print("          Install it with: pip install yt-dlp")
        sys.exit(1)
    except Exception as exc:
        print(f"  [error] Unexpected error for {url}: {exc}")

    return results


def scrape_source(source: dict) -> tuple[str, list[dict]]:
    """
    Scrape all URLs for a single source, deduplicate by video_id,
    and return (source_name, list_of_video_dicts).
    """
    name = source["name"]
    label = source["label"]
    urls = source["urls"]

    print(f"\n{'='*60}")
    print(f"  Scraping: {label}  ({len(urls)} URL(s))")
    print(f"{'='*60}")

    seen_ids: set[str] = set()
    all_videos: list[dict] = []

    for url in urls:
        print(f"  -> {url}")
        t0 = time.monotonic()
        videos = run_ytdlp(url)
        elapsed = time.monotonic() - t0

        new_count = 0
        for v in videos:
            if v["video_id"] not in seen_ids:
                seen_ids.add(v["video_id"])
                all_videos.append(v)
                new_count += 1

        print(f"     Found {len(videos)} videos, {new_count} new  ({elapsed:.1f}s)")

    print(f"  Total unique for {label}: {len(all_videos)}")
    return name, all_videos


def write_csv(filepath: Path, videos: list[dict]) -> None:
    """Write a list of video dicts to a CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video_id", "title", "url"])
        writer.writeheader()
        writer.writerows(videos)
    print(f"  Wrote {len(videos)} rows -> {filepath}")


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect YouTube video URLs for pawn shop TV shows."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write CSV files (default: {DEFAULT_OUTPUT_DIR}/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Also ensure the persistent storage directory exists
    PERSISTENT_DIR.mkdir(parents=True, exist_ok=True)

    print("Pawn Shop Show URL Collector")
    print(f"  Output directory : {output_dir}")
    print(f"  Persistent copy  : {PERSISTENT_DIR}")
    print(f"  yt-dlp binary    : {YT_DLP_BIN}")
    print(f"  Sources          : {len(SOURCES)}")

    # ── Parallel scraping ────────────────────────
    results: dict[str, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=len(SOURCES)) as pool:
        futures = {
            pool.submit(scrape_source, src): src["name"] for src in SOURCES
        }
        for future in as_completed(futures):
            source_name = futures[future]
            try:
                name, videos = future.result()
                results[name] = videos
            except Exception as exc:
                print(f"\n  [error] Source '{source_name}' raised: {exc}")
                results[source_name] = []

    # ── Write per-source CSVs ────────────────────
    print(f"\n{'='*60}")
    print("  Writing CSV files")
    print(f"{'='*60}")

    combined_seen: set[str] = set()
    combined_videos: list[dict] = []

    for source in SOURCES:
        name = source["name"]
        videos = results.get(name, [])

        # Write to output-dir
        write_csv(output_dir / f"{name}.csv", videos)

        # Write to persistent storage
        write_csv(PERSISTENT_DIR / f"{name}.csv", videos)

        # Accumulate for combined file (deduplicate across sources)
        for v in videos:
            if v["video_id"] not in combined_seen:
                combined_seen.add(v["video_id"])
                combined_videos.append(v)

    # ── Write combined CSV ───────────────────────
    write_csv(output_dir / "all_pawn_shows.csv", combined_videos)
    write_csv(PERSISTENT_DIR / "all_pawn_shows.csv", combined_videos)

    # ── Summary ──────────────────────────────────
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for source in SOURCES:
        name = source["name"]
        label = source["label"]
        count = len(results.get(name, []))
        print(f"    {label:25s} : {count:>5} videos")
    print(f"    {'Combined (deduplicated)':25s} : {len(combined_videos):>5} videos")
    print(f"\nDone. Files saved to:")
    print(f"  {output_dir}")
    print(f"  {PERSISTENT_DIR}")


if __name__ == "__main__":
    main()
