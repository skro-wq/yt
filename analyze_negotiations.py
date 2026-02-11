#!/usr/bin/env python3
"""
analyze_negotiations.py - Extract negotiation segments from transcripts.

Reads all batch result JSON files from results/ directories.
Identifies negotiation segments using keyword patterns for:
  initial_ask, first_offer, counter, expert_appraisal, deal
Extracts dollar amounts from text.
Records: video_id, show, segment_type, start_time, end_time, text, dollar_amounts, sentences.
Outputs results/negotiations_analysis.json.

Outputs are stored via symlinks to the persistent path:
  /root/.claude/projects/-workspaces-youtube-transcript-batch/data/
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

BASE_DIR = "/workspaces/youtube-transcript-batch"
PERSISTENT_DIR = "/root/.claude/projects/-workspaces-youtube-transcript-batch/data"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SHOW_DIRS = ["pawn_stars", "cajun_pawn_stars", "hardcore_pawn"]

# Negotiation keyword patterns
SEGMENT_PATTERNS = {
    "initial_ask": [
        r"how\s+much",
        r"looking\s+for",
        r"want\s+for",
        r"asking\s+for",
        r"worth",
        r"i'?d\s+like",
    ],
    "first_offer": [
        r"i\s+can\s+do",
        r"i'?ll\s+give\s+you",
        r"best\s+i\s+can\s+do",
        r"i\s+can\s+offer",
        r"i'?d\s+go",
    ],
    "counter": [
        r"how\s+about",
        r"what\s+about",
        r"meet\s+in\s+the\s+middle",
        r"split\s+the\s+difference",
        r"come\s+up",
    ],
    "expert_appraisal": [
        r"expert",
        r"appraiser",
        r"worth\s+about",
        r"value\s+at",
        r"retail\s+for",
        r"auction",
    ],
    "deal": [
        r"deal",
        r"you\s+got\s+a\s+deal",
        r"sold",
        r"i'?ll\s+take\s+it",
        r"shake\s+on\s+it",
    ],
}

# Dollar amount patterns: $N, $N,NNN, $N,NNN,NNN, $N.NN etc.
DOLLAR_PATTERN = re.compile(
    r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)'
)

# Also match written amounts like "500 dollars", "five hundred dollars"
DOLLAR_WORDS_PATTERN = re.compile(
    r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:dollars?|bucks?|grand)',
    re.IGNORECASE,
)


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


def find_all_json_files():
    """Find all batch JSON result files."""
    json_files = []

    # Show-specific directories
    for show in SHOW_DIRS:
        show_dir = os.path.join(RESULTS_DIR, show)
        if os.path.isdir(show_dir):
            for f in glob.glob(os.path.join(show_dir, "*.json")):
                json_files.append((f, show))

    # Root batch files
    for f in glob.glob(os.path.join(BASE_DIR, "batch_results_*.json")):
        json_files.append((f, "unknown"))

    # Master dataset
    master_file = os.path.join(BASE_DIR, "master_dataset", "all_transcripts.json")
    if os.path.isfile(master_file):
        json_files.append((master_file, "master"))

    return json_files


def extract_videos_from_file(filepath, default_show):
    """Extract video records from a JSON file."""
    videos = []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and "videos" in data:
            for video in data["videos"]:
                if "show" not in video:
                    video["show"] = default_show
                videos.append(video)
        elif isinstance(data, list):
            for video in data:
                if isinstance(video, dict):
                    if "show" not in video:
                        video["show"] = default_show
                    videos.append(video)
        elif isinstance(data, dict) and "video_id" in data:
            if "show" not in data:
                data["show"] = default_show
            videos.append(data)
    except (json.JSONDecodeError, IOError) as e:
        log(f"  WARNING: Could not read {filepath}: {e}")

    return videos


def extract_dollar_amounts(text):
    """Extract dollar amounts from text."""
    amounts = []

    # Match $N,NNN patterns
    for match in DOLLAR_PATTERN.finditer(text):
        amount_str = match.group(1).replace(",", "")
        try:
            amounts.append(float(amount_str))
        except ValueError:
            pass

    # Match "N dollars" patterns
    for match in DOLLAR_WORDS_PATTERN.finditer(text):
        amount_str = match.group(1).replace(",", "")
        try:
            val = float(amount_str)
            if val not in amounts:
                amounts.append(val)
        except ValueError:
            pass

    return amounts


def classify_segment(text):
    """Classify text into negotiation segment types."""
    text_lower = text.lower()
    matches = []

    for seg_type, patterns in SEGMENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                matches.append(seg_type)
                break  # Only count each type once

    return matches


def get_sentences_from_video(video):
    """Extract sentence-level data from a video record."""
    sentences = []

    # Try analysis_fields.sentence_boundaries first
    if "analysis_fields" in video and video["analysis_fields"]:
        boundaries = video["analysis_fields"].get("sentence_boundaries", [])
        for sb in boundaries:
            sentences.append({
                "text": sb.get("text", ""),
                "start": sb.get("start", 0),
                "end": sb.get("end", 0),
                "speaker": sb.get("speaker", "unknown"),
            })

    # Fall back to raw transcript entries
    if not sentences and "transcript" in video and video["transcript"]:
        t = video["transcript"]
        raw = t if isinstance(t, list) else t.get("raw_entries", [])
        for entry in raw:
            sentences.append({
                "text": entry.get("text", ""),
                "start": entry.get("start", 0),
                "end": entry.get("end", entry.get("start", 0) + entry.get("duration", 0)),
                "speaker": "unknown",
            })

    return sentences


def analyze_video_negotiations(video):
    """Analyze a single video for negotiation segments."""
    video_id = video.get("video_id", "")
    show = video.get("show", "unknown")
    sentences = get_sentences_from_video(video)

    if not sentences:
        return []

    segments = []
    for sent in sentences:
        text = sent["text"]
        seg_types = classify_segment(text)
        dollar_amounts = extract_dollar_amounts(text)

        for seg_type in seg_types:
            segments.append({
                "video_id": video_id,
                "show": show,
                "segment_type": seg_type,
                "start_time": sent["start"],
                "end_time": sent["end"],
                "text": text.strip(),
                "dollar_amounts": dollar_amounts,
                "speaker": sent.get("speaker", "unknown"),
                "sentences": [text.strip()],
            })

    return segments


def merge_adjacent_segments(segments):
    """Merge adjacent segments of the same type within a video."""
    if not segments:
        return segments

    # Sort by video_id, segment_type, start_time
    segments.sort(key=lambda s: (s["video_id"], s["segment_type"], s["start_time"]))

    merged = []
    current = None

    for seg in segments:
        if (current is not None
                and current["video_id"] == seg["video_id"]
                and current["segment_type"] == seg["segment_type"]
                and seg["start_time"] - current["end_time"] < 10.0):
            # Merge: extend current segment
            current["end_time"] = max(current["end_time"], seg["end_time"])
            current["text"] += " " + seg["text"]
            current["dollar_amounts"] = list(
                set(current["dollar_amounts"] + seg["dollar_amounts"])
            )
            current["sentences"].extend(seg["sentences"])
        else:
            if current is not None:
                merged.append(current)
            current = dict(seg)

    if current is not None:
        merged.append(current)

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Extract negotiation segments from pawn shop transcripts."
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help=f"Directory containing result files (default: {RESULTS_DIR})"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(RESULTS_DIR, "negotiations_analysis.json"),
        help="Output file path"
    )
    parser.add_argument(
        "--merge-window",
        type=float,
        default=10.0,
        help="Time window (seconds) for merging adjacent segments (default: 10)"
    )
    args = parser.parse_args()

    log("=" * 60)
    log("NEGOTIATION ANALYSIS - Pawn Shop Transcripts")
    log("=" * 60)

    # Ensure results directory symlinks
    for show in SHOW_DIRS:
        persistent_show = os.path.join(PERSISTENT_DIR, "results", show)
        workspace_show = os.path.join(RESULTS_DIR, show)
        ensure_symlink(workspace_show, persistent_show)

    # Find all JSON files
    log("\nScanning for batch result files...")
    json_files = find_all_json_files()
    log(f"  Found {len(json_files)} file(s)")

    if not json_files:
        log("\nNo JSON files found. Run merge_results.py first or add batch result files.")
        sys.exit(0)

    # Extract all videos (deduplicate by video_id)
    log("\nExtracting videos...")
    all_videos = {}
    for filepath, show in json_files:
        videos = extract_videos_from_file(filepath, show)
        for v in videos:
            vid = v.get("video_id")
            if vid and vid not in all_videos:
                all_videos[vid] = v
    log(f"  Unique videos: {len(all_videos)}")

    # Analyze each video
    log("\nAnalyzing negotiations...")
    all_segments = []
    videos_with_negotiations = 0

    for i, (vid, video) in enumerate(all_videos.items()):
        # Skip videos without transcripts
        if not video.get("transcript"):
            continue

        segments = analyze_video_negotiations(video)
        if segments:
            videos_with_negotiations += 1
            all_segments.extend(segments)

        if (i + 1) % 50 == 0:
            log(f"  Processed {i + 1}/{len(all_videos)} videos...")

    log(f"\n  Total raw segments found: {len(all_segments)}")

    # Merge adjacent segments
    log("\nMerging adjacent segments...")
    merged_segments = merge_adjacent_segments(all_segments)
    log(f"  Merged segments: {len(merged_segments)}")

    # Compute statistics
    from collections import Counter
    type_counts = Counter(s["segment_type"] for s in merged_segments)
    show_counts = Counter(s["show"] for s in merged_segments)
    segments_with_dollars = sum(1 for s in merged_segments if s["dollar_amounts"])

    # Build output
    output = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_videos_analyzed": len(all_videos),
            "videos_with_negotiations": videos_with_negotiations,
            "total_segments": len(merged_segments),
            "segments_with_dollar_amounts": segments_with_dollars,
            "by_type": dict(type_counts),
            "by_show": dict(show_counts),
        },
        "segments": merged_segments,
    }

    # Write output
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log(f"\nWriting output to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    file_size_kb = os.path.getsize(output_path) / 1024
    log(f"  File size: {file_size_kb:.1f} KB")

    # Print summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  Videos analyzed:              {len(all_videos)}")
    log(f"  Videos with negotiations:     {videos_with_negotiations}")
    log(f"  Total negotiation segments:   {len(merged_segments)}")
    log(f"  Segments with $ amounts:      {segments_with_dollars}")
    log(f"\n  Segments by type:")
    for seg_type, count in sorted(type_counts.items()):
        log(f"    {seg_type}: {count}")
    log(f"\n  Segments by show:")
    for show, count in sorted(show_counts.items()):
        log(f"    {show}: {count}")

    # Show sample segments
    if merged_segments:
        log(f"\n  Sample segments:")
        for seg in merged_segments[:5]:
            amounts = ", ".join(f"${a:,.0f}" for a in seg["dollar_amounts"]) or "none"
            text_preview = seg["text"][:80] + ("..." if len(seg["text"]) > 80 else "")
            log(f"    [{seg['segment_type']}] {seg['video_id']}: \"{text_preview}\" (amounts: {amounts})")

    log("\n" + "=" * 60)
    log("Done!")


if __name__ == "__main__":
    main()
