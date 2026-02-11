#!/usr/bin/env python3
"""
analyze_sentiment.py - Facial sentiment analysis using DeepFace on extracted key frames.

Reads key_frames/manifest.json.
Uses DeepFace.analyze() with enforce_detection=False for each frame.
Extracts: dominant_emotion, emotion scores (angry, disgust, fear, happy, sad, surprise, neutral).
Uses ThreadPoolExecutor with configurable workers.
Outputs:
  analysis/sentiment_per_frame.csv
  analysis/sentiment_per_negotiation.csv (aggregated by video/segment)
Incremental: skips already-analyzed frames.

Outputs are stored via symlinks to the persistent path:
  /root/.claude/projects/-workspaces-youtube-transcript-batch/data/
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

import numpy as np

BASE_DIR = "/workspaces/youtube-transcript-batch"
PERSISTENT_DIR = "/root/.claude/projects/-workspaces-youtube-transcript-batch/data"
KEY_FRAMES_DIR = os.path.join(BASE_DIR, "key_frames")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")

EMOTION_KEYS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


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


def load_existing_results(csv_path):
    """Load already-analyzed frame paths from existing CSV."""
    analyzed = set()
    if os.path.isfile(csv_path):
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    analyzed.add(row.get("frame_file", ""))
        except (IOError, csv.Error):
            pass
    return analyzed


def analyze_frame(frame_info, key_frames_dir):
    """Analyze a single frame using DeepFace."""
    # Import here to avoid slow import at module level
    from deepface import DeepFace

    frame_file = frame_info.get("file", "")
    frame_path = os.path.join(key_frames_dir, frame_file)

    if not os.path.isfile(frame_path):
        return None, f"Frame file not found: {frame_path}"

    try:
        results = DeepFace.analyze(
            img_path=frame_path,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )

        # DeepFace returns a list of results (one per face detected)
        if isinstance(results, list):
            if not results:
                return None, f"No faces detected in {frame_file}"
            result = results[0]  # Take first/most prominent face
        else:
            result = results

        emotion_scores = result.get("emotion", {})
        dominant_emotion = result.get("dominant_emotion", "unknown")

        row = {
            "frame_file": frame_file,
            "video_id": frame_info.get("video_id", ""),
            "timestamp": frame_info.get("timestamp", 0),
            "moment_type": frame_info.get("moment_type", ""),
            "amount": frame_info.get("amount", ""),
            "speaker": frame_info.get("speaker", ""),
            "dominant_emotion": dominant_emotion,
        }

        # Add individual emotion scores
        for emo in EMOTION_KEYS:
            row[f"emotion_{emo}"] = round(emotion_scores.get(emo, 0.0), 4)

        # Compute face region info if available
        region = result.get("region", {})
        if region:
            row["face_x"] = region.get("x", 0)
            row["face_y"] = region.get("y", 0)
            row["face_w"] = region.get("w", 0)
            row["face_h"] = region.get("h", 0)
        else:
            row["face_x"] = row["face_y"] = row["face_w"] = row["face_h"] = 0

        # Confidence score
        row["face_confidence"] = round(result.get("face_confidence", 0.0), 4)

        return row, None

    except Exception as e:
        return None, f"DeepFace error for {frame_file}: {e}"


def aggregate_by_negotiation(frame_rows):
    """Aggregate frame-level results by video_id and moment_type."""
    groups = defaultdict(list)

    for row in frame_rows:
        key = (row["video_id"], row["moment_type"])
        groups[key].append(row)

    agg_rows = []
    for (video_id, moment_type), rows in groups.items():
        agg = {
            "video_id": video_id,
            "moment_type": moment_type,
            "frame_count": len(rows),
        }

        # Aggregate emotion scores
        for emo in EMOTION_KEYS:
            col = f"emotion_{emo}"
            vals = [r[col] for r in rows if col in r]
            if vals:
                agg[f"mean_{emo}"] = round(float(np.mean(vals)), 4)
                agg[f"std_{emo}"] = round(float(np.std(vals)), 4) if len(vals) > 1 else 0.0
            else:
                agg[f"mean_{emo}"] = 0.0
                agg[f"std_{emo}"] = 0.0

        # Dominant emotion (mode)
        emotions = [r["dominant_emotion"] for r in rows]
        if emotions:
            from collections import Counter
            emotion_counts = Counter(emotions)
            agg["dominant_emotion"] = emotion_counts.most_common(1)[0][0]
            agg["emotion_agreement"] = round(
                emotion_counts.most_common(1)[0][1] / len(emotions), 4
            )
        else:
            agg["dominant_emotion"] = "unknown"
            agg["emotion_agreement"] = 0.0

        # Average timestamp
        timestamps = [r["timestamp"] for r in rows]
        agg["mean_timestamp"] = round(float(np.mean(timestamps)), 2) if timestamps else 0.0

        # Dollar amounts
        amounts = [r["amount"] for r in rows if r.get("amount")]
        agg["amounts"] = "|".join(str(a) for a in set(amounts)) if amounts else ""

        agg_rows.append(agg)

    return agg_rows


def write_csv(filepath, rows, fieldnames=None):
    """Write rows to CSV."""
    if not rows:
        log(f"  No data to write for {filepath}")
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    log(f"  Written: {filepath} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Facial sentiment analysis using DeepFace on extracted key frames."
    )
    parser.add_argument(
        "--manifest",
        default=os.path.join(KEY_FRAMES_DIR, "manifest.json"),
        help="Path to key_frames manifest.json"
    )
    parser.add_argument(
        "--key-frames-dir",
        default=KEY_FRAMES_DIR,
        help=f"Directory containing key frames (default: {KEY_FRAMES_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        default=ANALYSIS_DIR,
        help=f"Output directory (default: {ANALYSIS_DIR})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of frames to analyze (0 = no limit)"
    )
    args = parser.parse_args()

    log("=" * 60)
    log("FACIAL SENTIMENT ANALYSIS - DeepFace")
    log("=" * 60)

    # Ensure output directory symlink
    persistent_analysis = os.path.join(PERSISTENT_DIR, "analysis")
    ensure_symlink(args.output_dir, persistent_analysis)

    # Read manifest
    log(f"\nReading manifest: {args.manifest}")
    if not os.path.isfile(args.manifest):
        log(f"ERROR: Manifest not found: {args.manifest}")
        log("Run extract_key_frames.py first.")
        sys.exit(1)

    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    frames = manifest.get("frames", [])
    log(f"  Total frames in manifest: {len(frames)}")

    if not frames:
        log("No frames to analyze.")
        sys.exit(0)

    # Load existing results for incremental processing
    frame_csv_path = os.path.join(args.output_dir, "sentiment_per_frame.csv")
    already_analyzed = load_existing_results(frame_csv_path)
    log(f"  Already analyzed: {len(already_analyzed)} frame(s)")

    # Filter to unanalyzed frames
    to_analyze = [f for f in frames if f.get("file", "") not in already_analyzed]
    log(f"  Frames to analyze: {len(to_analyze)}")

    if args.limit > 0:
        to_analyze = to_analyze[:args.limit]
        log(f"  Limited to: {len(to_analyze)} frame(s)")

    if not to_analyze:
        log("\nAll frames already analyzed. Regenerating aggregations...")
        # Still need to load existing data for aggregation
        existing_rows = []
        if os.path.isfile(frame_csv_path):
            with open(frame_csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    for key in row:
                        if key.startswith("emotion_") or key in ("timestamp", "face_confidence"):
                            try:
                                row[key] = float(row[key])
                            except (ValueError, TypeError):
                                pass
                    existing_rows.append(row)

        if existing_rows:
            agg_rows = aggregate_by_negotiation(existing_rows)
            agg_csv_path = os.path.join(args.output_dir, "sentiment_per_negotiation.csv")
            write_csv(agg_csv_path, agg_rows)

        log("Done!")
        return

    # Pre-import DeepFace to download models
    log("\nInitializing DeepFace...")
    try:
        from deepface import DeepFace
        log("  DeepFace ready")
    except ImportError:
        log("ERROR: DeepFace not installed. Run: pip install deepface")
        sys.exit(1)

    # Process frames
    log(f"\nAnalyzing {len(to_analyze)} frame(s) with {args.workers} worker(s)...")
    frame_rows = []
    errors = []
    results_lock = Lock()

    # Load existing rows if any
    existing_rows = []
    if os.path.isfile(frame_csv_path):
        with open(frame_csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in row:
                    if key.startswith("emotion_") or key in ("timestamp", "face_confidence"):
                        try:
                            row[key] = float(row[key])
                        except (ValueError, TypeError):
                            pass
                existing_rows.append(row)

    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_frame = {}

        for frame_info in to_analyze:
            future = executor.submit(analyze_frame, frame_info, args.key_frames_dir)
            future_to_frame[future] = frame_info

        for future in as_completed(future_to_frame):
            frame_info = future_to_frame[future]

            try:
                row, error = future.result()
                if row:
                    with results_lock:
                        frame_rows.append(row)
                if error:
                    errors.append(error)
            except Exception as e:
                errors.append(f"Exception for {frame_info.get('file', '?')}: {e}")

            completed += 1
            if completed % 50 == 0:
                log(f"  Processed {completed}/{len(to_analyze)} frames "
                    f"({len(frame_rows)} successful, {len(errors)} errors)")

    log(f"\n  Completed: {completed}")
    log(f"  Successful: {len(frame_rows)}")
    log(f"  Errors: {len(errors)}")

    if errors[:5]:
        log("\n  Sample errors:")
        for err in errors[:5]:
            log(f"    {err}")

    # Combine with existing rows
    all_rows = existing_rows + frame_rows
    log(f"\n  Total frame rows (existing + new): {len(all_rows)}")

    # Write frame-level CSV
    log("\nWriting outputs...")
    write_csv(frame_csv_path, all_rows)

    # Aggregate by video/segment
    agg_rows = aggregate_by_negotiation(all_rows)
    agg_csv_path = os.path.join(args.output_dir, "sentiment_per_negotiation.csv")
    write_csv(agg_csv_path, agg_rows)

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  Total frames analyzed:      {len(all_rows)}")
    log(f"  New frames this run:        {len(frame_rows)}")
    log(f"  Aggregated negotiations:    {len(agg_rows)}")

    if all_rows:
        # Emotion distribution
        from collections import Counter
        emotion_dist = Counter(r.get("dominant_emotion", "unknown") for r in all_rows)
        log(f"\n  Dominant emotion distribution:")
        for emo, count in emotion_dist.most_common():
            pct = count / len(all_rows) * 100
            log(f"    {emo}: {count} ({pct:.1f}%)")

    log("\n" + "=" * 60)
    log("Done!")


if __name__ == "__main__":
    main()
