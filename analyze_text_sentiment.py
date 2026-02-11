#!/usr/bin/env python3
"""
analyze_text_sentiment.py - Full text sentiment analysis with Kahneman behavioral
economics and Nisbett language pattern features.

Reads negotiations_analysis.json.
Uses NLTK VADER for sentiment per sentence.
Computes Kahneman features (loss/gain frames, anchor shifts, expert shock, trajectory).
Detects Nisbett analytic vs holistic markers.
Classifies roles per sentence.
Outputs:
  analysis/text_sentiment_per_sentence.csv
  analysis/text_sentiment_per_negotiation.csv
  analysis/text_analysis_report.txt

Outputs are stored via symlinks to the persistent path:
  /root/.claude/projects/-workspaces-youtube-transcript-batch/data/
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

# NLTK VADER
import nltk
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure vader_lexicon is downloaded
try:
    sid_test = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

BASE_DIR = "/workspaces/youtube-transcript-batch"
PERSISTENT_DIR = "/root/.claude/projects/-workspaces-youtube-transcript-batch/data"
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ---------- Nisbett Analytic Markers (Western/individualistic) ----------
NISBETT_ANALYTIC_PATTERNS = {
    "specific_numbers": re.compile(
        r'\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\b\d+\s*(?:percent|%|years?|dollars?|bucks?)\b',
        re.IGNORECASE,
    ),
    "conditions": re.compile(
        r'\b(?:if|unless|provided\s+that|on\s+condition|assuming|given\s+that)\b',
        re.IGNORECASE,
    ),
    "categories": re.compile(
        r'\b(?:type|kind|grade|class|category|model|series|edition|version|variant)\b',
        re.IGNORECASE,
    ),
    "rules": re.compile(
        r'\b(?:always|never|every\s+time|without\s+exception|guaranteed|definitely|certainly)\b',
        re.IGNORECASE,
    ),
    "comparisons": re.compile(
        r'\b(?:better\s+than|more\s+than|less\s+than|worse\s+than|higher\s+than|lower\s+than|compared\s+to|versus|rather\s+than)\b',
        re.IGNORECASE,
    ),
}

# ---------- Nisbett Holistic Markers (Eastern/collectivistic) ----------
NISBETT_HOLISTIC_PATTERNS = {
    "family_story": re.compile(
        r'\b(?:family|grandfather|grandmother|father|mother|dad|mom|grandpa|grandma|son|daughter|brother|sister|uncle|aunt|generation|inherited|passed\s+down|story|told\s+me)\b',
        re.IGNORECASE,
    ),
    "feelings": re.compile(
        r'\b(?:feel|feeling|feels|felt|sense|sensing|gut|instinct|intuition|heart|emotion|passionate|love|hate)\b',
        re.IGNORECASE,
    ),
    "relationships": re.compile(
        r'\b(?:trust|bond|friendship|loyal|loyalty|relationship|connect|connection|together|partner|community|respect)\b',
        re.IGNORECASE,
    ),
    "aesthetic": re.compile(
        r'\b(?:beautiful|gorgeous|amazing|incredible|stunning|wonderful|magnificent|fantastic|awesome|cool|neat|special|unique|rare|one\s+of\s+a\s+kind)\b',
        re.IGNORECASE,
    ),
    "context": re.compile(
        r'\b(?:history|historical|memories|memory|remember|reminds|nostalgia|era|period|time\s+period|back\s+in\s+the\s+day|vintage|antique|classic|heritage|tradition)\b',
        re.IGNORECASE,
    ),
}

# ---------- Role Classification Patterns ----------
ROLE_PATTERNS = {
    "asking": re.compile(
        r'\b(?:how\s+much|looking\s+for|want\s+for|asking\s+for|i\'?d\s+like|hoping\s+to\s+get|what\s+do\s+you\s+want|price)\b',
        re.IGNORECASE,
    ),
    "offering": re.compile(
        r'\b(?:i\s+can\s+do|i\'?ll\s+give\s+you|best\s+i\s+can\s+do|i\s+can\s+offer|i\'?d\s+go|i\'?ll\s+go|my\s+offer|willing\s+to\s+pay)\b',
        re.IGNORECASE,
    ),
    "countering": re.compile(
        r'\b(?:how\s+about|what\s+about|meet\s+in\s+the\s+middle|split\s+the\s+difference|come\s+up|come\s+down|counter|go\s+a\s+little)\b',
        re.IGNORECASE,
    ),
    "appraising": re.compile(
        r'\b(?:expert|appraiser|worth\s+about|value\s+at|retail\s+for|auction|apprais|estimate|market\s+value|insurance\s+value)\b',
        re.IGNORECASE,
    ),
    "closing": re.compile(
        r'\b(?:deal|you\s+got\s+a\s+deal|sold|i\'?ll\s+take\s+it|shake\s+on\s+it|done|agreed|pleasure\s+doing\s+business)\b',
        re.IGNORECASE,
    ),
    "rejecting": re.compile(
        r'\b(?:no\s+way|can\'?t\s+do|too\s+low|too\s+high|not\s+enough|no\s+deal|pass|walk\s+away|forget\s+it|sorry|can\'?t\s+go)\b',
        re.IGNORECASE,
    ),
}

# Dollar extraction
DOLLAR_PATTERN = re.compile(r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)')
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


def extract_dollar_amounts(text):
    """Extract dollar amounts from text."""
    amounts = []
    for match in DOLLAR_PATTERN.finditer(text):
        try:
            amounts.append(float(match.group(1).replace(",", "")))
        except ValueError:
            pass
    for match in DOLLAR_WORDS_PATTERN.finditer(text):
        try:
            val = float(match.group(1).replace(",", ""))
            if val not in amounts:
                amounts.append(val)
        except ValueError:
            pass
    return amounts


def classify_role(text):
    """Classify sentence into negotiation role."""
    roles = []
    for role, pattern in ROLE_PATTERNS.items():
        if pattern.search(text):
            roles.append(role)
    return roles if roles else ["neutral"]


def compute_nisbett_analytic(text):
    """Count Nisbett analytic marker matches."""
    scores = {}
    total = 0
    for marker, pattern in NISBETT_ANALYTIC_PATTERNS.items():
        matches = pattern.findall(text)
        scores[f"analytic_{marker}"] = len(matches)
        total += len(matches)
    scores["analytic_total"] = total
    return scores


def compute_nisbett_holistic(text):
    """Count Nisbett holistic marker matches."""
    scores = {}
    total = 0
    for marker, pattern in NISBETT_HOLISTIC_PATTERNS.items():
        matches = pattern.findall(text)
        scores[f"holistic_{marker}"] = len(matches)
        total += len(matches)
    scores["holistic_total"] = total
    return scores


def split_into_sentences(text):
    """Simple sentence splitter."""
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def analyze_segments(segments):
    """Analyze all negotiation segments, returning sentence-level and segment-level data."""
    sid = SentimentIntensityAnalyzer()

    sentence_rows = []
    segment_rows = []

    # Group segments by video_id for Kahneman features
    video_segments = defaultdict(list)
    for seg in segments:
        video_segments[seg["video_id"]].append(seg)

    total_segments = len(segments)

    for seg_idx, seg in enumerate(segments):
        video_id = seg["video_id"]
        show = seg.get("show", "unknown")
        segment_type = seg["segment_type"]
        start_time = seg.get("start_time", 0)
        end_time = seg.get("end_time", 0)
        full_text = seg.get("text", "")
        dollar_amounts = seg.get("dollar_amounts", [])
        speaker = seg.get("speaker", "unknown")

        # Split into sentences
        sentences = split_into_sentences(full_text)
        if not sentences:
            sentences = [full_text] if full_text else []

        seg_sentiments = []
        seg_analytic_total = 0
        seg_holistic_total = 0

        for sent_idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # VADER sentiment
            vader = sid.polarity_scores(sentence)
            compound = vader["compound"]
            seg_sentiments.append(compound)

            # Nisbett markers
            analytic = compute_nisbett_analytic(sentence)
            holistic = compute_nisbett_holistic(sentence)
            seg_analytic_total += analytic["analytic_total"]
            seg_holistic_total += holistic["holistic_total"]

            # Role classification
            roles = classify_role(sentence)
            primary_role = roles[0]

            # Dollar amounts in this sentence
            sent_dollars = extract_dollar_amounts(sentence)

            row = {
                "video_id": video_id,
                "show": show,
                "segment_type": segment_type,
                "sentence_index": sent_idx,
                "start_time": start_time,
                "end_time": end_time,
                "speaker": speaker,
                "sentence": sentence,
                "vader_compound": round(compound, 4),
                "vader_positive": round(vader["pos"], 4),
                "vader_negative": round(vader["neg"], 4),
                "vader_neutral": round(vader["neu"], 4),
                "primary_role": primary_role,
                "all_roles": "|".join(roles),
                "dollar_amounts": "|".join(str(d) for d in sent_dollars),
            }

            # Add Nisbett analytic markers
            for key, val in analytic.items():
                row[key] = val

            # Add Nisbett holistic markers
            for key, val in holistic.items():
                row[key] = val

            sentence_rows.append(row)

        # Segment-level aggregates
        if seg_sentiments:
            mean_sentiment = float(np.mean(seg_sentiments))
            std_sentiment = float(np.std(seg_sentiments)) if len(seg_sentiments) > 1 else 0.0
            min_sentiment = float(np.min(seg_sentiments))
            max_sentiment = float(np.max(seg_sentiments))

            # Sentiment trajectory (slope via linear regression)
            if len(seg_sentiments) >= 2:
                x = np.arange(len(seg_sentiments))
                slope = float(np.polyfit(x, seg_sentiments, 1)[0])
            else:
                slope = 0.0
        else:
            mean_sentiment = std_sentiment = min_sentiment = max_sentiment = slope = 0.0

        seg_row = {
            "video_id": video_id,
            "show": show,
            "segment_type": segment_type,
            "start_time": start_time,
            "end_time": end_time,
            "speaker": speaker,
            "sentence_count": len(sentences),
            "mean_sentiment": round(mean_sentiment, 4),
            "std_sentiment": round(std_sentiment, 4),
            "min_sentiment": round(min_sentiment, 4),
            "max_sentiment": round(max_sentiment, 4),
            "sentiment_trajectory": round(slope, 4),
            "analytic_total": seg_analytic_total,
            "holistic_total": seg_holistic_total,
            "dollar_amounts": "|".join(str(d) for d in dollar_amounts),
            "text_preview": full_text[:200],
        }

        segment_rows.append(seg_row)

        if (seg_idx + 1) % 100 == 0:
            log(f"  Processed {seg_idx + 1}/{total_segments} segments...")

    return sentence_rows, segment_rows


def compute_kahneman_features(segments, segment_rows):
    """Compute Kahneman behavioral economics features across videos."""
    # Group segments by video_id
    video_segs = defaultdict(list)
    for seg in segments:
        video_segs[seg["video_id"]].append(seg)

    # Group segment rows by video_id
    video_seg_rows = defaultdict(list)
    for row in segment_rows:
        video_seg_rows[row["video_id"]].append(row)

    kahneman_features = {}

    for video_id, v_segs in video_segs.items():
        features = {
            "video_id": video_id,
            "loss_frame": False,
            "gain_frame": False,
            "anchor_sentiment_shift": None,
            "expert_sentiment_shock": None,
            "sentiment_trajectory": None,
        }

        # Gather dollar amounts by segment type
        type_amounts = defaultdict(list)
        type_sentiments = {}
        for seg in v_segs:
            type_amounts[seg["segment_type"]].extend(seg.get("dollar_amounts", []))

        for row in video_seg_rows.get(video_id, []):
            st = row["segment_type"]
            if st not in type_sentiments:
                type_sentiments[st] = row["mean_sentiment"]

        # Loss/Gain frame detection
        initial_amounts = type_amounts.get("initial_ask", [])
        expert_amounts = type_amounts.get("expert_appraisal", [])

        if initial_amounts and expert_amounts:
            avg_initial = np.mean(initial_amounts)
            avg_expert = np.mean(expert_amounts)
            # Loss frame: expert value < initial ask (seller loses relative to expectation)
            features["loss_frame"] = bool(avg_expert < avg_initial)
            # Gain frame: expert value > initial ask (seller gains relative to expectation)
            features["gain_frame"] = bool(avg_expert > avg_initial)

        # Anchor sentiment shift: sentiment change from initial_ask to first_offer
        initial_sent = type_sentiments.get("initial_ask")
        offer_sent = type_sentiments.get("first_offer")
        if initial_sent is not None and offer_sent is not None:
            features["anchor_sentiment_shift"] = round(offer_sent - initial_sent, 4)

        # Expert sentiment shock: sentiment change around expert appraisal
        expert_sent = type_sentiments.get("expert_appraisal")
        if expert_sent is not None and initial_sent is not None:
            features["expert_sentiment_shock"] = round(expert_sent - initial_sent, 4)

        # Overall sentiment trajectory across all segments
        v_rows = video_seg_rows.get(video_id, [])
        if len(v_rows) >= 2:
            sentiments_ordered = [r["mean_sentiment"] for r in sorted(v_rows, key=lambda r: r["start_time"])]
            x = np.arange(len(sentiments_ordered))
            slope = float(np.polyfit(x, sentiments_ordered, 1)[0])
            features["sentiment_trajectory"] = round(slope, 4)

        kahneman_features[video_id] = features

    return kahneman_features


def write_csv(filepath, rows, fieldnames=None):
    """Write rows as CSV."""
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


def write_report(filepath, sentence_rows, segment_rows, kahneman_features, segments):
    """Write text analysis report."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("TEXT SENTIMENT & BEHAVIORAL ECONOMICS ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")

        # Overall stats
        f.write("OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total sentences analyzed:     {len(sentence_rows)}\n")
        f.write(f"Total segments analyzed:       {len(segment_rows)}\n")
        f.write(f"Total videos:                  {len(kahneman_features)}\n\n")

        # Sentiment by segment type
        f.write("SENTIMENT BY SEGMENT TYPE\n")
        f.write("-" * 40 + "\n")
        type_sentiments = defaultdict(list)
        for row in segment_rows:
            type_sentiments[row["segment_type"]].append(row["mean_sentiment"])

        for seg_type in sorted(type_sentiments.keys()):
            vals = type_sentiments[seg_type]
            f.write(f"  {seg_type}:\n")
            f.write(f"    Count:    {len(vals)}\n")
            f.write(f"    Mean:     {np.mean(vals):.4f}\n")
            f.write(f"    Std:      {np.std(vals):.4f}\n")
            f.write(f"    Min:      {np.min(vals):.4f}\n")
            f.write(f"    Max:      {np.max(vals):.4f}\n\n")

        # Kahneman features summary
        f.write("KAHNEMAN BEHAVIORAL ECONOMICS FEATURES\n")
        f.write("-" * 40 + "\n")
        loss_count = sum(1 for v in kahneman_features.values() if v["loss_frame"])
        gain_count = sum(1 for v in kahneman_features.values() if v["gain_frame"])
        f.write(f"  Loss frame videos:           {loss_count}\n")
        f.write(f"  Gain frame videos:           {gain_count}\n")

        anchor_shifts = [v["anchor_sentiment_shift"] for v in kahneman_features.values()
                         if v["anchor_sentiment_shift"] is not None]
        if anchor_shifts:
            f.write(f"  Anchor sentiment shifts:     {len(anchor_shifts)}\n")
            f.write(f"    Mean shift:                {np.mean(anchor_shifts):.4f}\n")

        expert_shocks = [v["expert_sentiment_shock"] for v in kahneman_features.values()
                         if v["expert_sentiment_shock"] is not None]
        if expert_shocks:
            f.write(f"  Expert sentiment shocks:     {len(expert_shocks)}\n")
            f.write(f"    Mean shock:                {np.mean(expert_shocks):.4f}\n")

        trajectories = [v["sentiment_trajectory"] for v in kahneman_features.values()
                        if v["sentiment_trajectory"] is not None]
        if trajectories:
            f.write(f"  Sentiment trajectories:      {len(trajectories)}\n")
            f.write(f"    Mean trajectory:           {np.mean(trajectories):.4f}\n")
        f.write("\n")

        # Nisbett language patterns
        f.write("NISBETT LANGUAGE PATTERNS\n")
        f.write("-" * 40 + "\n")
        total_analytic = sum(row.get("analytic_total", 0) for row in sentence_rows)
        total_holistic = sum(row.get("holistic_total", 0) for row in sentence_rows)
        f.write(f"  Total analytic markers:      {total_analytic}\n")
        f.write(f"  Total holistic markers:      {total_holistic}\n")

        if total_analytic + total_holistic > 0:
            ratio = total_analytic / (total_analytic + total_holistic)
            f.write(f"  Analytic ratio:              {ratio:.4f}\n")
        f.write("\n")

        # Analytic marker breakdown
        f.write("  Analytic marker breakdown:\n")
        for marker in NISBETT_ANALYTIC_PATTERNS:
            col = f"analytic_{marker}"
            total = sum(row.get(col, 0) for row in sentence_rows)
            f.write(f"    {marker}: {total}\n")

        f.write("\n  Holistic marker breakdown:\n")
        for marker in NISBETT_HOLISTIC_PATTERNS:
            col = f"holistic_{marker}"
            total = sum(row.get(col, 0) for row in sentence_rows)
            f.write(f"    {marker}: {total}\n")
        f.write("\n")

        # Role distribution
        f.write("ROLE CLASSIFICATION\n")
        f.write("-" * 40 + "\n")
        role_counts = defaultdict(int)
        for row in sentence_rows:
            role_counts[row["primary_role"]] += 1
        for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {role}: {count}\n")
        f.write("\n")

        # Role sentiment
        f.write("SENTIMENT BY ROLE\n")
        f.write("-" * 40 + "\n")
        role_sentiments = defaultdict(list)
        for row in sentence_rows:
            role_sentiments[row["primary_role"]].append(row["vader_compound"])
        for role in sorted(role_sentiments.keys()):
            vals = role_sentiments[role]
            f.write(f"  {role}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, n={len(vals)}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")

    log(f"  Written: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Text sentiment analysis with Kahneman & Nisbett features."
    )
    parser.add_argument(
        "--input",
        default=os.path.join(RESULTS_DIR, "negotiations_analysis.json"),
        help="Input negotiations analysis JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default=ANALYSIS_DIR,
        help=f"Output directory (default: {ANALYSIS_DIR})"
    )
    args = parser.parse_args()

    log("=" * 60)
    log("TEXT SENTIMENT ANALYSIS")
    log("Kahneman Behavioral Economics + Nisbett Language Patterns")
    log("=" * 60)

    # Ensure output directory symlink
    persistent_analysis = os.path.join(PERSISTENT_DIR, "analysis")
    ensure_symlink(args.output_dir, persistent_analysis)

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

    if not segments:
        log("No segments to analyze.")
        sys.exit(0)

    # Initialize VADER
    log("\nInitializing VADER sentiment analyzer...")
    # (initialized inside analyze_segments)

    # Analyze segments
    log("\nAnalyzing text sentiment...")
    sentence_rows, segment_rows = analyze_segments(segments)
    log(f"  Sentence-level rows: {len(sentence_rows)}")
    log(f"  Segment-level rows:  {len(segment_rows)}")

    # Compute Kahneman features
    log("\nComputing Kahneman behavioral economics features...")
    kahneman_features = compute_kahneman_features(segments, segment_rows)
    log(f"  Videos analyzed: {len(kahneman_features)}")

    # Write outputs
    log("\nWriting outputs...")

    sentence_csv = os.path.join(args.output_dir, "text_sentiment_per_sentence.csv")
    write_csv(sentence_csv, sentence_rows)

    segment_csv = os.path.join(args.output_dir, "text_sentiment_per_negotiation.csv")
    write_csv(segment_csv, segment_rows)

    report_path = os.path.join(args.output_dir, "text_analysis_report.txt")
    write_report(report_path, sentence_rows, segment_rows, kahneman_features, segments)

    # Print summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)

    if sentence_rows:
        all_compounds = [r["vader_compound"] for r in sentence_rows]
        log(f"  Overall mean sentiment:      {np.mean(all_compounds):.4f}")
        log(f"  Overall std sentiment:       {np.std(all_compounds):.4f}")

    total_analytic = sum(r.get("analytic_total", 0) for r in sentence_rows)
    total_holistic = sum(r.get("holistic_total", 0) for r in sentence_rows)
    log(f"  Total analytic markers:      {total_analytic}")
    log(f"  Total holistic markers:      {total_holistic}")

    loss_count = sum(1 for v in kahneman_features.values() if v["loss_frame"])
    gain_count = sum(1 for v in kahneman_features.values() if v["gain_frame"])
    log(f"  Loss frame negotiations:     {loss_count}")
    log(f"  Gain frame negotiations:     {gain_count}")

    log("\n" + "=" * 60)
    log("Done!")


if __name__ == "__main__":
    main()
