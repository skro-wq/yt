#!/usr/bin/env python3
"""
analyze_multimodal.py - Combine text sentiment and facial sentiment for multimodal analysis.

Reads text_sentiment_per_sentence.csv and sentiment_per_frame.csv.
Matches text and facial data by video_id and timestamp (within 5-second window).
Calculates congruence score: alignment between text sentiment (VADER) and facial emotion.
Kahneman multimodal features: emotional alignment during loss/gain frames.
Generates reports:
  analysis/multimodal_analysis.csv
  analysis/behavioral_economics_report.txt
  analysis/language_patterns_report.txt
Creates matplotlib visualizations:
  analysis/emotion_progression.png
  analysis/text_vs_face_sentiment.png
  analysis/loss_aversion_multimodal.png

Outputs are stored via symlinks to the persistent path:
  /root/.claude/projects/-workspaces-youtube-transcript-batch/data/
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "/workspaces/youtube-transcript-batch"
PERSISTENT_DIR = "/root/.claude/projects/-workspaces-youtube-transcript-batch/data"
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TIMESTAMP_WINDOW = 5.0  # seconds for matching text to face data

# Emotion-to-valence mapping for congruence calculation
EMOTION_VALENCE = {
    "happy": 1.0,
    "surprise": 0.3,
    "neutral": 0.0,
    "sad": -0.5,
    "fear": -0.6,
    "disgust": -0.7,
    "angry": -0.8,
}

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


def load_csv(filepath):
    """Load CSV file into list of dicts with numeric conversion."""
    rows = []
    if not os.path.isfile(filepath):
        return rows

    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in list(row.keys()):
                if row[key] == "":
                    continue
                try:
                    if "." in str(row[key]):
                        row[key] = float(row[key])
                    elif str(row[key]).lstrip("-").isdigit():
                        row[key] = int(row[key])
                except (ValueError, TypeError):
                    pass
            rows.append(row)

    return rows


def match_text_and_face(text_rows, face_rows, window=TIMESTAMP_WINDOW):
    """Match text sentiment and facial sentiment by video_id and timestamp."""
    # Index face rows by video_id
    face_by_video = defaultdict(list)
    for row in face_rows:
        vid = row.get("video_id", "")
        if vid:
            face_by_video[vid].append(row)

    # Sort face rows by timestamp within each video
    for vid in face_by_video:
        face_by_video[vid].sort(key=lambda r: float(r.get("timestamp", 0)))

    matched = []
    unmatched_text = 0
    unmatched_face = 0

    for text_row in text_rows:
        vid = text_row.get("video_id", "")
        text_time = float(text_row.get("start_time", 0))

        # Find closest face frame within window
        best_face = None
        best_dist = float("inf")

        for face_row in face_by_video.get(vid, []):
            face_time = float(face_row.get("timestamp", 0))
            dist = abs(face_time - text_time)
            if dist <= window and dist < best_dist:
                best_face = face_row
                best_dist = dist

        if best_face is not None:
            # Compute congruence
            vader_compound = float(text_row.get("vader_compound", 0))
            dominant_emotion = best_face.get("dominant_emotion", "neutral")
            face_valence = EMOTION_VALENCE.get(dominant_emotion, 0.0)

            # Congruence: 1.0 if text and face sentiment agree, 0.0 if opposite
            # Normalized difference: closer to 0 = more congruent
            raw_diff = abs(vader_compound - face_valence)
            congruence = 1.0 - min(raw_diff / 2.0, 1.0)

            # Build matched row
            mrow = {
                "video_id": vid,
                "timestamp": text_time,
                "face_timestamp": float(best_face.get("timestamp", 0)),
                "time_diff": round(best_dist, 2),
                "segment_type": text_row.get("segment_type", ""),
                "show": text_row.get("show", ""),
                "speaker": text_row.get("speaker", ""),
                "sentence": text_row.get("sentence", ""),
                "primary_role": text_row.get("primary_role", ""),
                # Text sentiment
                "vader_compound": round(vader_compound, 4),
                "vader_positive": float(text_row.get("vader_positive", 0)),
                "vader_negative": float(text_row.get("vader_negative", 0)),
                # Face sentiment
                "dominant_emotion": dominant_emotion,
                "face_valence": round(face_valence, 4),
            }

            # Add individual emotion scores
            for emo in EMOTION_KEYS:
                col = f"emotion_{emo}"
                mrow[col] = float(best_face.get(col, 0))

            # Congruence metrics
            mrow["congruence_score"] = round(congruence, 4)
            mrow["text_face_diff"] = round(vader_compound - face_valence, 4)

            # Nisbett markers from text
            mrow["analytic_total"] = int(text_row.get("analytic_total", 0))
            mrow["holistic_total"] = int(text_row.get("holistic_total", 0))

            # Dollar amounts
            mrow["dollar_amounts"] = text_row.get("dollar_amounts", "")

            matched.append(mrow)
        else:
            unmatched_text += 1

    log(f"  Matched pairs:      {len(matched)}")
    log(f"  Unmatched text:     {unmatched_text}")

    return matched


def compute_kahneman_multimodal(matched_rows, negotiations_path):
    """Compute Kahneman multimodal features: emotional alignment during loss/gain frames."""
    features = {}

    # Load negotiations for dollar amount context
    negotiations = {}
    if os.path.isfile(negotiations_path):
        with open(negotiations_path, "r") as f:
            data = json.load(f)
        for seg in data.get("segments", []):
            vid = seg.get("video_id", "")
            if vid not in negotiations:
                negotiations[vid] = defaultdict(list)
            negotiations[vid][seg["segment_type"]].append(seg)

    # Group matched rows by video
    video_rows = defaultdict(list)
    for row in matched_rows:
        video_rows[row["video_id"]].append(row)

    for vid, rows in video_rows.items():
        vid_neg = negotiations.get(vid, {})

        # Determine loss/gain frame from dollar amounts
        initial_amounts = []
        expert_amounts = []
        for seg in vid_neg.get("initial_ask", []):
            initial_amounts.extend(seg.get("dollar_amounts", []))
        for seg in vid_neg.get("expert_appraisal", []):
            expert_amounts.extend(seg.get("dollar_amounts", []))

        loss_frame = False
        gain_frame = False
        if initial_amounts and expert_amounts:
            avg_initial = np.mean(initial_amounts)
            avg_expert = np.mean(expert_amounts)
            loss_frame = avg_expert < avg_initial
            gain_frame = avg_expert > avg_initial

        # Compute emotional alignment metrics
        seg_congruences = defaultdict(list)
        seg_vader = defaultdict(list)
        seg_face_valence = defaultdict(list)

        for row in rows:
            st = row.get("segment_type", "")
            seg_congruences[st].append(row["congruence_score"])
            seg_vader[st].append(row["vader_compound"])
            seg_face_valence[st].append(row["face_valence"])

        feat = {
            "video_id": vid,
            "loss_frame": loss_frame,
            "gain_frame": gain_frame,
            "total_matched": len(rows),
            "mean_congruence": round(float(np.mean([r["congruence_score"] for r in rows])), 4),
        }

        # Congruence during different frames
        for st in ["initial_ask", "first_offer", "counter", "expert_appraisal", "deal"]:
            if st in seg_congruences:
                feat[f"{st}_congruence"] = round(float(np.mean(seg_congruences[st])), 4)
                feat[f"{st}_vader_mean"] = round(float(np.mean(seg_vader[st])), 4)
                feat[f"{st}_face_valence_mean"] = round(float(np.mean(seg_face_valence[st])), 4)
            else:
                feat[f"{st}_congruence"] = None
                feat[f"{st}_vader_mean"] = None
                feat[f"{st}_face_valence_mean"] = None

        features[vid] = feat

    return features


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


def write_behavioral_economics_report(filepath, kahneman_features, matched_rows):
    """Write Kahneman behavioral economics multimodal report."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("BEHAVIORAL ECONOMICS MULTIMODAL REPORT\n")
        f.write("Kahneman Framework Analysis\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")

        features = list(kahneman_features.values())
        total = len(features)

        f.write("OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total videos analyzed:       {total}\n")
        f.write(f"Total matched pairs:         {len(matched_rows)}\n\n")

        # Loss/Gain frame analysis
        loss_vids = [v for v in features if v["loss_frame"]]
        gain_vids = [v for v in features if v["gain_frame"]]

        f.write("LOSS/GAIN FRAMING\n")
        f.write("-" * 40 + "\n")
        f.write(f"Loss frame videos:           {len(loss_vids)}\n")
        f.write(f"Gain frame videos:           {len(gain_vids)}\n")
        f.write(f"Neutral frame videos:        {total - len(loss_vids) - len(gain_vids)}\n\n")

        # Congruence in loss vs gain frames
        if loss_vids:
            loss_cong = [v["mean_congruence"] for v in loss_vids]
            f.write(f"Loss frame mean congruence:  {np.mean(loss_cong):.4f} (n={len(loss_cong)})\n")
        if gain_vids:
            gain_cong = [v["mean_congruence"] for v in gain_vids]
            f.write(f"Gain frame mean congruence:  {np.mean(gain_cong):.4f} (n={len(gain_cong)})\n")
        f.write("\n")

        # Congruence by segment type
        f.write("CONGRUENCE BY NEGOTIATION STAGE\n")
        f.write("-" * 40 + "\n")
        for stage in ["initial_ask", "first_offer", "counter", "expert_appraisal", "deal"]:
            vals = [v[f"{stage}_congruence"] for v in features if v.get(f"{stage}_congruence") is not None]
            if vals:
                f.write(f"  {stage}:\n")
                f.write(f"    Congruence: {np.mean(vals):.4f} +/- {np.std(vals):.4f} (n={len(vals)})\n")

                vader_vals = [v[f"{stage}_vader_mean"] for v in features if v.get(f"{stage}_vader_mean") is not None]
                face_vals = [v[f"{stage}_face_valence_mean"] for v in features if v.get(f"{stage}_face_valence_mean") is not None]
                if vader_vals:
                    f.write(f"    Text sentiment: {np.mean(vader_vals):.4f}\n")
                if face_vals:
                    f.write(f"    Face valence:   {np.mean(face_vals):.4f}\n")
                f.write("\n")

        # Loss aversion signature
        f.write("LOSS AVERSION SIGNATURE\n")
        f.write("-" * 40 + "\n")
        f.write("(Lower congruence during loss frames suggests emotional leakage,\n")
        f.write(" consistent with Kahneman's prospect theory predictions)\n\n")

        if loss_vids and gain_vids:
            loss_mean = np.mean([v["mean_congruence"] for v in loss_vids])
            gain_mean = np.mean([v["mean_congruence"] for v in gain_vids])
            diff = gain_mean - loss_mean
            f.write(f"  Gain frame congruence:     {gain_mean:.4f}\n")
            f.write(f"  Loss frame congruence:     {loss_mean:.4f}\n")
            f.write(f"  Difference (gain - loss):  {diff:.4f}\n")
            if diff > 0:
                f.write("  >> Loss aversion signal detected: lower congruence in loss frames\n")
            else:
                f.write("  >> No clear loss aversion signal in congruence data\n")
        else:
            f.write("  Insufficient data for loss aversion analysis\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")

    log(f"  Written: {filepath}")


def write_language_patterns_report(filepath, matched_rows):
    """Write Nisbett language patterns multimodal report."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("LANGUAGE PATTERNS MULTIMODAL REPORT\n")
        f.write("Nisbett Framework Analysis\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")

        total = len(matched_rows)
        f.write("OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total matched text-face pairs: {total}\n\n")

        # Analytic vs holistic totals
        total_analytic = sum(int(r.get("analytic_total", 0)) for r in matched_rows)
        total_holistic = sum(int(r.get("holistic_total", 0)) for r in matched_rows)

        f.write("NISBETT ANALYTIC vs HOLISTIC MARKERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total analytic markers:     {total_analytic}\n")
        f.write(f"  Total holistic markers:     {total_holistic}\n")
        if total_analytic + total_holistic > 0:
            ratio = total_analytic / (total_analytic + total_holistic)
            f.write(f"  Analytic ratio:             {ratio:.4f}\n")
        f.write("\n")

        # Congruence by language style
        analytic_rows = [r for r in matched_rows if int(r.get("analytic_total", 0)) > 0]
        holistic_rows = [r for r in matched_rows if int(r.get("holistic_total", 0)) > 0]
        neutral_rows = [r for r in matched_rows
                        if int(r.get("analytic_total", 0)) == 0
                        and int(r.get("holistic_total", 0)) == 0]

        f.write("CONGRUENCE BY LANGUAGE STYLE\n")
        f.write("-" * 40 + "\n")

        if analytic_rows:
            cong = [r["congruence_score"] for r in analytic_rows]
            f.write(f"  Analytic sentences:  congruence={np.mean(cong):.4f} (n={len(cong)})\n")
        if holistic_rows:
            cong = [r["congruence_score"] for r in holistic_rows]
            f.write(f"  Holistic sentences:  congruence={np.mean(cong):.4f} (n={len(cong)})\n")
        if neutral_rows:
            cong = [r["congruence_score"] for r in neutral_rows]
            f.write(f"  Neutral sentences:   congruence={np.mean(cong):.4f} (n={len(cong)})\n")
        f.write("\n")

        # Emotion distribution by language style
        f.write("FACIAL EMOTION BY LANGUAGE STYLE\n")
        f.write("-" * 40 + "\n")

        from collections import Counter
        for label, rows in [("Analytic", analytic_rows), ("Holistic", holistic_rows)]:
            if not rows:
                continue
            f.write(f"\n  {label} language moments:\n")
            emo_counts = Counter(r["dominant_emotion"] for r in rows)
            for emo, count in emo_counts.most_common():
                pct = count / len(rows) * 100
                f.write(f"    {emo}: {count} ({pct:.1f}%)\n")

        # Sentiment by role and language style
        f.write("\nSENTIMENT BY ROLE\n")
        f.write("-" * 40 + "\n")
        role_data = defaultdict(list)
        for r in matched_rows:
            role = r.get("primary_role", "neutral")
            role_data[role].append(r)

        for role in sorted(role_data.keys()):
            rows = role_data[role]
            vaders = [r["vader_compound"] for r in rows]
            congs = [r["congruence_score"] for r in rows]
            f.write(f"  {role}: sentiment={np.mean(vaders):.4f}, "
                    f"congruence={np.mean(congs):.4f} (n={len(rows)})\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")

    log(f"  Written: {filepath}")


def create_visualizations(matched_rows, kahneman_features, output_dir):
    """Create matplotlib visualizations."""
    log("\nCreating visualizations...")

    # 1. Emotion Progression
    log("  Creating emotion_progression.png...")
    try:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Group by segment type and compute mean emotions
        seg_types = ["initial_ask", "first_offer", "counter", "expert_appraisal", "deal"]
        seg_labels = ["Initial Ask", "First Offer", "Counter", "Expert Appraisal", "Deal"]

        seg_vader = {}
        seg_face = {}
        seg_congruence = {}
        for st in seg_types:
            rows = [r for r in matched_rows if r.get("segment_type") == st]
            if rows:
                seg_vader[st] = np.mean([r["vader_compound"] for r in rows])
                seg_face[st] = np.mean([r["face_valence"] for r in rows])
                seg_congruence[st] = np.mean([r["congruence_score"] for r in rows])

        present_types = [st for st in seg_types if st in seg_vader]
        present_labels = [seg_labels[seg_types.index(st)] for st in present_types]

        if present_types:
            x = np.arange(len(present_types))
            width = 0.35

            # Top: text vs face sentiment
            ax1 = axes[0]
            vader_vals = [seg_vader[st] for st in present_types]
            face_vals = [seg_face[st] for st in present_types]
            bars1 = ax1.bar(x - width/2, vader_vals, width, label="Text (VADER)", color="#2196F3", alpha=0.8)
            bars2 = ax1.bar(x + width/2, face_vals, width, label="Face Valence", color="#FF9800", alpha=0.8)
            ax1.set_ylabel("Sentiment / Valence")
            ax1.set_title("Emotion Progression Across Negotiation Stages")
            ax1.set_xticks(x)
            ax1.set_xticklabels(present_labels, rotation=15)
            ax1.legend()
            ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax1.grid(axis="y", alpha=0.3)

            # Bottom: congruence
            ax2 = axes[1]
            cong_vals = [seg_congruence.get(st, 0) for st in present_types]
            bars3 = ax2.bar(x, cong_vals, width * 1.5, color="#4CAF50", alpha=0.8)
            ax2.set_ylabel("Congruence Score")
            ax2.set_title("Text-Face Congruence by Stage")
            ax2.set_xticks(x)
            ax2.set_xticklabels(present_labels, rotation=15)
            ax2.set_ylim(0, 1)
            ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "emotion_progression.png"), dpi=150, bbox_inches="tight")
        plt.close()
        log("    Saved emotion_progression.png")
    except Exception as e:
        log(f"    ERROR creating emotion_progression.png: {e}")

    # 2. Text vs Face Sentiment Scatter
    log("  Creating text_vs_face_sentiment.png...")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        if matched_rows:
            vader_vals = [r["vader_compound"] for r in matched_rows]
            face_vals = [r["face_valence"] for r in matched_rows]

            # Color by segment type
            seg_colors = {
                "initial_ask": "#2196F3",
                "first_offer": "#FF9800",
                "counter": "#9C27B0",
                "expert_appraisal": "#F44336",
                "deal": "#4CAF50",
            }

            for st, color in seg_colors.items():
                st_rows = [r for r in matched_rows if r.get("segment_type") == st]
                if st_rows:
                    vv = [r["vader_compound"] for r in st_rows]
                    fv = [r["face_valence"] for r in st_rows]
                    ax.scatter(vv, fv, c=color, label=st.replace("_", " ").title(),
                               alpha=0.5, s=20)

            # Other types
            other_rows = [r for r in matched_rows if r.get("segment_type") not in seg_colors]
            if other_rows:
                vv = [r["vader_compound"] for r in other_rows]
                fv = [r["face_valence"] for r in other_rows]
                ax.scatter(vv, fv, c="gray", label="Other", alpha=0.3, s=10)

            # Diagonal line (perfect congruence)
            ax.plot([-1, 1], [-1, 1], "k--", alpha=0.3, label="Perfect Congruence")

            ax.set_xlabel("Text Sentiment (VADER Compound)")
            ax.set_ylabel("Facial Emotion Valence")
            ax.set_title("Text Sentiment vs. Facial Emotion")
            ax.legend(loc="upper left", fontsize=8)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "text_vs_face_sentiment.png"), dpi=150, bbox_inches="tight")
        plt.close()
        log("    Saved text_vs_face_sentiment.png")
    except Exception as e:
        log(f"    ERROR creating text_vs_face_sentiment.png: {e}")

    # 3. Loss Aversion Multimodal
    log("  Creating loss_aversion_multimodal.png...")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        features = list(kahneman_features.values())

        loss_vids = [v for v in features if v["loss_frame"]]
        gain_vids = [v for v in features if v["gain_frame"]]
        neutral_vids = [v for v in features if not v["loss_frame"] and not v["gain_frame"]]

        # Panel 1: Congruence distribution by frame type
        ax1 = axes[0]
        data_groups = []
        labels_groups = []
        colors_groups = []

        if loss_vids:
            data_groups.append([v["mean_congruence"] for v in loss_vids])
            labels_groups.append(f"Loss\n(n={len(loss_vids)})")
            colors_groups.append("#F44336")
        if gain_vids:
            data_groups.append([v["mean_congruence"] for v in gain_vids])
            labels_groups.append(f"Gain\n(n={len(gain_vids)})")
            colors_groups.append("#4CAF50")
        if neutral_vids:
            data_groups.append([v["mean_congruence"] for v in neutral_vids])
            labels_groups.append(f"Neutral\n(n={len(neutral_vids)})")
            colors_groups.append("#9E9E9E")

        if data_groups:
            bp = ax1.boxplot(data_groups, labels=labels_groups, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors_groups):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax1.set_ylabel("Congruence Score")
        ax1.set_title("Congruence by Frame Type")
        ax1.grid(axis="y", alpha=0.3)

        # Panel 2: Stage-specific congruence for loss vs gain
        ax2 = axes[1]
        stages = ["initial_ask", "first_offer", "counter", "expert_appraisal", "deal"]
        stage_labels = ["Ask", "Offer", "Counter", "Expert", "Deal"]

        if loss_vids:
            loss_cong = []
            for st in stages:
                vals = [v[f"{st}_congruence"] for v in loss_vids if v.get(f"{st}_congruence") is not None]
                loss_cong.append(np.mean(vals) if vals else 0)
            ax2.plot(stage_labels, loss_cong, "o-", color="#F44336", label="Loss Frame", linewidth=2)

        if gain_vids:
            gain_cong = []
            for st in stages:
                vals = [v[f"{st}_congruence"] for v in gain_vids if v.get(f"{st}_congruence") is not None]
                gain_cong.append(np.mean(vals) if vals else 0)
            ax2.plot(stage_labels, gain_cong, "s-", color="#4CAF50", label="Gain Frame", linewidth=2)

        ax2.set_ylabel("Mean Congruence")
        ax2.set_title("Congruence Across Stages")
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.tick_params(axis="x", rotation=15)

        # Panel 3: Text vs Face sentiment in loss vs gain
        ax3 = axes[2]
        categories = []
        vader_means = []
        face_means = []
        bar_colors = []

        for label, vids, color in [("Loss", loss_vids, "#F44336"),
                                     ("Gain", gain_vids, "#4CAF50"),
                                     ("Neutral", neutral_vids, "#9E9E9E")]:
            if vids:
                vid_ids = set(v["video_id"] for v in vids)
                vid_rows = [r for r in matched_rows if r["video_id"] in vid_ids]
                if vid_rows:
                    categories.append(label)
                    vader_means.append(np.mean([r["vader_compound"] for r in vid_rows]))
                    face_means.append(np.mean([r["face_valence"] for r in vid_rows]))
                    bar_colors.append(color)

        if categories:
            x = np.arange(len(categories))
            width = 0.35
            ax3.bar(x - width/2, vader_means, width, label="Text", alpha=0.8,
                    color=[c for c in bar_colors])
            ax3.bar(x + width/2, face_means, width, label="Face", alpha=0.5,
                    color=[c for c in bar_colors], hatch="//")
            ax3.set_xticks(x)
            ax3.set_xticklabels(categories)
            ax3.set_ylabel("Mean Sentiment/Valence")
            ax3.set_title("Text vs Face by Frame Type")
            ax3.legend()
            ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax3.grid(axis="y", alpha=0.3)

        plt.suptitle("Loss Aversion Multimodal Analysis", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_aversion_multimodal.png"), dpi=150, bbox_inches="tight")
        plt.close()
        log("    Saved loss_aversion_multimodal.png")
    except Exception as e:
        log(f"    ERROR creating loss_aversion_multimodal.png: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine text sentiment and facial sentiment for multimodal analysis."
    )
    parser.add_argument(
        "--text-csv",
        default=os.path.join(ANALYSIS_DIR, "text_sentiment_per_sentence.csv"),
        help="Path to text sentiment CSV"
    )
    parser.add_argument(
        "--face-csv",
        default=os.path.join(ANALYSIS_DIR, "sentiment_per_frame.csv"),
        help="Path to facial sentiment CSV"
    )
    parser.add_argument(
        "--negotiations",
        default=os.path.join(RESULTS_DIR, "negotiations_analysis.json"),
        help="Path to negotiations analysis JSON"
    )
    parser.add_argument(
        "--output-dir",
        default=ANALYSIS_DIR,
        help=f"Output directory (default: {ANALYSIS_DIR})"
    )
    parser.add_argument(
        "--window",
        type=float,
        default=TIMESTAMP_WINDOW,
        help=f"Timestamp matching window in seconds (default: {TIMESTAMP_WINDOW})"
    )
    args = parser.parse_args()

    log("=" * 60)
    log("MULTIMODAL ANALYSIS - Text + Facial Sentiment")
    log("=" * 60)

    # Ensure output directory symlink
    persistent_analysis = os.path.join(PERSISTENT_DIR, "analysis")
    ensure_symlink(args.output_dir, persistent_analysis)

    # Load text sentiment data
    log(f"\nLoading text sentiment: {args.text_csv}")
    if not os.path.isfile(args.text_csv):
        log(f"ERROR: Text sentiment CSV not found: {args.text_csv}")
        log("Run analyze_text_sentiment.py first.")
        sys.exit(1)
    text_rows = load_csv(args.text_csv)
    log(f"  Loaded {len(text_rows)} text rows")

    # Load facial sentiment data
    log(f"\nLoading facial sentiment: {args.face_csv}")
    if not os.path.isfile(args.face_csv):
        log(f"ERROR: Facial sentiment CSV not found: {args.face_csv}")
        log("Run analyze_sentiment.py first.")
        sys.exit(1)
    face_rows = load_csv(args.face_csv)
    log(f"  Loaded {len(face_rows)} face rows")

    # Match text and face data
    log(f"\nMatching text and face data (window={args.window}s)...")
    matched_rows = match_text_and_face(text_rows, face_rows, window=args.window)

    if not matched_rows:
        log("\nNo matched pairs found. Check that text and face data share video_ids.")
        log("Text video_ids sample: " + ", ".join(set(r.get("video_id", "") for r in text_rows[:5])))
        log("Face video_ids sample: " + ", ".join(set(r.get("video_id", "") for r in face_rows[:5])))
        sys.exit(0)

    # Compute Kahneman multimodal features
    log("\nComputing Kahneman multimodal features...")
    kahneman_features = compute_kahneman_multimodal(matched_rows, args.negotiations)
    log(f"  Videos with features: {len(kahneman_features)}")

    # Write outputs
    log("\nWriting outputs...")

    # Multimodal CSV
    multimodal_csv = os.path.join(args.output_dir, "multimodal_analysis.csv")
    write_csv(multimodal_csv, matched_rows)

    # Behavioral economics report
    be_report = os.path.join(args.output_dir, "behavioral_economics_report.txt")
    write_behavioral_economics_report(be_report, kahneman_features, matched_rows)

    # Language patterns report
    lp_report = os.path.join(args.output_dir, "language_patterns_report.txt")
    write_language_patterns_report(lp_report, matched_rows)

    # Visualizations
    create_visualizations(matched_rows, kahneman_features, args.output_dir)

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  Text rows:              {len(text_rows)}")
    log(f"  Face rows:              {len(face_rows)}")
    log(f"  Matched pairs:          {len(matched_rows)}")
    log(f"  Videos analyzed:        {len(kahneman_features)}")

    if matched_rows:
        mean_cong = np.mean([r["congruence_score"] for r in matched_rows])
        log(f"  Mean congruence:        {mean_cong:.4f}")

    loss_count = sum(1 for v in kahneman_features.values() if v["loss_frame"])
    gain_count = sum(1 for v in kahneman_features.values() if v["gain_frame"])
    log(f"  Loss frame videos:      {loss_count}")
    log(f"  Gain frame videos:      {gain_count}")

    log(f"\nOutputs written to: {args.output_dir}")
    log("  - multimodal_analysis.csv")
    log("  - behavioral_economics_report.txt")
    log("  - language_patterns_report.txt")
    log("  - emotion_progression.png")
    log("  - text_vs_face_sentiment.png")
    log("  - loss_aversion_multimodal.png")

    log("\n" + "=" * 60)
    log("Done!")


if __name__ == "__main__":
    main()
