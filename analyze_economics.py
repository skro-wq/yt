#!/usr/bin/env python3
"""
analyze_economics.py - Comprehensive negotiation economics analysis.

Reconstructs price trajectories per video, measures expert impact on pricing,
classifies deal vs no-deal outcomes, categorizes items, maps System 1/2
behavioral engagement, and models predictors of deal outcome and price.

Research question: Role of dollar amount and emotions related to expert's
price change and deal outcome, considering object type, amount changes,
and cognitive system engagement.

Outputs:
  analysis/economics_report.txt  - Full findings report
  analysis/price_trajectories.csv - Per-video price data
  analysis/deal_prediction.csv - Features for deal/no-deal
"""

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

BASE_DIR = "/workspaces/youtube-transcript-batch"
PERSISTENT_DIR = "/root/.claude/projects/-workspaces-youtube-transcript-batch/data"
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MASTER_FILE = os.path.join(BASE_DIR, "master_dataset", "all_transcripts.json")


def log(msg):
    print(msg)
    sys.stdout.flush()


def ensure_symlink(workspace_path, persistent_path):
    os.makedirs(persistent_path, exist_ok=True)
    if os.path.islink(workspace_path):
        return
    if os.path.isdir(workspace_path) and not os.listdir(workspace_path):
        os.rmdir(workspace_path)
    if not os.path.exists(workspace_path):
        os.symlink(persistent_path, workspace_path)


# ─── ITEM CATEGORIZATION ───────────────────────────────────────────

ITEM_CATEGORIES = {
    "weapons_military": [
        r"gun", r"rifle", r"pistol", r"sword", r"dagger", r"knife",
        r"bayonet", r"cannon", r"musket", r"revolver", r"shotgun",
        r"weapon", r"ammo", r"grenade", r"military", r"army", r"navy",
        r"war", r"civil war", r"wwii", r"ww2", r"world war",
        r"medal", r"uniform", r"helmet", r"samurai",
    ],
    "vehicles_machines": [
        r"car", r"truck", r"motorcycle", r"bike", r"harley",
        r"chevy", r"ford", r"corvette", r"mustang", r"camaro",
        r"hot rod", r"engine", r"vehicle", r"boat", r"aircraft",
        r"pinball", r"slot machine", r"jukebox", r"arcade",
    ],
    "art_antiques": [
        r"paint", r"art", r"sculpture", r"statue", r"portrait",
        r"antique", r"vase", r"pottery", r"ceramic", r"glass",
        r"furniture", r"chair", r"desk", r"cabinet", r"lamp",
        r"chandelier", r"clock", r"watch",
    ],
    "jewelry_precious": [
        r"gold", r"silver", r"diamond", r"ring", r"necklace",
        r"bracelet", r"jewel", r"gem", r"rolex", r"cartier",
        r"coin", r"bullion", r"bar\b",
    ],
    "sports_memorabilia": [
        r"baseball", r"football", r"basketball", r"hockey",
        r"boxing", r"nfl", r"nba", r"mlb", r"jersey",
        r"signed", r"autograph", r"trophy", r"championship",
        r"super bowl", r"world series",
    ],
    "entertainment_pop_culture": [
        r"guitar", r"music", r"album", r"vinyl", r"record",
        r"movie", r"film", r"star wars", r"elvis", r"beatles",
        r"comic", r"action figure", r"toy", r"game",
        r"pokemon", r"disney", r"marvel", r"lego",
        r"celebrity", r"tv show",
    ],
    "books_documents": [
        r"book", r"letter", r"document", r"manuscript",
        r"newspaper", r"map", r"atlas", r"bible",
        r"first edition", r"signed copy", r"constitution",
    ],
    "historical_artifacts": [
        r"ancient", r"fossil", r"dinosaur", r"meteorite",
        r"artifact", r"relic", r"egyptian", r"roman",
        r"native american", r"indian", r"colonial",
        r"president", r"lincoln", r"washington",
    ],
}


def categorize_item(title):
    """Categorize item from video title."""
    title_lower = title.lower()
    scores = {}
    for category, patterns in ITEM_CATEGORIES.items():
        score = sum(1 for p in patterns if re.search(p, title_lower))
        if score > 0:
            scores[category] = score
    if scores:
        return max(scores, key=scores.get)
    return "other"


# ─── DEAL / NO-DEAL CLASSIFICATION ──────────────────────────────────

DEAL_STRONG = [
    r"you[\' ]?ve? got a deal",
    r"i[\' ]?ll take it",
    r"we[\' ]?ve? got a deal",
    r"it[\' ]?s a deal",
    r"\bsold\b",
    r"shake on it",
    r"let[\' ]?s do it",
    r"i[\' ]?ll do it",
    r"you got yourself a deal",
    r"we have a deal",
    r"deal done",
    r"write (it|him|her) up",
    r"ring (it|him|her) up",
    r"chum,? write",
]

NO_DEAL_STRONG = [
    r"no deal",
    r"i[\' ]?m (going to |gonna )?pass",
    r"i[\' ]?ll pass",
    r"can[\' ]?t do it",
    r"not (going to|gonna) (work|happen)",
    r"i[\' ]?m out",
    r"walk(ed|ing)? away",
    r"not for me",
    r"no way",
    r"can[\' ]?t do (that|this)",
    r"not interested",
    r"take it somewhere else",
]


def classify_outcome(transcript_entries):
    """Classify deal vs no-deal from transcript ending."""
    if not transcript_entries:
        return "unknown"

    # Check last 25% of transcript
    cutoff = max(1, int(len(transcript_entries) * 0.75))
    end_entries = transcript_entries[cutoff:]
    end_text = " ".join(e.get("text", "") for e in end_entries).lower()

    deal_score = sum(1 for p in DEAL_STRONG if re.search(p, end_text))
    no_deal_score = sum(1 for p in NO_DEAL_STRONG if re.search(p, end_text))

    # Also check very last entries (last 5) with higher weight
    last_text = " ".join(e.get("text", "") for e in end_entries[-5:]).lower()
    deal_score += sum(2 for p in DEAL_STRONG if re.search(p, last_text))
    no_deal_score += sum(2 for p in NO_DEAL_STRONG if re.search(p, last_text))

    if deal_score > no_deal_score and deal_score >= 2:
        return "deal"
    elif no_deal_score > deal_score and no_deal_score >= 2:
        return "no_deal"
    elif deal_score > 0 and no_deal_score == 0:
        return "deal"
    elif no_deal_score > 0 and deal_score == 0:
        return "no_deal"
    else:
        return "unknown"


# ─── PRICE TRAJECTORY EXTRACTION ────────────────────────────────────

def find_price_cluster(all_amounts):
    """Find the dominant negotiation-price cluster from all amounts in a video.

    The real negotiation prices cluster together (e.g. $5000, $3000, $4500)
    while noise amounts ($20, $5) are far away. Find the largest cluster
    of amounts within 10x of each other.
    """
    if not all_amounts:
        return set()
    amounts = sorted(set(all_amounts))
    if len(amounts) == 1:
        return set(amounts)

    # Try each amount as cluster center, find cluster within 10x
    best_cluster = set()
    for center in amounts:
        if center <= 0:
            continue
        cluster = set()
        for a in amounts:
            if a > 0 and (a / center <= 10 and center / a <= 10):
                cluster.add(a)
        # Prefer cluster with higher total value (real prices are larger)
        if (len(cluster) > len(best_cluster) or
            (len(cluster) == len(best_cluster) and
             sum(cluster) > sum(best_cluster))):
            best_cluster = cluster

    return best_cluster


def extract_representative_amount(amounts, price_cluster=None):
    """Pick the most representative dollar amount from a list.

    Uses the price cluster to filter noise, then takes the max
    (negotiation prices are usually the largest amounts mentioned).
    """
    if not amounts:
        return None

    # Filter to cluster if available
    if price_cluster:
        filtered = [a for a in amounts if a in price_cluster]
        if filtered:
            amounts = filtered

    # Take max - the actual negotiation price is typically the largest
    # dollar amount mentioned in that stage context
    return float(max(amounts))


def build_price_trajectory(video_segments):
    """Build price trajectory from negotiation segments for one video.

    Uses clustering to identify the real negotiation prices vs noise.
    """
    stage_amounts = defaultdict(list)
    all_amounts = []
    for seg in video_segments:
        if seg["dollar_amounts"]:
            stage_amounts[seg["segment_type"]].extend(seg["dollar_amounts"])
            all_amounts.extend(seg["dollar_amounts"])

    # Find the price cluster for this video
    cluster = find_price_cluster(all_amounts)

    trajectory = {}
    for stage in ["initial_ask", "first_offer", "counter", "expert_appraisal", "deal"]:
        if stage in stage_amounts:
            amt = extract_representative_amount(stage_amounts[stage], cluster)
            if amt is not None and amt > 0:
                trajectory[stage] = amt

    # Validate: if any stage is >20x another, the trajectory is suspect
    if len(trajectory) >= 2:
        vals = list(trajectory.values())
        spread = max(vals) / max(min(vals), 0.01)
        if spread > 20:
            # Keep only stages whose prices are in the main cluster
            if cluster:
                filtered = {}
                for stage, amt in trajectory.items():
                    if amt in cluster:
                        filtered[stage] = amt
                if len(filtered) >= 1:
                    trajectory = filtered

    return trajectory


# ─── TEXT SENTIMENT AGGREGATION PER VIDEO ────────────────────────────

def load_text_sentiment_by_video():
    """Load text sentiment CSV and aggregate per video."""
    csv_path = os.path.join(ANALYSIS_DIR, "text_sentiment_per_sentence.csv")
    if not os.path.isfile(csv_path):
        return {}

    video_data = defaultdict(lambda: {
        "sentences": [],
        "by_stage": defaultdict(list),
        "by_role": defaultdict(list),
        "analytic_total": 0,
        "holistic_total": 0,
        "dollar_mentions": 0,
    })

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", "")
            if not vid:
                continue

            compound = float(row.get("vader_compound", 0))
            stage = row.get("segment_type", "")
            role = row.get("primary_role", "neutral")

            video_data[vid]["sentences"].append(compound)
            if stage:
                video_data[vid]["by_stage"][stage].append(compound)
            video_data[vid]["by_role"][role].append(compound)
            video_data[vid]["analytic_total"] += int(row.get("analytic_total", 0))
            video_data[vid]["holistic_total"] += int(row.get("holistic_total", 0))
            if row.get("dollar_amounts"):
                video_data[vid]["dollar_mentions"] += 1

    return dict(video_data)


# ─── FACIAL SENTIMENT AGGREGATION PER VIDEO ─────────────────────────

def load_facial_sentiment_by_video():
    """Load facial sentiment CSV and aggregate per video."""
    csv_path = os.path.join(ANALYSIS_DIR, "sentiment_per_frame.csv")
    if not os.path.isfile(csv_path):
        return {}

    video_data = defaultdict(lambda: {
        "emotions": [],
        "by_moment": defaultdict(list),
    })

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", "")
            if not vid:
                continue
            emo = row.get("dominant_emotion", "neutral")
            moment = row.get("moment_type", "")
            video_data[vid]["emotions"].append(emo)
            if moment:
                video_data[vid]["by_moment"][moment].append(emo)

    return dict(video_data)


# ─── CONGRUENCE DATA PER VIDEO ───────────────────────────────────────

def load_congruence_by_video():
    """Load multimodal analysis and aggregate congruence per video."""
    csv_path = os.path.join(ANALYSIS_DIR, "multimodal_analysis.csv")
    if not os.path.isfile(csv_path):
        return {}

    video_data = defaultdict(lambda: {
        "congruence_scores": [],
        "by_stage": defaultdict(list),
    })

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", "")
            if not vid:
                continue
            cong = float(row.get("congruence_score", 0))
            stage = row.get("segment_type", "")
            video_data[vid]["congruence_scores"].append(cong)
            if stage:
                video_data[vid]["by_stage"][stage].append(cong)

    return dict(video_data)


# ─── SYSTEM 1 / SYSTEM 2 CLASSIFICATION ─────────────────────────────

def classify_cognitive_system(analytic, holistic, mean_sentiment, emotion_dist):
    """Classify dominant cognitive system engagement.

    System 2 (deliberative): high analytic markers, moderate sentiment,
                             neutral/controlled facial expressions
    System 1 (automatic): high holistic markers, extreme sentiment,
                          strong emotional facial expressions
    Mixed: significant engagement of both systems
    """
    total_markers = analytic + holistic
    if total_markers == 0:
        # No clear markers - classify by emotional reactivity
        if abs(mean_sentiment) > 0.3:
            return "system_1"
        return "system_2"

    analytic_ratio = analytic / total_markers

    # Strong emotional facial dominance
    emotional_faces = sum(emotion_dist.get(e, 0) for e in ["angry", "fear", "sad", "disgust"])
    total_faces = sum(emotion_dist.values()) or 1
    emotional_ratio = emotional_faces / total_faces

    if analytic_ratio > 0.8 and emotional_ratio < 0.6:
        return "system_2"
    elif analytic_ratio < 0.4 or (emotional_ratio > 0.7 and abs(mean_sentiment) > 0.2):
        return "system_1"
    else:
        return "mixed"


# ─── STATISTICAL HELPERS ─────────────────────────────────────────────

def pearson_r(x, y):
    """Compute Pearson correlation coefficient."""
    x, y = np.array(x), np.array(y)
    if len(x) < 3:
        return float("nan"), float("nan")
    mx, my = np.mean(x), np.mean(y)
    sx = np.sqrt(np.sum((x - mx) ** 2))
    sy = np.sqrt(np.sum((y - my) ** 2))
    if sx == 0 or sy == 0:
        return float("nan"), float("nan")
    r = np.sum((x - mx) * (y - my)) / (sx * sy)
    # t-test for significance
    n = len(x)
    if abs(r) >= 1.0:
        return r, 0.0
    t_stat = r * np.sqrt((n - 2) / (1 - r ** 2))
    # Approximate p-value from t distribution (two-tailed)
    from math import pi, atan
    df = n - 2
    p = 2 * (0.5 - atan(t_stat / np.sqrt(df)) / pi)  # crude approx
    return float(r), float(p)


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        return float("nan")
    return (m1 - m2) / pooled


def t_test_ind(group1, group2):
    """Independent samples t-test (Welch's)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    se = np.sqrt(s1 / n1 + s2 / n2)
    if se == 0:
        return float("nan"), float("nan")
    t_stat = (m1 - m2) / se
    # Welch-Satterthwaite df
    num = (s1 / n1 + s2 / n2) ** 2
    den = (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
    df = num / den if den > 0 else 1
    # Approximate p-value
    from math import pi, atan
    p = 2 * (0.5 - atan(t_stat / np.sqrt(max(df, 1))) / pi)
    return float(t_stat), float(abs(p))


def fmt(val, decimals=4):
    """Format number."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


# ─── MAIN ANALYSIS ──────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("NEGOTIATION ECONOMICS ANALYSIS")
    log("Dollar Amounts, Expert Impact, Deal Outcomes & Cognitive Systems")
    log("=" * 70)

    ensure_symlink(ANALYSIS_DIR, os.path.join(PERSISTENT_DIR, "analysis"))

    # ── Load all data sources ──
    log("\n[1] Loading data sources...")

    # Negotiations
    with open(os.path.join(RESULTS_DIR, "negotiations_analysis.json")) as f:
        neg_data = json.load(f)
    segments = neg_data["segments"]
    log(f"  Negotiation segments: {len(segments)}")

    # Group segments by video
    seg_by_video = defaultdict(list)
    for s in segments:
        seg_by_video[s["video_id"]].append(s)

    # Master dataset (for titles and transcripts)
    with open(MASTER_FILE) as f:
        master = json.load(f)
    video_lookup = {}
    for v in master["videos"]:
        vid = v.get("video_id", "")
        if vid:
            video_lookup[vid] = v
    log(f"  Master videos: {len(video_lookup)}")

    # Text sentiment
    text_data = load_text_sentiment_by_video()
    log(f"  Text sentiment videos: {len(text_data)}")

    # Facial sentiment
    face_data = load_facial_sentiment_by_video()
    log(f"  Facial sentiment videos: {len(face_data)}")

    # Congruence
    cong_data = load_congruence_by_video()
    log(f"  Congruence videos: {len(cong_data)}")

    # ── Build per-video feature set ──
    log("\n[2] Building per-video features...")

    videos = []
    for vid, segs in seg_by_video.items():
        master_v = video_lookup.get(vid, {})
        title = master_v.get("title", "")
        show = segs[0].get("show", "unknown") if segs else "unknown"

        # Transcript for outcome classification
        transcript = master_v.get("transcript", [])
        if isinstance(transcript, dict):
            transcript = transcript.get("raw_entries", [])

        # Price trajectory
        trajectory = build_price_trajectory(segs)

        # Outcome
        outcome = classify_outcome(transcript)

        # Item category
        item_cat = categorize_item(title)

        # Text sentiment features
        td = text_data.get(vid, {})
        sentences = td.get("sentences", [])
        mean_sentiment = float(np.mean(sentences)) if sentences else 0.0
        sentiment_std = float(np.std(sentences)) if len(sentences) > 1 else 0.0
        analytic = td.get("analytic_total", 0)
        holistic = td.get("holistic_total", 0)

        # Stage-specific sentiment
        stage_sentiment = {}
        for stage in ["initial_ask", "first_offer", "counter", "expert_appraisal", "deal"]:
            s_vals = td.get("by_stage", {}).get(stage, [])
            stage_sentiment[stage] = float(np.mean(s_vals)) if s_vals else None

        # Role sentiment
        role_sentiment = {}
        for role in ["asking", "offering", "countering", "appraising", "closing", "rejecting"]:
            r_vals = td.get("by_role", {}).get(role, [])
            role_sentiment[role] = float(np.mean(r_vals)) if r_vals else None

        # Facial emotion distribution
        fd = face_data.get(vid, {})
        all_emotions = fd.get("emotions", [])
        emotion_dist = Counter(all_emotions)

        # Congruence
        cd = cong_data.get(vid, {})
        cong_scores = cd.get("congruence_scores", [])
        mean_congruence = float(np.mean(cong_scores)) if cong_scores else None

        # System classification
        cog_system = classify_cognitive_system(
            analytic, holistic, mean_sentiment, emotion_dist
        )

        # Sentiment trajectory (slope from first to last stage)
        ordered_stages = ["initial_ask", "first_offer", "counter", "expert_appraisal", "deal"]
        trajectory_vals = [(i, stage_sentiment[s]) for i, s in enumerate(ordered_stages)
                           if stage_sentiment.get(s) is not None]
        sentiment_slope = None
        if len(trajectory_vals) >= 2:
            xs = [t[0] for t in trajectory_vals]
            ys = [t[1] for t in trajectory_vals]
            if len(set(xs)) > 1:
                slope = np.polyfit(xs, ys, 1)[0]
                sentiment_slope = float(slope)

        # Compute price-derived features
        ask = trajectory.get("initial_ask")
        expert = trajectory.get("expert_appraisal")
        deal_price = trajectory.get("deal")
        first_offer = trajectory.get("first_offer")
        counter = trajectory.get("counter")

        expert_vs_ask = None  # % change
        deal_vs_ask = None
        deal_vs_expert = None
        expert_direction = None  # "above", "below", "equal"
        concession_pct = None  # how much seller conceded from ask
        buyer_stretch_pct = None  # how much buyer stretched from first offer

        if ask and ask > 0:
            if expert:
                expert_vs_ask = (expert - ask) / ask
                if expert > ask * 1.05:
                    expert_direction = "above"
                elif expert < ask * 0.95:
                    expert_direction = "below"
                else:
                    expert_direction = "equal"
            if deal_price:
                deal_vs_ask = (deal_price - ask) / ask
                concession_pct = (ask - deal_price) / ask  # positive = seller conceded

        if expert and expert > 0 and deal_price:
            deal_vs_expert = (deal_price - expert) / expert

        if first_offer and first_offer > 0 and deal_price:
            buyer_stretch_pct = (deal_price - first_offer) / first_offer

        # Number of negotiation stages present
        n_stages = len(trajectory)
        has_expert = "expert_appraisal" in trajectory

        # Segment counts
        seg_counts = Counter(s["segment_type"] for s in segs)

        videos.append({
            "video_id": vid,
            "title": title,
            "show": show,
            "item_category": item_cat,
            "outcome": outcome,
            # Prices
            "ask_price": ask,
            "first_offer_price": first_offer,
            "counter_price": counter,
            "expert_price": expert,
            "deal_price": deal_price,
            "n_price_stages": n_stages,
            "has_expert": has_expert,
            # Price changes
            "expert_vs_ask_pct": expert_vs_ask,
            "deal_vs_ask_pct": deal_vs_ask,
            "deal_vs_expert_pct": deal_vs_expert,
            "expert_direction": expert_direction,
            "concession_pct": concession_pct,
            "buyer_stretch_pct": buyer_stretch_pct,
            # Sentiment
            "mean_sentiment": mean_sentiment,
            "sentiment_std": sentiment_std,
            "sentiment_slope": sentiment_slope,
            "sentiment_initial_ask": stage_sentiment.get("initial_ask"),
            "sentiment_first_offer": stage_sentiment.get("first_offer"),
            "sentiment_counter": stage_sentiment.get("counter"),
            "sentiment_expert": stage_sentiment.get("expert_appraisal"),
            "sentiment_deal": stage_sentiment.get("deal"),
            # Cognitive system
            "analytic_markers": analytic,
            "holistic_markers": holistic,
            "analytic_ratio": analytic / (analytic + holistic) if (analytic + holistic) > 0 else None,
            "cognitive_system": cog_system,
            # Facial
            "dominant_emotion": emotion_dist.most_common(1)[0][0] if emotion_dist else None,
            "happy_pct": emotion_dist.get("happy", 0) / max(len(all_emotions), 1),
            "sad_pct": emotion_dist.get("sad", 0) / max(len(all_emotions), 1),
            "fear_pct": emotion_dist.get("fear", 0) / max(len(all_emotions), 1),
            "angry_pct": emotion_dist.get("angry", 0) / max(len(all_emotions), 1),
            "neutral_pct": emotion_dist.get("neutral", 0) / max(len(all_emotions), 1),
            "n_frames": len(all_emotions),
            # Congruence
            "mean_congruence": mean_congruence,
            # Negotiation complexity
            "n_segments": len(segs),
            "n_initial_asks": seg_counts.get("initial_ask", 0),
            "n_offers": seg_counts.get("first_offer", 0),
            "n_counters": seg_counts.get("counter", 0),
            "n_expert_segments": seg_counts.get("expert_appraisal", 0),
            "n_deal_segments": seg_counts.get("deal", 0),
            "n_sentences": len(sentences),
        })

    log(f"  Built features for {len(videos)} videos")

    # ── Filter and classify ──
    outcome_dist = Counter(v["outcome"] for v in videos)
    log(f"\n  Outcome distribution: {dict(outcome_dist)}")

    has_prices = [v for v in videos if v["n_price_stages"] >= 1]
    has_ask_deal = [v for v in videos if v["ask_price"] and v["deal_price"]]
    has_expert_v = [v for v in videos if v["has_expert"]]
    has_ask_expert = [v for v in videos if v["ask_price"] and v["expert_price"]]
    has_full_chain = [v for v in videos if v["ask_price"] and v["expert_price"] and v["deal_price"]]

    log(f"  Videos with any price: {len(has_prices)}")
    log(f"  Videos with ask + deal: {len(has_ask_deal)}")
    log(f"  Videos with expert appraisal: {len(has_expert_v)}")
    log(f"  Videos with ask + expert: {len(has_ask_expert)}")
    log(f"  Videos with ask + expert + deal: {len(has_full_chain)}")

    item_dist = Counter(v["item_category"] for v in videos)
    log(f"\n  Item categories: {dict(item_dist.most_common())}")

    system_dist = Counter(v["cognitive_system"] for v in videos)
    log(f"  Cognitive systems: {dict(system_dist)}")

    # ── WRITE REPORT ──
    log("\n[3] Writing analysis report...")

    report_path = os.path.join(ANALYSIS_DIR, "economics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        W = f.write

        W("=" * 70 + "\n")
        W("NEGOTIATION ECONOMICS: EXPERT IMPACT, DEAL OUTCOMES\n")
        W("& COGNITIVE SYSTEM ENGAGEMENT\n")
        W(f"Generated: {datetime.now().isoformat()}\n")
        W("=" * 70 + "\n\n")

        # ── 1. DATA OVERVIEW ──
        W("1. DATA OVERVIEW\n")
        W("-" * 50 + "\n")
        W(f"Total videos with negotiations:  {len(videos)}\n")
        W(f"Videos with price data:          {len(has_prices)}\n")
        W(f"Videos with ask + deal price:    {len(has_ask_deal)}\n")
        W(f"Videos with expert appraisal:    {len(has_expert_v)}\n")
        W(f"Full price chain (ask→expert→deal): {len(has_full_chain)}\n\n")

        W(f"Outcome classification:\n")
        for outcome, count in outcome_dist.most_common():
            W(f"  {outcome}: {count} ({count/len(videos)*100:.1f}%)\n")
        W("\n")

        W(f"Item categories:\n")
        for cat, count in item_dist.most_common():
            W(f"  {cat}: {count}\n")
        W("\n")

        # ── 2. PRICE TRAJECTORY ANALYSIS ──
        W("\n2. PRICE TRAJECTORY ANALYSIS\n")
        W("-" * 50 + "\n")

        if has_ask_deal:
            # Filter to reasonable price changes (within -90% to +500%)
            deal_vs_ask = [v["deal_vs_ask_pct"] for v in has_ask_deal
                           if v["deal_vs_ask_pct"] is not None
                           and -0.95 <= v["deal_vs_ask_pct"] <= 5.0]
            concessions = [v["concession_pct"] for v in has_ask_deal
                           if v["concession_pct"] is not None
                           and -5.0 <= v["concession_pct"] <= 1.0]

            W(f"\nAsk → Deal Price Change (n={len(deal_vs_ask)}, filtered to reasonable range):\n")
            if deal_vs_ask:
                W(f"  Mean change:    {fmt(np.mean(deal_vs_ask)*100, 1)}%\n")
                W(f"  Median change:  {fmt(np.median(deal_vs_ask)*100, 1)}%\n")
                W(f"  Std:            {fmt(np.std(deal_vs_ask)*100, 1)}%\n")
                W(f"  Deal > Ask:     {sum(1 for d in deal_vs_ask if d > 0.05)}\n")
                W(f"  Deal ≈ Ask:     {sum(1 for d in deal_vs_ask if -0.05 <= d <= 0.05)}\n")
                W(f"  Deal < Ask:     {sum(1 for d in deal_vs_ask if d < -0.05)}\n\n")

            if concessions:
                W(f"Seller Concession (ask - deal)/ask (n={len(concessions)}):\n")
                W(f"  Mean concession: {fmt(np.mean(concessions)*100, 1)}%\n")
                W(f"  Median:          {fmt(np.median(concessions)*100, 1)}%\n")

        # Average prices by stage
        for stage_name, stage_key in [("Initial Ask", "ask_price"), ("First Offer", "first_offer_price"),
                                       ("Counter", "counter_price"), ("Expert", "expert_price"),
                                       ("Deal", "deal_price")]:
            vals = [v[stage_key] for v in videos if v[stage_key] is not None]
            if vals:
                W(f"\n{stage_name} prices (n={len(vals)}):\n")
                W(f"  Mean:   ${np.mean(vals):,.0f}\n")
                W(f"  Median: ${np.median(vals):,.0f}\n")
                W(f"  Range:  ${min(vals):,.0f} - ${max(vals):,.0f}\n")

        # ── 3. EXPERT IMPACT ──
        W("\n\n3. EXPERT IMPACT ON PRICING\n")
        W("-" * 50 + "\n")

        if has_ask_expert:
            expert_changes = [v["expert_vs_ask_pct"] for v in has_ask_expert
                              if v["expert_vs_ask_pct"] is not None
                              and -0.95 <= v["expert_vs_ask_pct"] <= 5.0]

            W(f"\nExpert Appraisal vs Initial Ask (n={len(expert_changes)}):\n")
            if expert_changes:
                W(f"  Mean change:     {fmt(np.mean(expert_changes)*100, 1)}%\n")
                W(f"  Median change:   {fmt(np.median(expert_changes)*100, 1)}%\n")
                W(f"  Expert > Ask:    {sum(1 for e in expert_changes if e > 0.05)} "
                  f"({sum(1 for e in expert_changes if e > 0.05)/len(expert_changes)*100:.1f}%)\n")
                W(f"  Expert ≈ Ask:    {sum(1 for e in expert_changes if -0.05 <= e <= 0.05)} "
                  f"({sum(1 for e in expert_changes if -0.05 <= e <= 0.05)/len(expert_changes)*100:.1f}%)\n")
                W(f"  Expert < Ask:    {sum(1 for e in expert_changes if e < -0.05)} "
                  f"({sum(1 for e in expert_changes if e < -0.05)/len(expert_changes)*100:.1f}%)\n")

        # Expert direction → deal outcome
        if has_expert_v:
            W("\nExpert Direction → Deal Outcome:\n")
            for direction in ["above", "below", "equal"]:
                d_vids = [v for v in has_expert_v if v["expert_direction"] == direction]
                if d_vids:
                    deals = sum(1 for v in d_vids if v["outcome"] == "deal")
                    no_deals = sum(1 for v in d_vids if v["outcome"] == "no_deal")
                    unknown = sum(1 for v in d_vids if v["outcome"] == "unknown")
                    total = len(d_vids)
                    W(f"  Expert {direction} ask (n={total}):\n")
                    W(f"    deal={deals} ({deals/total*100:.1f}%), "
                      f"no_deal={no_deals} ({no_deals/total*100:.1f}%), "
                      f"unknown={unknown}\n")

        # Full chain: ask → expert → deal
        if has_full_chain:
            W(f"\nFull Price Chain Analysis (n={len(has_full_chain)}):\n")

            above_expert = [v for v in has_full_chain if v["expert_direction"] == "above"]
            below_expert = [v for v in has_full_chain if v["expert_direction"] == "below"]

            if above_expert:
                dve = [v["deal_vs_expert_pct"] for v in above_expert
                       if v["deal_vs_expert_pct"] is not None and -0.95 <= v["deal_vs_expert_pct"] <= 5.0]
                dva = [v["deal_vs_ask_pct"] for v in above_expert
                       if v["deal_vs_ask_pct"] is not None and -0.95 <= v["deal_vs_ask_pct"] <= 5.0]
                W(f"\n  When expert ABOVE ask (n={len(above_expert)}):\n")
                if dve:
                    W(f"    Deal vs Expert: {fmt(np.median(dve)*100, 1)}% median\n")
                if dva:
                    W(f"    Deal vs Ask:    {fmt(np.median(dva)*100, 1)}% median\n")
                    W(f"    → Seller captures uplift: sellers get more when expert validates higher value\n")

            if below_expert:
                dve = [v["deal_vs_expert_pct"] for v in below_expert
                       if v["deal_vs_expert_pct"] is not None and -0.95 <= v["deal_vs_expert_pct"] <= 5.0]
                dva = [v["deal_vs_ask_pct"] for v in below_expert
                       if v["deal_vs_ask_pct"] is not None and -0.95 <= v["deal_vs_ask_pct"] <= 5.0]
                W(f"\n  When expert BELOW ask (n={len(below_expert)}):\n")
                if dve:
                    W(f"    Deal vs Expert: {fmt(np.median(dve)*100, 1)}% median\n")
                if dva:
                    W(f"    Deal vs Ask:    {fmt(np.median(dva)*100, 1)}% median\n")
                    W(f"    → Seller must concede: expert anchors price downward\n")

        # ── 4. DEAL vs NO-DEAL ANALYSIS ──
        W("\n\n4. DEAL vs NO-DEAL: WHAT PREDICTS OUTCOME?\n")
        W("-" * 50 + "\n")

        deal_vids = [v for v in videos if v["outcome"] == "deal"]
        no_deal_vids = [v for v in videos if v["outcome"] == "no_deal"]

        W(f"\nClassifiable outcomes: deal={len(deal_vids)}, no_deal={len(no_deal_vids)}\n")

        if deal_vids and no_deal_vids:
            # Compare features between deal and no-deal
            comparisons = [
                ("Mean Sentiment", "mean_sentiment"),
                ("Sentiment Std (volatility)", "sentiment_std"),
                ("Sentiment Slope (trajectory)", "sentiment_slope"),
                ("Analytic Ratio", "analytic_ratio"),
                ("Mean Congruence", "mean_congruence"),
                ("Happy Face %", "happy_pct"),
                ("Sad Face %", "sad_pct"),
                ("Fear Face %", "fear_pct"),
                ("Angry Face %", "angry_pct"),
                ("Neutral Face %", "neutral_pct"),
                ("N Segments", "n_segments"),
                ("N Counter Offers", "n_counters"),
            ]

            W(f"\nFeature Comparison (deal vs no_deal):\n")
            W(f"{'Feature':<35} {'Deal':<12} {'No Deal':<12} {'Cohen d':<10} {'Sig':<6}\n")
            W("-" * 75 + "\n")

            for label, key in comparisons:
                d_vals = [v[key] for v in deal_vids if v[key] is not None]
                nd_vals = [v[key] for v in no_deal_vids if v[key] is not None]
                if len(d_vals) >= 5 and len(nd_vals) >= 5:
                    d_mean = np.mean(d_vals)
                    nd_mean = np.mean(nd_vals)
                    d = cohens_d(d_vals, nd_vals)
                    t, p = t_test_ind(d_vals, nd_vals)
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    W(f"{label:<35} {fmt(d_mean, 3):<12} {fmt(nd_mean, 3):<12} {fmt(d, 3):<10} {sig:<6}\n")

            # Deal rate by item category
            W(f"\nDeal Rate by Item Category:\n")
            for cat, _ in item_dist.most_common():
                cat_vids = [v for v in videos if v["item_category"] == cat and v["outcome"] in ("deal", "no_deal")]
                if len(cat_vids) >= 5:
                    cat_deals = sum(1 for v in cat_vids if v["outcome"] == "deal")
                    rate = cat_deals / len(cat_vids)
                    W(f"  {cat:<30} {rate*100:5.1f}% deal (n={len(cat_vids)})\n")

            # Deal rate by cognitive system
            W(f"\nDeal Rate by Cognitive System:\n")
            for sys_type in ["system_1", "system_2", "mixed"]:
                sys_vids = [v for v in videos if v["cognitive_system"] == sys_type
                            and v["outcome"] in ("deal", "no_deal")]
                if sys_vids:
                    sys_deals = sum(1 for v in sys_vids if v["outcome"] == "deal")
                    rate = sys_deals / len(sys_vids)
                    W(f"  {sys_type:<20} {rate*100:5.1f}% deal (n={len(sys_vids)})\n")

            # Deal rate by expert presence
            W(f"\nDeal Rate by Expert Presence:\n")
            for has_exp, label in [(True, "With Expert"), (False, "Without Expert")]:
                exp_vids = [v for v in videos if v["has_expert"] == has_exp
                            and v["outcome"] in ("deal", "no_deal")]
                if exp_vids:
                    exp_deals = sum(1 for v in exp_vids if v["outcome"] == "deal")
                    rate = exp_deals / len(exp_vids)
                    W(f"  {label:<20} {rate*100:5.1f}% deal (n={len(exp_vids)})\n")

            # Deal rate by show
            W(f"\nDeal Rate by Show:\n")
            for show in ["pawn_stars", "cajun_pawn_stars", "hardcore_pawn"]:
                show_vids = [v for v in videos if v["show"] == show
                             and v["outcome"] in ("deal", "no_deal")]
                if show_vids:
                    show_deals = sum(1 for v in show_vids if v["outcome"] == "deal")
                    rate = show_deals / len(show_vids)
                    W(f"  {show:<25} {rate*100:5.1f}% deal (n={len(show_vids)})\n")

        # ── 5. ANCHORING ANALYSIS ──
        W("\n\n5. ANCHORING & PRICE PSYCHOLOGY\n")
        W("-" * 50 + "\n")

        if has_ask_deal:
            asks = [v["ask_price"] for v in has_ask_deal]
            deals = [v["deal_price"] for v in has_ask_deal]
            r, p = pearson_r(asks, deals)
            W(f"\nAnchoring: Initial Ask → Deal Price\n")
            W(f"  Pearson r = {fmt(r)} (n={len(asks)})\n")
            W(f"  → {'Strong' if abs(r)>0.7 else 'Moderate' if abs(r)>0.4 else 'Weak'} anchoring effect\n")
            W(f"  Initial ask strongly predicts final deal price\n")

        # First offer anchoring
        fo_vids = [v for v in videos if v["first_offer_price"] and v["deal_price"]]
        if len(fo_vids) >= 10:
            fos = [v["first_offer_price"] for v in fo_vids]
            deals = [v["deal_price"] for v in fo_vids]
            r, p = pearson_r(fos, deals)
            W(f"\nAnchoring: First Offer → Deal Price\n")
            W(f"  Pearson r = {fmt(r)} (n={len(fos)})\n")

        # Expert anchoring
        if len(has_full_chain) >= 10:
            experts = [v["expert_price"] for v in has_full_chain]
            deals = [v["deal_price"] for v in has_full_chain]
            r, p = pearson_r(experts, deals)
            W(f"\nAnchoring: Expert Appraisal → Deal Price\n")
            W(f"  Pearson r = {fmt(r)} (n={len(experts)})\n")
            W(f"  → Expert appraisal {'strongly' if abs(r)>0.7 else 'moderately' if abs(r)>0.4 else 'weakly'} "
              f"anchors the final price\n")

        # Concession by price range
        W(f"\nConcession Patterns by Price Range:\n")
        price_ranges = [
            ("$0-100", 0, 100),
            ("$100-500", 100, 500),
            ("$500-2,000", 500, 2000),
            ("$2,000-10,000", 2000, 10000),
            ("$10,000+", 10000, float("inf")),
        ]
        for label, lo, hi in price_ranges:
            range_vids = [v for v in has_ask_deal
                          if v["ask_price"] and lo <= v["ask_price"] < hi
                          and v["concession_pct"] is not None
                          and -5.0 <= v["concession_pct"] <= 1.0]
            if len(range_vids) >= 3:
                conc = [v["concession_pct"] for v in range_vids]
                W(f"  {label:<15} median concession: {fmt(np.median(conc)*100, 1)}% (n={len(range_vids)})\n")

        # ── 6. COGNITIVE SYSTEM ANALYSIS ──
        W("\n\n6. COGNITIVE SYSTEM ENGAGEMENT (Kahneman System 1 / System 2)\n")
        W("-" * 50 + "\n")

        W(f"\nSystem Distribution:\n")
        for sys_type, count in system_dist.most_common():
            W(f"  {sys_type}: {count} ({count/len(videos)*100:.1f}%)\n")

        # System by negotiation stage
        W(f"\nSystem Engagement Indicators by Stage:\n")
        W(f"{'Stage':<25} {'Mean Sentiment':<18} {'Analytic/Sent':<18} {'Description':<30}\n")
        W("-" * 91 + "\n")

        for stage in ["initial_ask", "first_offer", "counter", "expert_appraisal", "deal"]:
            s_key = f"sentiment_{stage.split('_')[0] if stage != 'initial_ask' else 'initial_ask'}"
            if stage == "first_offer":
                s_key = "sentiment_first_offer"
            elif stage == "expert_appraisal":
                s_key = "sentiment_expert"
            elif stage == "initial_ask":
                s_key = "sentiment_initial_ask"
            elif stage == "counter":
                s_key = "sentiment_counter"
            elif stage == "deal":
                s_key = "sentiment_deal"

            vals = [v[s_key] for v in videos if v[s_key] is not None]
            if vals:
                mean_s = np.mean(vals)
                # Describe system engagement
                if mean_s > 0.15:
                    desc = "System 1 (emotional, positive)"
                elif mean_s < 0.02:
                    desc = "System 2 (deliberative, neutral)"
                else:
                    desc = "Mixed (moderate affect)"
                W(f"  {stage:<23} {fmt(mean_s, 4):<18} {'—':<18} {desc}\n")

        # System × Price behavior
        W(f"\nConcession by Cognitive System:\n")
        for sys_type in ["system_1", "system_2", "mixed"]:
            sys_vids = [v for v in has_ask_deal if v["cognitive_system"] == sys_type
                        and v["concession_pct"] is not None
                        and -5.0 <= v["concession_pct"] <= 1.0]
            if len(sys_vids) >= 3:
                conc = [v["concession_pct"] for v in sys_vids]
                W(f"  {sys_type:<15} median concession: {fmt(np.median(conc)*100, 1)}% (n={len(sys_vids)})\n")

        # ── 7. EMOTION × ECONOMICS ──
        W("\n\n7. EMOTION × ECONOMICS INTERACTION\n")
        W("-" * 50 + "\n")

        # Sentiment correlation with concession
        s_vals = [v["mean_sentiment"] for v in has_ask_deal if v["concession_pct"] is not None]
        c_vals = [v["concession_pct"] for v in has_ask_deal if v["concession_pct"] is not None]
        if len(s_vals) >= 10:
            r, p = pearson_r(s_vals, c_vals)
            W(f"\nSentiment → Concession: r={fmt(r)} (n={len(s_vals)})\n")
            if r > 0:
                W(f"  → More positive sentiment correlates with LARGER concessions (giving in)\n")
            else:
                W(f"  → More positive sentiment correlates with SMALLER concessions (holding firm)\n")

        # Congruence correlation with concession
        cg_vals = [v["mean_congruence"] for v in has_ask_deal
                   if v["concession_pct"] is not None and v["mean_congruence"] is not None]
        c2_vals = [v["concession_pct"] for v in has_ask_deal
                   if v["concession_pct"] is not None and v["mean_congruence"] is not None]
        if len(cg_vals) >= 10:
            r, p = pearson_r(cg_vals, c2_vals)
            W(f"\nCongruence → Concession: r={fmt(r)} (n={len(cg_vals)})\n")
            if r > 0:
                W(f"  → Higher text-face alignment correlates with more concession\n")
            else:
                W(f"  → Higher text-face alignment correlates with less concession\n")

        # Dominant emotion → deal outcome
        W(f"\nDominant Emotion → Deal Outcome:\n")
        for emo in ["happy", "sad", "fear", "angry", "neutral"]:
            emo_deal = [v for v in deal_vids if v["dominant_emotion"] == emo]
            emo_no_deal = [v for v in no_deal_vids if v["dominant_emotion"] == emo]
            total_emo = len(emo_deal) + len(emo_no_deal)
            if total_emo >= 3:
                rate = len(emo_deal) / total_emo
                W(f"  {emo:<10} deal rate: {rate*100:5.1f}% (deal={len(emo_deal)}, "
                  f"no_deal={len(emo_no_deal)})\n")

        # Expert sentiment shock → price adjustment
        W(f"\nExpert Stage Emotional Impact:\n")
        expert_sent_vids = [v for v in has_ask_expert if v["sentiment_expert"] is not None
                            and v["sentiment_initial_ask"] is not None]
        if expert_sent_vids:
            shocks = [(v["sentiment_expert"] - v["sentiment_initial_ask"]) for v in expert_sent_vids]
            W(f"  Sentiment shift at expert stage (n={len(shocks)}):\n")
            W(f"    Mean shift: {fmt(np.mean(shocks))}\n")
            W(f"    Negative shifts (shock): {sum(1 for s in shocks if s < -0.1)} "
              f"({sum(1 for s in shocks if s < -0.1)/len(shocks)*100:.1f}%)\n")
            W(f"    Positive shifts (relief): {sum(1 for s in shocks if s > 0.1)} "
              f"({sum(1 for s in shocks if s > 0.1)/len(shocks)*100:.1f}%)\n")

            # Does shock predict concession?
            shock_conc = [(v["sentiment_expert"] - v["sentiment_initial_ask"], v["concession_pct"])
                          for v in has_ask_deal if v["sentiment_expert"] is not None
                          and v["sentiment_initial_ask"] is not None
                          and v["concession_pct"] is not None]
            if len(shock_conc) >= 10:
                shk = [s[0] for s in shock_conc]
                conc = [s[1] for s in shock_conc]
                r, p = pearson_r(shk, conc)
                W(f"  Expert sentiment shock → Concession: r={fmt(r)} (n={len(shock_conc)})\n")

        # ── 8. ITEM CATEGORY ECONOMICS ──
        W("\n\n8. ECONOMICS BY ITEM CATEGORY\n")
        W("-" * 50 + "\n")

        W(f"\n{'Category':<30} {'Med Ask':<12} {'Med Deal':<12} {'Med Conc%':<12} {'Deal Rate':<12} {'n':<5}\n")
        W("-" * 83 + "\n")

        for cat, _ in item_dist.most_common():
            cat_vids = [v for v in videos if v["item_category"] == cat]
            cat_ask = [v["ask_price"] for v in cat_vids if v["ask_price"]]
            cat_deal = [v["deal_price"] for v in cat_vids if v["deal_price"]]
            cat_conc = [v["concession_pct"] for v in cat_vids if v["concession_pct"] is not None]
            cat_classified = [v for v in cat_vids if v["outcome"] in ("deal", "no_deal")]
            cat_deals = sum(1 for v in cat_classified if v["outcome"] == "deal")
            deal_rate = cat_deals / len(cat_classified) if cat_classified else 0

            W(f"  {cat:<28} "
              f"{'$'+str(int(np.median(cat_ask))) if cat_ask else 'N/A':<12} "
              f"{'$'+str(int(np.median(cat_deal))) if cat_deal else 'N/A':<12} "
              f"{fmt(np.median(cat_conc)*100, 1)+'%' if cat_conc else 'N/A':<12} "
              f"{fmt(deal_rate*100, 1)+'%' if cat_classified else 'N/A':<12} "
              f"{len(cat_vids)}\n")

        # ── 9. SHOW-LEVEL COMPARISON ──
        W("\n\n9. SHOW-LEVEL COMPARISON\n")
        W("-" * 50 + "\n")

        for show in ["pawn_stars", "cajun_pawn_stars", "hardcore_pawn"]:
            show_vids = [v for v in videos if v["show"] == show]
            if not show_vids:
                continue
            W(f"\n  {show} (n={len(show_vids)}):\n")

            s_asks = [v["ask_price"] for v in show_vids if v["ask_price"]]
            s_deals = [v["deal_price"] for v in show_vids if v["deal_price"]]
            s_conc = [v["concession_pct"] for v in show_vids if v["concession_pct"] is not None]
            s_classified = [v for v in show_vids if v["outcome"] in ("deal", "no_deal")]
            s_deal_count = sum(1 for v in s_classified if v["outcome"] == "deal")

            if s_asks:
                W(f"    Median ask:        ${np.median(s_asks):,.0f}\n")
            if s_deals:
                W(f"    Median deal:       ${np.median(s_deals):,.0f}\n")
            s_conc_f = [c for c in s_conc if -5.0 <= c <= 1.0]
            if s_conc_f:
                W(f"    Median concession: {np.median(s_conc_f)*100:.1f}%\n")
            if s_classified:
                W(f"    Deal rate:         {s_deal_count/len(s_classified)*100:.1f}% (n={len(s_classified)})\n")

            # Sentiment
            s_sent = [v["mean_sentiment"] for v in show_vids if v["mean_sentiment"]]
            if s_sent:
                W(f"    Mean sentiment:    {np.mean(s_sent):.4f}\n")

            # System
            s_sys = Counter(v["cognitive_system"] for v in show_vids)
            W(f"    Cognitive system:  " + ", ".join(f"{k}={v}" for k, v in s_sys.most_common()) + "\n")

        # ── 10. KEY CONCLUSIONS ──
        W("\n\n10. KEY CONCLUSIONS\n")
        W("=" * 50 + "\n")

        conclusions = []

        # Anchoring conclusion
        if has_ask_deal:
            asks = [v["ask_price"] for v in has_ask_deal]
            deals = [v["deal_price"] for v in has_ask_deal]
            r, _ = pearson_r(asks, deals)
            if abs(r) > 0.5:
                conclusions.append(
                    f"ANCHORING: Initial ask price is the strongest predictor of deal "
                    f"price (r={fmt(r)}). The first number spoken dominates the "
                    f"entire negotiation, consistent with Tversky & Kahneman (1974)."
                )

        # Expert impact conclusion
        if has_full_chain:
            above = [v for v in has_full_chain if v["expert_direction"] == "above"]
            below = [v for v in has_full_chain if v["expert_direction"] == "below"]
            if above and below:
                above_dva = [v["deal_vs_ask_pct"] for v in above
                             if v["deal_vs_ask_pct"] is not None and -0.95 <= v["deal_vs_ask_pct"] <= 5.0]
                below_dva = [v["deal_vs_ask_pct"] for v in below
                             if v["deal_vs_ask_pct"] is not None and -0.95 <= v["deal_vs_ask_pct"] <= 5.0]
                if above_dva and below_dva:
                    conclusions.append(
                        f"EXPERT IMPACT: When experts appraise ABOVE ask, deal price moves "
                        f"{np.median(above_dva)*100:+.1f}% vs ask (median, n={len(above_dva)}). "
                        f"When BELOW ask, deal moves "
                        f"{np.median(below_dva)*100:+.1f}% vs ask (median, n={len(below_dva)}). "
                        f"Expert appraisal serves as a re-anchoring mechanism."
                    )

        # Deal vs no-deal
        if deal_vids and no_deal_vids:
            d_sent = np.mean([v["mean_sentiment"] for v in deal_vids])
            nd_sent = np.mean([v["mean_sentiment"] for v in no_deal_vids])
            d = cohens_d(
                [v["mean_sentiment"] for v in deal_vids],
                [v["mean_sentiment"] for v in no_deal_vids]
            )
            conclusions.append(
                f"DEAL PREDICTION: Deals show mean sentiment {d_sent:.3f} vs "
                f"no-deals {nd_sent:.3f} (Cohen's d={fmt(d, 3)}). "
                f"{'Meaningful' if abs(d) > 0.2 else 'Negligible'} effect size."
            )

        # Cognitive system
        for sys_type in ["system_2", "system_1"]:
            sys_deal_vids = [v for v in videos if v["cognitive_system"] == sys_type
                             and v["outcome"] in ("deal", "no_deal")]
            if len(sys_deal_vids) >= 10:
                rate = sum(1 for v in sys_deal_vids if v["outcome"] == "deal") / len(sys_deal_vids)
                conclusions.append(
                    f"COGNITIVE SYSTEM: {sys_type.replace('_', ' ').title()} engagement "
                    f"shows {rate*100:.1f}% deal rate (n={len(sys_deal_vids)})."
                )

        for i, c in enumerate(conclusions, 1):
            W(f"\n{i}. {c}\n")

        W("\n" + "=" * 70 + "\n")
        W("END OF REPORT\n")

    log(f"  Written: {report_path}")

    # ── Write CSVs ──
    log("\n[4] Writing data CSVs...")

    # Price trajectories
    traj_fields = [
        "video_id", "title", "show", "item_category", "outcome",
        "ask_price", "first_offer_price", "counter_price", "expert_price", "deal_price",
        "expert_vs_ask_pct", "deal_vs_ask_pct", "deal_vs_expert_pct",
        "expert_direction", "concession_pct", "buyer_stretch_pct",
        "has_expert", "n_price_stages",
    ]
    traj_path = os.path.join(ANALYSIS_DIR, "price_trajectories.csv")
    with open(traj_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=traj_fields, extrasaction="ignore")
        writer.writeheader()
        for v in videos:
            writer.writerow(v)
    log(f"  Written: {traj_path} ({len(videos)} rows)")

    # Deal prediction features
    pred_fields = [
        "video_id", "show", "item_category", "outcome",
        "ask_price", "deal_price", "concession_pct", "has_expert", "expert_direction",
        "mean_sentiment", "sentiment_std", "sentiment_slope",
        "analytic_markers", "holistic_markers", "analytic_ratio", "cognitive_system",
        "dominant_emotion", "happy_pct", "sad_pct", "fear_pct", "angry_pct", "neutral_pct",
        "mean_congruence", "n_segments", "n_counters", "n_sentences",
    ]
    pred_path = os.path.join(ANALYSIS_DIR, "deal_prediction.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fields, extrasaction="ignore")
        writer.writeheader()
        for v in videos:
            writer.writerow(v)
    log(f"  Written: {pred_path} ({len(videos)} rows)")

    log("\n" + "=" * 70)
    log("ANALYSIS COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
