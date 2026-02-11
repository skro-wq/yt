# YouTube Transcript Batch Processor - Pipeline Tracker

## Project Goal
Multimodal negotiation analysis of pawn shop TV shows using **500+ quality videos with actual negotiation content**.
- **Target**: Videos with actual buy/sell negotiations (initial ask, offer, counter, expert appraisal, deal)
- **NOT**: Short-form clips, compilations without negotiations, unrelated content
- **Kahneman behavioral economics**: loss aversion, anchoring, prospect theory
- **Nisbett language patterns**: analytic vs holistic reasoning
- **Research question**: Role of dollar amount and emotions related to expert's price change and deal outcome

## Quality Requirements
- Need 500+ videos with negotiation segments (have 719 currently)
- Each video must have: transcript, negotiation keywords, dollar amounts
- Key frames extracted at negotiation moments for facial sentiment
- Multimodal analysis combines text sentiment + facial emotion at matched timestamps

## Data Persistence
- **Primary storage**: `/root/.claude/projects/-workspaces-youtube-transcript-batch/data/` (host-mounted, survives Docker rebuilds)
- **Workspace symlinks**: `results/`, `analysis/`, `key_frames/`, `master_dataset/` → persistent storage
- **GitHub**: https://github.com/skro-wq/yt.git
- **CRITICAL**: Docker overlay loses data on rebuild. ALWAYS store data in persistent path.

## Pipeline Status - ALL COMPLETE

| Step | Script | Status | Output |
|------|--------|--------|--------|
| 0 | collect_urls.py | COMPLETE | 3,285 URLs (2360 PS + 554 Cajun + 471 HC) |
| 1 | process_all_batches.py | COMPLETE | 803/3285 transcripts downloaded |
| 2 | merge_results.py | COMPLETE | 877 unique videos (772 with transcripts) |
| 3 | analyze_negotiations.py | COMPLETE | 21,604 segments from 719 videos |
| 4 | analyze_text_sentiment.py | COMPLETE | 29,514 sentences from 719 videos |
| 5 | extract_key_frames.py | COMPLETE | 18,488 valid frames from 599 videos |
| 6 | analyze_sentiment.py | COMPLETE | 18,373 frames analyzed (99.4%), 3,109 negotiations |
| 7 | analyze_multimodal.py | COMPLETE | 22,780 matched pairs from 599 videos |

## Key Findings

### Behavioral Economics (Kahneman)
- **Loss aversion signal**: Loss frame congruence=0.6753 vs Gain frame=0.6760 (diff=0.0007)
- 46 loss frame videos, 17 gain frame videos, 536 neutral
- Text-face congruence increases through negotiation: initial_ask=0.666 → counter=0.712 → deal=0.699
- Face valence consistently negative across all stages (-0.20 to -0.28)
- Text sentiment consistently positive (0.04 to 0.18), highest at initial_ask

### Language Patterns (Nisbett)
- Analytic ratio: 0.901 (heavily analytic, as expected for business negotiation)
- Holistic language: lower congruence (0.633) than analytic (0.673) or neutral (0.691)
- Analytic markers: 7,256; Holistic markers: 795

### Facial Emotion Distribution
- sad: 30.1%, fear: 21.6%, neutral: 17.8%, happy: 15.2%, angry: 13.0%
- surprise: 1.9%, disgust: 0.4%

### Sentiment by Role
- Rejecting: most negative text sentiment (-0.109), highest congruence (0.714)
- Asking: most positive text sentiment (0.171)
- Countering: lowest text sentiment among non-rejecting (0.049), highest congruence (0.712)

## Environment
- Python 3.11.2 (Debian bookworm), venv at `./venv/`
- Key packages: yt-dlp, youtube_transcript_api, deepface, tf-keras, nltk, matplotlib, sklearn
- ffmpeg installed via apt
- After Docker rebuild: recreate venv + reinstall packages + restore symlinks

## Resume Instructions
- All scripts are incremental (progress.json / manifest.json)
- Transcript download: `venv/bin/python3 process_all_batches.py <csv> --output-dir /root/.claude/.../data/results --workers 4`
- Key frames: `venv/bin/python3 extract_key_frames.py --workers 8`
- Facial sentiment: `venv/bin/python3 analyze_sentiment.py --workers 4`
- Multimodal: `venv/bin/python3 analyze_multimodal.py`
