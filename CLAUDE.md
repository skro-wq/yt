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

## Pipeline Status

| Step | Script | Status | Output |
|------|--------|--------|--------|
| 0 | collect_urls.py | COMPLETE | 3,285 URLs (2360 PS + 554 Cajun + 471 HC) |
| 1 | process_all_batches.py | DONE (partial) | 803/3285 transcripts downloaded |
| 2 | merge_results.py | COMPLETE | 877 unique videos (772 with transcripts) |
| 3 | analyze_negotiations.py | COMPLETE | 21,604 segments from 719 videos |
| 4 | analyze_text_sentiment.py | IN PROGRESS | Running... |
| 5 | extract_key_frames.py | IN PROGRESS | Running (4 workers)... |
| 6 | analyze_sentiment.py | PENDING | → analysis/sentiment_per_frame.csv |
| 7 | analyze_multimodal.py | PENDING | → analysis/multimodal_analysis.csv + reports |

## Negotiation Analysis Results
- 719 videos with negotiation content (target: 500+)
- 21,604 total segments (9,557 initial_ask, 5,952 deal, 2,242 expert, 2,036 counter, 1,817 first_offer)
- 2,700 segments with dollar amounts
- By show: pawn_stars 14,178, hardcore_pawn 6,017, cajun_pawn_stars 1,409

## Environment
- Python 3.11.2 (Debian bookworm), venv at `./venv/`
- Key packages: yt-dlp, youtube_transcript_api, deepface, tf-keras, nltk, matplotlib, sklearn
- ffmpeg installed via apt
- After Docker rebuild: recreate venv + reinstall packages + restore symlinks

## Resume Instructions
- All scripts are incremental (progress.json / manifest.json)
- Transcript download: `venv/bin/python3 process_all_batches.py <csv> --output-dir /root/.claude/.../data/results --workers 4`
- Key frames: `venv/bin/python3 extract_key_frames.py --workers 4`
