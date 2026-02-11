# YouTube Transcript Batch Processor - Pipeline Tracker

## Project Goal
Multimodal negotiation analysis of pawn shop TV shows (Pawn Stars, Cajun Pawn Stars, Hardcore Pawn).
- **Kahneman behavioral economics**: loss aversion, anchoring, prospect theory
- **Nisbett language patterns**: analytic vs holistic reasoning
- **Research question**: Role of dollar amount and emotions related to expert's price change and deal outcome

## Data Persistence
- **Primary storage**: `/root/.claude/projects/-workspaces-youtube-transcript-batch/data/` (host-mounted, survives Docker rebuilds)
- **Workspace symlinks**: `results/`, `analysis/`, `key_frames/`, `master_dataset/` → persistent storage
- **GitHub**: https://github.com/skro-wq/yt.git (scripts only, needs PAT with `repo` scope for push)

## Pipeline Status

| Step | Script | Status | Output |
|------|--------|--------|--------|
| 0 | collect_urls.py | COMPLETE | 3,285 URLs (2360 PS + 554 Cajun + 471 HC) |
| 1 | process_all_batches.py | IN PROGRESS | ~543/3285 transcripts (rate-limit cycles needed) |
| 2 | merge_results.py | PENDING | → master_dataset/all_transcripts.json |
| 3 | analyze_negotiations.py | PENDING | → results/negotiations_analysis.json |
| 4 | analyze_text_sentiment.py | PENDING | → analysis/text_sentiment_per_sentence.csv |
| 5 | extract_key_frames.py | PENDING | → key_frames/{video_id}/*.jpg + manifest.json |
| 6 | analyze_sentiment.py | PENDING | → analysis/sentiment_per_frame.csv |
| 7 | analyze_multimodal.py | PENDING | → analysis/multimodal_analysis.csv + reports |

## Current Progress (Step 1 - Transcript Download)
- pawn_stars: 190 completed, 2170 remaining
- cajun_pawn_stars: 196 completed, 358 remaining
- hardcore_pawn: 157 completed, 314 remaining
- Rate limiting: YouTube blocks after ~150-200 videos per run with parallel workers
- Strategy: Run with --workers 2, re-run after rate limit clears

## Environment
- Python 3.11.2 (Debian bookworm), venv at `./venv/`
- Key packages: yt-dlp, youtube_transcript_api, deepface, tf-keras, nltk, matplotlib, sklearn
- ffmpeg installed via apt
- NLTK VADER lexicon for text sentiment

## Scripts (all in project root)
1. `collect_urls.py` - YouTube URL collection via yt-dlp search
2. `process_all_batches.py` - Parallel transcript download with rate-limit detection
3. `merge_results.py` - Combine batch results into master dataset
4. `analyze_negotiations.py` - Extract negotiation segments from transcripts
5. `analyze_text_sentiment.py` - VADER + Kahneman + Nisbett text analysis
6. `extract_key_frames.py` - Parallel video frame extraction at key moments
7. `analyze_sentiment.py` - DeepFace facial emotion analysis on frames
8. `analyze_multimodal.py` - Combine text + facial sentiment, generate reports

## Resume Instructions
After rate limit: `venv/bin/python3 process_all_batches.py <csv> --output-dir /root/.claude/.../data/results --workers 2`
Scripts are incremental - they skip already-processed items via progress.json/manifest.json.
