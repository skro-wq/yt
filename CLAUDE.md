# YouTube Transcript Batch Processor - Pipeline Tracker

## Project Goal
Multimodal negotiation analysis of pawn shop TV shows: **role of dollar amounts and emotions related to expert's price change and deal outcome**.
- **Research question**: How do expert appraisals shift prices, and what psychological/behavioral markers predict deal vs no-deal?
- **Kahneman behavioral economics**: anchoring, loss aversion, prospect theory, System 1/2
- **Nisbett language patterns**: analytic vs holistic reasoning
- **Shows**: Pawn Stars, Cajun Pawn Stars, Hardcore Pawn

## Data Persistence
- **Primary storage**: `/root/.claude/projects/-workspaces-youtube-transcript-batch/data/` (host-mounted, survives Docker rebuilds)
- **Workspace symlinks**: `results/`, `analysis/`, `key_frames/`, `master_dataset/` → persistent storage
- **GitHub**: https://github.com/skro-wq/yt.git
- **CRITICAL**: Docker overlay loses data on rebuild. ALWAYS store data in persistent path.

## Pipeline Status - ALL COMPLETE

| Step | Script | Status | Output |
|------|--------|--------|--------|
| 0 | collect_urls.py | COMPLETE | 3,285 URLs |
| 1 | process_all_batches.py | COMPLETE | 803 transcripts |
| 2 | merge_results.py | COMPLETE | 877 unique videos |
| 3 | analyze_negotiations.py | COMPLETE | 21,604 segments from 719 videos |
| 4 | analyze_text_sentiment.py | COMPLETE | 29,514 sentences (VADER + Kahneman + Nisbett) |
| 5 | extract_key_frames.py | COMPLETE | 18,488 frames from 599 videos |
| 6 | analyze_sentiment.py | COMPLETE | 18,373 facial frames (DeepFace) |
| 7 | **analyze_economics.py** | COMPLETE | Economics report + deal prediction |

## Key Findings (from analyze_economics.py)

### 1. Anchoring (Strongest Effect)
- Initial ask → deal price: **r = 0.823** (strong anchoring)
- Expert appraisal → deal price: **r = 0.947** (strongest single predictor)
- First offer → deal price: **r = 0.848**
- The first number spoken dominates the entire negotiation (Tversky & Kahneman, 1974)

### 2. Expert Impact on Pricing
- Experts appraise BELOW seller's ask **60.7%** of the time
- When expert is below ask → deal closes at **-50% vs ask** (median)
- When expert is above ask → deal closes at **+43% vs ask** (median)
- Expert presence correlates with higher deal rate: **69.7% vs 46.9%**
- Expert sentiment shock: mean shift = **-0.091** (48% negative, only 8% positive)

### 3. Deal vs No-Deal Predictors
- **Sad face %**: strongest discriminator (d = -0.383) — no-deals have more sadness
- **Happy face %**: deals show less happiness (d = -0.305) — counterintuitive
- **Expert presence**: +22.8% deal rate when expert is involved
- **System 2 engagement**: 65.9% deal rate vs System 1's 47.4%
- **Counter offers**: more counters correlate with no-deal (d = -0.196)
- Text sentiment: negligible predictor (d = 0.094)

### 4. Seller Concession
- Median concession: **50%** from asking price
- By price range: $100-500 items → 77% concession; $10K+ → 60%
- System 2 negotiators: 60% median concession; Mixed: 50%

### 5. Show-Level Differences
- **Pawn Stars**: highest asks ($4K median), 62.4% deal rate, System 2 dominant
- **Hardcore Pawn**: lowest asks ($500 median), 34.5% deal rate, Mixed system dominant
- **Cajun Pawn Stars**: mid-range, 62.3% deal rate, highest System 1 presence

### 6. Item Category Economics
- Vehicles: 81.8% deal rate (highest)
- Weapons/military: 61.5% deal rate, deal > ask (weapons hold value)
- Jewelry: 36.0% deal rate (lowest classifiable), highest concession
- Books: 100% deal rate (small n=5) but deal often exceeds ask

## Scripts (active)
| Script | Purpose |
|--------|---------|
| `collect_urls.py` | YouTube URL collection via yt-dlp |
| `process_all_batches.py` | Parallel transcript download |
| `merge_results.py` | Combine batch results into master dataset |
| `analyze_negotiations.py` | Extract negotiation segments + dollar amounts |
| `analyze_text_sentiment.py` | VADER sentiment + Kahneman/Nisbett markers |
| `extract_key_frames.py` | Extract video frames at negotiation moments |
| `analyze_sentiment.py` | DeepFace facial emotion on frames |
| `analyze_economics.py` | **Core analysis**: price trajectories, expert impact, deal prediction |

Archived: `archive/analyze_multimodal.py` (superseded by analyze_economics.py)
Preliminary outputs: `analysis/preliminary/` (early-stage reports before economics analysis)

## Environment
- Python 3.11.2 (Debian bookworm), venv at `./venv/`
- Key packages: yt-dlp, youtube_transcript_api, deepface, tf-keras, nltk, matplotlib, sklearn
- ffmpeg installed via apt
- After Docker rebuild: recreate venv + reinstall packages + restore symlinks

## Resume Instructions
- All scripts are incremental (progress.json / manifest.json)
- Full re-run: `venv/bin/python3 analyze_economics.py`
