# AudioBuddy

AudioBuddy is a small local Python CLI tool that summarizes one MP3 audio file
with the Google Gemini API.

## Phase 1

Phase 1 supports a single local MP3 file at a time.

The script:

- reads `GEMINI_API_KEY` from the environment or a local `.env` file
- rejects missing files and non-MP3 inputs
- parses the date and time from filenames like `2026-08-30 11_10_13.mp3`
- reads the original audio duration with `ffprobe`
- skips silence trimming for files that are 10 minutes or shorter
- trims long silence with FFmpeg for files longer than 10 minutes
- uploads the audio with the Google GenAI Python SDK
- asks a current Gemini Flash model for an English Markdown summary using `prompt.md`
- writes the summary to `D:\Documents\Personal\audio\out`

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create a local `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Do not commit `.env`. It is ignored by Git.

## Usage

Run the script with one MP3 file path:

```powershell
python summarize_audio.py "D:\Documents\Personal\audio\in\2026-08-30 11_10_13.mp3"
```

Input filenames must match this format:

```text
YYYY-MM-DD HH_MM_SS.mp3
```

For example, `2026-08-30 11_10_13.mp3` is treated as:

```text
Date: 30 August 2026
Time: 11:10:13
```

## Output

Summaries are written to:

```text
D:\Documents\Personal\audio\out
```

The output filename is based on the source filename:

```text
2026-08-30 11_10_13-summary.md
```

If that file already exists, AudioBuddy creates a numbered alternative:

```text
2026-08-30 11_10_13-summary-1.md
```

Phase 2-2 also maintains a master summary file:

```text
D:\Documents\Personal\audio\out\master-summary.md
```

Master entries use hidden Markdown markers so entries can be replaced safely:

```markdown
<!-- AUDIOBUDDY_ENTRY_START filename="2026-08-30 11_10_13.mp3" -->
<!-- AUDIOBUDDY_ENTRY_END -->
```

Each master entry includes the source audio path, the newest individual summary
path, the Gemini model, audio processing metrics, and the generated summary.

## Summary Format

The Gemini prompt lives in `prompt.md`. Treat that file as the authoritative
source for the prompt and expected output structure.

The generated Markdown uses this fixed heading structure:

```markdown
# Audio Summary

## Date and Time

## Audio Processing

## Executive Summary

## Verbose Summary

## Key Discussion Points

## Decisions

## Action Items

## Open Questions
```

Unknown action item owners are labeled `unknown`.

## Common Errors

- Missing API key: set `GEMINI_API_KEY` in the environment or `.env`.
- Nonexistent input file: check the file path.
- Unsupported file: Phase 1 accepts only `.mp3` files.
- Missing FFmpeg: install FFmpeg and ensure `ffmpeg` and `ffprobe` are on PATH.
- Gemini API failure: retry later or check quota, billing, and API-key access.

## Silence Trimming

For files longer than 10 minutes, AudioBuddy uses FFmpeg to remove long silence
before upload. The Phase 2-1 default is conservative: silence must last at least
30 seconds and be at or below `-30dB`.

The original audio file is never modified.
Temporary trimmed files are deleted after processing completes or fails.

## Phase 2 Ideas

These are intentionally not implemented in Phase 1:

- Split longer audio into smaller chunks, for example 30-minute segments.
- Scan the input directory and summarize only new audio files.
- Track filenames, timestamps, and summaries in one master file.
