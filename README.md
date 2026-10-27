# AudioBuddy

AudioBuddy is a local Python CLI for turning timestamp-named MP3 and OGG
recordings into English meeting summaries or turn-by-turn transcripts with the
Google Gemini API.

It supports three summary workflows: process one named file, scan the configured
input folder for new recordings, or deliberately rescan the complete input
archive after changing `prompt.md`. It also has an independent, one-file
transcript command. Summary results are maintained both as individual Markdown
files and in a chronological master summary; transcript results stay separate.

## Summary Processing

For each summary, AudioBuddy:

- reads `GEMINI_API_KEY` from the environment or a local `.env` file
- rejects missing files and unsupported audio formats
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

Summarize one MP3 or OGG file explicitly, including an intentional reprocessing
of an existing file:

```powershell
python summarize_audio.py "D:\Documents\Personal\audio\in\2026-08-30 11_10_13.mp3"
```

To scan the top level of `D:\Documents\Personal\audio\in` and process only valid
files not already represented in the master summary:

```powershell
.venv\Scripts\python.exe summarize_audio.py --scan
```

To deliberately regenerate every valid top-level input file using the current
`prompt.md`, run a rescan:

```powershell
.venv\Scripts\python.exe summarize_audio.py --rescan
```

`--scan` processes selected files sequentially in chronological filename order.
It skips files already represented in the master, reports invalid inputs, and
continues past per-file failures. Transient Gemini failures in a scan retry up
to three times after 2, 4, and 8 seconds.

`--rescan` first reports its selection and asks for confirmation before starting
paid Gemini work. Only `y` or `yes` proceeds; every other response cancels
without processing. Use it after an intentional `prompt.md` change or whenever
you need fresh summaries for the whole input archive.

You can double-click `run-audiobuddy-scan.bat` to scan for new files, or
`run-audiobuddy-rescan.bat` to start the same confirmed rescan flow. Both leave
the command window open so the final result can be reviewed.

Input filenames must match this format:

```text
YYYY-MM-DD HH_MM_SS.mp3
YYYY-MM-DD HH_MM_SS.ogg
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

AudioBuddy also maintains a master summary file:

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
If the master file markers are malformed, AudioBuddy fails clearly rather than
guessing how to replace entries.

Entries are kept in chronological order using the timestamp in each source
filename. A newly processed latest recording is appended. Processing an older
recording, or retrying a source filename already present, rewrites the master
file in chronological order; the retried entry points to its newest individual
summary file.

Run the local master-summary checks without contacting Gemini:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_master_summary
```

## Summary Format

The Gemini prompt lives in `prompt.md`. Treat that file as the authoritative
source for the prompt and expected output structure.

The generated Markdown uses this fixed heading structure:

```markdown
# Audio Summary

## Date and Time

## Audio Processing

## Detailed Summary

## Discussion Path

## Decisions

## Action Items

## Open Questions
```

Unknown action item owners are labeled `unknown`.

## Transcripts

Use the separate transcription command for one recording when a complete,
turn-by-turn transcript is needed:

```powershell
.venv\Scripts\python.exe transcribe_audio.py "D:\Documents\Personal\audio\in\2026-08-30 11_10_13.mp3"
```

For a double-clickable option, run `run-audiobuddy-transcript.bat` and enter
only the audio filename from `D:\Documents\Personal\audio\in` when prompted.
For example, enter `2026-08-30 11_10_13.mp3`, not the full path.

It accepts the same timestamp-named MP3 and OGG inputs as the summary command,
uses the same Gemini reliability behavior, and writes an individual file such
as:

```text
D:\Documents\Personal\audio\out\2026-08-30 11_10_13-transcript.md
```

Existing transcript names receive a numeric suffix. Transcript runs do not add
to, replace, or rebuild `master-summary.md`.

Transcripts deliberately skip silence trimming so their timestamps remain tied
to the original recording timeline.

`transcript-prompt.md` is the authoritative source for the transcript prompt.
It instructs Gemini to return only timestamped turns, translate non-English
speech faithfully to English, use stable numbered speaker labels by default,
mark confident guesses in brackets, and mark wholly unintelligible speech as
`[inaudible]`.

## Common Errors

- Missing API key: set `GEMINI_API_KEY` in the environment or `.env`.
- Nonexistent input file: check the file path.
- Unsupported file: AudioBuddy accepts `.mp3` and `.ogg` files.
- Missing FFmpeg: install FFmpeg and ensure `ffmpeg` and `ffprobe` are on PATH.
- Gemini API failure: retry later or check quota, billing, and API-key access.

## Gemini Diagnostics

AudioBuddy writes timestamped Gemini status messages to the terminal for each
upload and model-generation attempt. Upload requests have a five-minute
deadline; summary and transcript generation requests have ten-minute deadlines.
Transcript generation uses Gemini's Generate Content streaming endpoint with
low thinking to reduce latency. A transient model failure, including a stalled
transcript stream, tries Gemini 3.8 Flash first and Gemini 3.6 Flash second.
For batch `--scan` and `--rescan` work, a source file that still fails after
that model fallback receives up to three additional attempts after 2, 4, and 8
seconds. This model order is scheduled for review on 1 April 2027; see
`docs/Gemini Model Decision.txt` for the reason.

## Silence Trimming

For summary files longer than 10 minutes, AudioBuddy uses FFmpeg to remove long
silence before upload. The default is conservative: silence must last at least
30 seconds and be at or below `-30dB`. Transcripts never trim silence, so their
timestamps remain tied to the original recording.

The original audio file is never modified.
Temporary trimmed files are deleted after processing completes or fails.

## Project Status

Phases 1 through 3 are complete: single-file summaries, silence handling,
master-summary tracking, `--scan`, confirmed `--rescan`, and the independent
single-file transcript command are available now. Phase 4 remains planning for
a friend-accessible web workflow; see `docs/Phase 4 Plan.txt`.

A possible later enhancement is splitting unusually long recordings into
smaller chunks.
