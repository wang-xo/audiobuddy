"""Summarize one local MP3 audio file with Gemini.

Phase 1-4 validates one MP3 input, uploads it to Gemini, asks for a fixed
Markdown summary, and saves the result in the configured output folder.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


API_KEY_NAME = "GEMINI_API_KEY"
GEMINI_MODELS = ("gemini-3.7-flash", "gemini-3.6-flash")
INPUT_DIR = Path(r"D:\Documents\Personal\audio\in")
OUTPUT_DIR = Path(r"D:\Documents\Personal\audio\out")
MASTER_SUMMARY_PATH = OUTPUT_DIR / "master-summary.md"
PROMPT_PATH = Path(__file__).resolve().parent / "prompt.md"
TRIM_DURATION_THRESHOLD_SECONDS = 10 * 60
SILENCE_DURATION_SECONDS = 30
SILENCE_THRESHOLD_DB = "-30dB"
RETAINED_SILENCE_SECONDS = 1.0
MINIMUM_TRIMMED_SECONDS = 1.0
TRANSIENT_RETRY_DELAYS_SECONDS = (2, 4, 8)
FILENAME_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}_\d{2}_\d{2})$"
)
MASTER_ENTRY_START_TEMPLATE = '<!-- AUDIOBUDDY_ENTRY_START filename="{filename}" -->'
MASTER_ENTRY_END = "<!-- AUDIOBUDDY_ENTRY_END -->"
MASTER_SUMMARY_TITLE = "# AudioBuddy Master Summary"


class ValidationError(Exception):
    """Raised when local CLI validation fails."""


class PromptError(Exception):
    """Raised when the prompt template cannot be loaded or rendered."""


class AudioToolError(Exception):
    """Raised when required local audio processing tools are unavailable."""


class MasterSummaryError(Exception):
    """Raised when the master summary file cannot be parsed safely."""


class GeminiIntegrationError(Exception):
    """Raised when the Gemini SDK, upload, or model check fails."""


@dataclass(frozen=True)
class AudioInput:
    path: Path
    date_text: str
    time_text: str


@dataclass(frozen=True)
class SkippedInputFile:
    path: Path
    reason: str


@dataclass(frozen=True)
class InputDiscovery:
    valid_audio: list[AudioInput]
    skipped_files: list[SkippedInputFile]


@dataclass(frozen=True)
class InputSelection:
    new_audio: list[AudioInput]
    already_processed: list[AudioInput]


@dataclass(frozen=True)
class AudioMetadata:
    original_duration_seconds: float


@dataclass(frozen=True)
class AudioProcessingResult:
    upload_path: Path
    original_duration_seconds: float
    processed_duration_seconds: float
    removed_seconds: float
    reduction_percent: float
    trimmed: bool
    reason: str
    temporary_file: bool = False


@dataclass(frozen=True)
class SilenceInterval:
    start_seconds: float
    end_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class GeminiSummary:
    markdown: str
    model: str


@dataclass(frozen=True)
class ProcessingOutcome:
    audio_input: AudioInput
    audio_processing: AudioProcessingResult
    summary: GeminiSummary
    output_path: Path
    master_update_action: str
    cleanup_path: Path | None


@dataclass(frozen=True)
class BatchFailure:
    audio_input: AudioInput
    error: Exception


@dataclass(frozen=True)
class BatchProcessingResult:
    selected_audio: list[AudioInput]
    processed: list[ProcessingOutcome]
    skipped_files: list[SkippedInputFile]
    failures: list[BatchFailure]


@dataclass(frozen=True)
class RescanPreflight:
    selected_audio: list[AudioInput]
    skipped_files: list[SkippedInputFile]


@dataclass(frozen=True)
class MasterSummaryEntry:
    source_filename: str
    source_audio_path: Path
    individual_summary_path: Path
    date_text: str
    time_text: str
    gemini_model: str
    audio_processing: AudioProcessingResult
    summary_markdown: str


@dataclass(frozen=True)
class ParsedMasterEntry:
    source_filename: str
    content: str


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize one local MP3 audio file with Gemini."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to one MP3 audio file to summarize or reprocess.",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Process new audio files from the configured input folder.",
    )
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Reprocess all valid audio files from the configured input folder.",
    )
    parsed_args = parser.parse_args(arguments)

    selected_modes = sum(
        (
            parsed_args.audio_file is not None,
            parsed_args.scan,
            parsed_args.rescan,
        )
    )
    if selected_modes != 1:
        parser.error("provide exactly one audio file path, --scan, or --rescan")

    return parsed_args


def load_dotenv(env_path: Path) -> None:
    """Load simple KEY=VALUE entries from .env without overriding the shell."""
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def require_api_key() -> None:
    api_key = os.environ.get(API_KEY_NAME, "").strip()
    if not api_key:
        raise ValidationError(
            f"Missing {API_KEY_NAME}. Set it in the environment or in .env."
        )


def parse_filename_timestamp(audio_path: Path) -> tuple[str, str]:
    match = FILENAME_TIMESTAMP_PATTERN.match(audio_path.stem)
    if not match:
        raise ValidationError(
            "Audio filename must match 'YYYY-MM-DD HH_MM_SS.mp3', "
            "for example '2026-08-30 11_10_13.mp3'."
        )

    raw_datetime = f"{match.group('date')} {match.group('time')}"
    parsed = datetime.strptime(raw_datetime, "%Y-%m-%d %H_%M_%S")
    return parsed.strftime("%d %B %Y"), parsed.strftime("%H:%M:%S")


def parse_source_filename_datetime(source_filename: str) -> datetime:
    """Return the sortable timestamp encoded in a master entry filename."""
    match = FILENAME_TIMESTAMP_PATTERN.match(Path(source_filename).stem)
    if not match:
        raise MasterSummaryError(
            "Master entry filename must match 'YYYY-MM-DD HH_MM_SS.mp3': "
            f"{source_filename}"
        )

    return datetime.strptime(
        f"{match.group('date')} {match.group('time')}", "%Y-%m-%d %H_%M_%S"
    )


def validate_audio_file(audio_file: str) -> AudioInput:
    audio_path = Path(audio_file).expanduser()

    if not audio_path.is_file():
        raise ValidationError(f"Input file does not exist: {audio_path}")

    if audio_path.suffix.lower() != ".mp3":
        raise ValidationError("Unsupported file format. Phase 1 accepts only .mp3 files.")

    date_text, time_text = parse_filename_timestamp(audio_path)
    return AudioInput(path=audio_path, date_text=date_text, time_text=time_text)


def discover_input_files(input_dir: Path = INPUT_DIR) -> InputDiscovery:
    """Discover valid top-level audio inputs and record skipped entries."""
    if not input_dir.is_dir():
        raise ValidationError(f"Input directory does not exist: {input_dir}")

    valid_audio: list[AudioInput] = []
    skipped_files: list[SkippedInputFile] = []

    for input_path in sorted(input_dir.iterdir(), key=lambda path: path.name.lower()):
        if not input_path.is_file():
            skipped_files.append(
                SkippedInputFile(input_path, "not a regular file")
            )
            continue

        if input_path.suffix.lower() != ".mp3":
            skipped_files.append(SkippedInputFile(input_path, "not an MP3 file"))
            continue

        try:
            valid_audio.append(validate_audio_file(str(input_path)))
        except ValidationError as error:
            skipped_files.append(SkippedInputFile(input_path, str(error)))

    valid_audio.sort(key=lambda audio_input: audio_input.path.name)
    return InputDiscovery(valid_audio=valid_audio, skipped_files=skipped_files)


def require_audio_tools() -> None:
    missing_tools = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if not missing_tools:
        return

    missing = ", ".join(missing_tools)
    raise AudioToolError(
        f"Missing required audio tool(s): {missing}. Install FFmpeg and make sure "
        "ffmpeg and ffprobe are on PATH. On Windows, try: winget install Gyan.FFmpeg "
        "then reopen your terminal."
    )


def get_audio_duration_seconds(audio_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    if result.returncode != 0:
        details = result.stderr.strip() or "ffprobe did not return duration details."
        raise AudioToolError(f"Could not read audio duration: {details}")

    try:
        duration = float(result.stdout.strip())
    except ValueError as error:
        raise AudioToolError(
            f"Could not parse audio duration from ffprobe output: {result.stdout.strip()}"
        ) from error

    if duration <= 0:
        raise AudioToolError(f"Invalid audio duration reported by ffprobe: {duration}")

    return duration


def get_audio_metadata(audio_input: AudioInput) -> AudioMetadata:
    return AudioMetadata(original_duration_seconds=get_audio_duration_seconds(audio_input.path))


def create_temp_mp3_path(audio_path: Path) -> Path:
    temp_file = tempfile.NamedTemporaryFile(
        prefix=f"{audio_path.stem}-trimmed-", suffix=".mp3", delete=False
    )
    temp_path = Path(temp_file.name)
    temp_file.close()
    return temp_path


def detect_long_silences(audio_path: Path) -> list[SilenceInterval]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(audio_path),
            "-af",
            f"silencedetect=noise={SILENCE_THRESHOLD_DB}:d={SILENCE_DURATION_SECONDS}",
            "-f",
            "null",
            "NUL",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    if result.returncode != 0:
        details = result.stderr.strip() or "ffmpeg did not return silence details."
        raise AudioToolError(f"Could not detect long silence: {details}")

    starts: list[float] = []
    intervals: list[SilenceInterval] = []

    for line in result.stderr.splitlines():
        start_match = re.search(r"silence_start: (?P<start>[0-9.]+)", line)
        if start_match:
            starts.append(float(start_match.group("start")))
            continue

        end_match = re.search(
            r"silence_end: (?P<end>[0-9.]+) \| silence_duration: (?P<duration>[0-9.]+)",
            line,
        )
        if end_match and starts:
            intervals.append(
                SilenceInterval(
                    start_seconds=starts.pop(0),
                    end_seconds=float(end_match.group("end")),
                    duration_seconds=float(end_match.group("duration")),
                )
            )

    return intervals


def build_keep_ranges(
    original_duration: float, silence_intervals: list[SilenceInterval]
) -> list[tuple[float, float]]:
    keep_ranges: list[tuple[float, float]] = []
    cursor = 0.0

    for interval in silence_intervals:
        retained_silence_end = min(
            interval.start_seconds + RETAINED_SILENCE_SECONDS,
            interval.end_seconds,
        )
        keep_end = max(cursor, retained_silence_end)
        if keep_end > cursor:
            keep_ranges.append((cursor, keep_end))
        cursor = max(cursor, interval.end_seconds)

    if cursor < original_duration:
        keep_ranges.append((cursor, original_duration))

    return [(start, end) for start, end in keep_ranges if end - start > 0.01]


def trim_long_silence(audio_path: Path, original_duration: float) -> Path | None:
    silence_intervals = detect_long_silences(audio_path)
    if not silence_intervals:
        return None

    keep_ranges = build_keep_ranges(original_duration, silence_intervals)
    if not keep_ranges:
        raise AudioToolError("Silence trimming would remove the entire audio file.")

    output_path = create_temp_mp3_path(audio_path)
    trim_parts = []
    concat_inputs = []

    for index, (start, end) in enumerate(keep_ranges):
        label = f"a{index}"
        trim_parts.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=N/SR/TB[{label}]"
        )
        concat_inputs.append(f"[{label}]")

    filter_complex = (
        ";".join(trim_parts)
        + ";"
        + "".join(concat_inputs)
        + f"concat=n={len(keep_ranges)}:v=0:a=1[outa]"
    )

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[outa]",
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        details = result.stderr.strip() or "ffmpeg did not return error details."
        raise AudioToolError(f"Could not trim long silence: {details}")

    if not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise AudioToolError("FFmpeg did not create a usable trimmed audio file.")

    return output_path


def plan_audio_processing(
    audio_input: AudioInput, audio_metadata: AudioMetadata
) -> AudioProcessingResult:
    original_duration = audio_metadata.original_duration_seconds

    if original_duration <= TRIM_DURATION_THRESHOLD_SECONDS:
        return AudioProcessingResult(
            upload_path=audio_input.path,
            original_duration_seconds=original_duration,
            processed_duration_seconds=original_duration,
            removed_seconds=0.0,
            reduction_percent=0.0,
            trimmed=False,
            reason="duration is 10 minutes or shorter; skipping silence trimming",
        )

    trimmed_path = trim_long_silence(audio_input.path, original_duration)
    if trimmed_path is None:
        return AudioProcessingResult(
            upload_path=audio_input.path,
            original_duration_seconds=original_duration,
            processed_duration_seconds=original_duration,
            removed_seconds=0.0,
            reduction_percent=0.0,
            trimmed=False,
            reason=(
                f"duration is longer than 10 minutes, but no silence of at least "
                f"{SILENCE_DURATION_SECONDS} seconds at {SILENCE_THRESHOLD_DB} was detected"
            ),
        )

    processed_duration = get_audio_duration_seconds(trimmed_path)
    removed_seconds = max(0.0, original_duration - processed_duration)
    reduction_percent = (removed_seconds / original_duration) * 100

    if removed_seconds < MINIMUM_TRIMMED_SECONDS:
        trimmed_path.unlink(missing_ok=True)
        return AudioProcessingResult(
            upload_path=audio_input.path,
            original_duration_seconds=original_duration,
            processed_duration_seconds=original_duration,
            removed_seconds=0.0,
            reduction_percent=0.0,
            trimmed=False,
            reason=(
                f"duration is longer than 10 minutes, but no silence of at least "
                f"{SILENCE_DURATION_SECONDS} seconds at {SILENCE_THRESHOLD_DB} was trimmed"
            ),
        )

    return AudioProcessingResult(
        upload_path=trimmed_path,
        original_duration_seconds=original_duration,
        processed_duration_seconds=processed_duration,
        removed_seconds=removed_seconds,
        reduction_percent=reduction_percent,
        trimmed=True,
        reason=(
            f"duration is longer than 10 minutes; removed silence of at least "
            f"{SILENCE_DURATION_SECONDS} seconds at {SILENCE_THRESHOLD_DB}"
        ),
        temporary_file=True,
    )


def format_duration(seconds: float) -> str:
    rounded_seconds = int(round(seconds))
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:d}:{seconds:02d}"


def unique_summary_path(audio_path: Path, output_dir: Path) -> Path:
    summary_stem = f"{audio_path.stem}-summary"
    candidate = output_dir / f"{summary_stem}.md"
    counter = 1

    while candidate.exists():
        candidate = output_dir / f"{summary_stem}-{counter}.md"
        counter += 1

    return candidate


def master_entry_start_marker(filename: str) -> str:
    return MASTER_ENTRY_START_TEMPLATE.format(filename=filename)


def master_entry_end_marker() -> str:
    return MASTER_ENTRY_END


def render_master_entry(entry: MasterSummaryEntry) -> str:
    return "\n".join(
        [
            master_entry_start_marker(entry.source_filename),
            f"## {entry.date_text} {entry.time_text} - {entry.source_filename}",
            "",
            f"- Source file: `{entry.source_audio_path}`",
            f"- Individual summary: `{entry.individual_summary_path}`",
            f"- Gemini model: `{entry.gemini_model}`",
            "",
            "### Audio Processing",
            "",
            format_audio_processing_summary(entry.audio_processing),
            "",
            "### Summary",
            "",
            entry.summary_markdown.strip(),
            "",
            master_entry_end_marker(),
            "",
        ]
    )


def parse_master_entries(master_markdown: str) -> list[ParsedMasterEntry]:
    entries: list[ParsedMasterEntry] = []
    position = 0
    start_pattern = re.compile(
        r'<!-- AUDIOBUDDY_ENTRY_START filename="(?P<filename>[^"]+)" -->'
    )

    while True:
        start_match = start_pattern.search(master_markdown, position)
        if start_match is None:
            remaining = master_markdown[position:]
            if MASTER_ENTRY_END in remaining:
                raise MasterSummaryError("Found master entry end marker without start marker.")
            break

        unmatched_text = master_markdown[position : start_match.start()]
        if MASTER_ENTRY_END in unmatched_text:
            raise MasterSummaryError("Found master entry end marker before next start marker.")

        end_index = master_markdown.find(MASTER_ENTRY_END, start_match.end())
        if end_index == -1:
            raise MasterSummaryError(
                f"Master entry for {start_match.group('filename')} is missing an end marker."
            )

        entry_end = end_index + len(MASTER_ENTRY_END)
        content = master_markdown[start_match.start() : entry_end].strip()
        entries.append(
            ParsedMasterEntry(
                source_filename=start_match.group("filename"),
                content=content,
            )
        )
        position = entry_end

    filenames = [entry.source_filename for entry in entries]
    if len(filenames) != len(set(filenames)):
        raise MasterSummaryError("Master summary contains duplicate source filename entries.")

    return entries


def read_master_entries(master_path: Path = MASTER_SUMMARY_PATH) -> list[ParsedMasterEntry]:
    if not master_path.is_file():
        return []

    return parse_master_entries(master_path.read_text(encoding="utf-8"))


def select_new_audio_files(
    discovery: InputDiscovery,
    master_path: Path = MASTER_SUMMARY_PATH,
) -> InputSelection:
    """Split discovered inputs according to source filenames in the master file."""
    processed_filenames = {
        entry.source_filename for entry in read_master_entries(master_path)
    }
    new_audio: list[AudioInput] = []
    already_processed: list[AudioInput] = []

    for audio_input in discovery.valid_audio:
        if audio_input.path.name in processed_filenames:
            already_processed.append(audio_input)
        else:
            new_audio.append(audio_input)

    return InputSelection(
        new_audio=new_audio,
        already_processed=already_processed,
    )


def select_all_audio_files(discovery: InputDiscovery) -> list[AudioInput]:
    """Select every valid discovered input for an intentional rescan."""
    return list(discovery.valid_audio)


def prepare_rescan(
    input_dir: Path = INPUT_DIR,
    master_path: Path = MASTER_SUMMARY_PATH,
) -> RescanPreflight:
    """Validate the master and prepare every valid input for confirmation."""
    discovery = discover_input_files(input_dir)
    read_master_entries(master_path)
    return RescanPreflight(
        selected_audio=select_all_audio_files(discovery),
        skipped_files=list(discovery.skipped_files),
    )


def confirm_rescan(selected_count: int) -> bool:
    """Ask for explicit approval before starting paid rescan work."""
    try:
        response = input(
            "This will reprocess "
            f"{selected_count} audio files using the current prompt.md. Continue? [y/N] "
        )
    except EOFError:
        return False

    return response.strip().lower() in {"y", "yes"}


def print_rescan_preflight(preflight: RescanPreflight) -> None:
    print(
        "Rescan preflight: "
        f"selected {len(preflight.selected_audio)}, "
        f"skipped {len(preflight.skipped_files)}."
    )


def replace_or_add_master_entry(
    existing_entries: list[ParsedMasterEntry],
    current_entry: MasterSummaryEntry,
) -> tuple[list[ParsedMasterEntry], bool]:
    """Replace the entry for a retried source file, or add a new entry."""
    rendered_entry = ParsedMasterEntry(
        source_filename=current_entry.source_filename,
        content=render_master_entry(current_entry).strip(),
    )
    replaced = False
    updated_entries: list[ParsedMasterEntry] = []

    for entry in existing_entries:
        if entry.source_filename == rendered_entry.source_filename:
            updated_entries.append(rendered_entry)
            replaced = True
        else:
            updated_entries.append(entry)

    if not replaced:
        updated_entries.append(rendered_entry)

    return updated_entries, replaced


def render_master_summary(entries: list[ParsedMasterEntry]) -> str:
    """Render a complete, chronologically ordered master summary file."""
    ordered_entries = sorted(
        entries,
        key=lambda entry: parse_source_filename_datetime(entry.source_filename),
    )
    entry_text = "\n\n".join(entry.content.strip() for entry in ordered_entries)
    return f"{MASTER_SUMMARY_TITLE}\n\n{entry_text}\n"


def update_master_summary(
    current_entry: MasterSummaryEntry,
    master_path: Path = MASTER_SUMMARY_PATH,
) -> str:
    """Create, append to, or rewrite the master summary as needed."""
    existing_entries = read_master_entries(master_path)
    updated_entries, replaced = replace_or_add_master_entry(
        existing_entries, current_entry
    )
    current_timestamp = parse_source_filename_datetime(current_entry.source_filename)

    master_path.parent.mkdir(parents=True, exist_ok=True)
    if not existing_entries:
        master_path.write_text(render_master_summary(updated_entries), encoding="utf-8")
        return "created"

    latest_timestamp = max(
        parse_source_filename_datetime(entry.source_filename)
        for entry in existing_entries
    )
    if not replaced and current_timestamp >= latest_timestamp:
        with master_path.open("a", encoding="utf-8") as master_file:
            master_file.write("\n" + updated_entries[-1].content.strip() + "\n")
        return "appended"

    master_path.write_text(render_master_summary(updated_entries), encoding="utf-8")
    return "rewritten"


def load_prompt_template(prompt_path: Path = PROMPT_PATH) -> str:
    if not prompt_path.is_file():
        raise PromptError(f"Prompt file not found: {prompt_path}")

    template = prompt_path.read_text(encoding="utf-8").strip()
    if not template:
        raise PromptError(f"Prompt file is empty: {prompt_path}")

    return template


def format_audio_processing_summary(audio_processing: AudioProcessingResult) -> str:
    return "\n".join(
        [
            f"- Original duration: {format_duration(audio_processing.original_duration_seconds)} "
            f"({audio_processing.original_duration_seconds:.2f} seconds)",
            f"- Processed duration: {format_duration(audio_processing.processed_duration_seconds)} "
            f"({audio_processing.processed_duration_seconds:.2f} seconds)",
            f"- Audio removed: {format_duration(audio_processing.removed_seconds)} "
            f"({audio_processing.removed_seconds:.2f} seconds)",
            f"- Reduction: {audio_processing.reduction_percent:.2f}%",
            f"- Trimmed: {'yes' if audio_processing.trimmed else 'no'}",
            f"- Processing note: {audio_processing.reason}",
        ]
    )


def build_summary_prompt(
    audio_input: AudioInput, audio_processing: AudioProcessingResult
) -> str:
    template = load_prompt_template()
    try:
        return template.format(
            date=audio_input.date_text,
            time=audio_input.time_text,
            audio_processing_summary=format_audio_processing_summary(audio_processing),
        )
    except KeyError as error:
        raise PromptError(f"Unknown placeholder in prompt.md: {{{error.args[0]}}}") from error


def clean_markdown_response(markdown: str) -> str:
    cleaned = markdown.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned + "\n"


def write_summary(audio_input: AudioInput, markdown: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = unique_summary_path(audio_input.path, OUTPUT_DIR)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def cleanup_temporary_audio(audio_processing: AudioProcessingResult | None) -> Path | None:
    if audio_processing is None or not audio_processing.temporary_file:
        return None

    try:
        audio_processing.upload_path.unlink(missing_ok=True)
        return audio_processing.upload_path
    except OSError as error:
        print(
            f"Warning: Could not clean up temporary audio file "
            f"{audio_processing.upload_path}: {error}",
            file=sys.stderr,
        )
        return None


def is_transient_gemini_error(error: Exception) -> bool:
    error_text = str(error).lower()
    return any(
        marker in error_text
        for marker in (
            "503",
            "500",
            "429",
            "unavailable",
            "high demand",
            "api_error",
            "too_many_requests",
            "server disconnected",
        )
    )


def generate_gemini_summary(
    audio_input: AudioInput, audio_processing: AudioProcessingResult
) -> GeminiSummary:
    try:
        from google import genai
    except ImportError as error:
        raise GeminiIntegrationError(
            "Missing dependency 'google-genai'. Install requirements.txt first."
        ) from error

    client = genai.Client(api_key=os.environ[API_KEY_NAME])

    try:
        uploaded_file = client.files.upload(file=audio_processing.upload_path)
    except Exception as error:
        raise GeminiIntegrationError(f"Audio upload failed: {error}") from error

    last_error: Exception | None = None
    for model in GEMINI_MODELS:
        try:
            response = client.interactions.create(
                model=model,
                input=[
                    {
                        "type": "text",
                        "text": build_summary_prompt(audio_input, audio_processing),
                    },
                    {
                        "type": "audio",
                        "uri": uploaded_file.uri,
                        "mime_type": uploaded_file.mime_type,
                    },
                ],
            )
            response_text = getattr(response, "output_text", "") or ""
            if not response_text.strip():
                raise GeminiIntegrationError("Gemini returned an empty summary.")
            return GeminiSummary(markdown=clean_markdown_response(response_text), model=model)
        except Exception as error:
            last_error = error
            if not is_transient_gemini_error(error):
                break

    if last_error is None:
        raise GeminiIntegrationError("No Gemini model was configured.")

    raise GeminiIntegrationError(str(last_error)) from last_error


def process_audio_file(
    audio_file: str,
    master_path: Path = MASTER_SUMMARY_PATH,
) -> ProcessingOutcome:
    """Process one audio file after the environment and API key are prepared."""
    audio_processing: AudioProcessingResult | None = None

    try:
        audio_input = validate_audio_file(audio_file)
        require_audio_tools()
        audio_metadata = get_audio_metadata(audio_input)
        audio_processing = plan_audio_processing(audio_input, audio_metadata)
        summary = generate_gemini_summary(audio_input, audio_processing)
        output_path = write_summary(audio_input, summary.markdown)
        master_update_action = update_master_summary(
            MasterSummaryEntry(
                source_filename=audio_input.path.name,
                source_audio_path=audio_input.path,
                individual_summary_path=output_path,
                date_text=audio_input.date_text,
                time_text=audio_input.time_text,
                gemini_model=summary.model,
                audio_processing=audio_processing,
                summary_markdown=summary.markdown,
            ),
            master_path,
        )
    finally:
        cleanup_path = cleanup_temporary_audio(audio_processing)

    return ProcessingOutcome(
        audio_input=audio_input,
        audio_processing=audio_processing,
        summary=summary,
        output_path=output_path,
        master_update_action=master_update_action,
        cleanup_path=cleanup_path,
    )


def print_processing_success(outcome: ProcessingOutcome) -> None:
    print("Success: summary created.")
    print(f"Audio file: {outcome.audio_input.path}")
    print(f"Summary file: {outcome.output_path}")
    print(
        f"Master summary file: {MASTER_SUMMARY_PATH} "
        f"({outcome.master_update_action})"
    )
    print(f"Date: {outcome.audio_input.date_text}")
    print(f"Time: {outcome.audio_input.time_text}")
    print("Audio processing:")
    print(format_audio_processing_summary(outcome.audio_processing))
    print(f"Uploaded audio file: {outcome.audio_processing.upload_path}")
    if outcome.cleanup_path is not None:
        print(f"Cleaned up temporary audio file: {outcome.cleanup_path}")
    print(f"Gemini model: {outcome.summary.model}")


def process_audio_with_transient_retries(
    audio_input: AudioInput,
    master_path: Path = MASTER_SUMMARY_PATH,
) -> ProcessingOutcome:
    """Process one file, retrying only transient Gemini integration failures."""
    for retry_index, delay_seconds in enumerate(
        (0, *TRANSIENT_RETRY_DELAYS_SECONDS)
    ):
        if retry_index:
            print(
                f"Retrying {audio_input.path.name} after transient Gemini failure "
                f"in {delay_seconds} seconds.",
                file=sys.stderr,
            )
            time.sleep(delay_seconds)

        try:
            return process_audio_file(str(audio_input.path), master_path)
        except GeminiIntegrationError as error:
            if (
                retry_index == len(TRANSIENT_RETRY_DELAYS_SECONDS)
                or not is_transient_gemini_error(error)
            ):
                raise

    raise GeminiIntegrationError("Transient Gemini retry loop ended unexpectedly.")


def process_new_input_files(
    input_dir: Path = INPUT_DIR,
    master_path: Path = MASTER_SUMMARY_PATH,
) -> BatchProcessingResult:
    """Process new top-level input files while continuing after file failures."""
    discovery = discover_input_files(input_dir)
    selection = select_new_audio_files(discovery, master_path)
    skipped_files = list(discovery.skipped_files)
    skipped_files.extend(
        SkippedInputFile(
            audio_input.path,
            "already represented in the master summary",
        )
        for audio_input in selection.already_processed
    )

    return process_selected_audio_files(
        selection.new_audio,
        skipped_files,
        master_path,
    )


def process_rescan(
    preflight: RescanPreflight,
    master_path: Path = MASTER_SUMMARY_PATH,
) -> BatchProcessingResult:
    """Reprocess every file approved by a completed rescan preflight."""
    return process_selected_audio_files(
        preflight.selected_audio,
        preflight.skipped_files,
        master_path,
    )


def process_selected_audio_files(
    selected_audio: list[AudioInput],
    skipped_files: list[SkippedInputFile],
    master_path: Path,
) -> BatchProcessingResult:
    """Process selected audio files sequentially while collecting failures."""
    processed: list[ProcessingOutcome] = []
    failures: list[BatchFailure] = []

    if not selected_audio:
        return BatchProcessingResult(
            selected_audio=[],
            processed=processed,
            skipped_files=skipped_files,
            failures=failures,
        )

    load_dotenv(Path(__file__).resolve().parent / ".env")
    require_api_key()

    for audio_input in selected_audio:
        try:
            processed.append(
                process_audio_with_transient_retries(audio_input, master_path)
            )
        except (
            ValidationError,
            PromptError,
            MasterSummaryError,
            AudioToolError,
            GeminiIntegrationError,
            OSError,
        ) as error:
            failures.append(BatchFailure(audio_input, error))

    return BatchProcessingResult(
        selected_audio=selected_audio,
        processed=processed,
        skipped_files=skipped_files,
        failures=failures,
    )


def print_batch_processing_result(result: BatchProcessingResult) -> None:
    if not result.selected_audio:
        print("No new audio files found.")

    for outcome in result.processed:
        print_processing_success(outcome)

    for skipped_file in result.skipped_files:
        print(f"Skipped: {skipped_file.path.name} ({skipped_file.reason})")

    for failure in result.failures:
        print(
            f"Failed: {failure.audio_input.path.name} ({failure.error})",
            file=sys.stderr,
        )

    print(
        "Batch complete: "
        f"processed {len(result.processed)}, "
        f"skipped {len(result.skipped_files)}, "
        f"failed {len(result.failures)}."
    )


def print_rescan_processing_result(result: BatchProcessingResult) -> None:
    for outcome in result.processed:
        print_processing_success(outcome)

    for skipped_file in result.skipped_files:
        print(f"Skipped: {skipped_file.path.name} ({skipped_file.reason})")

    for failure in result.failures:
        print(
            f"Failed: {failure.audio_input.path.name} ({failure.error})",
            file=sys.stderr,
        )

    print(
        "Rescan complete: "
        f"processed {len(result.processed)}, "
        f"skipped {len(result.skipped_files)}, "
        f"failed {len(result.failures)}."
    )


def print_rescan_cancelled(preflight: RescanPreflight) -> None:
    print("Rescan cancelled. No audio files were processed.")
    print(
        "Rescan complete: "
        f"processed 0, skipped {len(preflight.skipped_files)}, failed 0, cancelled."
    )


def main() -> int:
    args = parse_args()

    if args.rescan:
        try:
            rescan_preflight = prepare_rescan()
        except ValidationError as error:
            print(f"Failure: {error}", file=sys.stderr)
            return 1
        except MasterSummaryError as error:
            print(f"Failure: {error}", file=sys.stderr)
            return 1

        print_rescan_preflight(rescan_preflight)
        if not rescan_preflight.selected_audio:
            print("No valid audio files found.")
            print(
                "Rescan complete: "
                f"processed 0, skipped {len(rescan_preflight.skipped_files)}, failed 0."
            )
            return 0
        if not confirm_rescan(len(rescan_preflight.selected_audio)):
            print_rescan_cancelled(rescan_preflight)
            return 0

        try:
            rescan_result = process_rescan(rescan_preflight)
        except ValidationError as error:
            print(f"Failure: {error}", file=sys.stderr)
            return 1

        print_rescan_processing_result(rescan_result)
        return 1 if rescan_result.failures else 0

    if args.scan:
        try:
            batch_result = process_new_input_files()
        except ValidationError as error:
            print(f"Failure: {error}", file=sys.stderr)
            return 1
        except MasterSummaryError as error:
            print(f"Failure: {error}", file=sys.stderr)
            return 1

        print_batch_processing_result(batch_result)
        return 1 if batch_result.failures else 0

    try:
        load_dotenv(Path(__file__).resolve().parent / ".env")
        require_api_key()
        outcome = process_audio_file(args.audio_file)
    except ValidationError as error:
        print(f"Failure: {error}", file=sys.stderr)
        return 1
    except PromptError as error:
        print(f"Failure: {error}", file=sys.stderr)
        return 1
    except MasterSummaryError as error:
        print(f"Failure: {error}", file=sys.stderr)
        return 1
    except AudioToolError as error:
        print(f"Failure: {error}", file=sys.stderr)
        return 1
    except GeminiIntegrationError as error:
        print(f"Failure: Gemini API integration failed: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Failure: Could not write summary file: {error}", file=sys.stderr)
        return 1
    print_processing_success(outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
