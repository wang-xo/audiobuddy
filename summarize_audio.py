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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


API_KEY_NAME = "GEMINI_API_KEY"
GEMINI_MODELS = ("gemini-3.7-flash", "gemini-3.6-flash")
OUTPUT_DIR = Path(r"D:\Documents\Personal\audio\out")
PROMPT_PATH = Path(__file__).resolve().parent / "prompt.md"
TRIM_DURATION_THRESHOLD_SECONDS = 10 * 60
SILENCE_DURATION_SECONDS = 30
SILENCE_THRESHOLD_DB = "-30dB"
RETAINED_SILENCE_SECONDS = 1.0
MINIMUM_TRIMMED_SECONDS = 1.0
FILENAME_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}_\d{2}_\d{2})$"
)


class ValidationError(Exception):
    """Raised when local CLI validation fails."""


class PromptError(Exception):
    """Raised when the prompt template cannot be loaded or rendered."""


class AudioToolError(Exception):
    """Raised when required local audio processing tools are unavailable."""


class GeminiIntegrationError(Exception):
    """Raised when the Gemini SDK, upload, or model check fails."""


@dataclass(frozen=True)
class AudioInput:
    path: Path
    date_text: str
    time_text: str


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize one local MP3 audio file with Gemini."
    )
    parser.add_argument("audio_file", help="Path to the MP3 audio file to summarize.")
    return parser.parse_args()


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


def validate_audio_file(audio_file: str) -> AudioInput:
    audio_path = Path(audio_file).expanduser()

    if not audio_path.is_file():
        raise ValidationError(f"Input file does not exist: {audio_path}")

    if audio_path.suffix.lower() != ".mp3":
        raise ValidationError("Unsupported file format. Phase 1 accepts only .mp3 files.")

    date_text, time_text = parse_filename_timestamp(audio_path)
    return AudioInput(path=audio_path, date_text=date_text, time_text=time_text)


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


def load_prompt_template(prompt_path: Path = PROMPT_PATH) -> str:
    if not prompt_path.is_file():
        raise PromptError(f"Prompt file not found: {prompt_path}")

    template = prompt_path.read_text(encoding="utf-8").strip()
    if not template:
        raise PromptError(f"Prompt file is empty: {prompt_path}")

    return template


def build_summary_prompt(audio_input: AudioInput) -> str:
    template = load_prompt_template()
    try:
        return template.format(date=audio_input.date_text, time=audio_input.time_text)
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
                        "text": build_summary_prompt(audio_input),
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


def main() -> int:
    args = parse_args()

    try:
        load_dotenv(Path(__file__).resolve().parent / ".env")
        require_api_key()
        audio_input = validate_audio_file(args.audio_file)
        require_audio_tools()
        audio_metadata = get_audio_metadata(audio_input)
        audio_processing = plan_audio_processing(audio_input, audio_metadata)
        summary = generate_gemini_summary(audio_input, audio_processing)
        output_path = write_summary(audio_input, summary.markdown)
    except ValidationError as error:
        print(f"Failure: {error}", file=sys.stderr)
        return 1
    except PromptError as error:
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

    print("Success: summary created.")
    print(f"Audio file: {audio_input.path}")
    print(f"Summary file: {output_path}")
    print(f"Date: {audio_input.date_text}")
    print(f"Time: {audio_input.time_text}")
    print(
        "Original duration: "
        f"{format_duration(audio_metadata.original_duration_seconds)} "
        f"({audio_metadata.original_duration_seconds:.2f} seconds)"
    )
    print(f"Audio processing: {audio_processing.reason}")
    print(f"Uploaded audio file: {audio_processing.upload_path}")
    print(f"Gemini model: {summary.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
