"""Summarize one local MP3 audio file with Gemini.

Phase 1-4 validates one MP3 input, uploads it to Gemini, asks for a fixed
Markdown summary, and saves the result in the configured output folder.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


API_KEY_NAME = "GEMINI_API_KEY"
GEMINI_MODELS = ("gemini-3.7-flash", "gemini-3.6-flash")
OUTPUT_DIR = Path(r"D:\Documents\Personal\audio\out")
PROMPT_PATH = Path(__file__).resolve().parent / "prompt.md"
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


def generate_gemini_summary(audio_input: AudioInput) -> GeminiSummary:
    try:
        from google import genai
    except ImportError as error:
        raise GeminiIntegrationError(
            "Missing dependency 'google-genai'. Install requirements.txt first."
        ) from error

    client = genai.Client(api_key=os.environ[API_KEY_NAME])

    try:
        uploaded_file = client.files.upload(file=audio_input.path)
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
        summary = generate_gemini_summary(audio_input)
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
    print(f"Gemini model: {summary.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
