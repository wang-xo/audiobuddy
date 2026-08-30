"""Validate inputs for the AudioBuddy Phase 1 audio summarizer.

Phase 1-1 intentionally stops after local validation. Later stages will add
output-path handling and Gemini API calls.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


API_KEY_NAME = "GEMINI_API_KEY"
FILENAME_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}_\d{2}_\d{2})$"
)


class ValidationError(Exception):
    """Raised when local CLI validation fails."""


@dataclass(frozen=True)
class AudioInput:
    path: Path
    date_text: str
    time_text: str


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


def main() -> int:
    args = parse_args()

    try:
        load_dotenv(Path(__file__).resolve().parent / ".env")
        require_api_key()
        audio_input = validate_audio_file(args.audio_file)
    except ValidationError as error:
        print(f"Failure: {error}", file=sys.stderr)
        return 1

    print("Success: local validation passed.")
    print(f"Audio file: {audio_input.path}")
    print(f"Date: {audio_input.date_text}")
    print(f"Time: {audio_input.time_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
