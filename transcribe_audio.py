"""Create one chronological, English turn-by-turn transcript with Gemini."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from summarize_audio import (
    API_KEY_NAME,
    GEMINI_GENERATION_TIMEOUT_SECONDS,
    GEMINI_MODELS,
    GEMINI_UPLOAD_TIMEOUT_MS,
    OUTPUT_DIR,
    AudioInput,
    AudioProcessingResult,
    GeminiIntegrationError,
    PromptError,
    ValidationError,
    clean_markdown_response,
    cleanup_temporary_audio,
    get_audio_metadata,
    is_transient_gemini_error,
    load_dotenv,
    plan_audio_processing,
    print_gemini_timing,
    require_api_key,
    require_audio_tools,
    validate_audio_file,
)


TRANSCRIPT_PROMPT_PATH = Path(__file__).resolve().parent / "transcript-prompt.md"


@dataclass(frozen=True)
class GeminiTranscript:
    text: str
    model: str


@dataclass(frozen=True)
class TranscriptionOutcome:
    audio_input: AudioInput
    audio_processing: AudioProcessingResult
    transcript: GeminiTranscript
    output_path: Path
    cleanup_path: Path | None


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one turn-by-turn English transcript with Gemini."
    )
    parser.add_argument(
        "audio_file",
        help="Path to one MP3 or OGG audio file to transcribe.",
    )
    return parser.parse_args(arguments)


def load_transcript_prompt(prompt_path: Path = TRANSCRIPT_PROMPT_PATH) -> str:
    if not prompt_path.is_file():
        raise PromptError(f"Transcript prompt file not found: {prompt_path}")

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise PromptError(f"Transcript prompt file is empty: {prompt_path}")

    return prompt


def unique_transcript_path(audio_path: Path, output_dir: Path) -> Path:
    transcript_stem = f"{audio_path.stem}-transcript"
    candidate = output_dir / f"{transcript_stem}.md"
    counter = 1

    while candidate.exists():
        candidate = output_dir / f"{transcript_stem}-{counter}.md"
        counter += 1

    return candidate


def write_transcript(audio_input: AudioInput, transcript: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = unique_transcript_path(audio_input.path, OUTPUT_DIR)
    output_path.write_text(clean_markdown_response(transcript), encoding="utf-8")
    return output_path


def generate_gemini_transcript(
    audio_input: AudioInput, audio_processing: AudioProcessingResult
) -> GeminiTranscript:
    try:
        from google import genai
    except ImportError as error:
        raise GeminiIntegrationError(
            "Missing dependency 'google-genai'. Install requirements.txt first."
        ) from error

    client = genai.Client(
        api_key=os.environ[API_KEY_NAME],
        http_options={
            "timeout": GEMINI_UPLOAD_TIMEOUT_MS,
            "retry_options": {"attempts": 1},
        },
    )
    prompt = load_transcript_prompt()

    upload_started_at = time.monotonic()
    print_gemini_timing(
        f"transcript upload started for {audio_input.path.name} "
        f"(timeout {GEMINI_UPLOAD_TIMEOUT_MS // 1000}s)."
    )
    try:
        uploaded_file = client.files.upload(file=audio_processing.upload_path)
    except Exception as error:
        elapsed_seconds = time.monotonic() - upload_started_at
        print_gemini_timing(
            f"transcript upload failed for {audio_input.path.name} after "
            f"{elapsed_seconds:.1f}s: {type(error).__name__}."
        )
        raise GeminiIntegrationError(
            f"Audio upload failed after {elapsed_seconds:.1f}s: {error}"
        ) from error

    upload_elapsed_seconds = time.monotonic() - upload_started_at
    print_gemini_timing(
        f"transcript upload completed for {audio_input.path.name} in "
        f"{upload_elapsed_seconds:.1f}s."
    )

    last_error: Exception | None = None
    for model in GEMINI_MODELS:
        generation_started_at = time.monotonic()
        print_gemini_timing(
            f"transcript generation started for {audio_input.path.name} with "
            f"{model} (timeout {GEMINI_GENERATION_TIMEOUT_SECONDS}s)."
        )
        try:
            response = client.interactions.create(
                model=model,
                input=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "audio",
                        "uri": uploaded_file.uri,
                        "mime_type": uploaded_file.mime_type,
                    },
                ],
                timeout=GEMINI_GENERATION_TIMEOUT_SECONDS,
            )
            response_text = getattr(response, "output_text", "") or ""
            if not response_text.strip():
                raise GeminiIntegrationError("Gemini returned an empty transcript.")
            elapsed_seconds = time.monotonic() - generation_started_at
            print_gemini_timing(
                f"transcript generation completed for {audio_input.path.name} with "
                f"{model} in {elapsed_seconds:.1f}s."
            )
            return GeminiTranscript(text=response_text, model=model)
        except Exception as error:
            last_error = error
            elapsed_seconds = time.monotonic() - generation_started_at
            print_gemini_timing(
                f"transcript generation failed for {audio_input.path.name} with "
                f"{model} after {elapsed_seconds:.1f}s: {type(error).__name__}."
            )
            if not is_transient_gemini_error(error):
                break

    if last_error is None:
        raise GeminiIntegrationError("No Gemini model was configured.")
    raise GeminiIntegrationError(str(last_error)) from last_error


def transcribe_audio_file(audio_file: str) -> TranscriptionOutcome:
    audio_processing: AudioProcessingResult | None = None

    try:
        audio_input = validate_audio_file(audio_file)
        require_audio_tools()
        audio_metadata = get_audio_metadata(audio_input)
        audio_processing = plan_audio_processing(audio_input, audio_metadata)
        transcript = generate_gemini_transcript(audio_input, audio_processing)
        output_path = write_transcript(audio_input, transcript.text)
    finally:
        cleanup_path = cleanup_temporary_audio(audio_processing)

    return TranscriptionOutcome(
        audio_input=audio_input,
        audio_processing=audio_processing,
        transcript=transcript,
        output_path=output_path,
        cleanup_path=cleanup_path,
    )


def print_transcription_success(outcome: TranscriptionOutcome) -> None:
    print("Success: transcript created.")
    print(f"Audio file: {outcome.audio_input.path}")
    print(f"Transcript file: {outcome.output_path}")
    print(f"Gemini model: {outcome.transcript.model}")
    if outcome.cleanup_path is not None:
        print(f"Cleaned up temporary audio file: {outcome.cleanup_path}")


def main() -> int:
    args = parse_args()

    try:
        load_dotenv(Path(__file__).resolve().parent / ".env")
        require_api_key()
        outcome = transcribe_audio_file(args.audio_file)
    except ValidationError as error:
        print(f"Failure: {error}", file=sys.stderr)
        return 1
    except PromptError as error:
        print(f"Failure: {error}", file=sys.stderr)
        return 1
    except GeminiIntegrationError as error:
        print(f"Failure: Gemini API integration failed: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Failure: Could not write transcript file: {error}", file=sys.stderr)
        return 1

    print_transcription_success(outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
