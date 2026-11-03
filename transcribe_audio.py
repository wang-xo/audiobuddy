"""Create one chronological, English turn-by-turn transcript with Gemini."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from summarize_audio import (
    API_KEY_NAME,
    GEMINI_GENERATION_TIMEOUT_MS,
    GEMINI_GENERATION_TIMEOUT_SECONDS,
    GEMINI_MODELS,
    GEMINI_UPLOAD_TIMEOUT_MS,
    OUTPUT_DIR,
    AudioToolError,
    AudioInput,
    AudioProcessingResult,
    GeminiIntegrationError,
    PromptError,
    ValidationError,
    clean_markdown_response,
    cleanup_temporary_audio,
    detect_long_silences,
    get_audio_metadata,
    is_transient_gemini_error,
    load_dotenv,
    print_gemini_timing,
    require_api_key,
    require_audio_tools,
    validate_audio_file,
)


TRANSCRIPT_PROMPT_PATH = Path(__file__).resolve().parent / "transcript-prompt.md"
TRANSCRIPT_CHUNK_DURATION_SECONDS = 90 * 60
TRANSCRIPT_CHUNK_BOUNDARY_SEARCH_SECONDS = 5 * 60
TRANSCRIPT_CHUNK_OVERLAP_SECONDS = 30
TRANSCRIPT_TURN_PATTERN = re.compile(
    r"^\[(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})\] "
    r"(?P<speaker>[^:]+): (?P<content>\S.*)$"
)
TRANSCRIPT_CODE_FENCE_PATTERN = re.compile(r"^```(?:[a-zA-Z0-9_-]+)?$")
TRANSCRIPT_BOLD_LABEL_PATTERN = re.compile(
    r"^\*\*(?P<label>\[\d{2}:\d{2}:\d{2}\] [^:]+:)\*\*\s*"
    r"(?P<content>\S.*)$"
)
TRANSCRIPT_PREAMBLE_PATTERN = re.compile(
    r"^(?:#+\s*)?(?:complete\s+)?(?:turn-by-turn\s+)?transcript:?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GeminiTranscript:
    text: str
    model: str


@dataclass(frozen=True)
class TranscriptChunk:
    index: int
    source_start_seconds: float
    source_end_seconds: float
    ownership_start_seconds: float
    ownership_end_seconds: float


@dataclass(frozen=True)
class TranscriptTurn:
    timestamp_seconds: int
    speaker: str
    content: str


@dataclass(frozen=True)
class ChunkTranscript:
    chunk: TranscriptChunk
    transcript: GeminiTranscript


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


def build_transcript_prompt(chunk: TranscriptChunk | None = None) -> str:
    """Render the authoritative prompt with source-specific chunk context."""
    if chunk is None:
        context = (
            "This is the full recording. Timestamps must be relative to the "
            "start of the recording."
        )
    else:
        context = (
            f"This is chunk {chunk.index + 1}. Timestamps must be relative to "
            "the start of this chunk, where 00:00:00 is the beginning of the "
            "audio provided to you."
        )

    prompt = load_transcript_prompt()
    if "{chunk_context}" not in prompt:
        raise PromptError(
            "Transcript prompt must include the {chunk_context} placeholder."
        )
    return prompt.replace("{chunk_context}", context)


def format_timestamp(timestamp_seconds: int) -> str:
    hours, remainder = divmod(timestamp_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def normalize_transcript_response(transcript: str) -> str:
    """Remove harmless Markdown wrappers while preserving transcript content.

    Gemini occasionally encloses an otherwise valid response in a Markdown code
    fence or adds a simple ``Transcript`` heading despite the prompt. Those
    wrappers are presentation, not speech. Everything else remains subject to
    the strict turn-by-turn parser below.
    """
    normalized_lines: list[str] = []
    found_turn = False

    for raw_line in transcript.splitlines():
        line = raw_line.strip()
        if not line or TRANSCRIPT_CODE_FENCE_PATTERN.fullmatch(line):
            continue

        bold_label = TRANSCRIPT_BOLD_LABEL_PATTERN.fullmatch(line)
        if bold_label is not None:
            line = f"{bold_label.group('label')} {bold_label.group('content')}"

        if not found_turn and TRANSCRIPT_PREAMBLE_PATTERN.fullmatch(line):
            continue

        if TRANSCRIPT_TURN_PATTERN.match(line):
            found_turn = True
        normalized_lines.append(line)

    return "\n".join(normalized_lines) + ("\n" if normalized_lines else "")


def parse_transcript_turns(transcript: str) -> list[TranscriptTurn]:
    """Reject malformed model output before it can become a final transcript."""
    turns: list[TranscriptTurn] = []
    previous_timestamp = -1

    normalized_transcript = normalize_transcript_response(transcript)
    for line_number, line in enumerate(normalized_transcript.splitlines(), start=1):
        if not line.strip():
            continue

        match = TRANSCRIPT_TURN_PATTERN.match(line)
        if match is None:
            raise GeminiIntegrationError(
                "Gemini transcript has an invalid turn format on line "
                f"{line_number}. Expected '[HH:MM:SS] Speaker N: content'."
            )

        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        if minutes >= 60 or seconds >= 60:
            raise GeminiIntegrationError(
                f"Gemini transcript has an invalid timestamp on line {line_number}."
            )

        timestamp_seconds = (
            int(match.group("hours")) * 3600 + minutes * 60 + seconds
        )
        if timestamp_seconds < previous_timestamp:
            raise GeminiIntegrationError(
                "Gemini transcript timestamps are not in chronological order."
            )

        turns.append(
            TranscriptTurn(
                timestamp_seconds=timestamp_seconds,
                speaker=match.group("speaker"),
                content=match.group("content"),
            )
        )
        previous_timestamp = timestamp_seconds

    if not turns:
        raise GeminiIntegrationError("Gemini returned an empty transcript.")
    return turns


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


def prepare_transcript_audio(
    audio_input: AudioInput, original_duration_seconds: float
) -> AudioProcessingResult:
    """Keep source timing intact for a timestamped transcript."""
    return AudioProcessingResult(
        upload_path=audio_input.path,
        original_duration_seconds=original_duration_seconds,
        processed_duration_seconds=original_duration_seconds,
        removed_seconds=0,
        reduction_percent=0,
        trimmed=False,
        reason="Transcript preserves original timestamps; silence trimming was skipped.",
    )


def choose_chunk_boundary(
    target_seconds: float,
    minimum_seconds: float,
    maximum_seconds: float,
    silence_intervals: list[object],
) -> float:
    """Prefer a nearby long-silence midpoint, otherwise use the target cut."""
    candidates: list[float] = []
    for interval in silence_intervals:
        start_seconds = getattr(interval, "start_seconds")
        end_seconds = getattr(interval, "end_seconds")
        midpoint = (start_seconds + end_seconds) / 2
        if minimum_seconds <= midpoint <= maximum_seconds:
            candidates.append(midpoint)

    if not candidates:
        return target_seconds
    return min(candidates, key=lambda candidate: abs(candidate - target_seconds))


def plan_transcript_chunks(
    audio_path: Path, duration_seconds: float
) -> list[TranscriptChunk]:
    """Create source-timeline chunks with small contextual overlaps."""
    if duration_seconds <= TRANSCRIPT_CHUNK_DURATION_SECONDS:
        return []

    silences = detect_long_silences(audio_path)
    boundaries = [0.0]
    while boundaries[-1] + TRANSCRIPT_CHUNK_DURATION_SECONDS < duration_seconds:
        target_seconds = boundaries[-1] + TRANSCRIPT_CHUNK_DURATION_SECONDS
        remaining_after_target = duration_seconds - target_seconds
        if (
            len(boundaries) > 1
            and remaining_after_target < TRANSCRIPT_CHUNK_DURATION_SECONDS / 2
        ):
            break
        boundary = choose_chunk_boundary(
            target_seconds,
            max(
                boundaries[-1] + TRANSCRIPT_CHUNK_DURATION_SECONDS / 2,
                target_seconds - TRANSCRIPT_CHUNK_BOUNDARY_SEARCH_SECONDS,
            ),
            min(
                duration_seconds - TRANSCRIPT_CHUNK_DURATION_SECONDS / 2,
                target_seconds + TRANSCRIPT_CHUNK_BOUNDARY_SEARCH_SECONDS,
            ),
            silences,
        )
        boundaries.append(boundary)

    boundaries.append(duration_seconds)
    chunks: list[TranscriptChunk] = []
    for index, (ownership_start, ownership_end) in enumerate(
        zip(boundaries, boundaries[1:])
    ):
        chunks.append(
            TranscriptChunk(
                index=index,
                source_start_seconds=max(
                    0.0, ownership_start - TRANSCRIPT_CHUNK_OVERLAP_SECONDS
                ),
                source_end_seconds=min(
                    duration_seconds, ownership_end + TRANSCRIPT_CHUNK_OVERLAP_SECONDS
                ),
                ownership_start_seconds=ownership_start,
                ownership_end_seconds=ownership_end,
            )
        )
    return chunks


def transcript_checkpoint_directory(audio_input: AudioInput) -> Path:
    return OUTPUT_DIR / f".{audio_input.path.stem}-transcript-checkpoint"


def transcript_checkpoint_path(audio_input: AudioInput) -> Path:
    return transcript_checkpoint_directory(audio_input) / "manifest.json"


def transcript_chunk_path(checkpoint_directory: Path, chunk: TranscriptChunk) -> Path:
    return checkpoint_directory / f"chunk-{chunk.index + 1:03d}.ogg"


def transcript_chunk_result_path(
    checkpoint_directory: Path, chunk: TranscriptChunk
) -> Path:
    return checkpoint_directory / f"chunk-{chunk.index + 1:03d}.json"


def transcript_checkpoint_payload(
    audio_input: AudioInput,
    duration_seconds: float,
    chunks: list[TranscriptChunk],
) -> dict[str, object]:
    stat = audio_input.path.stat()
    return {
        "source_filename": audio_input.path.name,
        "source_size_bytes": stat.st_size,
        "source_modified_ns": stat.st_mtime_ns,
        "source_duration_seconds": duration_seconds,
        "chunks": [
            {
                "index": chunk.index,
                "source_start_seconds": chunk.source_start_seconds,
                "source_end_seconds": chunk.source_end_seconds,
                "ownership_start_seconds": chunk.ownership_start_seconds,
                "ownership_end_seconds": chunk.ownership_end_seconds,
            }
            for chunk in chunks
        ],
    }


def prepare_transcript_checkpoint(
    audio_input: AudioInput,
    duration_seconds: float,
    chunks: list[TranscriptChunk],
) -> Path:
    checkpoint_directory = transcript_checkpoint_directory(audio_input)
    checkpoint_path = transcript_checkpoint_path(audio_input)
    payload = transcript_checkpoint_payload(audio_input, duration_seconds, chunks)

    if checkpoint_path.is_file():
        try:
            existing = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if existing == payload:
            return checkpoint_directory
        shutil.rmtree(checkpoint_directory)

    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return checkpoint_directory


def create_transcript_chunk(
    audio_input: AudioInput, chunk: TranscriptChunk, output_path: Path
) -> None:
    """Extract one temporary OGG chunk while keeping source timing in the manifest."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{chunk.source_start_seconds:.3f}",
            "-t",
            f"{chunk.source_end_seconds - chunk.source_start_seconds:.3f}",
            "-i",
            str(audio_input.path),
            "-vn",
            "-c:a",
            "libopus",
            "-b:a",
            "64k",
            str(output_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        details = result.stderr.strip() or "ffmpeg did not create a usable chunk."
        raise AudioToolError(f"Could not create transcript chunk: {details}")


def collect_streamed_transcript(
    chunks: object, generation_started_at: float
) -> str:
    """Collect transcript text from Gemini Generate Content response chunks."""
    text_parts: list[str] = []

    for chunk in chunks:
        elapsed_seconds = time.monotonic() - generation_started_at
        if elapsed_seconds >= GEMINI_GENERATION_TIMEOUT_SECONDS:
            raise GeminiIntegrationError(
                "Gemini transcript stream did not complete within "
                f"{GEMINI_GENERATION_TIMEOUT_SECONDS}s."
            )

        text = getattr(chunk, "text", "") or ""
        if text:
            text_parts.append(text)

    transcript = "".join(text_parts)
    if not transcript.strip():
        raise GeminiIntegrationError("Gemini returned an empty transcript.")
    return transcript


def generate_gemini_transcript(
    audio_input: AudioInput,
    audio_processing: AudioProcessingResult,
    chunk: TranscriptChunk | None = None,
) -> GeminiTranscript:
    try:
        from google import genai
    except ImportError as error:
        raise GeminiIntegrationError(
            "Missing dependency 'google-genai'. Install requirements.txt first."
        ) from error

    upload_client = genai.Client(
        api_key=os.environ[API_KEY_NAME],
        http_options={
            "timeout": GEMINI_UPLOAD_TIMEOUT_MS,
            "retry_options": {"attempts": 1},
        },
    )
    prompt = build_transcript_prompt(chunk)

    upload_started_at = time.monotonic()
    print_gemini_timing(
        f"transcript upload started for {audio_input.path.name} "
        f"(timeout {GEMINI_UPLOAD_TIMEOUT_MS // 1000}s)."
    )
    try:
        uploaded_file = upload_client.files.upload(file=audio_processing.upload_path)
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
    generation_client = genai.Client(
        api_key=os.environ[API_KEY_NAME],
        http_options={
            "timeout": GEMINI_GENERATION_TIMEOUT_MS,
            "retry_options": {"attempts": 1},
        },
    )
    for model in GEMINI_MODELS:
        generation_started_at = time.monotonic()
        print_gemini_timing(
            f"transcript generation started for {audio_input.path.name} with "
            f"{model} (timeout {GEMINI_GENERATION_TIMEOUT_SECONDS}s)."
        )
        try:
            chunks = generation_client.models.generate_content_stream(
                model=model,
                contents=[prompt, uploaded_file],
                config={
                    "thinking_config": {"thinking_level": "low"},
                    "automatic_function_calling": {"disable": True},
                },
            )
            print_gemini_timing(
                f"transcript content stream opened for {audio_input.path.name} "
                f"with {model} (low thinking)."
            )
            response_text = normalize_transcript_response(collect_streamed_transcript(
                chunks, generation_started_at
            ))
            parse_transcript_turns(response_text)
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


def read_chunk_transcript(result_path: Path) -> GeminiTranscript | None:
    if not result_path.is_file():
        return None

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        transcript = GeminiTranscript(
            text=str(payload["text"]), model=str(payload["model"])
        )
        parse_transcript_turns(transcript.text)
        return transcript
    except (KeyError, TypeError, ValueError, GeminiIntegrationError):
        result_path.unlink(missing_ok=True)
        return None


def write_chunk_transcript(result_path: Path, transcript: GeminiTranscript) -> None:
    result_path.write_text(
        json.dumps({"text": transcript.text, "model": transcript.model}) + "\n",
        encoding="utf-8",
    )


def normalize_turn_content(content: str) -> str:
    return re.sub(r"\W+", "", content).lower()


def reconcile_chunk_speakers(
    previous_turns: list[TranscriptTurn], current_turns: list[TranscriptTurn]
) -> dict[str, str]:
    """Map only unmistakable overlap matches; uncertainty gets a new label."""
    matches: dict[str, set[str]] = {}
    for current_turn in current_turns:
        if not re.fullmatch(r"Speaker \d+", current_turn.speaker):
            continue
        current_content = normalize_turn_content(current_turn.content)
        if len(current_content) < 12:
            continue
        for previous_turn in previous_turns:
            if (
                normalize_turn_content(previous_turn.content) == current_content
                and abs(previous_turn.timestamp_seconds - current_turn.timestamp_seconds)
                <= 3
            ):
                matches.setdefault(current_turn.speaker, set()).add(previous_turn.speaker)

    return {
        speaker: next(iter(candidates))
        for speaker, candidates in matches.items()
        if len(candidates) == 1
    }


def next_speaker_number(turns: list[TranscriptTurn]) -> int:
    numbers = [
        int(match.group(1))
        for turn in turns
        if (match := re.fullmatch(r"Speaker (\d+)", turn.speaker))
    ]
    return max(numbers, default=0) + 1


def combine_chunk_transcripts(chunk_transcripts: list[ChunkTranscript]) -> GeminiTranscript:
    """Offset turns to source time and deterministically drop overlap duplicates."""
    combined_turns: list[TranscriptTurn] = []
    previous_chunk_turns: list[TranscriptTurn] = []
    models: list[str] = []

    for chunk_transcript in chunk_transcripts:
        chunk = chunk_transcript.chunk
        source_turns = [
            TranscriptTurn(
                timestamp_seconds=int(round(chunk.source_start_seconds))
                + turn.timestamp_seconds,
                speaker=turn.speaker,
                content=turn.content,
            )
            for turn in parse_transcript_turns(chunk_transcript.transcript.text)
        ]
        reconciliation = reconcile_chunk_speakers(previous_chunk_turns, source_turns)
        next_number = next_speaker_number(combined_turns)
        speaker_map: dict[str, str] = dict(reconciliation)

        translated_turns: list[TranscriptTurn] = []
        for turn in source_turns:
            speaker = turn.speaker
            if re.fullmatch(r"Speaker \d+", speaker):
                if speaker not in speaker_map:
                    speaker_map[speaker] = f"Speaker {next_number}"
                    next_number += 1
                speaker = speaker_map[speaker]
            translated_turns.append(
                TranscriptTurn(
                    timestamp_seconds=turn.timestamp_seconds,
                    speaker=speaker,
                    content=turn.content,
                )
            )

        for turn in translated_turns:
            is_last_chunk = chunk_transcript is chunk_transcripts[-1]
            if (
                turn.timestamp_seconds >= int(round(chunk.ownership_start_seconds))
                and (
                    is_last_chunk
                    or turn.timestamp_seconds
                    < int(round(chunk.ownership_end_seconds))
                )
            ):
                combined_turns.append(turn)

        previous_chunk_turns = translated_turns
        models.append(chunk_transcript.transcript.model)

    if not combined_turns:
        raise GeminiIntegrationError("No transcript turns remained after chunk recombination.")

    rendered = "\n".join(
        f"[{format_timestamp(turn.timestamp_seconds)}] {turn.speaker}: {turn.content}"
        for turn in combined_turns
    )
    model = models[0] if len(set(models)) == 1 else "mixed Gemini models"
    return GeminiTranscript(text=rendered + "\n", model=model)


def generate_chunked_transcript(
    audio_input: AudioInput, duration_seconds: float
) -> GeminiTranscript:
    chunks = plan_transcript_chunks(audio_input.path, duration_seconds)
    if not chunks:
        raise GeminiIntegrationError("Transcript chunking was requested without chunks.")

    checkpoint_directory = prepare_transcript_checkpoint(
        audio_input, duration_seconds, chunks
    )
    chunk_transcripts: list[ChunkTranscript] = []

    try:
        for chunk in chunks:
            result_path = transcript_chunk_result_path(checkpoint_directory, chunk)
            transcript = read_chunk_transcript(result_path)
            if transcript is None:
                chunk_path = transcript_chunk_path(checkpoint_directory, chunk)
                print_gemini_timing(
                    f"transcript chunk {chunk.index + 1}/{len(chunks)} started for "
                    f"{audio_input.path.name} ({format_timestamp(int(chunk.ownership_start_seconds))} "
                    f"to {format_timestamp(int(chunk.ownership_end_seconds))})."
                )
                try:
                    create_transcript_chunk(audio_input, chunk, chunk_path)
                    chunk_input = AudioInput(
                        path=chunk_path,
                        date_text=audio_input.date_text,
                        time_text=audio_input.time_text,
                    )
                    chunk_processing = prepare_transcript_audio(
                        chunk_input,
                        chunk.source_end_seconds - chunk.source_start_seconds,
                    )
                    transcript = generate_gemini_transcript(
                        chunk_input, chunk_processing, chunk
                    )
                    write_chunk_transcript(result_path, transcript)
                finally:
                    chunk_path.unlink(missing_ok=True)
            else:
                print_gemini_timing(
                    f"transcript chunk {chunk.index + 1}/{len(chunks)} resumed for "
                    f"{audio_input.path.name}."
                )
            chunk_transcripts.append(ChunkTranscript(chunk, transcript))

        return combine_chunk_transcripts(chunk_transcripts)
    except Exception:
        print_gemini_timing(
            "transcript chunking stopped; completed chunks were kept for a later resume."
        )
        raise


def transcribe_audio_file(audio_file: str) -> TranscriptionOutcome:
    audio_processing: AudioProcessingResult | None = None
    checkpoint_directory: Path | None = None

    try:
        audio_input = validate_audio_file(audio_file)
        require_audio_tools()
        audio_metadata = get_audio_metadata(audio_input)
        audio_processing = prepare_transcript_audio(
            audio_input, audio_metadata.original_duration_seconds
        )
        if (
            audio_metadata.original_duration_seconds
            > TRANSCRIPT_CHUNK_DURATION_SECONDS
        ):
            checkpoint_directory = transcript_checkpoint_directory(audio_input)
            transcript = generate_chunked_transcript(
                audio_input, audio_metadata.original_duration_seconds
            )
        else:
            transcript = generate_gemini_transcript(audio_input, audio_processing)
        output_path = write_transcript(audio_input, transcript.text)
        if checkpoint_directory is not None:
            shutil.rmtree(checkpoint_directory, ignore_errors=True)
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
    except AudioToolError as error:
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
