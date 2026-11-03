"""Local checks for source-timeline transcript chunking and recombination."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from summarize_audio import AudioInput, SilenceInterval
from transcribe_audio import (
    GeminiIntegrationError,
    GeminiTranscript,
    ChunkTranscript,
    TranscriptChunk,
    combine_chunk_transcripts,
    parse_transcript_turns,
    plan_transcript_chunks,
    prepare_transcript_checkpoint,
)


class TranscriptChunkingTests(unittest.TestCase):
    @patch("transcribe_audio.detect_long_silences")
    def test_plan_uses_nearby_silence_and_preserves_overlap(self, detect_long_silences) -> None:
        detect_long_silences.return_value = [
            SilenceInterval(5350, 5410, 60),
        ]

        chunks = plan_transcript_chunks(Path("recording.ogg"), 17_000)

        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(chunks[0].ownership_start_seconds, 0)
        self.assertEqual(chunks[-1].ownership_end_seconds, 17_000)
        self.assertEqual(chunks[0].ownership_end_seconds, 5380)
        self.assertEqual(chunks[1].source_start_seconds, 5350)
        self.assertEqual(chunks[0].source_end_seconds, 5410)

    @patch("transcribe_audio.detect_long_silences", return_value=[])
    def test_plan_avoids_a_tiny_final_chunk(self, _detect_long_silences) -> None:
        chunks = plan_transcript_chunks(Path("recording.ogg"), 11_000)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].ownership_end_seconds, 5400)
        self.assertEqual(chunks[1].ownership_end_seconds, 11_000)

    @patch("transcribe_audio.detect_long_silences", return_value=[])
    def test_plan_splits_a_two_hour_recording(self, _detect_long_silences) -> None:
        chunks = plan_transcript_chunks(Path("recording.ogg"), 7_548)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].ownership_end_seconds, 5400)
        self.assertEqual(chunks[1].ownership_start_seconds, 5400)

    def test_recombine_offsets_timestamps_and_owns_each_overlap_once(self) -> None:
        first = TranscriptChunk(
            index=0,
            source_start_seconds=0,
            source_end_seconds=130,
            ownership_start_seconds=0,
            ownership_end_seconds=100,
        )
        second = TranscriptChunk(
            index=1,
            source_start_seconds=70,
            source_end_seconds=200,
            ownership_start_seconds=100,
            ownership_end_seconds=200,
        )
        combined = combine_chunk_transcripts(
            [
                ChunkTranscript(
                    first,
                    GeminiTranscript(
                        "[00:01:30] Speaker 1: Shared boundary statement.\n"
                        "[00:01:40] Speaker 2: This belongs to the next chunk.\n",
                        "gemini-3.8-flash",
                    ),
                ),
                ChunkTranscript(
                    second,
                    GeminiTranscript(
                        "[00:00:20] Speaker 1: Shared boundary statement.\n"
                        "[00:00:40] Speaker 1: Continued after the boundary.\n",
                        "gemini-3.8-flash",
                    ),
                ),
            ]
        )

        self.assertEqual(
            combined.text,
            "[00:01:30] Speaker 1: Shared boundary statement.\n"
            "[00:01:50] Speaker 1: Continued after the boundary.\n",
        )

    def test_rejects_malformed_transcript_turns(self) -> None:
        with self.assertRaisesRegex(GeminiIntegrationError, "invalid turn format"):
            parse_transcript_turns("Speaker 1: Missing timestamp.\n")

    def test_accepts_harmless_markdown_wrappers(self) -> None:
        turns = parse_transcript_turns(
            "```markdown\n"
            "# Transcript\n"
            "**[00:00:00] Speaker 1:** Hello.\n"
            "```\n"
        )

        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].speaker, "Speaker 1")
        self.assertEqual(turns[0].content, "Hello.")

    def test_does_not_discard_unexpected_preamble(self) -> None:
        with self.assertRaisesRegex(GeminiIntegrationError, "invalid turn format"):
            parse_transcript_turns(
                "Here is the requested transcript:\n"
                "[00:00:00] Speaker 1: Hello.\n"
            )

    def test_checkpoint_is_reused_only_for_the_same_source_and_plan(self) -> None:
        chunk = TranscriptChunk(0, 0, 100, 0, 100)
        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            audio_path = temporary_path / "2026-08-30 11_10_13.ogg"
            audio_path.write_bytes(b"audio")
            audio_input = AudioInput(audio_path, "30 August 2026", "11:10:13")

            with patch("transcribe_audio.OUTPUT_DIR", temporary_path):
                first = prepare_transcript_checkpoint(audio_input, 100, [chunk])
                second = prepare_transcript_checkpoint(audio_input, 100, [chunk])

            self.assertEqual(first, second)
            self.assertTrue((first / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
