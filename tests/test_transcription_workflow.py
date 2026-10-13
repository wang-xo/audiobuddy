"""Local checks for the separate turn-by-turn transcription workflow."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from summarize_audio import AudioInput, AudioMetadata, AudioProcessingResult, GeminiIntegrationError
from transcribe_audio import GeminiTranscript, transcribe_audio_file


class TranscriptionWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_input = AudioInput(
            path=Path("2026-08-30 13_53_39.mp3"),
            date_text="30 August 2026",
            time_text="13:53:39",
        )
        self.audio_processing = AudioProcessingResult(
            upload_path=Path("temporary-trimmed.mp3"),
            original_duration_seconds=700,
            processed_duration_seconds=600,
            removed_seconds=100,
            reduction_percent=14.29,
            trimmed=True,
            reason="Test trimming.",
            temporary_file=True,
        )
        self.transcript = GeminiTranscript(
            text="[00:00:00] Speaker 1: Test transcript.\n",
            model="test-model",
        )

    @patch("transcribe_audio.cleanup_temporary_audio")
    @patch("transcribe_audio.write_transcript", return_value=Path("transcript.md"))
    @patch("transcribe_audio.generate_gemini_transcript")
    @patch("transcribe_audio.get_audio_metadata")
    @patch("transcribe_audio.require_audio_tools")
    @patch("transcribe_audio.validate_audio_file")
    def test_processes_one_file_without_updating_the_master_summary(
        self,
        validate_audio_file,
        require_audio_tools,
        get_audio_metadata,
        generate_gemini_transcript,
        write_transcript,
        cleanup_temporary_audio,
    ) -> None:
        validate_audio_file.return_value = self.audio_input
        get_audio_metadata.return_value = AudioMetadata(original_duration_seconds=700)
        generate_gemini_transcript.return_value = self.transcript
        cleanup_temporary_audio.return_value = None

        outcome = transcribe_audio_file(str(self.audio_input.path))

        self.assertEqual(outcome.audio_input, self.audio_input)
        self.assertEqual(outcome.audio_processing.upload_path, self.audio_input.path)
        self.assertFalse(outcome.audio_processing.trimmed)
        self.assertEqual(outcome.output_path, Path("transcript.md"))
        self.assertIsNone(outcome.cleanup_path)
        write_transcript.assert_called_once_with(self.audio_input, self.transcript.text)
        cleanup_temporary_audio.assert_called_once_with(outcome.audio_processing)

    @patch("transcribe_audio.cleanup_temporary_audio")
    @patch("transcribe_audio.generate_gemini_transcript")
    @patch("transcribe_audio.get_audio_metadata")
    @patch("transcribe_audio.require_audio_tools")
    @patch("transcribe_audio.validate_audio_file")
    def test_cleans_up_when_transcript_generation_fails(
        self,
        validate_audio_file,
        require_audio_tools,
        get_audio_metadata,
        generate_gemini_transcript,
        cleanup_temporary_audio,
    ) -> None:
        validate_audio_file.return_value = self.audio_input
        get_audio_metadata.return_value = AudioMetadata(original_duration_seconds=700)
        generate_gemini_transcript.side_effect = GeminiIntegrationError("test failure")

        with self.assertRaises(GeminiIntegrationError):
            transcribe_audio_file(str(self.audio_input.path))

        cleanup_temporary_audio.assert_called_once()
        self.assertFalse(cleanup_temporary_audio.call_args.args[0].trimmed)


if __name__ == "__main__":
    unittest.main()
