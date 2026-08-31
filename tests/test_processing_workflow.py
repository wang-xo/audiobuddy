"""Local checks for the reusable one-file processing workflow."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from summarize_audio import (
    AudioInput,
    AudioMetadata,
    AudioProcessingResult,
    GeminiIntegrationError,
    GeminiSummary,
    process_audio_file,
)


class ProcessingWorkflowTests(unittest.TestCase):
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
        self.summary = GeminiSummary(markdown="# Summary\n", model="test-model")

    @patch("summarize_audio.cleanup_temporary_audio")
    @patch("summarize_audio.update_master_summary", return_value="created")
    @patch("summarize_audio.write_summary", return_value=Path("summary.md"))
    @patch("summarize_audio.generate_gemini_summary")
    @patch("summarize_audio.plan_audio_processing")
    @patch("summarize_audio.get_audio_metadata")
    @patch("summarize_audio.require_audio_tools")
    @patch("summarize_audio.validate_audio_file")
    def test_processes_one_file_and_returns_its_outcome(
        self,
        validate_audio_file,
        require_audio_tools,
        get_audio_metadata,
        plan_audio_processing,
        generate_gemini_summary,
        write_summary,
        update_master_summary,
        cleanup_temporary_audio,
    ) -> None:
        validate_audio_file.return_value = self.audio_input
        get_audio_metadata.return_value = AudioMetadata(original_duration_seconds=700)
        plan_audio_processing.return_value = self.audio_processing
        generate_gemini_summary.return_value = self.summary
        cleanup_temporary_audio.return_value = self.audio_processing.upload_path

        outcome = process_audio_file(str(self.audio_input.path))

        self.assertEqual(outcome.audio_input, self.audio_input)
        self.assertEqual(outcome.output_path, Path("summary.md"))
        self.assertEqual(outcome.master_update_action, "created")
        self.assertEqual(outcome.cleanup_path, self.audio_processing.upload_path)
        write_summary.assert_called_once_with(self.audio_input, self.summary.markdown)
        update_master_summary.assert_called_once()
        cleanup_temporary_audio.assert_called_once_with(self.audio_processing)

    @patch("summarize_audio.cleanup_temporary_audio")
    @patch("summarize_audio.generate_gemini_summary")
    @patch("summarize_audio.plan_audio_processing")
    @patch("summarize_audio.get_audio_metadata")
    @patch("summarize_audio.require_audio_tools")
    @patch("summarize_audio.validate_audio_file")
    def test_cleans_up_when_summary_generation_fails(
        self,
        validate_audio_file,
        require_audio_tools,
        get_audio_metadata,
        plan_audio_processing,
        generate_gemini_summary,
        cleanup_temporary_audio,
    ) -> None:
        validate_audio_file.return_value = self.audio_input
        get_audio_metadata.return_value = AudioMetadata(original_duration_seconds=700)
        plan_audio_processing.return_value = self.audio_processing
        generate_gemini_summary.side_effect = GeminiIntegrationError("test failure")

        with self.assertRaises(GeminiIntegrationError):
            process_audio_file(str(self.audio_input.path))

        cleanup_temporary_audio.assert_called_once_with(self.audio_processing)


if __name__ == "__main__":
    unittest.main()
