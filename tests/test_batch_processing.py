"""Local checks for batch processing, retries, and reporting."""

from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import call, patch

from summarize_audio import (
    AudioInput,
    AudioProcessingResult,
    BatchFailure,
    BatchProcessingResult,
    GeminiIntegrationError,
    GeminiSummary,
    InputDiscovery,
    InputSelection,
    ProcessingOutcome,
    SkippedInputFile,
    print_batch_processing_result,
    process_audio_with_transient_retries,
    process_new_input_files,
)


class BatchProcessingTests(unittest.TestCase):
    def make_audio_input(self, filename: str) -> AudioInput:
        return AudioInput(
            path=Path(filename),
            date_text="30 August 2026",
            time_text="13:53:39",
        )

    def make_outcome(self, audio_input: AudioInput) -> ProcessingOutcome:
        processing = AudioProcessingResult(
            upload_path=audio_input.path,
            original_duration_seconds=60,
            processed_duration_seconds=60,
            removed_seconds=0,
            reduction_percent=0,
            trimmed=False,
            reason="No trimming required.",
        )
        return ProcessingOutcome(
            audio_input=audio_input,
            audio_processing=processing,
            summary=GeminiSummary(markdown="# Summary\n", model="test-model"),
            output_path=Path(f"{audio_input.path.stem}-summary.md"),
            master_update_action="appended",
            cleanup_path=None,
        )

    def test_no_new_files_never_loads_the_api_key(self) -> None:
        already_processed = self.make_audio_input("2026-08-30 11_10_13.mp3")
        discovery = InputDiscovery(
            valid_audio=[already_processed],
            skipped_files=[SkippedInputFile(Path("notes.txt"), "not an MP3 file")],
        )
        selection = InputSelection(new_audio=[], already_processed=[already_processed])

        with (
            patch("summarize_audio.discover_input_files", return_value=discovery),
            patch("summarize_audio.select_new_audio_files", return_value=selection),
            patch("summarize_audio.load_dotenv") as load_dotenv,
            patch("summarize_audio.require_api_key") as require_api_key,
        ):
            result = process_new_input_files(Path("input"), Path("master.md"))

        self.assertEqual(result.selected_audio, [])
        self.assertEqual(result.processed, [])
        self.assertEqual(result.failures, [])
        self.assertEqual(len(result.skipped_files), 2)
        load_dotenv.assert_not_called()
        require_api_key.assert_not_called()

    def test_continues_after_a_file_failure_and_preserves_totals(self) -> None:
        successful_audio = self.make_audio_input("2026-08-30 13_53_39.mp3")
        failed_audio = self.make_audio_input("2026-08-30 13_56_27.mp3")
        outcome = self.make_outcome(successful_audio)
        discovery = InputDiscovery(valid_audio=[successful_audio, failed_audio], skipped_files=[])
        selection = InputSelection(
            new_audio=[successful_audio, failed_audio], already_processed=[]
        )

        with (
            patch("summarize_audio.discover_input_files", return_value=discovery),
            patch("summarize_audio.select_new_audio_files", return_value=selection),
            patch("summarize_audio.load_dotenv"),
            patch("summarize_audio.require_api_key"),
            patch(
                "summarize_audio.process_audio_with_transient_retries",
                side_effect=[outcome, GeminiIntegrationError("upload failed")],
            ) as process_audio,
        ):
            result = process_new_input_files(Path("input"), Path("master.md"))

        self.assertEqual(result.processed, [outcome])
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].audio_input, failed_audio)
        process_audio.assert_has_calls(
            [call(successful_audio, Path("master.md")), call(failed_audio, Path("master.md"))]
        )

    def test_retries_transient_errors_with_configured_waits(self) -> None:
        audio_input = self.make_audio_input("2026-08-30 13_53_39.mp3")

        with (
            patch(
                "summarize_audio.process_audio_file",
                side_effect=[
                    GeminiIntegrationError("server disconnected"),
                    GeminiIntegrationError("server disconnected"),
                    GeminiIntegrationError("server disconnected"),
                    GeminiIntegrationError("server disconnected"),
                ],
            ) as process_audio_file,
            patch("summarize_audio.time.sleep") as sleep,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(GeminiIntegrationError):
                process_audio_with_transient_retries(audio_input, Path("master.md"))

        self.assertEqual(process_audio_file.call_count, 4)
        self.assertEqual(sleep.call_args_list, [call(2), call(4), call(8)])

    def test_does_not_retry_non_transient_gemini_errors(self) -> None:
        audio_input = self.make_audio_input("2026-08-30 13_53_39.mp3")

        with (
            patch(
                "summarize_audio.process_audio_file",
                side_effect=GeminiIntegrationError("invalid request"),
            ) as process_audio_file,
            patch("summarize_audio.time.sleep") as sleep,
        ):
            with self.assertRaises(GeminiIntegrationError):
                process_audio_with_transient_retries(audio_input, Path("master.md"))

        process_audio_file.assert_called_once()
        sleep.assert_not_called()

    def test_reports_no_new_files_and_final_totals(self) -> None:
        skipped = SkippedInputFile(Path("notes.txt"), "not an MP3 file")
        result = BatchProcessingResult(
            selected_audio=[],
            processed=[],
            skipped_files=[skipped],
            failures=[],
        )
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            print_batch_processing_result(result)

        report = stdout.getvalue()
        self.assertIn("No new audio files found.", report)
        self.assertIn("Skipped: notes.txt (not an MP3 file)", report)
        self.assertIn("Batch complete: processed 0, skipped 1, failed 0.", report)

    def test_reports_processed_skipped_and_failed_totals(self) -> None:
        processed_audio = self.make_audio_input("2026-08-30 13_53_39.mp3")
        failed_audio = self.make_audio_input("2026-08-30 13_56_27.mp3")
        result = BatchProcessingResult(
            selected_audio=[processed_audio, failed_audio],
            processed=[self.make_outcome(processed_audio)],
            skipped_files=[SkippedInputFile(Path("notes.txt"), "not an MP3 file")],
            failures=[BatchFailure(failed_audio, GeminiIntegrationError("upload failed"))],
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch("summarize_audio.print_processing_success") as print_success,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            print_batch_processing_result(result)

        print_success.assert_called_once_with(result.processed[0])
        self.assertIn("Skipped: notes.txt (not an MP3 file)", stdout.getvalue())
        self.assertIn("Batch complete: processed 1, skipped 1, failed 1.", stdout.getvalue())
        self.assertIn("Failed: 2026-08-30 13_56_27.mp3 (upload failed)", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
