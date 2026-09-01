"""Local checks for intentional full-input rescans."""

from __future__ import annotations

import argparse
import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import call, patch

from summarize_audio import (
    AudioInput,
    AudioProcessingResult,
    GeminiIntegrationError,
    GeminiSummary,
    InputDiscovery,
    MasterSummaryError,
    ProcessingOutcome,
    RescanPreflight,
    SkippedInputFile,
    confirm_rescan,
    main,
    prepare_rescan,
    process_rescan,
)


class RescanTests(unittest.TestCase):
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
            output_path=Path(f"{audio_input.path.stem}-summary-2.md"),
            master_update_action="rewritten",
            cleanup_path=None,
        )

    def test_preflight_selects_all_valid_audio_after_master_validation(self) -> None:
        first_audio = self.make_audio_input("2026-08-29 16_51_37.mp3")
        second_audio = self.make_audio_input("2026-08-30 13_53_39.mp3")
        skipped_file = SkippedInputFile(Path("notes.txt"), "not an MP3 file")
        discovery = InputDiscovery(
            valid_audio=[first_audio, second_audio],
            skipped_files=[skipped_file],
        )

        with (
            patch("summarize_audio.discover_input_files", return_value=discovery),
            patch("summarize_audio.read_master_entries", return_value=[]) as read_master,
        ):
            preflight = prepare_rescan(Path("input"), Path("master.md"))

        self.assertEqual(preflight.selected_audio, [first_audio, second_audio])
        self.assertEqual(preflight.skipped_files, [skipped_file])
        read_master.assert_called_once_with(Path("master.md"))

    def test_malformed_master_stops_preflight_before_confirmation_or_processing(self) -> None:
        args = argparse.Namespace(audio_file=None, scan=False, rescan=True)

        with (
            patch("summarize_audio.parse_args", return_value=args),
            patch(
                "summarize_audio.prepare_rescan",
                side_effect=MasterSummaryError("missing an end marker"),
            ),
            patch("summarize_audio.confirm_rescan") as confirm,
            patch("summarize_audio.process_rescan") as process,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 1)
        confirm.assert_not_called()
        process.assert_not_called()

    def test_confirmation_accepts_only_y_or_yes(self) -> None:
        for response, expected in (
            ("y", True),
            (" YES ", True),
            ("n", False),
            ("", False),
            ("continue", False),
        ):
            with self.subTest(response=response):
                with patch("builtins.input", return_value=response):
                    self.assertEqual(confirm_rescan(3), expected)

    def test_cancelled_rescan_does_not_start_processing(self) -> None:
        audio_input = self.make_audio_input("2026-08-30 13_53_39.mp3")
        preflight = RescanPreflight(selected_audio=[audio_input], skipped_files=[])
        args = argparse.Namespace(audio_file=None, scan=False, rescan=True)
        stdout = io.StringIO()

        with (
            patch("summarize_audio.parse_args", return_value=args),
            patch("summarize_audio.prepare_rescan", return_value=preflight),
            patch("summarize_audio.confirm_rescan", return_value=False),
            patch("summarize_audio.process_rescan") as process,
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        process.assert_not_called()
        self.assertIn("Rescan cancelled. No audio files were processed.", stdout.getvalue())

    def test_rescan_processes_sequentially_and_continues_after_failure(self) -> None:
        successful_audio = self.make_audio_input("2026-08-30 13_53_39.mp3")
        failed_audio = self.make_audio_input("2026-08-30 13_56_27.mp3")
        outcome = self.make_outcome(successful_audio)
        preflight = RescanPreflight(
            selected_audio=[successful_audio, failed_audio],
            skipped_files=[SkippedInputFile(Path("notes.txt"), "not an MP3 file")],
        )

        with (
            patch("summarize_audio.load_dotenv") as load_dotenv,
            patch("summarize_audio.require_api_key") as require_api_key,
            patch(
                "summarize_audio.process_audio_with_transient_retries",
                side_effect=[outcome, GeminiIntegrationError("upload failed")],
            ) as process_audio,
        ):
            result = process_rescan(preflight, Path("master.md"))

        self.assertEqual(result.processed, [outcome])
        self.assertEqual(result.skipped_files, preflight.skipped_files)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].audio_input, failed_audio)
        load_dotenv.assert_called_once()
        require_api_key.assert_called_once()
        process_audio.assert_has_calls(
            [call(successful_audio, Path("master.md")), call(failed_audio, Path("master.md"))]
        )


if __name__ == "__main__":
    unittest.main()
