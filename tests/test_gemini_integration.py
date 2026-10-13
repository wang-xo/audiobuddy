"""Local checks for Gemini timing, timeout, and fallback behavior."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from summarize_audio import (
    API_KEY_NAME,
    GEMINI_GENERATION_TIMEOUT_SECONDS,
    GEMINI_UPLOAD_TIMEOUT_MS,
    AudioInput,
    AudioProcessingResult,
    GeminiIntegrationError,
    generate_gemini_summary,
    is_transient_gemini_error,
)


class GeminiIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_input = AudioInput(
            path=Path("2026-08-30 11_10_13.mp3"),
            date_text="30 August 2026",
            time_text="11:10:13",
        )
        self.audio_processing = AudioProcessingResult(
            upload_path=self.audio_input.path,
            original_duration_seconds=60,
            processed_duration_seconds=60,
            removed_seconds=0,
            reduction_percent=0,
            trimmed=False,
            reason="No trimming required.",
        )

    def fake_genai(self, upload_result: object, interaction_results: list[object]):
        client = SimpleNamespace(
            files=SimpleNamespace(upload=Mock(return_value=upload_result)),
            interactions=SimpleNamespace(
                create=Mock(side_effect=interaction_results)
            ),
        )
        return client, SimpleNamespace(Client=Mock(return_value=client))

    def test_records_upload_and_generation_timings_with_explicit_deadlines(self) -> None:
        uploaded_file = SimpleNamespace(uri="files/test-audio", mime_type="audio/mpeg")
        response = SimpleNamespace(output_text="# Audio Summary\n")
        client, genai = self.fake_genai(uploaded_file, [response])
        stderr = io.StringIO()

        with (
            patch.dict(sys.modules, {"google": SimpleNamespace(genai=genai)}),
            patch.dict(os.environ, {API_KEY_NAME: "test-key"}),
            patch("summarize_audio.time.monotonic", side_effect=[10, 12.5, 13, 18]),
            contextlib.redirect_stderr(stderr),
        ):
            summary = generate_gemini_summary(self.audio_input, self.audio_processing)

        self.assertEqual(summary.model, "gemini-3.8-flash")
        genai.Client.assert_called_once_with(
            api_key="test-key",
            http_options={
                "timeout": GEMINI_UPLOAD_TIMEOUT_MS,
                "retry_options": {"attempts": 1},
            },
        )
        client.files.upload.assert_called_once_with(file=self.audio_input.path)
        self.assertEqual(
            client.interactions.create.call_args.kwargs["timeout"],
            GEMINI_GENERATION_TIMEOUT_SECONDS,
        )
        report = stderr.getvalue()
        self.assertIn("upload started", report)
        self.assertIn("upload completed", report)
        self.assertIn("generation started", report)
        self.assertIn("generation completed", report)

    def test_generation_timeout_falls_back_to_the_next_model(self) -> None:
        uploaded_file = SimpleNamespace(uri="files/test-audio", mime_type="audio/mpeg")
        response = SimpleNamespace(output_text="# Audio Summary\n")
        client, genai = self.fake_genai(
            uploaded_file,
            [TimeoutError("request timed out"), response],
        )

        with (
            patch.dict(sys.modules, {"google": SimpleNamespace(genai=genai)}),
            patch.dict(os.environ, {API_KEY_NAME: "test-key"}),
            patch("summarize_audio.time.monotonic", return_value=1),
            patch("summarize_audio.print_gemini_timing"),
        ):
            summary = generate_gemini_summary(self.audio_input, self.audio_processing)

        self.assertEqual(summary.model, "gemini-3.6-flash")
        self.assertEqual(
            [call_args.kwargs["model"] for call_args in client.interactions.create.call_args_list],
            ["gemini-3.8-flash", "gemini-3.6-flash"],
        )

    def test_upload_timeout_is_reported_with_its_elapsed_time(self) -> None:
        client, genai = self.fake_genai(None, [])
        client.files.upload.side_effect = TimeoutError("request timed out")

        with (
            patch.dict(sys.modules, {"google": SimpleNamespace(genai=genai)}),
            patch.dict(os.environ, {API_KEY_NAME: "test-key"}),
            patch("summarize_audio.time.monotonic", side_effect=[10, 15]),
            patch("summarize_audio.print_gemini_timing") as print_timing,
        ):
            with self.assertRaisesRegex(
                GeminiIntegrationError, "Audio upload failed after 5.0s"
            ):
                generate_gemini_summary(self.audio_input, self.audio_processing)

        self.assertIn("upload failed", print_timing.call_args.args[0])

    def test_timeouts_are_transient_for_file_level_retries(self) -> None:
        self.assertTrue(is_transient_gemini_error(TimeoutError("request timed out")))

    def test_generation_deadline_errors_are_transient_for_file_level_retries(self) -> None:
        self.assertTrue(
            is_transient_gemini_error(
                GeminiIntegrationError("504 DEADLINE_EXCEEDED")
            )
        )

    def test_stalled_transcript_streams_are_transient_for_file_level_retries(self) -> None:
        self.assertTrue(
            is_transient_gemini_error(
                GeminiIntegrationError("Gemini transcript stream did not complete within 600s.")
            )
        )


if __name__ == "__main__":
    unittest.main()
