"""Local checks for Gemini transcript timing, deadlines, and fallback."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from summarize_audio import (
    API_KEY_NAME,
    GEMINI_GENERATION_TIMEOUT_MS,
    GEMINI_GENERATION_TIMEOUT_SECONDS,
    GEMINI_UPLOAD_TIMEOUT_MS,
    AudioInput,
    AudioProcessingResult,
)
from transcribe_audio import generate_gemini_transcript


class TranscriptGeminiIntegrationTests(unittest.TestCase):
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

    def fake_genai(self, create_results: list[object]):
        uploaded_file = SimpleNamespace(uri="files/test-audio", mime_type="audio/mpeg")
        client = SimpleNamespace(
            files=SimpleNamespace(upload=Mock(return_value=uploaded_file)),
            models=SimpleNamespace(
                generate_content_stream=Mock(side_effect=create_results)
            ),
        )
        return client, SimpleNamespace(Client=Mock(return_value=client))

    def test_uses_the_authoritative_prompt_and_explicit_deadline(self) -> None:
        chunks = [
            SimpleNamespace(text="[00:00:00] Speaker 1: "),
            SimpleNamespace(text="Hello.\n"),
        ]
        client, genai = self.fake_genai([chunks])

        with (
            patch.dict(sys.modules, {"google": SimpleNamespace(genai=genai)}),
            patch.dict(os.environ, {API_KEY_NAME: "test-key"}),
            patch("transcribe_audio.print_gemini_timing"),
        ):
            transcript = generate_gemini_transcript(
                self.audio_input, self.audio_processing
            )

        self.assertEqual(transcript.model, "gemini-3.8-flash")
        self.assertEqual(transcript.text, "[00:00:00] Speaker 1: Hello.\n")
        genai.Client.assert_has_calls(
            [
                call(
                    api_key="test-key",
                    http_options={
                        "timeout": GEMINI_UPLOAD_TIMEOUT_MS,
                        "retry_options": {"attempts": 1},
                    },
                ),
                call(
                    api_key="test-key",
                    http_options={
                        "timeout": GEMINI_GENERATION_TIMEOUT_MS,
                        "retry_options": {"attempts": 1},
                    },
                ),
            ]
        )
        request = client.models.generate_content_stream.call_args.kwargs
        self.assertEqual(request["model"], "gemini-3.8-flash")
        self.assertIn("fully transcribe everything", request["contents"][0])
        self.assertEqual(request["contents"][1], client.files.upload.return_value)
        self.assertEqual(
            request["config"],
            {
                "thinking_config": {"thinking_level": "low"},
                "automatic_function_calling": {"disable": True},
            },
        )

    def test_transient_failure_falls_back_to_the_next_model(self) -> None:
        chunks = [SimpleNamespace(text="[00:00:00] Speaker 1: Hello.\n")]
        client, genai = self.fake_genai([TimeoutError("request timed out"), chunks])

        with (
            patch.dict(sys.modules, {"google": SimpleNamespace(genai=genai)}),
            patch.dict(os.environ, {API_KEY_NAME: "test-key"}),
            patch("transcribe_audio.print_gemini_timing"),
        ):
            transcript = generate_gemini_transcript(
                self.audio_input, self.audio_processing
            )

        self.assertEqual(transcript.model, "gemini-3.6-flash")
        self.assertEqual(
            [
                request.kwargs["model"]
                for request in client.models.generate_content_stream.call_args_list
            ],
            ["gemini-3.8-flash", "gemini-3.6-flash"],
        )

    def test_rejects_an_empty_content_stream(self) -> None:
        client, genai = self.fake_genai([[]])

        with (
            patch.dict(sys.modules, {"google": SimpleNamespace(genai=genai)}),
            patch.dict(os.environ, {API_KEY_NAME: "test-key"}),
            patch("transcribe_audio.print_gemini_timing"),
        ):
            with self.assertRaisesRegex(
                Exception, "Gemini returned an empty transcript"
            ):
                generate_gemini_transcript(self.audio_input, self.audio_processing)

    def test_generation_deadline_errors_are_transient_for_model_fallback(self) -> None:
        chunks = [SimpleNamespace(text="[00:00:00] Speaker 1: Hello.\n")]
        client, genai = self.fake_genai(
            [RuntimeError("504 DEADLINE_EXCEEDED"), chunks]
        )

        with (
            patch.dict(sys.modules, {"google": SimpleNamespace(genai=genai)}),
            patch.dict(os.environ, {API_KEY_NAME: "test-key"}),
            patch("transcribe_audio.print_gemini_timing"),
        ):
            transcript = generate_gemini_transcript(
                self.audio_input, self.audio_processing
            )

        self.assertEqual(transcript.model, "gemini-3.6-flash")


if __name__ == "__main__":
    unittest.main()
