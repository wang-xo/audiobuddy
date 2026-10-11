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

    def fake_genai(self, interaction_results: list[object]):
        uploaded_file = SimpleNamespace(uri="files/test-audio", mime_type="audio/mpeg")
        client = SimpleNamespace(
            files=SimpleNamespace(upload=Mock(return_value=uploaded_file)),
            interactions=SimpleNamespace(create=Mock(side_effect=interaction_results)),
        )
        return client, SimpleNamespace(Client=Mock(return_value=client))

    def test_uses_the_authoritative_prompt_and_explicit_deadline(self) -> None:
        response = SimpleNamespace(output_text="[00:00:00] Speaker 1: Hello.\n")
        client, genai = self.fake_genai([response])

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
        genai.Client.assert_called_once_with(
            api_key="test-key",
            http_options={
                "timeout": GEMINI_UPLOAD_TIMEOUT_MS,
                "retry_options": {"attempts": 1},
            },
        )
        request = client.interactions.create.call_args.kwargs
        self.assertEqual(request["timeout"], GEMINI_GENERATION_TIMEOUT_SECONDS)
        self.assertIn("fully transcribe everything", request["input"][0]["text"])

    def test_transient_failure_falls_back_to_the_next_model(self) -> None:
        response = SimpleNamespace(output_text="[00:00:00] Speaker 1: Hello.\n")
        client, genai = self.fake_genai([TimeoutError("request timed out"), response])

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
            [request.kwargs["model"] for request in client.interactions.create.call_args_list],
            ["gemini-3.8-flash", "gemini-3.6-flash"],
        )


if __name__ == "__main__":
    unittest.main()
