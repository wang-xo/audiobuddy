"""Local checks for the authoritative Gemini prompt template."""

from __future__ import annotations

import unittest
from pathlib import Path

from summarize_audio import (
    AudioInput,
    AudioProcessingResult,
    build_summary_prompt,
)


EXPECTED_HEADINGS = (
    "# Audio Summary",
    "## Date and Time",
    "## Audio Processing",
    "## Detailed Summary",
    "## Discussion Path",
    "## Decisions",
    "## Action Items",
    "## Open Questions",
)


class PromptTemplateTests(unittest.TestCase):
    def test_renders_revised_headings_and_runtime_placeholders(self) -> None:
        audio_input = AudioInput(
            path=Path("2026-08-30 13_53_39.mp3"),
            date_text="30 August 2026",
            time_text="13:53:39",
        )
        audio_processing = AudioProcessingResult(
            upload_path=audio_input.path,
            original_duration_seconds=60,
            processed_duration_seconds=60,
            removed_seconds=0,
            reduction_percent=0,
            trimmed=False,
            reason="No trimming required.",
        )

        prompt = build_summary_prompt(audio_input, audio_processing)

        self.assertIn("Date: 30 August 2026", prompt)
        self.assertIn("Time: 13:53:39", prompt)
        self.assertIn("- Original duration: 1:00 (60.00 seconds)", prompt)
        self.assertIn("under 1,500 words", prompt)
        self.assertNotIn("{date}", prompt)
        self.assertNotIn("{time}", prompt)
        self.assertNotIn("{audio_processing_summary}", prompt)
        self.assertNotIn("## Executive Summary", prompt)
        self.assertNotIn("## Verbose Summary", prompt)
        self.assertNotIn("## Key Discussion Points", prompt)
        self.assertEqual(
            [prompt.index(heading) for heading in EXPECTED_HEADINGS],
            sorted(prompt.index(heading) for heading in EXPECTED_HEADINGS),
        )


if __name__ == "__main__":
    unittest.main()
