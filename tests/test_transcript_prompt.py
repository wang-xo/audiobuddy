"""Local checks for the authoritative transcript prompt and output naming."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from transcribe_audio import load_transcript_prompt, unique_transcript_path


class TranscriptPromptTests(unittest.TestCase):
    def test_prompt_contains_the_agreed_transcription_contract(self) -> None:
        prompt = load_transcript_prompt()

        self.assertIn("fully transcribe everything", prompt)
        self.assertIn("complete transcript in English", prompt)
        self.assertIn("[HH:MM:SS] Speaker 1: Spoken content", prompt)
        self.assertIn("use [00:16:35], never [16:35]", prompt)
        self.assertIn("stable numbered speaker labels", prompt)
        self.assertIn("[inaudible]", prompt)
        self.assertIn("square brackets", prompt)
        self.assertIn("Do not add a title", prompt)

    def test_output_names_are_independent_from_summary_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            audio_path = Path("2026-08-30 11_10_13.mp3")

            first_path = unique_transcript_path(audio_path, output_dir)
            first_path.touch()
            second_path = unique_transcript_path(audio_path, output_dir)

        self.assertEqual(first_path.name, "2026-08-30 11_10_13-transcript.md")
        self.assertEqual(second_path.name, "2026-08-30 11_10_13-transcript-1.md")


if __name__ == "__main__":
    unittest.main()
