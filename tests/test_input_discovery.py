"""Local checks for top-level input-folder discovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from summarize_audio import ValidationError, discover_input_files, validate_audio_file


class InputDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.input_directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_discovers_valid_files_chronologically_and_skips_invalid_entries(self) -> None:
        (self.input_directory / "2026-08-30 11_10_13.mp3").write_text(
            "audio", encoding="utf-8"
        )
        (self.input_directory / "2026-08-29 16_51_37.mp3").write_text(
            "audio", encoding="utf-8"
        )
        (self.input_directory / "2026-08-30 12_00_00.ogg").write_text(
            "audio", encoding="utf-8"
        )
        (self.input_directory / "invalid.mp3").write_text("audio", encoding="utf-8")
        (self.input_directory / "notes.txt").write_text("notes", encoding="utf-8")
        nested_directory = self.input_directory / "nested"
        nested_directory.mkdir()
        (nested_directory / "2026-08-31 10_00_00.mp3").write_text(
            "audio", encoding="utf-8"
        )

        discovery = discover_input_files(self.input_directory)

        self.assertEqual(
            [audio.path.name for audio in discovery.valid_audio],
            [
                "2026-08-29 16_51_37.mp3",
                "2026-08-30 11_10_13.mp3",
                "2026-08-30 12_00_00.ogg",
            ],
        )
        self.assertEqual(
            {skipped.path.name for skipped in discovery.skipped_files},
            {"invalid.mp3", "nested", "notes.txt"},
        )

    def test_rejects_a_missing_input_directory(self) -> None:
        missing_directory = self.input_directory / "missing"

        with self.assertRaises(ValidationError):
            discover_input_files(missing_directory)



    def test_accepts_timestamp_named_ogg_and_rejects_other_extensions(self) -> None:
        ogg_path = self.input_directory / "2026-08-30 12_00_00.ogg"
        wav_path = self.input_directory / "2026-08-30 12_00_00.wav"
        ogg_path.write_text("audio", encoding="utf-8")
        wav_path.write_text("audio", encoding="utf-8")

        audio_input = validate_audio_file(str(ogg_path))

        self.assertEqual(audio_input.path, ogg_path)
        with self.assertRaisesRegex(ValidationError, r"accepts .mp3 and .ogg"):
            validate_audio_file(str(wav_path))


if __name__ == "__main__":
    unittest.main()
