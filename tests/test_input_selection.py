"""Local checks for selecting new audio files from a master summary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from summarize_audio import (
    InputDiscovery,
    MasterSummaryError,
    master_entry_end_marker,
    master_entry_start_marker,
    select_new_audio_files,
    validate_audio_file,
)


class InputSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.master_path = self.directory / "master-summary.md"
        self.first_audio = self.make_audio_file("2026-08-29 16_51_37.mp3")
        self.second_audio = self.make_audio_file("2026-08-30 13_53_39.mp3")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_audio_file(self, filename: str):
        audio_path = self.directory / filename
        audio_path.write_text("audio", encoding="utf-8")
        return validate_audio_file(str(audio_path))

    def write_master_entry(self, filename: str) -> None:
        self.master_path.write_text(
            "\n".join(
                [
                    "# AudioBuddy Master Summary",
                    "",
                    master_entry_start_marker(filename),
                    "entry content",
                    master_entry_end_marker(),
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def test_selects_only_filenames_missing_from_master(self) -> None:
        self.write_master_entry(self.first_audio.path.name)
        discovery = InputDiscovery(
            valid_audio=[self.first_audio, self.second_audio],
            skipped_files=[],
        )

        selection = select_new_audio_files(discovery, self.master_path)

        self.assertEqual(selection.new_audio, [self.second_audio])
        self.assertEqual(selection.already_processed, [self.first_audio])

    def test_treats_all_discovered_audio_as_new_without_a_master_file(self) -> None:
        discovery = InputDiscovery(
            valid_audio=[self.first_audio, self.second_audio],
            skipped_files=[],
        )

        selection = select_new_audio_files(discovery, self.master_path)

        self.assertEqual(selection.new_audio, [self.first_audio, self.second_audio])
        self.assertEqual(selection.already_processed, [])

    def test_stops_when_master_markers_are_malformed(self) -> None:
        self.master_path.write_text(
            master_entry_start_marker(self.first_audio.path.name), encoding="utf-8"
        )
        discovery = InputDiscovery(valid_audio=[self.first_audio], skipped_files=[])

        with self.assertRaises(MasterSummaryError):
            select_new_audio_files(discovery, self.master_path)


if __name__ == "__main__":
    unittest.main()
