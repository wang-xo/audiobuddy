"""Local checks for master-summary ordering and retry behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from summarize_audio import (
    AudioProcessingResult,
    MasterSummaryEntry,
    read_master_entries,
    unique_summary_path,
    update_master_summary,
)


class MasterSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_directory = Path(self.temporary_directory.name)
        self.master_path = self.output_directory / "master-summary.md"
        self.processing = AudioProcessingResult(
            upload_path=Path("upload.mp3"),
            original_duration_seconds=60,
            processed_duration_seconds=60,
            removed_seconds=0,
            reduction_percent=0,
            trimmed=False,
            reason="No trimming required.",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_entry(self, source_filename: str, summary_filename: str) -> MasterSummaryEntry:
        return MasterSummaryEntry(
            source_filename=source_filename,
            source_audio_path=Path(source_filename),
            individual_summary_path=Path(summary_filename),
            date_text="30 August 2026",
            time_text="11:10:13",
            gemini_model="test-model",
            audio_processing=self.processing,
            summary_markdown=f"# Summary for {summary_filename}",
        )

    def test_earlier_entry_is_inserted_in_chronological_order(self) -> None:
        first = self.make_entry("2026-08-30 11_10_13.mp3", "first-summary.md")
        later = self.make_entry("2026-08-31 11_10_13.mp3", "later-summary.md")
        earlier = self.make_entry("2026-08-29 11_10_13.mp3", "earlier-summary.md")

        self.assertEqual(update_master_summary(first, self.master_path), "created")
        self.assertEqual(update_master_summary(later, self.master_path), "appended")
        self.assertEqual(update_master_summary(earlier, self.master_path), "rewritten")

        self.assertEqual(
            [entry.source_filename for entry in read_master_entries(self.master_path)],
            [
                "2026-08-29 11_10_13.mp3",
                "2026-08-30 11_10_13.mp3",
                "2026-08-31 11_10_13.mp3",
            ],
        )

    def test_retry_replaces_existing_entry_without_duplication(self) -> None:
        original = self.make_entry("2026-08-30 11_10_13.mp3", "first-summary.md")
        retry = self.make_entry("2026-08-30 11_10_13.mp3", "retry-summary.md")

        update_master_summary(original, self.master_path)
        self.assertEqual(update_master_summary(retry, self.master_path), "rewritten")

        entries = read_master_entries(self.master_path)
        self.assertEqual(len(entries), 1)
        self.assertIn("retry-summary.md", entries[0].content)
        self.assertNotIn("first-summary.md", entries[0].content)

    def test_individual_summary_paths_remain_numbered(self) -> None:
        source_audio = self.output_directory / "2026-08-30 11_10_13.mp3"
        first_path = unique_summary_path(source_audio, self.output_directory)
        first_path.write_text("first", encoding="utf-8")
        second_path = unique_summary_path(source_audio, self.output_directory)

        self.assertEqual(first_path.name, "2026-08-30 11_10_13-summary.md")
        self.assertEqual(second_path.name, "2026-08-30 11_10_13-summary-1.md")


if __name__ == "__main__":
    unittest.main()
