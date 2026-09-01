"""Local checks for AudioBuddy command-line argument handling."""

from __future__ import annotations

import contextlib
import io
import unittest

from summarize_audio import parse_args


class CliArgumentTests(unittest.TestCase):
    def test_accepts_one_audio_file(self) -> None:
        arguments = parse_args(["2026-08-30 11_10_13.mp3"])

        self.assertEqual(arguments.audio_file, "2026-08-30 11_10_13.mp3")
        self.assertFalse(arguments.scan)
        self.assertFalse(arguments.rescan)

    def test_accepts_scan_mode(self) -> None:
        arguments = parse_args(["--scan"])

        self.assertIsNone(arguments.audio_file)
        self.assertTrue(arguments.scan)
        self.assertFalse(arguments.rescan)

    def test_accepts_rescan_mode(self) -> None:
        arguments = parse_args(["--rescan"])

        self.assertIsNone(arguments.audio_file)
        self.assertFalse(arguments.scan)
        self.assertTrue(arguments.rescan)

    def test_rejects_missing_mode(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args([])

    def test_rejects_audio_file_and_scan_together(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["2026-08-30 11_10_13.mp3", "--scan"])

    def test_rejects_scan_and_rescan_together(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--scan", "--rescan"])

    def test_rejects_audio_file_and_rescan_together(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["2026-08-30 11_10_13.mp3", "--rescan"])


if __name__ == "__main__":
    unittest.main()
