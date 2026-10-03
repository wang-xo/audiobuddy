"""Local checks for the standalone transcript command-line contract."""

from __future__ import annotations

import contextlib
import io
import unittest

from transcribe_audio import parse_args


class TranscriptCliTests(unittest.TestCase):
    def test_accepts_one_audio_file(self) -> None:
        arguments = parse_args(["2026-08-30 11_10_13.ogg"])

        self.assertEqual(arguments.audio_file, "2026-08-30 11_10_13.ogg")

    def test_rejects_a_missing_audio_file(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args([])


if __name__ == "__main__":
    unittest.main()
