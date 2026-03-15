#!/usr/bin/env python3
"""Tests for demo/app.py history management.

These tests verify that:
- History is persisted to disk on every save.
- History is re-read from disk on every load (the core refresh-bug fix).
- The load_on_refresh callback returns the latest history.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

# Ensure the project root is on the path so we can import demo.app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import demo.app as app


class TestHistoryPersistence(unittest.TestCase):
    """Tests that history is always read fresh from disk."""

    def setUp(self):
        """Create a temporary directory for history files."""
        self.tmpdir = tempfile.mkdtemp()
        self.history_file = os.path.join(self.tmpdir, "task_history.json")
        # Patch HISTORY_DIR and HISTORY_FILE so tests don't touch real files
        self._patches = [
            mock.patch.object(app, "HISTORY_DIR", self.tmpdir),
            mock.patch.object(app, "HISTORY_FILE", self.history_file),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        # Clean up temp files
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
        os.rmdir(self.tmpdir)

    # ------------------------------------------------------------------ #
    # Core bug-fix test: load_history always reads from disk
    # ------------------------------------------------------------------ #
    def test_load_history_reads_fresh_from_disk(self):
        """load_history must re-read the file every call, not return a
        cached startup snapshot."""
        # Initially empty
        self.assertEqual(load_history(), [])

        # Write one entry directly to disk (simulating another session)
        entry = [{"question": "first"}]
        with open(self.history_file, "w") as f:
            json.dump(entry, f)
        self.assertEqual(load_history(), entry)

        # Append a second entry
        entry.append({"question": "second"})
        with open(self.history_file, "w") as f:
            json.dump(entry, f)
        self.assertEqual(load_history(), entry)
        self.assertEqual(len(load_history()), 2)

    # ------------------------------------------------------------------ #
    # load_on_refresh returns latest history
    # ------------------------------------------------------------------ #
    def test_load_on_refresh_returns_latest(self):
        """load_on_refresh (used by demo.load) should always reflect the
        most recent disk state, not a stale startup cache."""
        # Start with no history
        chat, display = app.load_on_refresh()
        self.assertEqual(chat, [])
        self.assertIn("No task history", display)

        # Simulate a task being saved
        app.save_history([{
            "id": "1",
            "timestamp": "2026-01-01 00:00:00",
            "question": "test question",
            "response": "test response",
        }])

        chat, display = app.load_on_refresh()
        self.assertEqual(chat, [])  # chat window resets on refresh
        self.assertIn("test question", display)

    # ------------------------------------------------------------------ #
    # save / load round-trip
    # ------------------------------------------------------------------ #
    def test_save_and_load_roundtrip(self):
        entries = [
            {"id": "a", "question": "q1"},
            {"id": "b", "question": "q2"},
        ]
        app.save_history(entries)
        loaded = load_history()
        self.assertEqual(loaded, entries)

    def test_load_history_empty_when_no_file(self):
        self.assertEqual(load_history(), [])

    def test_load_history_handles_corrupt_file(self):
        with open(self.history_file, "w") as f:
            f.write("NOT VALID JSON{{{")
        self.assertEqual(load_history(), [])

    # ------------------------------------------------------------------ #
    # chat_respond persists to disk
    # ------------------------------------------------------------------ #
    def test_chat_respond_saves_to_disk(self):
        _, _, _ = app.chat_respond("hello", [])
        history = load_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["question"], "hello")

    def test_chat_respond_accumulates(self):
        app.chat_respond("first", [])
        app.chat_respond("second", [])
        history = load_history()
        self.assertEqual(len(history), 2)


def load_history():
    """Shortcut that calls app.load_history."""
    return app.load_history()


if __name__ == "__main__":
    unittest.main()
