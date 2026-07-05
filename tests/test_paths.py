import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mafia_framework.paths import resolve_repo_path


class TestResolveRepoPath(unittest.TestCase):

    def test_absolute_path_passes_through_unchanged(self):
        absolute = Path("/tmp/some/absolute/path.db")
        self.assertEqual(resolve_repo_path(absolute), absolute)

    def test_prefers_cwd_relative_path_when_it_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "data").mkdir()
            (Path(tmp) / "data" / "mafia.db").write_text("", encoding="utf-8")

            original_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                resolved = resolve_repo_path("data/mafia.db")
            finally:
                os.chdir(original_cwd)

            self.assertEqual(resolved, Path(tmp).resolve() / "data" / "mafia.db")

    def test_falls_back_to_repo_root_when_cwd_relative_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                # config.toml lives at the real repo root and won't exist
                # relative to this empty temp directory.
                resolved = resolve_repo_path("config.example.toml")
            finally:
                os.chdir(original_cwd)

            self.assertTrue(resolved.exists())
            self.assertEqual(resolved.name, "config.example.toml")

    def test_falls_back_to_cwd_relative_when_nothing_exists_anywhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                resolved = resolve_repo_path("totally_made_up_dir/made_up.db")
            finally:
                os.chdir(original_cwd)

            self.assertEqual(resolved, Path(tmp).resolve() / "totally_made_up_dir" / "made_up.db")
