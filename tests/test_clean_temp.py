"""Check the cleanup command using disposable files, never the study data."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "clean_temp.py"


class CleanupTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name) / "repository"
        self.root.mkdir()
        (self.root / "clean_temp.py").write_text(SCRIPT.read_text())

    def write(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("invented fixture")
        return path

    def run_cleanup(self, *args):
        result = subprocess.run([sys.executable, "-B", str(self.root / "clean_temp.py"), *args],
                                cwd=self.root.parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_preview_from_another_directory_changes_nothing(self):
        temporary = self.write(".DS_Store")
        self.assertIn(".DS_Store", self.run_cleanup())
        self.assertEqual(temporary.read_text(), "invented fixture")

    def test_apply_removes_only_temporary_files(self):
        keep = [self.write(name) for name in (
            "data/human/record.jsonl", "data/processed/rows.jsonl", "checks/.cache/comparison.json",
            "data/ai/responses.jsonl", "data/stimuli/puzzles.jsonl", ".env",
            "intermediate/processed/human.jsonl", "intermediate/statistics/summary.json",
            "intermediate/__pycache__/cached.pyc", "intermediate/.cache/snapshot.json",
            ".venv/__pycache__/library.pyc", ".git/cache.pyc", "analysis/method.py",
            "paper/main.tex", "paper/main.pdf", "paper/main.bbl", "paper/figs/board.png",
            "results/run/build/main.aux", "collection/events.log", "analysis/estimates.out")]
        remove = [self.write(name) for name in (
            "analysis/__pycache__/method.pyc", ".cache/fonts.json", ".DS_Store",
            "paper/main.aux", "paper/main.log", "paper/main.out")]
        self.run_cleanup("--apply")
        self.assertTrue(all(path.exists() for path in keep))
        self.assertTrue(all(not path.exists() for path in remove))
        self.assertIn("0 files/directories", self.run_cleanup("--apply"))

    def test_results_require_both_flags_to_be_deleted(self):
        run = self.write("results/run/checkpoints/fit.json")
        processed = self.write("data/processed/rows.jsonl")
        intermediate = [self.write("intermediate/processed/human.jsonl"),
                        self.write("intermediate/statistics/summary.json"),
                        self.write("intermediate/.cache/cached.pyc")]
        archive = self.write("data/human/record.jsonl")
        reference = self.write("checks/reference/statistics.json")
        self.assertIn("  intermediate", self.run_cleanup("--results"))
        self.assertTrue(run.exists())
        self.assertTrue(processed.exists())
        self.assertTrue(all(path.exists() for path in intermediate))
        self.run_cleanup("--apply")
        self.assertTrue(run.exists())
        self.assertTrue(processed.exists())
        self.assertTrue(all(path.exists() for path in intermediate))
        self.run_cleanup("--results", "--apply")
        self.assertFalse(run.exists())
        self.assertFalse(processed.exists())
        self.assertFalse((self.root / "intermediate").exists())
        self.assertTrue(archive.exists())
        self.assertTrue(reference.exists())
        self.assertIn("0 files/directories", self.run_cleanup("--results", "--apply"))

    def test_symlinks_do_not_expose_external_files_to_cleanup(self):
        outside = self.root.parent / "outside"
        outside.mkdir()
        original = outside / "keep.pyc"
        original.write_text("keep")
        for name in ("__pycache__", "intermediate", "results", "other"):
            (self.root / name).symlink_to(outside, target_is_directory=True)
        (self.root / "linked.pyc").symlink_to(original)
        self.run_cleanup("--results", "--apply")
        self.assertEqual(original.read_text(), "keep")
        self.assertTrue(all((self.root / name).is_symlink()
                            for name in ("__pycache__", "intermediate", "results", "other", "linked.pyc")))
        (outside / "processed").mkdir()
        (outside / "processed/rows.jsonl").write_text("keep")
        (self.root / "data").symlink_to(outside, target_is_directory=True)
        self.run_cleanup("--results", "--apply")
        self.assertTrue((outside / "processed/rows.jsonl").exists())
        (self.root / "data").unlink()
        (self.root / "data").mkdir()
        (self.root / "data/processed").symlink_to(outside / "processed", target_is_directory=True)
        self.run_cleanup("--results", "--apply")
        self.assertTrue((outside / "processed/rows.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
