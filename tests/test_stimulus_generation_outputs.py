import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageFont, features

from collection.models.build_solution_catalog import build_solution_catalog
from collection.models.render_api_images import render_api_images, render_single_trial


ROOT = Path(__file__).resolve().parents[1]
PUZZLE = {
    "id": "SYNTH_SUDOKU",
    "puzzle_type": "mini_sudoku",
    "difficulty_bucket": "easy",
    "num_solutions": 1,
    "machine_readable_instance": {
        "n": 4, "board": ["1...", "....", "....", "...."],
        "digit": 1, "box_rows": 2, "box_cols": 2,
    },
    "solutions": [{"solution_id": "SYNTH_SOLUTION", "coordinate": [3, 3], "digit": 1}],
}


class StimulusOutputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.main = self.root / "main.jsonl"
        self.main.write_text(json.dumps(PUZZLE) + "\n")
        self.modules = self.root / "modules.jsonl"
        self.modules.write_text("")
        self.blocks = self.root / "blocks.jsonl"
        self.blocks.write_text("")

    def render(self, out, manifest):
        return render_api_images(str(self.main), str(self.modules), str(self.blocks),
                                 str(out), str(manifest))

    @unittest.skipUnless(features.check('raqm'), 'Requires a Pillow build with both text-layout engines')
    def test_image_pixels_do_not_depend_on_raqm_availability(self):
        pixels = []
        for available in (True, False):
            output = self.root / f'layout_{available}.png'
            with patch.object(ImageFont.core, 'HAVE_RAQM', available):
                render_single_trial(PUZZLE, output, dataset_name='main')
            with Image.open(output) as image:
                pixels.append((image.size, image.convert('RGB').tobytes()))
        self.assertEqual(pixels[0], pixels[1])

    def test_cli_requires_explicit_output_destinations(self):
        commands = [
            (["build_solution_catalog.py"], "--out"),
            (["render_api_images.py"], "--out_dir"),
            (["render_api_images.py", "--out_dir", str(self.root / "images")], "--manifest"),
            (["render_api_images.py", "--manifest", str(self.root / "manifest.csv")], "--out_dir"),
            (["run.py", "render"], "--out_dir"),
        ]
        before = set(self.root.iterdir())
        for arguments, missing in commands:
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "collection/models" / arguments[0]), *arguments[1:]],
                    cwd=self.root, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(missing, result.stderr)
                self.assertEqual(set(self.root.iterdir()), before)

    def test_catalog_refuses_existing_file_and_input_path(self):
        existing = self.root / "existing.jsonl"
        existing.write_bytes(b"keep this catalog\n")
        for out in (existing, self.main):
            with self.subTest(out=out):
                before = out.read_bytes()
                with self.assertRaises(FileExistsError):
                    build_solution_catalog(str(self.main), str(out))
                self.assertEqual(out.read_bytes(), before)

    def test_catalog_refuses_symlink_even_when_target_is_missing(self):
        target = self.root / "missing.jsonl"
        link = self.root / "linked.jsonl"
        link.symlink_to(target)
        with self.assertRaises(FileExistsError):
            build_solution_catalog(str(self.main), str(link))
        self.assertTrue(link.is_symlink())
        self.assertFalse(target.exists())

    def test_render_refuses_existing_images_before_writing_manifest(self):
        out = self.root / "images"
        out.mkdir()
        manifest = self.root / "new" / "manifest.csv"
        for populated in (False, True):
            with self.subTest(populated=populated):
                image = out / "existing.png"
                if populated:
                    image.write_bytes(b"keep this image")
                with self.assertRaises(FileExistsError):
                    self.render(out, manifest)
                self.assertFalse(manifest.parent.exists())
                self.assertEqual(list(out.iterdir()), [image] if populated else [])
                if populated:
                    self.assertEqual(image.read_bytes(), b"keep this image")

    def test_render_refuses_existing_manifest_before_creating_images(self):
        manifest = self.root / "manifest.csv"
        manifest.write_bytes(b"keep this manifest\n")
        out = self.root / "new" / "images"
        with self.assertRaises(FileExistsError):
            self.render(out, manifest)
        self.assertFalse(out.parent.exists())
        self.assertEqual(manifest.read_bytes(), b"keep this manifest\n")

    def test_render_refuses_symlink_destinations_and_identical_paths(self):
        for destination in ("images", "manifest"):
            with self.subTest(destination=destination):
                out = self.root / f"{destination}_images"
                manifest = self.root / f"{destination}_manifest.csv"
                link = out if destination == "images" else manifest
                target = self.root / f"{destination}_missing"
                link.symlink_to(target)
                with self.assertRaises(FileExistsError):
                    self.render(out, manifest)
                self.assertFalse(target.exists())
                self.assertFalse((manifest if destination == "images" else out).exists())
        same = self.root / "same"
        with self.assertRaises(ValueError):
            self.render(same, same)
        self.assertFalse(same.exists())

    def test_fresh_destinations_produce_catalog_image_and_matching_manifest(self):
        original = self.main.read_bytes()
        catalog_path = self.root / "catalogs" / "new.jsonl"
        catalog = build_solution_catalog(str(self.main), str(catalog_path))
        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog[0]["canonical_answer"], [3, 3])
        self.assertEqual(catalog[0]["solution_id"], "SYNTH_SOLUTION")
        self.assertEqual(json.loads(catalog_path.read_text()), catalog[0])

        out = self.root / "images"
        manifest = out / "manifest.csv"
        counts = self.render(out, manifest)
        self.assertEqual(counts["total_images"], 1)
        image = out / "SYNTH_SUDOKU.png"
        with Image.open(image) as opened:
            self.assertEqual(opened.size, (1280, 960))
            opened.verify()
        with manifest.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image_path"], str(image))
        self.assertEqual(rows[0]["image_sha256"], hashlib.sha256(image.read_bytes()).hexdigest())
        self.assertEqual(self.main.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
