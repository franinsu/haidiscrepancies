import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import collection.models.build_api_run_manifest as manifest_builder
from collection.models.run_api_collection import _score_attempt
import processing.response_parser as response_parser
from processing.score_responses import _expand_response_row, _score_one_response, score_responses
from puzzles.common import stable_hash


def _single_coordinate_puzzle(puzzle_type):
    if puzzle_type == "minesweeper_lite":
        instance = {"board": ["?0"], "target_kind": "safe"}
    elif puzzle_type == "mini_sudoku":
        instance = {"board": ["....", "....", "....", "...."], "digit": 1}
    else:
        raise AssertionError(f"Unsupported synthetic puzzle type: {puzzle_type}")
    return {
        "id": f"SYNTH_{puzzle_type}",
        "puzzle_type": puzzle_type,
        "machine_readable_instance": instance,
        "solutions": [{"solution_id": "synthetic_solution", "coordinate": [1, 1]}],
    }


def _grid_placement_puzzle():
    return {
        "id": "SYNTH_grid_placement",
        "puzzle_type": "grid_placement",
        "machine_readable_instance": {"board": ["..", ".."], "m": 2},
        "solutions": [
            {
                "solution_id": "synthetic_grid_solution",
                "coordinates": [[1, 1], [2, 2]],
            }
        ],
    }


def _arithmetic_puzzle():
    expression = "8*3+4-4"
    return {
        "id": "SYNTH_arithmetic24",
        "puzzle_type": "arithmetic24",
        "machine_readable_instance": {"numbers": [3, 4, 4, 8], "target": 24},
        "solutions": [
            {
                "solution_id": "synthetic_arithmetic_solution",
                "canonical_expression": response_parser.canonicalize_arithmetic_expression(expression),
            }
        ],
    }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class ResponseParserRegressionTests(unittest.TestCase):
    def _scoring_fixture(self, root):
        puzzle = _single_coordinate_puzzle("mini_sudoku")
        puzzle["solutions"][0]["abstract_solution_id"] = "abstract_synthetic"
        data, catalog, responses, out = [root / name for name in ("data.jsonl", "catalog.jsonl", "responses.jsonl", "scored.jsonl")]
        _write_jsonl(data, [puzzle])
        _write_jsonl(catalog, [{"solution_id": "synthetic_solution", "abstract_solution_id": "abstract_synthetic"}])
        _write_jsonl(responses, [{"puzzle_id": puzzle["id"], "raw_answer": "(1,1)"}])
        return data, catalog, responses, out

    def test_direct_scorer_rejects_missing_incomplete_or_duplicate_catalog_before_write(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            data, catalog, responses, out = self._scoring_fixture(Path(tmp))
            with self.assertRaises(FileNotFoundError):
                score_responses(str(data), str(responses), str(out), str(Path(tmp) / "absent.jsonl"))
            for rows in [[], [{"solution_id": "synthetic_solution", "abstract_solution_id": "wrong"}],
                         [{"solution_id": "synthetic_solution", "abstract_solution_id": "abstract_synthetic"}] * 2]:
                _write_jsonl(catalog, rows)
                with self.assertRaises(ValueError):
                    score_responses(str(data), str(responses), str(out), str(catalog))
                self.assertFalse(out.exists())

    def test_direct_scorer_rejects_source_aliases_without_changing_inputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            data, catalog, responses, out = self._scoring_fixture(root)
            originals = {p: p.read_bytes() for p in (data, catalog, responses)}
            link = root / "linked_output.jsonl"
            link.symlink_to(responses)
            hard_link = root / "hardlinked_output.jsonl"
            os.link(responses, hard_link)
            for target in (data, catalog, responses, link, hard_link):
                with self.subTest(target=target.name), self.assertRaisesRegex(ValueError, "overwrite source"):
                    score_responses(str(data), str(responses), str(target), str(catalog))
                self.assertEqual({p: p.read_bytes() for p in originals}, originals)

    def test_direct_scorer_preserves_intentional_unknown_puzzle_skipping(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            data, catalog, responses, out = self._scoring_fixture(Path(tmp))
            current = json.loads(responses.read_text())
            _write_jsonl(responses, [current, {"puzzle_id": "UNKNOWN", "raw_answer": "(1,1)"}])
            scored = score_responses(str(data), str(responses), str(out), str(catalog), True)
            self.assertEqual(len(scored), 1)
            self.assertTrue(scored[0]["is_valid"])
            self.assertEqual(scored[0]["abstract_solution_id"], "abstract_synthetic")

    def test_sequence_missing_answers_stay_empty_in_collection_and_processing(self):
        puzzle = _single_coordinate_puzzle("mini_sudoku")
        puzzles = {"A": puzzle, "B": puzzle}
        cases = [
            ("A: (1,1)", ["(1,1)", ""]),
            ("B: (1,1)", ["", "(1,1)"]),
            ("A: (1,1)\nB:", ["(1,1)", ""]),
            ("A:\nB: (1,1)", ["", "(1,1)"]),
            ("B: (1,1)\nOne answer only.", ["", "(1,1)\nOne answer only."]),
            ("(1,1)", ["", ""]),
            ("", ["", ""]),
        ]
        for text, expected_answers in cases:
            with self.subTest(text=text):
                raw = {"input_kind": "sequence", "trial_ids": "A;B", "puzzle_id": "PAIR", "raw_response": text}
                split = response_parser.split_sequence_response(text)
                self.assertEqual(split["answers"], expected_answers)
                self.assertEqual(split["parse_confidence"], "low")
                scored = [_score_one_response(part, response_index=i, puzzles=puzzles, catalog={})
                          for i, part in enumerate(_expand_response_row(raw), start=1)]
                self.assertEqual([row["raw_answer"] for row in scored], expected_answers)
                self.assertTrue(all(row["raw_response"] == text for row in scored))
                self.assertEqual([row["is_valid"] for row in scored], [bool(answer) for answer in expected_answers])
                for row, answer in zip(scored, expected_answers):
                    if not answer:
                        self.assertEqual(row["invalid_reason"], "empty")
                        self.assertIsNone(row["solution_id"])
                online = _score_attempt(raw, text, puzzles)
                for online_row, offline_row in zip(online, scored):
                    self.assertEqual(online_row, {key: offline_row[key] for key in online_row})

    def test_complete_sequences_preserve_labels_and_line_order(self):
        for text, expected, confidence in [
            ("A: (1,1)\nB: (2,2)", ["(1,1)", "(2,2)"], "high"),
            ("B: (2,2)\nA: (1,1)", ["(1,1)", "(2,2)"], "high"),
            ("(1,1)\n(2,2)", ["(1,1)", "(2,2)"], "medium"),
        ]:
            with self.subTest(text=text):
                split = response_parser.split_sequence_response(text)
                self.assertEqual(split["answers"], expected)
                self.assertEqual(split["parse_confidence"], confidence)

    def test_single_answer_raw_response_fallback_remains_supported(self):
        puzzle = _single_coordinate_puzzle("mini_sudoku")
        raw = {"input_kind": "single", "trial_ids": "A", "puzzle_id": "A", "raw_answer": "", "raw_response": "(1,1)"}
        direct = response_parser.score_raw_answer(puzzle, "", raw_response="(1,1)")
        offline = _score_one_response(raw, response_index=1, puzzles={"A": puzzle}, catalog={})
        online = _score_attempt(raw, "(1,1)", {"A": puzzle})[0]
        self.assertTrue(direct["is_valid"])
        self.assertEqual(direct, online)
        self.assertEqual(direct, {key: offline[key] for key in direct})

    def test_single_coordinate_puzzles_reject_multiple_coordinates(self):
        for puzzle_type in ("minesweeper_lite", "mini_sudoku"):
            with self.subTest(puzzle_type=puzzle_type):
                result = response_parser.score_raw_answer(
                    _single_coordinate_puzzle(puzzle_type),
                    "(1,1), (2,2)",
                )
                self.assertFalse(result["is_valid"])
                self.assertEqual(result["invalid_reason"], "wrong_number_of_tokens")
                self.assertEqual(result["parsed_answer"], [[1, 1], [2, 2]])
                self.assertEqual(result["parser_version"], "response_parser_v5")

    def test_grid_placement_still_accepts_multiple_coordinates(self):
        result = response_parser.score_raw_answer(
            _grid_placement_puzzle(),
            "(1,1), (2,2)",
        )
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["solution_id"], "synthetic_grid_solution")
        self.assertEqual(result["parser_version"], "response_parser_v5")

    def test_single_sentence_terminal_period_is_ignored_for_semantic_scoring(self):
        result = response_parser.score_raw_answer(
            _arithmetic_puzzle(),
            "ANSWER: expression=8*3+4-4.",
        )
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["solution_id"], "synthetic_arithmetic_solution")
        self.assertEqual(result["extracted_answer"], "8*3+4-4")
        self.assertIn("Ignored one sentence-terminal period.", result["parse_notes"])
        self.assertEqual(result["parser_version"], "response_parser_v5")

    def test_ellipsis_is_not_treated_as_one_sentence_terminal_period(self):
        result = response_parser.score_raw_answer(
            _arithmetic_puzzle(),
            "ANSWER: expression=8*3+4-4..",
        )
        self.assertFalse(result["is_valid"])
        self.assertNotIn("Ignored one sentence-terminal period.", result["parse_notes"])

    def test_new_manifest_freezes_parser_version_and_source_hash(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            image_path = root / "synthetic.img"
            image_path.write_bytes(b"synthetic image bytes")

            single_prompt = root / "synthetic_single.txt"
            sequence_prompt = root / "synthetic_sequence.txt"
            single_prompt.write_text("Solve the synthetic puzzle.", encoding="utf-8")
            sequence_prompt.write_text("Solve both synthetic puzzles.", encoding="utf-8")

            prompt_manifest_path = root / "prompt_manifest.json"
            _write_json(
                prompt_manifest_path,
                {
                    "synthetic_single_v1": {
                        "prompt_condition": "direct_solve",
                        "input_kind": "single",
                        "file": single_prompt.name,
                        "sha256": _sha256(single_prompt),
                    },
                    "synthetic_sequence_v1": {
                        "prompt_condition": "direct_solve",
                        "input_kind": "sequence",
                        "file": sequence_prompt.name,
                        "sha256": _sha256(sequence_prompt),
                    },
                },
            )

            image_manifest_path = root / "image_manifest.csv"
            image_row = {
                "image_id": "synthetic_image",
                "puzzle_id": "synthetic_puzzle",
                "sequence_id": "",
                "dataset_name": "main",
                "module": "synthetic",
                "condition": "synthetic",
                "puzzle_type": "mini_sudoku",
                "trial_ids": "synthetic_puzzle",
                "input_kind": "single",
                "prompt_version": "synthetic_single_v1",
                "prompt_file": single_prompt.name,
                "image_path": str(image_path),
                "image_sha256": _sha256(image_path),
                "render_version": "api_image_v9",
                "width": "1",
                "height": "1",
            }
            with image_manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(image_row))
                writer.writeheader()
                writer.writerow(image_row)

            model_conditions_path = root / "model_conditions.json"
            _write_json(
                model_conditions_path,
                {
                    "SYNTH_MODEL": {
                        "provider": "openai",
                        "api": "responses",
                        "model": "synthetic-model",
                        "max_output_tokens": 16,
                        "reasoning": {"effort": "low"},
                        "text": {"format": {"type": "text"}},
                    }
                },
            )

            main_rows = [{"id": "synthetic_puzzle", "puzzle_type": "mini_sudoku"}]
            main_data_path = root / "main.jsonl"
            modules_data_path = root / "modules.jsonl"
            blocks_path = root / "blocks.jsonl"
            _write_jsonl(main_data_path, main_rows)
            _write_jsonl(modules_data_path, [])
            _write_jsonl(blocks_path, [])

            dataset_manifest_path = root / "dataset_manifest.json"
            _write_json(dataset_manifest_path, {"dataset_hash": stable_hash(main_rows)})

            with patch.object(
                manifest_builder,
                "_prompt_path",
                side_effect=lambda prompt_file: root / Path(prompt_file).name,
            ):
                manifest = manifest_builder.build_api_run_manifest(
                    run_id="SYNTH_RUN",
                    out_root=str(root / "runs"),
                    image_manifest_path=str(image_manifest_path),
                    model_conditions_path=str(model_conditions_path),
                    prompt_manifest_path=str(prompt_manifest_path),
                    main_data_path=str(main_data_path),
                    modules_data_path=str(modules_data_path),
                    blocks_path=str(blocks_path),
                    dataset_manifest_path=str(dataset_manifest_path),
                    model_conditions=["SYNTH_MODEL"],
                    prompt_conditions=["direct_solve"],
                    samples=1,
                    dataset_scope="main",
                )

            parser_hash = _sha256(response_parser.__file__)
            self.assertEqual(manifest["parser_version"], "response_parser_v5")
            self.assertEqual(manifest["hashes"]["response_parser_sha256"], parser_hash)

            written_manifest = json.loads(
                (root / "runs" / "SYNTH_RUN" / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(written_manifest["parser_version"], "response_parser_v5")
            self.assertEqual(written_manifest["hashes"]["response_parser_sha256"], parser_hash)


if __name__ == "__main__":
    unittest.main()
