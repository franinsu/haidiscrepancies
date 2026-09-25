"""Recorded gzip exports and new plain collections must produce identical rows."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from processing import run as processing
from processing.score_responses import score_responses
from puzzles.common import read_jsonl


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def puzzle(identifier):
    return {
        'id': identifier, 'puzzle_type': 'mini_sudoku',
        'machine_readable_instance': {'board': ['....'] * 4, 'digit': 1},
        'solutions': [{'solution_id': f'solution_{identifier}',
                       'abstract_solution_id': f'abstract_{identifier}', 'coordinate': [1, 1]}],
    }


def model_fixture(data):
    main, modules = [puzzle('main')], [puzzle('A'), puzzle('B')]
    write_rows(data / 'stimuli/all_puzzles.jsonl', main)
    write_rows(data / 'stimuli/solution_catalog.jsonl', [row['solutions'][0] for row in main])
    write_rows(data / 'stimuli/modules/all_module_trials.jsonl', modules)
    write_rows(data / 'stimuli/modules/solution_catalog_modules.jsonl', [row['solutions'][0] for row in modules])
    single = {'request_id': 'single-request', 'input_kind': 'single', 'puzzle_id': 'main',
              'raw_answer': '(1,1)', 'prompt_condition': 'plain', 'reasoning_effort': 'low'}
    sequence = {'request_id': 'paired-request', 'input_kind': 'sequence', 'puzzle_id': 'pair',
                'trial_ids': 'A;B', 'sequence_id': 'paired-sequence',
                'raw_response': '(1,1)\n(1,1)', 'prompt_condition': 'plain', 'reasoning_effort': 'low'}
    for provider in ('openai', 'anthropic'):
        for cohort, response in (('main', single), ('module', sequence)):
            write_rows(data / 'ai' / provider / cohort / 'responses.jsonl', [response])
    write_rows(data / 'ai/gemini/formal/responses.jsonl', [single, sequence])


class CompressedModelInputTests(unittest.TestCase):
    def test_plain_and_gzip_processing_and_direct_scoring_agree(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'data'
            model_fixture(data)
            plain = processing.run(data, root / 'plain', stage='models')
            response_path = data / 'ai/openai/module/responses.jsonl'
            direct_arguments = (str(data / 'stimuli/modules/all_module_trials.jsonl'),
                                str(data / 'stimuli/modules/solution_catalog_modules.jsonl'))
            direct_plain = score_responses(direct_arguments[0], str(response_path),
                                           str(root / 'direct_plain.jsonl'), direct_arguments[1])
            for path in data.glob('ai/*/*/responses.jsonl'):
                path.with_suffix('.jsonl.gz').write_bytes(gzip.compress(path.read_bytes(), mtime=0))
                path.unlink()
            compressed = processing.run(data, root / 'compressed', stage='models')
            direct_compressed = score_responses(direct_arguments[0], str(response_path) + '.gz',
                                                str(root / 'direct_gzip.jsonl'), direct_arguments[1])
            self.assertEqual(direct_plain, direct_compressed)
            self.assertEqual(plain['status'], 'passed')
            self.assertEqual(compressed['status'], 'passed')
            for provider in ('openai', 'anthropic', 'gemini'):
                for cohort in ('main', 'module'):
                    name = f'{provider}_{cohort}.jsonl'
                    self.assertEqual((root / 'plain/ai' / name).read_bytes(),
                                     (root / 'compressed/ai' / name).read_bytes())
                paired = read_jsonl(root / 'compressed/ai' / f'{provider}_module.jsonl')
                self.assertEqual([row['puzzle_id'] for row in paired], ['A', 'B'])
                self.assertEqual([row['sequence_position'] for row in paired], [1, 2])
                self.assertTrue(all(row['is_valid'] and row['request_id'] == 'paired-request'
                                    and row['sequence_id'] == 'paired-sequence' for row in paired))
            self.assertTrue(all(path.endswith('.jsonl.gz') for path in compressed['input_sha256'] if '/ai/' in path))

    def test_corrupt_or_truncated_gzip_fails_processing(self):
        for corruption, expected in (('crc', gzip.BadGzipFile), ('truncated', EOFError)):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                data = root / 'data'
                model_fixture(data)
                response = data / 'ai/openai/main/responses.jsonl'
                content = bytearray(gzip.compress(response.read_bytes(), mtime=0))
                if corruption == 'crc':
                    content[-8] ^= 1
                else:
                    del content[-4:]
                response.with_suffix('.jsonl.gz').write_bytes(content)
                response.unlink()
                with self.assertRaises(expected):
                    processing.run(data, root / 'output', stage='models')
                manifest = json.loads((root / 'output/processing_manifest.json').read_text())
                self.assertEqual(manifest['status'], 'failed')
                with self.assertRaises(expected):
                    read_jsonl(response.with_suffix('.jsonl.gz'))

    def test_ambiguous_and_missing_inputs_fail_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'data'
            model_fixture(data)
            response = data / 'ai/openai/main/responses.jsonl'
            compressed = response.with_suffix('.jsonl.gz')
            compressed.write_bytes(gzip.compress(response.read_bytes(), mtime=0))
            with self.assertRaisesRegex(ValueError, 'Ambiguous model inputs'):
                processing.run(data, root / 'output', stage='models')
            self.assertFalse((root / 'output').exists())
            response.unlink()
            compressed.unlink()
            with self.assertRaisesRegex(FileNotFoundError, r'responses\.jsonl or .*responses\.jsonl\.gz'):
                processing.run(data, root / 'output', stage='models')
            self.assertFalse((root / 'output').exists())


if __name__ == '__main__':
    unittest.main()
