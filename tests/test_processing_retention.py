from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

from processing.human import retained_users
from processing.retention import retention_by_user
from processing.run import run


def participant(name='P', cohort='main', count=40, source='formal'):
    return {'username': name, 'cohort': cohort, 'assigned_count': count, 'participant_source': source}


def responses(count=40):
    modules = ['cue', 'spatial', 'formulation', 'pair', 'transfer']
    return [{'username': 'P', 'puzzle_id': f'trial{i}', 'puzzle_type': f'type{i % 5}',
             'module': modules[i % 5], 'is_correct': True, 'bad_record': False} for i in range(count)]


class RetentionTests(unittest.TestCase):
    def test_main_exclusion_boundaries_and_distinct_completion(self):
        rows = responses()
        for row in rows[:30]: row['bad_record'] = True
        self.assertEqual(retained_users('main', [participant()], rows), {'P'})
        rows[30]['bad_record'] = True
        self.assertEqual(retained_users('main', [participant()], rows), set())
        rows = responses()
        for row in rows[:15]: row.update(bad_record=True, puzzle_type='same')
        self.assertEqual(retained_users('main', [participant()], rows), set())
        self.assertEqual(retained_users('main', [participant()], responses(39) + [responses(1)[0]]), set())

    def test_module_requires_40_correct_and_two_in_every_module(self):
        rows = responses()
        self.assertEqual(retained_users('module', [participant(cohort='module')], rows), {'P'})
        rows[0]['is_correct'] = False
        self.assertEqual(retained_users('module', [participant(cohort='module')], rows), set())
        rows = responses(45)
        for row in rows:
            if row['module'] == 'cue': row['module'] = 'spatial'
        rows[0]['module'] = 'cue'
        self.assertEqual(retained_users('module', [participant(cohort='module', count=45)], rows), set())
        rows[1]['module'] = 'cue'
        self.assertEqual(retained_users('module', [participant(cohort='module', count=45)], rows), {'P'})

    def test_tester_exclusion_and_same_policy_for_server_summary(self):
        rows = responses()
        person = participant(source='tester')
        summary = retention_by_user([person], rows)['P']
        self.assertFalse(summary['analysis_retained'])
        self.assertIn('tester', summary['analysis_exclusion_reasons'])
        self.assertEqual(retained_users('main', [person], rows), set())

    def test_processing_refuses_raw_output_and_partial_human_cohort(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            data = Path(directory)
            for name in ['private', 'ai', 'stimuli']:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'separate'):
                    run(data, data/name/'output', stage='human')
            (data/'private').mkdir()
            (data/'alias').symlink_to(data/'private', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'separate'):
                run(data, data/'alias/output', stage='human')
            with self.assertRaisesRegex(ValueError, '--limit'):
                run(data, data/'stage', stage='human', limit=1)

    def test_models_cli_from_outside_root_and_smoke_counts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory).resolve()
            data = root/'inputs'
            puzzle = {'id': 'P', 'puzzle_type': 'mini_sudoku',
                      'machine_readable_instance': {'board': ['....'] * 4, 'digit': 1},
                      'solutions': [{'solution_id': 'S', 'abstract_solution_id': 'AS', 'coordinate': [1, 1]}]}
            files = {'stimuli/all_puzzles.jsonl': puzzle, 'stimuli/modules/all_module_trials.jsonl': puzzle,
                     'stimuli/solution_catalog.jsonl': {'solution_id': 'S', 'abstract_solution_id': 'AS'},
                     'stimuli/modules/solution_catalog_modules.jsonl': {'solution_id': 'S', 'abstract_solution_id': 'AS'}}
            for provider in ['openai', 'anthropic', 'gemini']:
                for cohort in (['formal'] if provider == 'gemini' else ['main', 'module']):
                    files[f'ai/{provider}/{cohort}/responses.jsonl'] = {'puzzle_id': 'P', 'raw_answer': '(1,1)'}
            for relative, value in files.items():
                path = data/relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value)+'\n')
            command = [sys.executable, str(Path(__file__).resolve().parents[1]/'processing/run.py'),
                       '--data-dir', str(data), '--output-dir', str(root/'output'),
                       '--stage', 'models', '--limit', '1']
            completed = subprocess.run(command, cwd=root, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((root/'output/processing_manifest.json').read_text())
            self.assertTrue(manifest['smoke_only'])
            self.assertEqual(len(manifest['models']), 6)
            self.assertTrue(all(row['rows'] == row['valid_rows'] == 1 for row in manifest['models']))
            completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)


if __name__ == '__main__':
    unittest.main()
