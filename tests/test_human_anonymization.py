"""Sanitization tests use invented archives only; never collected human data."""
import copy
import csv
import json
from pathlib import Path
import stat
import tempfile
import unittest

from processing.anonymize_humans import MAP_FIELDS, run
from processing import human


class HumanAnonymizationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original = self.root/'private/original'
        self.output = self.root/'fresh_export'
        self.map_path = self.root/'private/id_map.csv'
        self.participants = {
            'main': [self.person('MAIN_A_PRIVATE', 'z_last', 2),
                     self.person('MAIN_B_PRIVATE', 'a_first', 2),
                     self.person('MAIN_INCOMPLETE_PRIVATE', None, 3),
                     self.person('MAIN_TESTER_PRIVATE', 'm_tester', 2, source='tester')],
            'module': [self.person('MODULE_INCLUDED_PRIVATE', 'z_module', 40),
                       self.person('MODULE_EXCLUDED_PRIVATE', 'a_module', 40)]}
        self.responses = {'main': [], 'module': []}
        for cohort, people in self.participants.items():
            for person in people:
                count = 1 if 'INCOMPLETE' in person['username'] else person['assigned_count']
                for trial in range(count):
                    correct = not ('EXCLUDED' in person['username'] and trial == 0)
                    self.responses[cohort].append(self.response(person['username'], trial,
                        len(self.responses[cohort]) + 101, correct))
        self.write_inputs()

    @staticmethod
    def person(username, analysis_id, count, source='recruitment_service'):
        return {'username': username, 'analysis_id': analysis_id, 'assigned_count': count,
                'participant_source': source, 'prolific_pid': username + '_PID',
                'assignment_json': json.dumps({'contact': 'NESTED_PRIVATE'}),
                'registered_at': '2020-01-02T03:04:05Z', 'email': 'CONTACT_PRIVATE'}

    @staticmethod
    def response(username, trial, database_id, correct):
        payload = {'puzzle_id': f'TRIAL_{trial}', 'puzzle_type': 'mini_sudoku',
                   'module': ['cue', 'spatial', 'formulation', 'pair', 'transfer'][trial % 5],
                   'family_id': 'FAMILY', 'block_id': 'BLOCK', 'condition': 'original',
                   'sequence_id': f'SEQ_{trial // 2}', 'sequence_position': trial % 2 + 1,
                   'solution_id': 'S', 'abstract_solution_id': 'AS',
                   'accepted_solution_id': 'CANONICAL_S', 'accepted_abstract_solution_id': 'CANONICAL_AS',
                   'username': username, 'participant_id': username, 'raw_answer': 'ANSWER_PRIVATE',
                   'nested': {'contact': 'NESTED_PRIVATE'}, 'wrong_attempts': ['ANSWER_PRIVATE'],
                   'trial_end_time_iso': '2020-01-02T03:04:05Z'}
        return {'id': database_id, 'username': username, 'puzzle_id': payload['puzzle_id'],
                'is_correct': int(correct), 'bad_record': int(not correct), 'timeout': 0, 'gave_up': 0,
                'response_time_ms': 1200 + trial, 'row_json': json.dumps(payload),
                'trial_end_time_iso': '2020-01-02T03:04:05Z', 'session_id': 'SESSION_PRIVATE'}

    def write_inputs(self):
        for cohort in ('main', 'module'):
            (self.original/cohort).mkdir(parents=True, exist_ok=True)
            for name, records in [('participants', self.participants[cohort]), ('responses', self.responses[cohort])]:
                (self.original/cohort/f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))

    def read_map(self):
        with self.map_path.open(newline='') as stream:
            return list(csv.DictReader(stream))

    def write_map(self, path, entries, fields=MAP_FIELDS):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(entries)
        path.chmod(0o600)

    def test_replay_map_is_deterministic_private_and_order_preserving(self):
        result = run(self.original, self.output, self.map_path)
        entries = self.read_map()
        self.assertEqual(len(entries), 6)
        self.assertEqual(len({r['new_id'] for r in entries}), 6)
        for cohort in ('main', 'module'):
            ordered = sorted((r for r in entries if r['cohort'] == cohort), key=lambda r:r['old_analysis_id'])
            self.assertEqual([r['new_id'] for r in ordered], sorted(r['new_id'] for r in ordered))
        fallback = next(r for r in entries if r['old_username'] == 'MAIN_INCOMPLETE_PRIVATE')
        self.assertEqual(fallback['old_analysis_id'], 'anonymous-0003')
        self.assertEqual(stat.S_IMODE(self.map_path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o700)
        old_map = self.map_path.read_bytes()
        second = self.root/'second'
        self.assertEqual(run(self.original, second, self.map_path), result)
        self.assertEqual(self.map_path.read_bytes(), old_map)
        for path in self.output.rglob('*'):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (second/path.relative_to(self.output)).read_bytes())
                self.assertNotIn(b'PRIVATE', path.read_bytes())
                self.assertNotIn(b'2020-', path.read_bytes())
                self.assertNotIn(str(self.original).encode(), path.read_bytes())
        self.assertEqual(set(result), {'counts', 'outputs'})

    def test_retention_and_human_processing_preserve_all_scientific_fields(self):
        run(self.original, self.output, self.map_path)
        mapping = {(r['cohort'], r['old_username']): r['new_id'] for r in self.read_map()}
        for cohort in ('main', 'module'):
            original_people = human.read_jsonl(self.original/cohort/'participants.jsonl')
            new_people = human.read_jsonl(self.output/cohort/'participants.jsonl')
            old_rows, new_rows = human.payloads(self.original/cohort), human.payloads(self.output/cohort)
            self.assertEqual(len(old_rows), len(new_rows))
            self.assertEqual([p['username'] for p in new_people],
                             [mapping[(cohort, p['username'])] for p in original_people])
            outer_rows = human.read_jsonl(self.output/cohort/'responses.jsonl')
            self.assertEqual([r['id'] for r in outer_rows], list(range(1, len(new_rows) + 1)))
            old_retained = human.retained_users(cohort, original_people, old_rows)
            new_retained = human.retained_users(cohort, new_people, new_rows)
            self.assertEqual(new_retained, {mapping[(cohort, name)] for name in old_retained})
            self.assertEqual(len(new_retained), 2 if cohort == 'main' else 1)
            old_slim = human.deidentify(old_rows, original_people, old_retained)
            new_slim = human.deidentify(new_rows, new_people, new_retained)
            for before, after in zip(old_slim, new_slim):
                for field in ('participant_id', 'db_id', 'trial_end_time_iso'):
                    before.pop(field, None); after.pop(field, None)
                self.assertEqual(before, after)
            self.assertTrue(all(not row.get('trial_end_time_iso') for row in new_rows))
        excluded = mapping[('module', 'MODULE_EXCLUDED_PRIVATE')]
        self.assertTrue(any(r['username'] == excluded and not r['is_correct']
                            for r in human.payloads(self.output/'module')))

    def test_invalid_maps_fail_before_output_and_are_never_overwritten(self):
        run(self.original, self.output, self.map_path)
        good = self.read_map()
        variants = []
        variants.append(good[:-1])
        variants.append(good + [good[0]])
        for field, value in [('new_id', 'malformed'), ('old_analysis_id', 'wrong'), ('prolific_pid', 'wrong')]:
            bad = copy.deepcopy(good); bad[0][field] = value; variants.append(bad)
        bad = copy.deepcopy(good); bad[1]['new_id'] = bad[0]['new_id']; variants.append(bad)
        bad = copy.deepcopy(good)
        bad[0]['new_id'], bad[1]['new_id'] = bad[1]['new_id'], bad[0]['new_id']
        variants.append(bad)
        for index, entries in enumerate(variants):
            with self.subTest(index=index):
                map_path = self.root/f'bad{index}.csv'; self.write_map(map_path, entries)
                before = map_path.read_bytes(); destination = self.root/f'output{index}'
                with self.assertRaises(ValueError):run(self.original, destination, map_path)
                self.assertFalse(destination.exists()); self.assertEqual(before, map_path.read_bytes())
        malformed = self.root/'columns.csv'; malformed.write_text('wrong,columns\n'); malformed.chmod(0o600)
        with self.assertRaises(ValueError):run(self.original, self.root/'bad-columns', malformed)
        self.assertFalse((self.root/'bad-columns').exists())

    def test_invalid_response_preflight_does_not_create_map_or_output(self):
        good = copy.deepcopy(self.responses)
        for case in ('unknown', 'duplicate_id', 'duplicate_trial', 'nested_metadata', 'bad_flag', 'bad_duration'):
            with self.subTest(case=case):
                self.responses = copy.deepcopy(good)
                row = self.responses['module'][-1]
                if case == 'unknown':row['username'] = 'UNKNOWN_PRIVATE'
                if case == 'duplicate_id':row['id'] = self.responses['module'][0]['id']
                if case == 'duplicate_trial':row['puzzle_id'] = self.responses['module'][-2]['puzzle_id']
                if case == 'nested_metadata':
                    payload = json.loads(row['row_json']); payload['solution_id'] = {'secret': 'PRIVATE'}
                    row['row_json'] = json.dumps(payload)
                if case == 'bad_flag':row['is_correct'] = 'false'
                if case == 'bad_duration':row['response_time_ms'] = float('nan')
                self.write_inputs()
                with self.assertRaises(ValueError):run(self.original, self.output, self.map_path)
                self.assertFalse(self.output.exists()); self.assertFalse(self.map_path.exists())

    def test_path_isolation_and_fresh_output_include_symlinks(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):run(self.original, self.output, self.map_path)
        self.output.rmdir()
        for output, map_path in [(self.original/'replay', self.map_path),
                                 (self.output, self.original/'map.csv'),
                                 (self.output, self.output/'map.csv')]:
            with self.subTest(output=output, map_path=map_path):
                with self.assertRaises(ValueError):run(self.original, output, map_path)
                self.assertFalse(output.exists()); self.assertFalse(map_path.exists())
        alias = self.root/'input_alias'; alias.symlink_to(self.original, target_is_directory=True)
        with self.assertRaises(ValueError):run(self.original, alias/'replay', self.map_path)
        self.output.symlink_to(self.root/'absent', target_is_directory=True)
        with self.assertRaises(FileExistsError):run(self.original, self.output, self.map_path)
        self.output.unlink()
        self.map_path.symlink_to(self.root/'absent.csv')
        with self.assertRaises(ValueError):run(self.original, self.output, self.map_path)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
