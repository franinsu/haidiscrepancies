import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from processing.export_models import ARCHIVES, public_response, run


def response():
    return dict(source='api', request_id='a' * 20, puzzle_id='synthetic',
                raw_answer='A: (1,2)\nB: (2,3)', raw_response='A: (1,2)\nB: (2,3)',
                input_kind='sequence', sample_index=2, reasoning_tokens=17,
                image_path='collection/models/images/synthetic.png', image_sha256='b' * 64,
                prompt_file='collection/models/prompts/synthetic.txt',
                prompt_file_used='collection/models/prompts/synthetic.txt', prompt_sha256='c' * 64,
                provider_response_json={'opaque': 'private provider payload'},
                provider_response_id='private provider identifier')


class ModelExportTests(unittest.TestCase):
    def test_preserves_pairing_answers_effort_and_hashes_without_provider_data_or_paths(self):
        original = response()
        removed = {'provider_response_json', 'provider_response_id',
                   'image_path', 'prompt_file', 'prompt_file_used'}
        expected = {k: v for k, v in original.items() if k not in removed}
        self.assertEqual(public_response(original), expected)
        self.assertTrue(removed <= original.keys())

    def test_refuses_unreviewed_data_and_personal_paths(self):
        for change in ({'access_token': 'secret'}, {'api_error': 'sensitive error'},
                       {'source': 'human'}, {'request_id': None},
                       {'image_path': '/Users/example/image.png'},
                       {'prompt_file': '.. /prompt.txt'.replace(' ', '')},
                       {'image_path': 'C:\\Users\\example\\image.png'},
                       {'image_path': 'C:/Users/example/image.png'},
                       {'image_path': 'file:///Users/example/image.png'},
                       {'image_path': '~/image.png'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                public_response({**response(), **change})

    def test_export_is_deterministic_complete_and_never_overwrites_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'original'
            for name in ARCHIVES:
                path = source / name / 'responses.jsonl'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(response()) + '\n')
                (path.parent / 'run_manifest.json').write_text('{"private": true}')
            first, second = root / 'first', root / 'second'
            manifest = run(source, first)
            self.assertEqual(run(source, second), manifest)
            self.assertEqual(set(manifest['removed_fields']),
                             {'provider_response_json', 'provider_response_id',
                              'image_path', 'prompt_file', 'prompt_file_used'})
            expected_paths = {'manifest.json'}
            for name in ARCHIVES:
                relative = f'{name}/responses.jsonl.gz'
                expected_paths.add(relative)
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
                entry = manifest['files'][relative]
                self.assertEqual(entry['requests'], 1)
                self.assertEqual(entry['bytes'], (first / relative).stat().st_size)
                self.assertEqual(entry['sha256'], hashlib.sha256((first / relative).read_bytes()).hexdigest())
                self.assertEqual(entry['original_sha256'],
                                 hashlib.sha256((source / name / 'responses.jsonl').read_bytes()).hexdigest())
                with gzip.open(first / relative, 'rt', encoding='utf-8') as stream:
                    self.assertEqual([json.loads(line) for line in stream], [public_response(response())])
                self.assertEqual(json.loads((source / name / 'responses.jsonl').read_text()), response())
            self.assertEqual({str(p.relative_to(first)) for p in first.rglob('*') if p.is_file()}, expected_paths)
            for destination in (source, source / 'nested', first):
                with self.subTest(destination=destination), self.assertRaises((ValueError, FileExistsError)):
                    run(source, destination)


if __name__ == '__main__':
    unittest.main()
