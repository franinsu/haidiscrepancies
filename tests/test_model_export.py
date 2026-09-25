import gzip
import json
import tempfile
import unittest
from pathlib import Path

from processing.export_models import ARCHIVES, public_response, run


def response():
    return dict(source='api', request_id='a' * 20, puzzle_id='synthetic',
                raw_answer='A: (1,2)\nB: (2,3)', raw_response='A: (1,2)\nB: (2,3)',
                input_kind='sequence', sample_index=2, reasoning_tokens=17,
                provider_response_json={'opaque': 'private provider payload'},
                provider_response_id='private provider identifier')


class ModelExportTests(unittest.TestCase):
    def test_preserves_pairing_answers_and_effort_and_removes_provider_data(self):
        original = response()
        expected = {k: v for k, v in original.items() if not k.startswith('provider_response')}
        self.assertEqual(public_response(original), expected)
        self.assertIn('provider_response_json', original)

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
            expected_paths = {'manifest.json'}
            for name in ARCHIVES:
                relative = f'{name}/responses.jsonl.gz'
                expected_paths.add(relative)
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
                with gzip.open(first / relative, 'rt', encoding='utf-8') as stream:
                    self.assertEqual([json.loads(line) for line in stream], [public_response(response())])
                self.assertEqual(json.loads((source / name / 'responses.jsonl').read_text()), response())
            self.assertEqual({str(p.relative_to(first)) for p in first.rglob('*') if p.is_file()}, expected_paths)
            for destination in (source, source / 'nested', first):
                with self.subTest(destination=destination), self.assertRaises((ValueError, FileExistsError)):
                    run(source, destination)


if __name__ == '__main__':
    unittest.main()
