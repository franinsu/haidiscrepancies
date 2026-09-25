"""Root entry-point isolation and truthful output selection."""
from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import reproduce
from analysis import DEFAULT_L2_PENALTY

ROOT = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def full_arguments(self, folder):
        root = Path(folder)
        return Namespace(data_dir=root / 'data', intermediate_dir=root / 'intermediate',
                         output_dir=root / 'results', mode='all', bootstrap=5000,
                         regression_draws=2000, permutations=10000, mi_permutations=2000,
                         workers=1, l2_penalty=DEFAULT_L2_PENALTY)

    def test_regularization_default_and_explicit_zero_reach_analysis(self):
        from analysis import run as analysis_run

        for arguments, expected in (([], 1e-4), (['--l2-penalty', '0'], 0.)):
            for command in ('analyze', 'full'):
                with self.subTest(command=command, expected=expected):
                    argv = ['reproduce.py', command, *arguments]
                    if command == 'analyze':
                        argv += ['--processed-dir', 'unused-inputs', '--output-dir', 'unused-output']
                    with patch.object(sys, 'argv', argv), patch.object(reproduce, 'invoke') as invoke, \
                            patch.object(reproduce, 'pipeline') as pipeline:
                        reproduce.main()
                    if command == 'analyze':
                        forwarded = invoke.call_args.args[1]
                        self.assertEqual(forwarded[forwarded.index('--l2-penalty') + 1], expected)
                    else:
                        self.assertEqual(pipeline.call_args.args[0].l2_penalty, expected)

            # Exercise the standalone analysis parser and real manifest writer,
            # replacing only the expensive numerical run.
            with self.subTest(command='analysis/run.py', expected=expected), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                argv = ['analysis/run.py', '--mode', 'regressions', '--data-dir', str(root/'data'),
                        '--processed-dir', str(root/'processed'), '--output-dir', str(root/'statistics'), *arguments]
                with patch.object(sys, 'argv', argv), patch.object(analysis_run.regressions, 'run') as run:
                    analysis_run.main()
                self.assertEqual(run.call_args.kwargs['l2_penalty'], expected)
                manifest = json.loads((root/'statistics/analysis_manifest.json').read_text())
                self.assertEqual(manifest['parameters']['l2_penalty'], expected)

    def test_relative_output_is_relative_to_callers_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(
                [sys.executable, '-B', str(ROOT / 'reproduce.py'), 'render',
                 '--input-dir', 'unused-inputs', '--output-dir', 'results',
                 '--only', 'fig_procedure'], cwd=folder, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(folder) / 'results/figures/fig_procedure.pdf').is_file())

    def test_full_keeps_individual_records_out_of_final_results(self):
        """Stub expensive inference, then exercise the real rendering boundary."""
        with tempfile.TemporaryDirectory() as folder:
            args = self.full_arguments(folder)
            args.data_dir.mkdir()
            raw = args.data_dir / 'raw.json'
            raw.write_text('original input')
            invoke = reproduce.invoke

            def stages(script, arguments):
                destination = Path(arguments[arguments.index('--output-dir') + 1])
                if script == 'processing/run.py':
                    destination.mkdir()
                    (destination / 'individual.json').write_text('private trial fixture')
                elif script == 'analysis/run.py':
                    self.assertEqual(arguments[arguments.index('--l2-penalty') + 1], DEFAULT_L2_PENALTY)
                    processed = Path(arguments[arguments.index('--processed-dir') + 1])
                    self.assertEqual((processed / 'individual.json').read_text(), 'private trial fixture')
                    destination.mkdir()
                    (destination / 'analysis_manifest.json').write_text('{"parameters": {}}')
                else:
                    invoke(script, [*arguments, '--only', 'fig_procedure'])

            with patch.object(reproduce, 'invoke', side_effect=stages):
                reproduce.pipeline(args)
            self.assertEqual(raw.read_text(), 'original input')
            self.assertEqual({p.name for p in args.intermediate_dir.iterdir()},
                             {'processed', 'statistics', 'run.json'})
            self.assertEqual({p.name for p in args.output_dir.iterdir()},
                             {'figures', 'tables', 'source_data', 'render_manifest.json'})
            self.assertTrue((args.output_dir / 'figures/fig_procedure.pdf').is_file())
            record = json.loads((args.intermediate_dir / 'run.json').read_text())
            self.assertEqual(record['status'], 'passed')
            self.assertEqual(record['settings']['l2_penalty'], DEFAULT_L2_PENALTY)
            self.assertEqual(record['completed_stages'], ['processing', 'analysis', 'figures'])

    def test_full_rejects_overlapping_or_existing_destinations_before_writing(self):
        for case in ('inside-data', 'contains-data', 'same', 'results-inside-intermediate',
                     'intermediate-inside-results', 'existing-intermediate', 'existing-results'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                args = self.full_arguments(folder)
                args.data_dir.mkdir()
                (args.data_dir / 'raw.json').write_text('original input')
                if case == 'inside-data':
                    args.intermediate_dir = args.data_dir / 'derived'
                elif case == 'contains-data':
                    args.output_dir = Path(folder)
                elif case == 'same':
                    args.output_dir = args.intermediate_dir
                elif case == 'results-inside-intermediate':
                    args.output_dir = args.intermediate_dir / 'results'
                elif case == 'intermediate-inside-results':
                    args.intermediate_dir = args.output_dir / 'intermediate'
                elif case == 'existing-intermediate':
                    args.intermediate_dir.mkdir()
                else:
                    args.output_dir.mkdir()
                before = set(Path(folder).rglob('*'))
                with patch.object(reproduce, 'invoke') as invoke:
                    with self.assertRaises((ValueError, FileExistsError)):
                        reproduce.pipeline(args)
                    invoke.assert_not_called()
                self.assertEqual(set(Path(folder).rglob('*')), before)
                self.assertEqual((args.data_dir / 'raw.json').read_text(), 'original input')

    def test_failed_run_is_recorded_only_in_intermediate(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.full_arguments(folder)
            with patch.object(reproduce, 'invoke', side_effect=RuntimeError('fixture failure')):
                with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                    reproduce.pipeline(args)
            record = json.loads((args.intermediate_dir / 'run.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertEqual(record['completed_stages'], [])
            self.assertEqual(list(args.output_dir.iterdir()), [])

if __name__ == '__main__':
    unittest.main()
