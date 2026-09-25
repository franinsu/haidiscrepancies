import hashlib
import importlib
import json
from contextlib import chdir
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from processing import response_parser
from collection.human import assignment_scheduler, server

ROOT = Path(__file__).resolve().parents[1]


class CollectionTests(unittest.TestCase):
    def test_single_v5_parser_is_shared_with_collection(self):
        runner = importlib.import_module('collection.models.run_api_collection')
        self.assertIs(runner.score_raw_answer, response_parser.score_raw_answer)
        self.assertIs(server.score_raw_answer, response_parser.score_raw_answer)
        self.assertEqual(response_parser.PARSER_VERSION, 'response_parser_v5')
        self.assertEqual(hashlib.sha256(Path(response_parser.__file__).read_bytes()).hexdigest(),
                         '92d73cd85c915bae89c2957fc39a8fc6365ac97482fe9b201b5ed0cccb0983c7')

    def test_study_modules_import_only_clean_project_sources(self):
        names = ['collection.models.' + name for name in
                 ['build_api_run_manifest', 'run_api_collection', 'run_api_batch', 'run_integrity', 'score_api_run', 'render_api_images']]
        names += ['collection.stimuli.generate_all', 'collection.stimuli.generate_study_modules',
                  'collection.stimuli.validate_dataset', 'collection.stimuli.validate_study_modules']
        for name in names:
            module = importlib.import_module(name)
            self.assertTrue(Path(module.__file__).resolve().is_relative_to(ROOT), name)

    def test_human_assignment_and_quality_checks_use_copied_stimuli(self):
        self.assertEqual(len(assignment_scheduler.make_assignment('M_SYNTH', 'main')['puzzle_ids']), 100)
        assignment = assignment_scheduler.make_assignment('E_SYNTH', 'module')
        self.assertIn(len(assignment['puzzle_ids']), [61, 62])
        self.assertEqual(set(assignment['module_counts']), {'cue', 'spatial', 'formulation', 'pair', 'transfer'})
        checks = server.load_quality_check_index()
        self.assertTrue(checks)
        self.assertTrue(all(row['solutions'] for row in checks.values()))

    def test_empty_database_initializes_without_network_or_archive(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            db = server.StudyDatabase(Path(directory)/'synthetic.sqlite3')
            with db.connect() as connection:
                self.assertEqual(db.analysis_retention_by_user(connection), {})
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute('SELECT 1')
            with self.assertRaisesRegex(ValueError, 'synthetic failure'):
                with db.connect() as failed_connection:
                    raise ValueError('synthetic failure')
            with self.assertRaises(sqlite3.ProgrammingError):
                failed_connection.execute('SELECT 1')
            self.assertTrue(Path(db.backup_database()['backup_path']).is_file())

    def test_registration_scoring_and_private_payload_boundary(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            db = server.StudyDatabase(Path(directory)/'synthetic.sqlite3')
            assignment = db.register('M_SYNTH', 'main', participant_source='tester')
            public = server.public_assignment_payload(assignment)
            self.assertNotIn('completion_code', public)
            self.assertEqual(len(public['public_puzzles']), 100)
            self.assertTrue(all('solutions' not in puzzle for puzzle in public['public_puzzles']))
            first_id = assignment['puzzle_ids'][0]
            puzzle = server.load_puzzle_index()[first_id]
            solution = puzzle['solutions'][0]
            if 'expression' in solution:
                answer = solution['expression']
            elif 'moves' in solution:
                answer = solution['moves']
            else:
                coords = solution.get('coordinates') or [solution['coordinate']]
                answer = ', '.join(f'({r},{c})' for r, c in coords)
            saved = db.save_response({'username': 'M_SYNTH', 'puzzle_id': 'T001',
                                      'raw_answer': answer, 'response_time_ms': 1000,
                                      'trial_start_time_iso': '2026-01-01T00:00:00',
                                      'trial_end_time_iso': '2026-01-01T00:00:01'})
            self.assertTrue(saved['is_correct'])
            with db.connect() as connection:
                summary = db.analysis_retention_by_user(connection)['M_SYNTH']
            self.assertFalse(summary['analysis_retained'])
            self.assertIn('tester', summary['analysis_exclusion_reasons'])
            exported = db.export_rows_page('responses', cohort='main')
            self.assertEqual(exported['total_count'], 1)
            self.assertTrue(json.loads(exported['rows'][0]['row_json'])['is_correct'])
            self.assertEqual(db.export_rows_page('responses', cohort='module')['rows'], [])

    def test_minimal_admin_exports_require_authentication(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            db = server.StudyDatabase(Path(directory) / 'synthetic.sqlite3')
            db.register('M_SYNTH', 'main', participant_source='tester')
            handler = server.StudyHandler.__new__(server.StudyHandler)
            handler.db = db
            handler.headers = {}
            handler.path = '/api/admin/export_rows?table=participants&cohort=main'
            handler.send_json = Mock()
            handler.handle_read_request()
            self.assertEqual(handler.send_json.call_args.kwargs['status'], 401)
            self.assertNotIn('rows', handler.send_json.call_args.args[0])
            handler.admin_is_authenticated = lambda: True
            handler.handle_read_request()
            self.assertEqual(handler.send_json.call_args.args[0]['total_count'], 1)
            handler.path = '/api/admin/status'
            handler.handle_read_request()
            self.assertEqual(handler.send_json.call_args.args[0]['counts']['participants'], 1)

    def test_three_study_providers_prepare_batches_and_mock_without_network(self):
        from collection.models import build_api_run_manifest, run_api_batch, run_api_collection, run_integrity
        with chdir(ROOT), tempfile.TemporaryDirectory(dir=ROOT / 'tests') as directory:
            config = json.loads((ROOT / 'collection/models/templates/api_model_conditions.json').read_text())
            self.assertEqual(len(config), 6)
            self.assertEqual({item['provider'] for item in config.values()}, {'openai', 'anthropic', 'google'})
            with self.assertRaises(run_integrity.RunIntegrityError):
                run_integrity.validate_static_assets(directory)
            for name in [name for name in config if name.endswith('_LOW')]:
                build_api_run_manifest.build_api_run_manifest(
                    run_id=name, out_root=directory, model_conditions=[name],
                    samples=1, dataset_scope='main', prompt_conditions=['direct_solve'])
                run_dir = Path(directory) / name
                manifest = run_api_batch.prepare_api_batches(str(run_dir), limit=1)
                self.assertEqual(len(manifest['batches']), 1)
                self.assertEqual(manifest['batches'][0]['provider'], config[name]['provider'])
                with patch.object(run_api_collection, '_live_response', side_effect=AssertionError('network forbidden')):
                    result = run_api_collection.run_api_collection(str(run_dir), mode='mock', limit=1)
                self.assertEqual(result['raw_attempt_rows_written_this_run'], 1)
                self.assertEqual(result['api_errors_this_run'], 0)
                self.assertFalse((run_dir / 'raw.jsonl').exists())

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for the browser-script smoke check')
    def test_formal_browser_bootstrap_uses_only_existing_controls(self):
        web = ROOT / 'collection/human/study_web'
        for page, script in [('app', 'app.js'), ('admin', 'admin.js'), ('admin_test', 'admin_test.js')]:
            html = (web / page / 'index.html').read_text()
            source = (web / page / script).read_text()
            ids = set(re.findall(r'id="([^"]+)"', html))
            self.assertTrue(set(re.findall(r'\$\("([^"]+)"\)', source)) <= ids, page)
            subprocess.run(['node', '--check', str(web / page / script)], check=True, capture_output=True)
        program = r'''
const fs = require('fs'), vm = require('vm');
const web = process.argv[1];
const ids = [...fs.readFileSync(web+'/app/index.html','utf8').matchAll(/id="([^"]+)"/g)].map(m=>m[1]);
const quality = JSON.parse(fs.readFileSync(web+'/app/quality_checks.json','utf8'));
(async () => {
  for (const search of ['?study=main', '?study=module', '?study=module&tester=1']) {
    const elements = Object.fromEntries(ids.map(id=>[id, {value:'',checked:false,textContent:'',
      classList:{add(){},remove(){},toggle(){}},addEventListener(){},replaceChildren(){},focus(){}}]));
    const context = {URLSearchParams, console, setTimeout, performance:{getEntriesByType:()=>[]},
      window:{location:{search,hostname:'localhost',pathname:'/study_web/app/',hash:''},history:{replaceState(){}}},
      sessionStorage:{getItem:()=>null,setItem(){}},
      document:{body:{classList:{add(){}}},getElementById:id=>{if(!elements[id])throw Error(id);return elements[id];}},
      fetch:async path=>{if(path!='/study_web/app/quality_checks.json')throw Error(path);return {ok:true,json:async()=>quality};}};
    vm.createContext(context);
    vm.runInContext(fs.readFileSync(web+'/app/app.js','utf8'),context);
    await new Promise(resolve=>setImmediate(resolve));
    if(elements.loadStatus.textContent!=='Ready to register.')throw Error(elements.loadStatus.textContent);
    if(vm.runInContext('state.formalMode',context)!==true)throw Error('not formal');
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
        subprocess.run(['node', '-e', program, str(web)], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
