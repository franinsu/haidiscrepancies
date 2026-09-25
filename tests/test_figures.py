"""Checks for figure provenance, public exports and numerical presentation."""
import json
import hashlib
import os
from pathlib import Path
import re
import tempfile
import unittest

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'clean-study-matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'clean-study-cache'))
from figures import tv_geometry, tables, tables_divergence
from figures.run import run

ROOT = Path(__file__).resolve().parents[1]


class FigureTests(unittest.TestCase):
    def test_manifest_declares_public_inputs_and_unique_outputs(self):
        """The reproduction inventory is usable without a manuscript checkout."""
        manifest=json.loads((ROOT/'figures/manifest.json').read_text())
        entries=manifest['outputs']
        self.assertEqual(len(entries),len({row['name'] for row in entries}))
        self.assertEqual(len(entries),len({row['output'] for row in entries}))
        for row in entries:
            with self.subTest(name=row['name']):
                output=Path(row['output'])
                self.assertEqual(output.parent,Path('figures' if row['kind']=='figure' else 'tables'))
                self.assertEqual(output.stem,row['name'])
                self.assertTrue(row['inputs'])
                self.assertIn(row['public_input_class'],{'study_design','display_summary'})
                for name in row['inputs']:
                    self.assertFalse(Path(name).is_absolute())
                    self.assertNotIn('..',Path(name).parts)
                    if row['public_input_class']=='study_design':
                        self.assertTrue((ROOT/row['input_base']/name).is_file())
                if row['public_input_class']=='display_summary':
                    self.assertEqual(row['input_base'],'numerical input directory')
                    self.assertEqual(row['source_data'],f"source_data/{row['name']}.json")

    def test_figure2_replaces_historical_numerical_labels(self):
        """Changed inputs must change the plotted labels, including every CI."""
        matrix=[[0.0 if i==j else .44 for j in range(5)] for i in range(5)]
        intervals=[[None if i==j else [.31,.57] for j in range(5)] for i in range(5)]
        stats={'global_tv_primary':{'matrix':matrix,'ci95_puzzle_matrix':intervals},
               'pairwise_tv_by_type_primary':{f:{'matrix':matrix,'ci95_matrix':intervals} for f in tv_geometry.FAMILY_KEYS}}
        geometry={name:{'coordinates':[[i*.08-.2,i*.04-.1] for i in range(5)],'shares':[.61,.27]} for name in ['mean_tv_mds','tv_family_geometry_cka']}
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'test.pdf'
            payload=tv_geometry.render(stats,geometry,output)
            import fitz
            with fitz.open(output) as document:text=document[0].get_text()
            self.assertEqual(text.count('[0.31,0.57]'),20)
            self.assertIn('MDS 1 (61.0%)',text)
            self.assertEqual(len(payload['family_intervals']),50)
            self.assertTrue(all(r['ci95']==[.31,.57] for r in payload['family_intervals']))
            self.assertFalse(output.with_suffix('.png').exists())
            tv_geometry.render(stats,geometry,output,previews=True)
            self.assertTrue(output.with_suffix('.png').exists())

    def test_only_selected_canonical_board_format_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'result'
            run(Path(folder)/'inputs',output,ROOT/'data/stimuli',only=['fBboard_types'])
            self.assertEqual({p.name for p in (output/'figures').iterdir()},{'fBboard_types.pdf'})

    def test_reference_mode_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'require --reference'):
                run(Path(folder)/'saved-statistics',Path(folder)/'result',ROOT/'data/stimuli',only=['tv_source_geometry'])

    def test_existing_output_is_not_silently_mixed(self):
        for collision in ['figures/old.pdf','tables/old.tex','source_data/old.json',
                          'render_manifest.json','figures','tables','source_data']:
            with self.subTest(collision=collision), tempfile.TemporaryDirectory() as folder:
                output=Path(folder)/'output'
                old=output/collision;old.parent.mkdir(parents=True);old.write_text('old')
                before=set(output.rglob('*'))
                with self.assertRaises(FileExistsError):
                    run(Path(folder)/'inputs',output,ROOT/'data/stimuli',only=['fig_procedure'])
                self.assertEqual(old.read_text(),'old')
                self.assertEqual(set(output.rglob('*')),before)

    def test_render_preserves_statistics_and_other_result_siblings(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'results';inputs=output/'statistics';inputs.mkdir(parents=True)
            sentinels={inputs/'sentinel.json':'{"keep": true}',
                       inputs/'analysis_manifest.json':'{"parameters": {"bootstrap": 5000}}',
                       output/'processed/raw.txt':'untouched',output/'run.json':'{"status": "analysis complete"}'}
            for path,text in sentinels.items():
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
            (output/'tables').mkdir()  # Existing empty managed folders are safe.
            result=run(inputs,output,ROOT/'data/stimuli',only=['fig_procedure'])
            for path,text in sentinels.items():self.assertEqual(path.read_text(),text)
            for relative,digest in result['outputs'].items():
                self.assertEqual(hashlib.sha256((output/relative).read_bytes()).hexdigest(),digest)
            self.assertEqual(json.loads((output/'render_manifest.json').read_text()),result)
            self.assertTrue((output/'source_data/fig_procedure.json').is_file())

    def test_inputs_cannot_overlap_managed_outputs(self):
        for relative in ['results','results/figures','results/figures/inputs',
                         'results/tables','results/source_data/inputs','results/render_manifest.json','.']:
            with self.subTest(input=relative), tempfile.TemporaryDirectory() as folder:
                base=Path(folder);output=base/'results';inputs=base/relative
                inputs.mkdir(parents=True,exist_ok=True)
                sentinel=inputs/'sentinel.json';sentinel.write_text('preserve')
                before=set(base.rglob('*'))
                with self.assertRaisesRegex(ValueError,'overlap'):
                    run(inputs,output,ROOT/'data/stimuli',only=['fig_procedure'])
                self.assertEqual(sentinel.read_text(),'preserve')
                self.assertEqual(set(base.rglob('*')),before)
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);output=base/'results'
            with self.assertRaisesRegex(ValueError,'Stimulus inputs overlap'):
                run(base/'statistics',output,output/'figures/stimuli',only=['fig_procedure'])
            self.assertFalse(output.exists())

    def test_managed_output_symlinks_are_not_followed(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);output=base/'results';output.mkdir()
            unrelated=base/'unrelated';unrelated.mkdir()
            (output/'source_data').symlink_to(unrelated,target_is_directory=True)
            with self.assertRaises(FileExistsError):
                run(base/'statistics',output,ROOT/'data/stimuli',only=['fig_procedure'])
            self.assertEqual(list(unrelated.iterdir()),[])
            self.assertEqual({p.name for p in output.iterdir()},{'source_data'})

    def test_accuracy_table_uses_current_manuscript_environments(self):
        fit={'cv_normalized_accuracy':.2,'cv_normalized_accuracy_ci_95':[.1,.3]}
        choices={source:fit for source in tables.SOURCES}
        model={source:fit for source in tables.SOURCES[1:]}
        def read(name):
            if name=='maze_three_feature_models':
                return {'fits':{'choice':dict(zip(tables.LABELS,[fit]*4)),
                                'source':dict(zip(tables.LABELS[1:],[fit]*3))}}
            return {'selected_solution_prediction':{'conditional':{'fits':choices}},
                    'source_prediction':{'binary_vs_human':{'conditional':{'fits':model}}}}
        text, values=tables.accuracy_table(read)
        self.assertEqual(set(re.findall(r'\\begin\{([^}]+)\}',text)),{'table','tabular'})
        self.assertEqual(len(values),35)
        self.assertIn(r'\label{tab:feature-accuracy-overview}',text)

    def test_mi_export_contains_only_displayed_summaries(self):
        records=[{'source':source,'context':context,'p_unadjusted':.01 if i<3 else .2,
                  'participant_id':'MUST_NOT_LEAK','raw_response':'MUST_NOT_LEAK'}
                 for source in tables.LABELS for context in ['Related','Unrelated'] for i in range(10)]
        text,export=tables.mi_table({'individual_independence_tests':records})
        self.assertEqual(len(export),8)
        self.assertTrue(all(r['below_005']==3 for r in export))
        self.assertNotIn('MUST_NOT_LEAK',json.dumps(export)+text)


if __name__=='__main__':unittest.main()
