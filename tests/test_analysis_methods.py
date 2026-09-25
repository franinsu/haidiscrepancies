"""Small scientific identities and stable-randomization checks."""
from pathlib import Path
import copy
from collections import Counter
import sys
import unittest
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis import metrics, entropy, mi, geometry, stimulus_modules, main_statistics
from analysis.perturbation_effects import apply_holm
from analysis.normalized_attraction import normalized_attraction
from analysis.regression_core import chance_to_ceiling_normalized_accuracy


def paired_fixture():
    data={'primary_sources':['Human','Model A','Model B','Model C'],'trials':{},'comparisons':[]}
    for index in range(10):
        for arm in ('related','unrelated_control'):
            a=f'A{index}_{arm}'; b=f'B{index}_{arm}'
            data['trials'][a]={'ids':['left','right']};data['trials'][b]={'ids':['left','right']}
            cells=[dict(a='left',b='left',n=7),dict(a='left',b='right',n=3),dict(a='right',b='left',n=2),dict(a='right',b='right',n=8)]
            data['comparisons'].append(dict(module='pair',block=f'block{index:02d}',family='minesweeper_lite' if index<5 else 'mini_sudoku',arm=arm,preceding=a,perturbed=b,
                joint_counts={source:copy.deepcopy(cells) for source in data['primary_sources']}))
    return data


class AnalysisMethodTests(unittest.TestCase):
    def test_tv_is_probability_mass_not_euclidean_distance(self):
        self.assertAlmostEqual(metrics.tv(Counter({'a':5,'b':5}),Counter({'b':5,'c':5})),.5)
        self.assertEqual(metrics.tv(Counter({'a':3}),Counter({'b':9})),1)

    def test_entropy_uses_catalog_size_even_for_unobserved_classes(self):
        self.assertAlmostEqual(entropy.normalized_entropy(1,3,'three catalog classes'),1/np.log2(3))
        self.assertEqual(entropy.normalized_entropy(0,3,'one observed class'),0)

    def test_normalization_preserves_negative_change_and_undefined_ceiling(self):
        self.assertEqual(normalized_attraction(1,0),None)
        self.assertAlmostEqual(normalized_attraction(.75,-.5),-2)

    def test_prediction_normalization_retains_below_chance(self):
        self.assertAlmostEqual(chance_to_ceiling_normalized_accuracy(.2,.5,.3),-1)
        self.assertEqual(chance_to_ceiling_normalized_accuracy(.5,.5,0),0)

    def test_mi_depends_on_pairing_not_only_marginals(self):
        dependent=[('a','a')]*5+[('b','b')]*5
        independent=[('a','a'),('a','b'),('b','a'),('b','b')]
        self.assertAlmostEqual(metrics.mutual_information(dependent),1)
        self.assertAlmostEqual(metrics.mutual_information(independent),0)

    def test_canonical_mi_is_invariant_to_input_cell_order(self):
        data=paired_fixture(); shuffled=copy.deepcopy(data)
        shuffled['comparisons'].reverse()
        for trial in shuffled['trials'].values():trial['ids'].reverse()
        for comparison in shuffled['comparisons']:
            for cells in comparison['joint_counts'].values():cells.reverse()
        self.assertEqual(mi.compute_tests(data,draws=16),mi.compute_tests(shuffled,draws=16))

    def test_holm_excludes_unrelated_context_before_adjustment(self):
        rows=[dict(row='cue',p_value=.01),dict(row='pair_related',p_value=.04),dict(row='pair_control',p_value=.0001)]
        apply_holm(rows)
        self.assertAlmostEqual(rows[0]['p_holm'],.02)
        self.assertAlmostEqual(rows[1]['p_holm'],.04)
        self.assertIsNone(rows[2]['p_holm'])

    def test_mi_seed_changes_for_distinct_tests(self):
        self.assertEqual(mi.stable_seed('Human','block','related'),mi.stable_seed('Human','block','related'))
        self.assertNotEqual(mi.stable_seed('Human','block','related')[0],mi.stable_seed('Human','block','unrelated_control')[0])

    def test_mi_rejects_unknown_classes_and_fractional_counts(self):
        data=paired_fixture();data['comparisons'][0]['joint_counts']['Human'][0]['a']='unknown'
        with self.assertRaises(ValueError):mi.compute_tests(data,draws=2)
        data=paired_fixture();data['comparisons'][0]['joint_counts']['Human'][0]['n']=1.5
        with self.assertRaises(ValueError):mi.compute_tests(data,draws=2)

    def test_centered_distance_kernel_recovers_euclidean_geometry(self):
        points=np.array([[0.,0.],[1.,0.],[0.,2.]])
        distances=np.linalg.norm(points[:,None]-points[None,:],axis=-1)
        kernel=-.5*geometry.centering_matrix(3)@(distances**2)@geometry.centering_matrix(3)
        centered=points-points.mean(axis=0)
        np.testing.assert_allclose(kernel,centered@centered.T,atol=1e-12)

    def test_human_pairs_use_retained_linkage_without_raw_history(self):
        trials = {name:SimpleNamespace(id=name,sequence_id='sequence') for name in ('A','B')}
        design = SimpleNamespace(trials=trials)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'human/module_retained.jsonl'
            path.parent.mkdir()
            rows = [dict(puzzle_id=name,participant_id='participant',sequence_id='sequence',
                         is_correct=True,abstract_solution_id=name.lower()) for name in trials]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            source = stimulus_modules.load_human_answers(Path(directory),design)
            self.assertEqual(stimulus_modules.paired_classes(source,trials['A'],trials['B']),[('a','b')])
            path.write_text(path.read_text()+json.dumps(rows[0])+'\n')
            with self.assertRaises(ValueError):
                stimulus_modules.load_human_answers(Path(directory),design)

    def test_module_loader_ignores_unreported_model_conditions(self):
        design = SimpleNamespace(trials={'P':SimpleNamespace(id='P')})
        providers = {'Model':{'slug':'model','conditions':{'low':'LOW','medium':'MEDIUM'}}}
        prompts = {'direct_solve':'plain','human_participant':'persona'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'ai/model_module.jsonl'
            path.parent.mkdir()
            rows = [dict(puzzle_id='P',api_model_condition='LOW',prompt_condition='direct_solve',
                         request_id='request',sample_index=0,is_valid=True,abstract_solution_id='answer'),
                    dict(puzzle_id='P',api_model_condition='MEDIUM',prompt_condition='direct_solve'),
                    dict(puzzle_id='P',api_model_condition='LOW',prompt_condition='human_participant')]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            sources = stimulus_modules.load_model_answers(Path(directory),design,providers,prompts,
                                                          lambda provider,effort,prompt:'Model|low|plain')
            self.assertEqual(set(sources),{'Model|low|plain'})
            self.assertEqual(sources['Model|low|plain'].counts('P'),Counter({'answer':1}))


class LogDensityValidationTests(unittest.TestCase):
    def test_constant_log_density_on_either_side_is_rejected(self):
        uniform = Counter(a=3, b=3, c=3)
        varied = Counter(a=5, b=3, c=1)
        for left, right in ((uniform, varied), (varied, uniform), (uniform, uniform)):
            with self.subTest(left=left, right=right):
                with self.assertRaisesRegex(ValueError, 'undefined for a constant vector'):
                    main_statistics.smoothed_log_density_correlation(left, right, ['a', 'b', 'c'])

    def test_unobserved_catalog_classes_are_not_mistaken_for_uniformity(self):
        # Equal counts on the observed support are nonconstant across the
        # complete catalog when another class has no observations.
        counts = Counter(a=3, b=3)
        self.assertAlmostEqual(main_statistics.smoothed_log_density_correlation(
            counts, counts, ['a', 'b', 'c']), 1.0)
        self.assertAlmostEqual(main_statistics.smoothed_log_density_correlation(
            Counter(a=5, b=1), Counter(a=1, b=5), ['a', 'b']), -1.0)

    def test_constant_puzzle_aborts_summary_instead_of_being_omitted(self):
        puzzles = {p: {'solutions': [{'solution_id': s} for s in ('a', 'b')]}
                   for p in ('valid', 'constant')}
        counters = {
            'Human': {'valid': Counter(a=5, b=1), 'constant': Counter(a=3, b=3)},
            'Model': {'valid': Counter(a=1, b=5), 'constant': Counter(a=5, b=1)},
        }
        with patch.object(main_statistics, 'PROVIDERS', {'Model': {}}), \
             patch.object(main_statistics, 'clustered_mean_ci') as intervals:
            with self.assertRaisesRegex(ValueError, 'undefined for a constant vector'):
                main_statistics.primary_log_density_correlations(
                    puzzles, ['valid', 'constant'], counters, {}, [], {})
            intervals.assert_not_called()


class ResponseValidationTests(unittest.TestCase):
    def test_empty_original_cell_fails_before_response_sampling(self):
        for empty_source in ('Human', 'Model|low|plain'):
            with self.subTest(source=empty_source):
                human = {'person': {'p': None if empty_source == 'Human' else 'a'}}
                model = {'Model|low|plain': {'p': [
                    None if empty_source == 'Model|low|plain' else 'a'
                ] * 100}}
                with patch.object(main_statistics.np.random, 'default_rng') as rng:
                    with self.assertRaises(ValueError) as caught:
                        main_statistics.global_tv_response_bootstrap_cis(
                            {'p': {'solutions': [{'solution_id': 'a'}, {'solution_id': 'b'}]}},
                            ['p'], ['Uniform', 'Human', 'Model'],
                            {'Model': 'Model|low|plain'}, [[('p',)]], human, model,
                            repeats=1,
                        )
                    rng.return_value.multinomial.assert_not_called()
                self.assertEqual(str(caught.exception),
                    f'No valid responses for required primary cell: {empty_source}, p')

    def test_compute_rejects_empty_primary_cell_before_metrics_or_puzzle_draws(self):
        for empty_source in ('Human', 'Model|low|plain'):
            with self.subTest(source=empty_source):
                rows = {
                    'all_puzzles.jsonl': [{'id': 'p'}],
                    'main_retained.jsonl': [dict(puzzle_id='p', participant_id='person',
                        is_correct=empty_source != 'Human', solution_id='a')],
                    'model_main.jsonl': [dict(puzzle_id='p', api_model_condition='LOW',
                        prompt_condition='direct_solve',
                        is_valid=empty_source != 'Model|low|plain', solution_id='a')],
                }
                with patch.object(main_statistics, 'PROVIDERS',
                                  {'Model': {'slug': 'model', 'conditions': {'low': 'LOW'}}}), \
                     patch.object(main_statistics, 'puzzle_metadata', return_value=({}, {})), \
                     patch.object(main_statistics.stimulus_modules, 'load_design', return_value=(None, None)), \
                     patch.object(main_statistics, 'read_jsonl', side_effect=lambda path: rows[path.name]), \
                     patch.object(main_statistics, 'cluster_bootstrap_draws') as draws, \
                     patch.object(main_statistics, 'entropy') as metric:
                    with self.assertRaises(ValueError) as caught:
                        main_statistics.compute(Path('data'), Path('processed'))
                    draws.assert_not_called()
                    metric.assert_not_called()
                self.assertEqual(str(caught.exception),
                    f'No valid responses for required primary cell: {empty_source}, p')

    def test_compute_rejects_empty_secondary_cells_before_metrics_or_puzzle_draws(self):
        secondary = [('low', 'human_participant'), ('medium', 'direct_solve'),
                     ('medium', 'human_participant')]
        for effort, prompt in secondary:
            for failure in ('all-invalid', 'missing'):
                with self.subTest(effort=effort, prompt=prompt, failure=failure):
                    model_rows = [dict(puzzle_id='p', api_model_condition=e.upper(),
                                      prompt_condition=p, is_valid=True, solution_id='a')
                                  for e in ('low', 'medium')
                                  for p in ('direct_solve', 'human_participant')]
                    target = next(row for row in model_rows
                                  if row['api_model_condition'] == effort.upper()
                                  and row['prompt_condition'] == prompt)
                    if failure == 'missing':
                        model_rows.remove(target)
                    else:
                        target['is_valid'] = False
                    rows = {
                        'all_puzzles.jsonl': [{'id': 'p'}],
                        'main_retained.jsonl': [dict(puzzle_id='p', participant_id='person',
                                                    is_correct=True, solution_id='a')],
                        'model_main.jsonl': model_rows,
                    }
                    with patch.object(main_statistics, 'PROVIDERS',
                                      {'Model': {'slug': 'model', 'conditions': {'low': 'LOW', 'medium': 'MEDIUM'}}}), \
                         patch.object(main_statistics, 'puzzle_metadata', return_value=({}, {})), \
                         patch.object(main_statistics.stimulus_modules, 'load_design', return_value=(None, None)), \
                         patch.object(main_statistics, 'read_jsonl', side_effect=lambda path: rows[path.name]), \
                         patch.object(main_statistics, 'stimulus_clusters') as clusters, \
                         patch.object(main_statistics, 'cluster_bootstrap_draws') as draws, \
                         patch.object(main_statistics, 'entropy') as entropy_metric, \
                         patch.object(main_statistics, 'tv') as tv_metric:
                        with self.assertRaises(ValueError) as caught:
                            main_statistics.compute(Path('data'), Path('processed'))
                        clusters.assert_not_called()
                        draws.assert_not_called()
                        entropy_metric.assert_not_called()
                        tv_metric.assert_not_called()
                    source = main_statistics.condition_key('Model', effort, prompt)
                    self.assertEqual(str(caught.exception),
                        f'No valid responses for required secondary cell: {source}, p')

    def test_mixed_valid_and_invalid_responses_still_redraw_empty_samples(self):
        # Both Human profiles are sampled together; some draws select only
        # the invalid profile. Every accepted distribution is concentrated on a.
        sampler = Mock(wraps=np.random.default_rng(12))
        with patch.object(main_statistics.np.random, 'default_rng', return_value=sampler):
            result = main_statistics.global_tv_response_bootstrap_cis(
                {'p': {'solutions': [{'solution_id': 'a'}, {'solution_id': 'b'}]}},
                ['p'], ['Uniform', 'Human', 'Model'], {'Model': 'Model|low|plain'},
                [[('p',)]] * 8, {'valid': {'p': 'a'}, 'invalid': {'p': None}},
                {'Model|low|plain': {'p': ['a'] * 50 + [None] * 50}}, repeats=8,
            )
        self.assertGreater(sampler.multinomial.call_count, 2)
        expected = [[None, [.5, .5], [.5, .5]],
                    [[.5, .5], None, [0., 0.]], [[.5, .5], [0., 0.], None]]
        self.assertEqual(result['ci95_trial_matrix'], expected)
        self.assertEqual(result['ci95_joint_matrix'], expected)


if __name__=='__main__':unittest.main()
