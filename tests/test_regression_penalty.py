"""Objective, independent optimizer and replay identity checks for default L2."""
from pathlib import Path
import copy
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import DEFAULT_L2_PENALTY, regression_core as core, regressions
from analysis.regression_bootstrap import batch_fit, bootstrap_validation, training_features
from analysis.regression_optimizer import LogitObjective, damped_newton, method_metadata, penalty_metadata

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    x = np.array([[[0., .3], [1., .3], [.2, .3]],
                  [[.7, 0.], [.7, 1.], [.7, .4]],
                  [[0., 1.], [.4, 0.], [1., .2]],
                  [[1., 0.], [0., 1.], [0., 0.]]])
    mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool)
    q = np.array([[.1, .6, .3], [.5, .2, .3], [.2, .3, .5], [.8, .2, 0.]])
    human = np.array([[.5, .2, .3], [.2, .5, .3], [.4, .4, .2], [.3, .7, 0.]])
    return x, mask, q, human


def binary_objective(design, q, human, multiplicity, strength):
    xx = np.repeat(design.reshape(-1, design.shape[-1]), 2, axis=0)
    y = np.tile([0., 1.], q.size)
    mass = np.stack((human, q), axis=-1) * multiplicity[:, None, None]
    return LogitObjective('source', xx, y, mass.ravel(), l2_penalty=strength)


class RegressionPenaltyTests(unittest.TestCase):
    def test_analytic_gradient_and_hessian_against_finite_differences(self):
        x, mask, q, human = fixture()
        weights = np.array([1., 2., 0., 1.])
        design = training_features(x, mask, weights[None], True)[0]
        objectives = [LogitObjective('choice', x, q, weights, mask, .03),
                      binary_objective(design, q, human, weights, .03)]
        for objective in objectives:
            with self.subTest(task=objective.task):
                beta = np.linspace(-.8, .6, objective.x.shape[-1])
                _, gradient, hessian = objective.evaluate(beta, True)
                step = 1e-5
                directions = np.eye(len(beta)) * step
                numeric_gradient = np.array([(objective.evaluate(beta+d)[0]-objective.evaluate(beta-d)[0])/(2*step)
                                             for d in directions])
                numeric_hessian = np.column_stack([(objective.evaluate(beta+d)[1]-objective.evaluate(beta-d)[1])/(2*step)
                                                  for d in directions])
                np.testing.assert_allclose(gradient, numeric_gradient, atol=2e-10, rtol=1e-7)
                np.testing.assert_allclose(hessian, numeric_hessian, atol=2e-10, rtol=1e-7)

    def test_intercept_is_not_penalized_and_weights_are_mean_normalized(self):
        x = np.array([[1., -.5], [1., .2], [1., 1.]])
        beta = np.array([1.7, -.8]); y = np.array([0., 1., 1.]); weights = np.array([2., 4., 1.])
        plain = LogitObjective('source', x, y, weights, l2_penalty=0.)
        ridge = LogitObjective('source', x, y, weights, l2_penalty=.2)
        f, g, h = plain.evaluate(beta, True); fr, gr, hr = ridge.evaluate(beta, True)
        self.assertAlmostEqual(fr-f, .2/2 * beta[1]**2)
        np.testing.assert_allclose(gr-g, [0., .2*beta[1]], atol=1e-15)
        np.testing.assert_allclose(hr-h, np.diag([0., .2]), atol=1e-15)
        scaled = LogitObjective('source', x, y, weights*123, l2_penalty=.2)
        for a, b in zip(ridge.evaluate(beta, True), scaled.evaluate(beta, True)):
            np.testing.assert_allclose(a, b, atol=1e-15)

    def test_separated_fit_has_finite_independently_known_optimum(self):
        strength = 1e-4
        expected = brentq(lambda beta: strength*beta-expit(-beta), 0., 30.)
        objectives = [LogitObjective('choice', np.array([[[0.], [1.]]]), np.array([[0., 1.]]),
                                     np.ones(1), np.ones((1, 2), dtype=bool), strength),
                      LogitObjective('source', np.array([[1., -1.], [1., 1.]]),
                                     np.array([0., 1.]), np.ones(2), l2_penalty=strength)]
        for objective in objectives:
            beta, _, diagnostic = damped_newton(objective)
            self.assertTrue(diagnostic['gradient_tolerance_met'])
            self.assertLess(abs(beta[-1]-expected), 1e-5)
            self.assertLess(abs(beta[-1]), 10)
            if objective.task == 'source':
                self.assertLess(abs(beta[0]), 1e-10)

    def test_batch_matches_independent_scalar_fits_with_fold_specific_constant_means(self):
        x, mask, q, human = fixture()
        multiplicities = np.array([[1., 1., 1., 1.], [2., 0., 1., 1.], [0., 2., 0., 2.]])
        strength = 1e-4
        for source in (False, True):
            design = training_features(x, mask, multiplicities, source)
            if source:
                for b, counts in enumerate(multiplicities):
                    expected = x.copy()
                    for feature in range(x.shape[-1]):
                        varying = [t for t in range(len(x)) if np.ptp(x[t, mask[t], feature]) > 0]
                        denom = sum(counts[t] for t in varying)
                        average = sum(counts[t]*np.mean(x[t, mask[t], feature]) for t in varying)/denom if denom else .5
                        for t in set(range(len(x)))-set(varying):
                            expected[t, :, feature] = average
                    np.testing.assert_allclose(design[b, :, :, 1:], expected, atol=1e-15)
            beta, diagnostics = batch_fit(x, mask, q, multiplicities, human if source else None, l2_penalty=strength)
            for index, counts in enumerate(multiplicities):
                objective = (binary_objective(design[index], q, human, counts, strength) if source else
                             LogitObjective('choice', x, q, counts, mask, strength))
                oracle = minimize(lambda value: objective.evaluate(value), np.zeros(beta.shape[-1]),
                                  jac=True, hess=lambda value: objective.evaluate(value, True)[2],
                                  method='trust-exact', options={'gtol': 1e-10})
                self.assertLess(np.max(np.abs(objective.evaluate(oracle.x)[1])), 1e-8)
                np.testing.assert_allclose(beta[index], oracle.x, atol=1e-6, rtol=1e-6)
                self.assertEqual(diagnostics[index]['penalty'], penalty_metadata(strength))

    def test_default_is_tested_small_penalty_in_scalar_and_batch_fits(self):
        x, mask, q, human = fixture(); multiplicities = np.array([[1., 2., 1., 0.]])
        self.assertEqual(DEFAULT_L2_PENALTY, 1e-4)
        self.assertEqual(method_metadata(), method_metadata(1e-4))
        for source in (False, True):
            first = batch_fit(x, mask, q, multiplicities, human if source else None)
            second = batch_fit(x, mask, q, multiplicities, human if source else None, l2_penalty=1e-4)
            np.testing.assert_array_equal(first[0], second[0]); self.assertEqual(first[1], second[1])
        scalar = LogitObjective('choice', x, q, multiplicities[0], mask)
        explicit = LogitObjective('choice', x, q, multiplicities[0], mask, 1e-4)
        for first, second in zip(damped_newton(scalar), damped_newton(explicit)):
            if isinstance(first, dict): self.assertEqual(first, second)
            else: np.testing.assert_array_equal(first, second)

    def test_explicit_zero_preserves_unpenalized_fits(self):
        x, mask, q, human = fixture(); multiplicities = np.array([[1., 2., 1., 0.]])
        self.assertIsNone(method_metadata(0.)['penalty'])
        # Recorded from the unpenalized implementation before changing defaults.
        legacy = ([1.002881303970414, -0.6791479329659925],
                  [-0.2378468525920776, 1.6050560215933571, -1.128955829431917])
        for source in (False, True):
            beta, diagnostics = batch_fit(x, mask, q, multiplicities,
                                           human if source else None, l2_penalty=0.)
            np.testing.assert_allclose(beta[0], legacy[source], rtol=1e-12, atol=1e-12)
            self.assertIsNone(diagnostics[0]['penalty'])
            design = training_features(x, mask, multiplicities, source)[0]
            objective = (binary_objective(design, q, human, multiplicities[0], 0.) if source else
                         LogitObjective('choice', x, q, multiplicities[0], mask, 0.))
            oracle = minimize(lambda value: objective.evaluate(value), np.zeros(beta.shape[-1]),
                              jac=True, hess=lambda value: objective.evaluate(value, True)[2],
                              method='trust-exact', options={'gtol': 1e-10})
            np.testing.assert_allclose(beta[0], oracle.x, rtol=1e-6, atol=1e-6)

    def test_bootstrap_validation_uses_same_draws_and_unpenalized_test_loss(self):
        x, mask, q, human = fixture(); repeats = 8; seed = 947; strength = .02
        draws = np.random.default_rng(seed).integers(0, len(x), size=(repeats, len(x)))
        multiplicities = np.array([np.bincount(row, minlength=len(x)) for row in draws], dtype=float)
        for source in (False, True):
            coefficients, validation, _ = bootstrap_validation(x, mask, q, np.random.default_rng(seed), repeats,
                                                               human if source else None, l2_penalty=strength)
            expected_coefficients, _ = batch_fit(x, mask, q, multiplicities, human if source else None,
                                                l2_penalty=strength)
            np.testing.assert_array_equal(coefficients, expected_coefficients[:, 1:] if source else expected_coefficients)
            for b, counts in enumerate(multiplicities):
                expected_loss = 0.
                for held in np.flatnonzero(counts):
                    train = counts.copy(); train[held] = 0.
                    features = training_features(x, mask, train[None], source)[0]
                    objective = (binary_objective(features, q, human, train, strength) if source else
                                 LogitObjective('choice', x, q, train, mask, strength))
                    beta, _, _ = damped_newton(objective)
                    scores = features[held] @ beta
                    if source:
                        p = np.clip(expit(scores), 1e-12, 1-1e-12)
                        loss = -.5*np.sum(human[held]*np.log1p(-p)+q[held]*np.log(p))
                    else:
                        scores = np.where(mask[held], scores, -np.inf)
                        p = np.exp(scores-logsumexp(scores))
                        loss = -np.sum(q[held]*np.log(np.clip(p, 1e-12, 1.)))
                    expected_loss += counts[held]/len(x)*loss
                self.assertAlmostEqual(validation[b]['log_loss'], expected_loss, places=8)

    def test_penalty_identifies_manifests_tasks_and_checkpoint_validation(self):
        with patch.object(regressions, 'digest', return_value='fixture-hash'):
            baseline = regressions.manifest(ROOT/'data', ROOT/'intermediate/processed', 2, 0.)
            ridge = regressions.manifest(ROOT/'data', ROOT/'intermediate/processed', 2)
        self.assertNotIn('l2_penalty', baseline['parameters'])
        self.assertEqual(ridge['parameters']['l2_penalty'], DEFAULT_L2_PENALTY)
        self.assertIn('analysis/__init__.py', ridge['code_sha256'])
        self.assertNotEqual(regressions.encoded(baseline), regressions.encoded(ridge))
        with tempfile.TemporaryDirectory() as folder:
            regressions.prepare_output(Path(folder), baseline)
            with self.assertRaisesRegex(ValueError, 'manifest differs'):
                regressions.prepare_output(Path(folder), ridge)
        task = regressions.tasks_for('sudoku', {'GPT': 'model'}, 2)[0]
        self.assertEqual(task['l2_penalty'], DEFAULT_L2_PENALTY)
        fit = {'bootstrap_validation_repeats': 2, 'bootstrap_rejected_fits': 0,
               'bootstrap_shared_draws_for_coefficients_and_validation': True,
               'optimizer': method_metadata(.0001), 'feature_coefficients': {'feature': 1.},
               'bootstrap_coefficients': [[1.], [2.]], 'bootstrap_validation': {'log_loss': [.5, .6]}}
        regressions.validate_fit(fit, task, ['feature'])
        with self.assertRaisesRegex(ValueError, 'requested regression'):
            regressions.validate_fit(fit, {**task, 'l2_penalty': .001}, ['feature'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.json'
            regressions.save(path, {'run_fingerprint': 'run', 'task': task, 'fit': fit,
                                    'fit_sha256': hashlib.sha256(regressions.encoded(fit)).hexdigest()})
            self.assertEqual(regressions.read_checkpoint(path, 'run', task, ['feature']), fit)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                regressions.read_checkpoint(path, 'run', {**task, 'l2_penalty': .001}, ['feature'])

        # A historical task with no penalty field still means unpenalized, even
        # though newly generated tasks now default to the positive strength.
        legacy_task = regressions.tasks_for('sudoku', {'GPT': 'model'}, 2, 0.)[0]
        self.assertNotIn('l2_penalty', legacy_task)
        legacy_fit = {**fit, 'optimizer': method_metadata(0.)}
        regressions.validate_fit(legacy_fit, legacy_task, ['feature'])
        with self.assertRaisesRegex(ValueError, 'requested regression'):
            regressions.validate_fit(fit, legacy_task, ['feature'])
        with patch.object(core, 'execute_fit_task', return_value=(None, legacy_fit)) as execute:
            self.assertEqual(regressions.fit_task(legacy_task), legacy_fit)
        self.assertEqual(execute.call_args.kwargs['l2_penalty'], 0.)

    def test_invalid_penalties_fail_before_creating_outputs(self):
        x, mask, q, _ = fixture()
        for value in (-1., float('nan'), float('inf')):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    LogitObjective('choice', x, q, np.ones(len(x)), mask, value)
                with self.assertRaises(ValueError):
                    batch_fit(x, mask, q, np.ones((1, len(x))), l2_penalty=value)
                with tempfile.TemporaryDirectory() as folder:
                    output = Path(folder)/'output'
                    for command in ([str(ROOT/'reproduce.py'), 'full'], [str(ROOT/'analysis/run.py'), '--processed-dir', folder]):
                        result = subprocess.run([sys.executable, '-B', *command, '--l2-penalty', str(value),
                                                 '--output-dir', str(output)], capture_output=True, text=True)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('finite and nonnegative', result.stderr)
                        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
