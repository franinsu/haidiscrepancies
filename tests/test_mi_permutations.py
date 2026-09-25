"""MI permutation sampling preserves the study's Monte Carlo conventions."""
from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import mi


def reference_null(pairs, draws, rng):
    """Independent, direct calculation of the original shuffled-pair null."""
    a, b = zip(*pairs)
    return np.array([
        mi.mutual_information(list(zip(a, rng.permutation(b))))
        for _ in range(draws)
    ])


class MiPermutationTests(unittest.TestCase):
    def test_randomized_pairings_preserve_seed_and_null(self):
        pairs = [('a', 'x')] * 8 + [('a', 'y')] * 2 + [('b', 'x')] * 3 + [('b', 'y')] * 7
        for seed in (0, 81):
            with self.subTest(seed=seed):
                rng = np.random.Generator(np.random.PCG64(seed))
                reference_rng = np.random.Generator(np.random.PCG64(seed))
                mean, null = mi.mi_permutation_null(pairs, 200, rng)
                reference = reference_null(pairs, 200, reference_rng)
                np.testing.assert_allclose(null, reference, atol=1e-15, rtol=0)
                self.assertEqual(mean, float(null.mean()))
                self.assertEqual(rng.bit_generator.state, reference_rng.bit_generator.state)

    def test_small_samples_keep_requested_random_draws(self):
        for pairs in ([('a', 'x')], [('a', 'x'), ('b', 'y')],
                      [('a', 'x'), ('b', 'y'), ('a', 'y')]):
            for draws in (math.factorial(len(pairs)), 20):
                with self.subTest(pairs=pairs, draws=draws):
                    rng = np.random.Generator(np.random.PCG64(0))
                    reference_rng = np.random.Generator(np.random.PCG64(0))
                    with patch.object(mi, 'permutation_test', side_effect=AssertionError('exact enumeration')):
                        mean, null = mi.mi_permutation_null(pairs, draws, rng)
                    self.assertEqual(len(null), draws)
                    np.testing.assert_allclose(null, reference_null(pairs, draws, reference_rng), atol=1e-15, rtol=0)
                    self.assertEqual(rng.bit_generator.state, reference_rng.bit_generator.state)
                    self.assertEqual(mean, float(null.mean()))

    def test_constant_answer_has_zero_information(self):
        pairs = [('a', 'x')] * 10 + [('a', 'y')] * 10
        mean, null = mi.mi_permutation_null(pairs, 100, np.random.default_rng(0))
        self.assertEqual(mean, 0)
        np.testing.assert_array_equal(null, np.zeros(100))

    def test_no_draws_returns_empty_null_without_consuming_randomness(self):
        rng = np.random.default_rng(0)
        state = rng.bit_generator.state
        mean, null = mi.mi_permutation_null([('a', 'x'), ('b', 'y')], 0, rng)
        self.assertEqual(mean, 0)
        self.assertEqual(null.size, 0)
        self.assertEqual(rng.bit_generator.state, state)

    def test_pvalues_keep_absolute_tie_tolerance_and_plus_one_correction(self):
        sources = ['Human', 'Model A', 'Model B', 'Model C']
        counts = dict(primary_sources=sources, trials={'a': {'ids': ['x']}, 'b': {'ids': ['y']}},
                      comparisons=[dict(module='pair', block=f'block{i}', arm=arm,
                                        family='family', preceding='a', perturbed='b',
                                        joint_counts={s: [dict(a='x', b='y', n=20)] for s in sources})
                                   for i in range(10) for arm in ('related', 'unrelated_control')])
        # Three draws exceed zero within the absolute 1e-15 tolerance.
        null = np.array([-2e-15, -0.5e-15, 0, 1e-15])
        with patch.object(mi, 'mi_permutation_null', return_value=(float(null.mean()), null)):
            records = mi.compute_tests(counts, draws=len(null))
        for record in records:
            self.assertEqual(record['null_exceedances'], 3)
            self.assertEqual(record['permutation_p'], 4 / 5)


if __name__ == '__main__':
    unittest.main()
