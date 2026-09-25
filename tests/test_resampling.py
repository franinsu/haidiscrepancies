"""Preserve strata, whole blocks and family weights during resampling."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import entropy, perturbation_effects, resampling, stimulus_modules


class ResamplingTests(unittest.TestCase):
    def test_seeded_draws_preserve_unequal_strata(self):
        draws = resampling.stratified_indices((2, 3), repeats=4, seed=0)
        np.testing.assert_array_equal(draws, [
            [1, 1, 2, 3, 4],
            [1, 1, 3, 3, 3],
            [0, 0, 3, 2, 2],
            [1, 0, 3, 2, 4],
        ])
        self.assertTrue(np.all(draws[:, :2] < 2))
        self.assertTrue(np.all(draws[:, 2:] >= 2))

    def test_whole_clusters_keep_paired_arms_and_sorted_strata(self):
        first = [("a_original", "a_perturbed"), ("b_original", "b_perturbed")]
        second = [("c",), ("d",), ("e",)]
        strata = {("module", "z", "family"): second,
                  ("module", "a", "family"): first}
        draws = resampling.cluster_draws(strata, repeats=40, seed=17)
        reordered = dict(reversed(list(strata.items())))
        self.assertEqual(draws, resampling.cluster_draws(reordered, repeats=40, seed=17))
        for draw in draws:
            self.assertEqual(len(draw), 5)
            self.assertTrue(all(cluster in first for cluster in draw[:2]))
            self.assertTrue(all(cluster in second for cluster in draw[2:]))

    def test_public_wrappers_resolve_current_repeat_settings(self):
        with patch.object(stimulus_modules, "BOOTSTRAP_REPEATS", 7):
            self.assertEqual(stimulus_modules.bootstrap_draws((2, 3)).shape, (7, 5))
        with patch.object(entropy, "BOOTSTRAP_REPEATS", 7):
            draws = entropy.cluster_bootstrap_draws({("main", "main", "family"): [("a",), ("b",)]})
            self.assertEqual(len(draws), 7)

    def test_unequal_family_sizes_keep_equal_family_weight(self):
        with patch.object(perturbation_effects, "NBOOT", 30):
            interval = perturbation_effects.bootstrap([np.zeros(1), np.ones(4)])
        self.assertEqual(interval, [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
