"""Shared bootstrap draws that preserve the study's strata and random stream."""

import random

import numpy as np


def stratified_indices(
    stratum_sizes: tuple[int, ...], *, repeats: int, seed: int
) -> np.ndarray:
    """Sample each stratum at its original size, using flattened indices.

    Strata retain their supplied order. Draws use the established Python
    random stream, consuming one index at a time within each replicate.
    """
    rng = random.Random(seed)
    offsets = np.cumsum((0, *stratum_sizes[:-1]))
    return np.asarray(
        [[int(offset) + rng.randrange(size)
          for offset, size in zip(offsets, stratum_sizes)
          for _ in range(size)] for _ in range(repeats)],
        dtype=int,
    )


def cluster_draws(
    clusters_by_stratum: dict[tuple[str, str, str], list[tuple[str, ...]]],
    *, repeats: int, seed: int,
) -> list[list[tuple[str, ...]]]:
    """Resample whole clusters within sorted strata, retaining cluster order."""
    strata = [clusters_by_stratum[key] for key in sorted(clusters_by_stratum)]
    clusters = [cluster for stratum in strata for cluster in stratum]
    indices = stratified_indices(tuple(map(len, strata)), repeats=repeats, seed=seed)
    return [[clusters[index] for index in draw] for draw in indices]
