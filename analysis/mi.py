"""Numerical mi for the study.

Extracted from the reviewed study implementation; presentation is separate.
"""

from __future__ import annotations

from collections import Counter


import hashlib
import math


import json


import numpy as np
from scipy.stats import permutation_test


from .metrics import mutual_information


def _mi_from_indices(a_index: np.ndarray, b_index: np.ndarray, k_a: int, k_b: int) -> float:
    total = len(a_index)
    joint = np.bincount(a_index * k_b + b_index, minlength=k_a * k_b).reshape(k_a, k_b) / total
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(joint > 0, joint * np.log2(joint / (pa * pb)), 0.0)
    return float(terms.sum())


def mi_permutation_null(pairs: list[tuple[str, str]], draws: int, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Plug-in MI under exchangeability of the B answers given the A answers."""
    a_labels = sorted({a for a, _ in pairs})
    b_labels = sorted({b for _, b in pairs})
    a_index = np.array([a_labels.index(a) for a, _ in pairs])
    b_index = np.array([b_labels.index(b) for _, b in pairs])
    observed = _mi_from_indices(a_index, b_index, len(a_labels), len(b_labels))
    reference = mutual_information(pairs)
    if not math.isclose(observed, reference, abs_tol=1e-9):
        raise ValueError("vectorized mutual information disagrees with the reference implementation")
    if draws <= 0:
        return 0.0, np.zeros(0)
    def statistic(permuted_b):
        return _mi_from_indices(a_index, permuted_b, len(a_labels), len(b_labels))

    if len(pairs) < 2 or draws >= math.factorial(len(pairs)):
        # Keep the requested Monte Carlo draws; SciPy otherwise enumerates
        # small samples exactly (and does not accept a singleton sample).
        null = np.array([statistic(rng.permutation(b_index)) for _ in range(draws)])
    else:
        null = permutation_test(
            (b_index,), statistic, permutation_type="pairings",
            alternative="greater", n_resamples=draws, vectorized=False, rng=rng,
        ).null_distribution
    return float(null.mean()), null


NAMESPACE = 'reasoning_discrepancy.context_mi.canonical_pairs.v1'


ARM_COMPONENT = {'related': 'mi_related', 'unrelated_control': 'mi_control'}


def encoded(value):
    """The exact serialization used for seed material and ordered-pair hashes."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('utf-8')


def stable_seed(source, block, arm, base_seed=0):
    material = dict(namespace=NAMESPACE, base_seed=base_seed,
                    source=source, block=block, arm=arm)
    checksum = hashlib.sha256(encoded(material)).hexdigest()
    return int(checksum, 16), material, checksum


def canonical_table(counts, comparison, source):
    """Expand aggregate cells in sorted (A class, B class) order, not row order."""
    a_classes = sorted(counts['trials'][comparison['preceding']]['ids'])
    b_classes = sorted(counts['trials'][comparison['perturbed']]['ids'])
    if len(a_classes) != len(set(a_classes)) or len(b_classes) != len(set(b_classes)):
        raise ValueError('Duplicate class IDs in a trial')
    joint = Counter()
    for cell in comparison['joint_counts'][source]:
        if cell['a'] not in a_classes or cell['b'] not in b_classes:
            raise ValueError('Pair cell contains an unknown solution class')
        n = cell['n']
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError('Pair counts must be nonnegative integers')
        joint[(cell['a'], cell['b'])] += n
    pairs = [pair for pair, n in sorted(joint.items()) for _ in range(n)]
    if not pairs:
        raise ValueError('Complete-pair table is empty')
    table = [[joint[(a, b)] for b in b_classes] for a in a_classes]
    return a_classes, b_classes, table, pairs


def compute_tests(counts, draws=2000, base_seed=0):
    if draws < 1 or base_seed < 0:
        raise ValueError('Draw count must be positive and base seed nonnegative')
    sources = counts['primary_sources']
    if len(sources) != 4 or len(set(sources)) != 4 or 'Human' not in sources:
        raise ValueError('Expected four distinct primary sources including Human')
    comparisons = sorted((c for c in counts['comparisons'] if c['module'] == 'pair'),
                         key=lambda c: (c['block'], c['arm']))
    identities = [(c['block'], c['arm']) for c in comparisons]
    blocks = {c['block'] for c in comparisons}
    if len(blocks) != 10 or len(comparisons) != 20 or len(set(identities)) != 20:
        raise ValueError('Expected ten context blocks with two comparisons each')
    if any({c['arm'] for c in comparisons if c['block'] == block} != set(ARM_COMPONENT)
           for block in blocks):
        raise ValueError('Every block must have related and unrelated context')
    records = []
    for source in sources:
        for comparison in comparisons:
            a_classes, b_classes, table, pairs = canonical_table(counts, comparison, source)
            seed, material, seed_hash = stable_seed(source, comparison['block'], comparison['arm'], base_seed)
            # PCG64 is explicit; default_rng and Python's salted hash are not used.
            rng = np.random.Generator(np.random.PCG64(seed))
            observed = mutual_information(pairs)
            null_mean, null = mi_permutation_null(pairs, draws, rng)
            # Preserve the study's absolute tie tolerance and Monte Carlo +1
            # correction rather than SciPy's relative-tolerance p-value.
            exceedances = int(np.count_nonzero(null >= observed - 1e-15))
            records.append(dict(source=source, block=comparison['block'], family=comparison['family'],
                arm=comparison['arm'], preceding_trial=comparison['preceding'],
                target_trial=comparison['perturbed'], a_classes=a_classes, b_classes=b_classes,
                joint_counts=table, n_pairs=len(pairs), mi_bits=observed,
                null_mean_bits=null_mean, null_exceedances=exceedances,
                permutation_p=(1+exceedances)/(1+draws), permutations=draws,
                ordered_aggregate_pairs_sha256=hashlib.sha256(encoded(pairs)).hexdigest(),
                null_draws_float64_little_endian_sha256=hashlib.sha256(null.astype('<f8').tobytes()).hexdigest(),
                seed_integer=seed, seed_material=material, seed_sha256=seed_hash))
    return records
