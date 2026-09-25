"""Training-only replacement of puzzle-constant, within-puzzle scaled features."""
import numpy as np


class ConstantFeatureMeans:
    """Use complete candidate sets, never response frequencies, to estimate means.

    Each nonconstant training puzzle has equal weight (including bootstrap
    multiplicity); its valid candidates have uniform weight. Constant cases
    do not enter the mean. Min/max scaling has already been done within puzzle.
    """

    def __init__(self, puzzle_ids, features):
        blocks = [np.asarray(list(features[p].values()), dtype=float)
                  for p in puzzle_ids]
        if not blocks or any(x.ndim != 2 or not len(x) or
                             not np.isfinite(x).all() for x in blocks):
            raise ValueError("Expected finite, nonempty candidate feature matrices")
        self.constant = np.asarray([np.ptp(x, axis=0) == 0 for x in blocks])
        self.puzzle_means = np.asarray([x.mean(axis=0) for x in blocks])

    def fit(self, multiplicity):
        m = np.asarray(multiplicity, dtype=float)
        if m.shape != (len(self.constant),) or not np.isfinite(m).all() or (m < 0).any():
            raise ValueError("Invalid training puzzle multiplicities")
        weights = m[:, None] * ~self.constant
        denominator = weights.sum(axis=0)
        # No feature variation in training: use the scale midpoint. The initially zero
        # slope has zero gradient; a missing replacement mean never drops a fold.
        return np.divide((weights * self.puzzle_means).sum(axis=0), denominator,
                         out=np.full(denominator.shape, 0.5), where=denominator > 0)

    def transform(self, matrix, groups, multiplicity):
        mean = self.fit(multiplicity)
        return np.where(self.constant[np.asarray(groups)], mean, matrix)

    def transform_candidates(self, tensor, multiplicity):
        mean = self.fit(multiplicity)
        return np.where(self.constant[:, None, :], mean, tensor)
