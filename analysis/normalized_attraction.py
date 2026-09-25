"""Target-probability normalization and whole-block family summaries."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from . import stimulus_modules as sm

def normalized_attraction(baseline_mass: float, attraction: float | None) -> float | None:
    """Return A/(1-p), or None if p=1 or the numerator is unavailable.

    Negative finite values are retained and not clipped. The quantity has
    upper bound one, but no common finite lower bound across baseline masses.
    """
    if not math.isfinite(baseline_mass) or not 0.0 <= baseline_mass <= 1.0:
        raise ValueError("baseline target mass must be a finite probability")
    if baseline_mass == 1.0 or attraction is None or not math.isfinite(attraction):
        return None
    return float(attraction / (1.0 - baseline_mass))


def equal_family_summary(values: dict[str, float], strata: list[tuple[str, list[str]]],
                         *, repeats: int = None,
                         seed: int = sm.BOOTSTRAP_SEED) -> dict[str, Any]:
    """Equal eligible blocks within family, then equal family weight.

    The bootstrap resamples complete eligible blocks within each family. If a
    family has no eligible blocks its summary and the cross-family mean are
    undefined, rather than silently redistributing its weight to other families.
    """
    if repeats is None:
        repeats = sm.BOOTSTRAP_REPEATS
    eligible = [(family, [block for block in ids if block in values]) for family, ids in strata]
    by_family: dict[str, Any] = {}
    for family, ids in eligible:
        if not ids:
            by_family[family] = {"estimate": None, "ci95": None, "n_blocks": 0}
            continue
        vector = np.array([values[block] for block in ids], dtype=float)
        draws = sm.bootstrap_draws((len(ids),), repeats=repeats, seed=seed)
        boot = vector[draws].mean(axis=1)
        by_family[family] = {
            "estimate": float(vector.mean()),
            "ci95": [float(x) for x in np.quantile(boot, (0.025, 0.975))],
            "n_blocks": len(ids),
        }
    result = {"estimate": None, "ci95": None, "n_blocks": len(values), "by_family": by_family}
    if any(not ids for _, ids in eligible):
        result["undefined_reason"] = "one_or_more_families_have_no_eligible_blocks"
        return result
    sizes = tuple(len(ids) for _, ids in eligible)
    vector = np.array([values[block] for _, ids in eligible for block in ids], dtype=float)
    draws = sm.bootstrap_draws(sizes, repeats=repeats, seed=seed)
    cuts = np.cumsum((0, *sizes))
    family_boot = np.stack([vector[draws[:, cuts[i]:cuts[i + 1]]].mean(axis=1)
                            for i in range(len(sizes))])
    result["estimate"] = float(np.mean([item["estimate"] for item in by_family.values()]))
    result["ci95"] = [float(x) for x in np.quantile(family_boot.mean(axis=0), (0.025, 0.975))]
    return result
