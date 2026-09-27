"""Numerical geometry for the study.

Extracted from the reviewed study implementation; presentation is separate.
"""

from __future__ import annotations

import math


from typing import Any


import numpy as np


TV_SOURCE_ORDER = (
    "Uniform",
    "Human",
    "GPT 5.6 Sol",
    "Claude Opus 4.8",
    "Gemini 3.5 Flash",
)


FAMILY_ORDER = (
    "arithmetic24",
    "maze",
    "grid_placement",
    "minesweeper_lite",
    "mini_sudoku",
)


FAMILY_LABELS = {
    "arithmetic24": "Arithmetic",
    "maze": "Maze",
    "grid_placement": "Rooks",
    "minesweeper_lite": "Minesweeper",
    "mini_sudoku": "Sudoku",
}


KERNEL_RULE = {
    "distance_squared_dissimilarity": "Q = D * D (entrywise)",
    "similarity_squared_dissimilarity": "Q = 2 * (J - S)",
    "centered_kernel": "K = -0.5 * C @ Q @ C",
    "aggregation": (
        "per-puzzle metric, then family/equal-family mean, then Q and K"
    ),
    "source_display": (
        "leading positive-eigenvalue coordinates; zero-pad a missing second "
        "component only in common-metric panels"
    ),
    "family_alignment": (
        "CKA of full PSD-projected kernels; clip negative eigenvalues to zero"
    ),
}


def centering_matrix(size: int) -> np.ndarray:
    return np.eye(size, dtype=float) - np.ones((size, size), dtype=float) / size


def validate_distance_matrix(matrix: np.ndarray, context: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{context} must be square, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{context} contains non-finite entries")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise ValueError(f"{context} is not symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"{context} has a nonzero diagonal")
    if np.min(matrix) < -1e-12 or np.max(matrix) > 1.0 + 1e-12:
        raise ValueError(f"{context} contains values outside [0, 1]")


def eigencoordinates(
    kernel: np.ndarray,
    dimensions: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
    positive = eigenvalues > tolerance
    if int(np.sum(positive)) < dimensions:
        raise ValueError(
            f"Embedding needs {dimensions} positive eigenvalues; found "
            f"{int(np.sum(positive))}"
        )
    selected = np.flatnonzero(positive)[:dimensions]
    coordinates = eigenvectors[:, selected] * np.sqrt(eigenvalues[selected])
    shares = eigenvalues[selected] / float(np.sum(eigenvalues[positive]))
    return coordinates, eigenvalues, shares


def orient_source_coordinates(
    coordinates: np.ndarray,
    source_order: tuple[str, ...],
) -> np.ndarray:
    """Give otherwise arbitrary eigenvector signs a stable interpretation."""
    oriented = coordinates.copy()
    indices = {source: source_order.index(source) for source in source_order}
    baseline_sources = [indices["Human"]]
    if "Uniform" in indices:
        baseline_sources.insert(0, indices["Uniform"])
    baseline = np.mean(oriented[baseline_sources, 0])
    model_mean = np.mean(
        oriented[
            [
                indices["GPT 5.6 Sol"],
                indices["Claude Opus 4.8"],
                indices["Gemini 3.5 Flash"],
            ],
            0,
        ]
    )
    if model_mean < baseline:
        oriented[:, 0] *= -1.0

    if oriented.shape[1] > 1:
        chatgpt = oriented[indices["GPT 5.6 Sol"], 1]
        other_models = np.mean(
            oriented[
                [indices["Claude Opus 4.8"], indices["Gemini 3.5 Flash"]],
                1,
            ]
        )
        if chatgpt < other_models:
            oriented[:, 1] *= -1.0
    return oriented


def orient_coordinates_by_largest_entry(coordinates: np.ndarray) -> np.ndarray:
    """Resolve each arbitrary eigenvector sign by its largest absolute entry."""
    oriented = coordinates.copy()
    for component in range(oriented.shape[1]):
        anchor = int(np.argmax(np.abs(oriented[:, component])))
        if oriented[anchor, component] < 0.0:
            oriented[:, component] *= -1.0
    return oriented


def stress_one(target: np.ndarray, coordinates: np.ndarray) -> float:
    """Target-normalized distance distortion, not Kruskal's Stress-1.

    Retain the function name and ``stress_1`` cache key for compatibility with
    frozen results. The denominator is the squared norm of target distances,
    rather than the squared norm of distances in the embedded configuration.
    """
    embedded = np.sqrt(
        np.sum(
            (coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2,
            axis=2,
        )
    )
    upper = np.triu_indices(target.shape[0], k=1)
    denominator = float(np.sum(target[upper] ** 2))
    if denominator <= 0.0:
        raise ValueError("Stress is undefined for a zero distance matrix")
    squared_error = float(np.sum((target[upper] - embedded[upper]) ** 2))
    return math.sqrt(squared_error / denominator)


def mean_tv_embedding(mean_tv: np.ndarray) -> dict[str, Any]:
    geometry = centered_metric_kernel(mean_tv, kind="distance")
    coordinates, eigenvalues, shares = eigencoordinates(geometry["kernel"])
    coordinates = orient_source_coordinates(coordinates, TV_SOURCE_ORDER)
    positive_mass = float(np.sum(eigenvalues[eigenvalues > 1e-12]))
    negative_mass = float(-np.sum(eigenvalues[eigenvalues < -1e-12]))
    return {
        "method": "classical MDS of the full-precision mean TV distance matrix",
        **geometry,
        "coordinates": coordinates,
        "eigenvalues": eigenvalues,
        "shares": shares,
        "positive_eigenvalue_mass": positive_mass,
        "negative_eigenvalue_mass": negative_mass,
        "stress_1": stress_one(geometry["target_dissimilarity"], coordinates),
    }


def validate_similarity_matrix(
    matrix: np.ndarray,
    context: str,
    *,
    lower: float,
) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{context} must be square, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{context} contains non-finite entries")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise ValueError(f"{context} is not symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"{context} does not have a unit diagonal")
    if np.min(matrix) < lower - 1e-12 or np.max(matrix) > 1.0 + 1e-12:
        raise ValueError(f"{context} contains values outside [{lower}, 1]")


def centered_metric_kernel(
    matrix: np.ndarray, *, kind: str
) -> dict[str, np.ndarray]:
    """Convert an already averaged metric matrix into the shared geometry.

    Apply this only after computing the per-puzzle metrics and their family or
    equal-family mean: averaging squared distances instead would change the
    analysis.  Similarities need not be PSD; negative kernel eigenvalues remain
    available for diagnostics and are handled by the downstream display/CKA.
    """
    if kind == "distance":
        validate_distance_matrix(matrix, "Geometry distance matrix")
        squared_dissimilarity = matrix**2
    elif kind == "similarity":
        validate_similarity_matrix(
            matrix, "Geometry similarity matrix", lower=-1.0
        )
        squared_dissimilarity = 2.0 * (np.ones_like(matrix) - matrix)
    else:
        raise ValueError(f"Unsupported geometry metric kind {kind!r}")
    center = centering_matrix(matrix.shape[0])
    kernel = -0.5 * center @ squared_dissimilarity @ center
    return {
        "squared_dissimilarity": squared_dissimilarity,
        "target_dissimilarity": np.sqrt(np.maximum(0.0, squared_dissimilarity)),
        "kernel": 0.5 * (kernel + kernel.T),
    }


def positive_spectral_projection(
    kernel: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project a centered symmetric matrix onto the PSD cone.

    Distance-derived family Gram matrices, and especially the historical
    shared-mode indicator, need not be positive semidefinite after averaging.
    CKA is therefore computed from the nearest PSD matrix in Frobenius norm,
    obtained by clipping negative eigenvalues.  The discarded spectrum is
    retained in the JSON diagnostics rather than hidden.
    """
    symmetric = 0.5 * (kernel + kernel.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = 1e-10 * scale
    clipped = np.maximum(eigenvalues, 0.0)
    projected = (eigenvectors * clipped) @ eigenvectors.T
    projected = 0.5 * (projected + projected.T)
    positive_mass = float(np.sum(eigenvalues[eigenvalues > tolerance]))
    negative_mass = float(-np.sum(eigenvalues[eigenvalues < -tolerance]))
    return projected, {
        "projection": "negative eigenvalues clipped to zero",
        "projection_applied": bool(np.any(eigenvalues < -tolerance)),
        "eigenvalues_before_projection": eigenvalues,
        "positive_eigenvalue_mass_before_projection": positive_mass,
        "negative_eigenvalue_mass_discarded": negative_mass,
    }


def family_cka_from_kernels(
    source_kernels: list[np.ndarray],
    *,
    method: str,
    kernel_definition: str,
) -> dict[str, Any]:
    normalized_kernels: list[np.ndarray] = []
    for family_index, kernel in enumerate(source_kernels):
        kernel = 0.5 * (kernel + kernel.T)
        tolerance = 1e-10 * max(1.0, float(np.linalg.norm(kernel, ord=2)))
        if float(np.min(np.linalg.eigvalsh(kernel))) < -tolerance:
            raise ValueError(
                f"Family {FAMILY_ORDER[family_index]} has a non-PSD source kernel"
            )
        norm = float(np.linalg.norm(kernel, ord="fro"))
        if norm <= 1e-15:
            raise ValueError(
                f"Family {FAMILY_ORDER[family_index]} has a constant source geometry"
            )
        normalized_kernels.append(kernel / norm)

    similarity = np.asarray(
        [
            [float(np.sum(left * right)) for right in normalized_kernels]
            for left in normalized_kernels
        ],
        dtype=float,
    )
    similarity = 0.5 * (similarity + similarity.T)
    if not np.allclose(np.diag(similarity), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("CKA similarity does not have a unit diagonal")
    if np.min(similarity) < -1e-10 or np.max(similarity) > 1.0 + 1e-10:
        raise ValueError("CKA similarity lies outside [0, 1]")

    family_center = centering_matrix(len(source_kernels))
    centered_similarity = family_center @ similarity @ family_center
    coordinates, eigenvalues, shares = eigencoordinates(centered_similarity)
    coordinates = orient_coordinates_by_largest_entry(coordinates)
    cka_distance = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * similarity))
    positive_mass = float(np.sum(eigenvalues[eigenvalues > 1e-12]))
    negative_mass = float(-np.sum(eigenvalues[eigenvalues < -1e-12]))
    return {
        "method": method,
        "kernel_definition": kernel_definition,
        "source_kernels": np.stack(source_kernels, axis=0),
        "similarity": similarity,
        "coordinates": coordinates,
        "eigenvalues": eigenvalues,
        "shares": shares,
        "positive_eigenvalue_mass": positive_mass,
        "negative_eigenvalue_mass": negative_mass,
        "stress_1": stress_one(cka_distance, coordinates),
    }


def family_geometry_from_metric_matrices(
    family_matrices: np.ndarray,
    *,
    kind: str,
    method: str,
) -> dict[str, Any]:
    """Apply the shared kernel rule and full PSD projection before family CKA."""
    if (
        family_matrices.ndim != 3
        or family_matrices.shape[0] != len(FAMILY_ORDER)
        or family_matrices.shape[1] != family_matrices.shape[2]
    ):
        raise ValueError("Expected one square metric matrix per puzzle family")
    raw_kernels = [
        centered_metric_kernel(matrix, kind=kind)["kernel"]
        for matrix in family_matrices
    ]
    projected_kernels: list[np.ndarray] = []
    projection_diagnostics: dict[str, Any] = {}
    for family, raw_kernel in zip(FAMILY_ORDER, raw_kernels, strict=True):
        projected, diagnostic = positive_spectral_projection(raw_kernel)
        projected_kernels.append(projected)
        projection_diagnostics[FAMILY_LABELS[family]] = diagnostic
    result = family_cka_from_kernels(
        projected_kernels,
        method=method,
        kernel_definition=(
            "K = -1/2 C_s Q^h C_s; "
            + (
                "Q^h = D^h odot D^h"
                if kind == "distance"
                else "Q^h = 2 (J - S^h)"
            )
        ),
    )
    result["source_kernels_before_projection"] = np.stack(raw_kernels, axis=0)
    result["kernel_projection_note"] = (
        "Each family kernel is projected onto the PSD cone before CKA by "
        "clipping negative eigenvalues; all positive components are retained, "
        "and per-family discarded spectral mass is reported in "
        "kernel_projection_diagnostics."
    )
    result["kernel_projection_diagnostics"] = projection_diagnostics
    result["retained_positive_inertia_share"] = float(np.sum(result["shares"]))
    return result


def family_geometry_cka(family_tv: np.ndarray) -> dict[str, Any]:
    """Compare five-source TV geometries using the shared squared-distance rule."""
    return family_geometry_from_metric_matrices(
        family_tv,
        kind="distance",
        method=(
            "kernel CKA between family-specific five-source PSD-projected "
            "shared-rule TV kernels, followed by eigendecomposition of the "
            "family-centered CKA matrix"
        ),
    )


def serializable_array_dict(result: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value

    return {key: convert(value) for key, value in result.items()}


def serializable_result(stats, mean_tv, family_tv, mean_tv_result, tv_cka_result):
    """The source and family TV geometry used by Figures 2 and 6."""
    return {
        'input':{
            'statistics':'figure_statistics.json',
            'condition':stats['primary_condition'],
            'n_puzzles':int(stats['global_tv_primary']['n_puzzles']),
            'kernel_rule':dict(KERNEL_RULE),
            'tv_source_order':list(TV_SOURCE_ORDER),
            'family_order':[FAMILY_LABELS[f] for f in FAMILY_ORDER],
            'mean_tv_matrix':mean_tv.tolist(),
            'family_tv_matrices':{FAMILY_LABELS[f]:family_tv[i].tolist() for i,f in enumerate(FAMILY_ORDER)},
            'tv_source_sign_convention':'dimension 1 points from Uniform/Human toward model centroid; dimension 2 points from Claude/Gemini toward GPT',
            'family_sign_convention':'largest absolute entry on each component is positive',
        },
        'mean_tv_mds':serializable_array_dict(mean_tv_result),
        'tv_family_geometry_cka':serializable_array_dict(tv_cka_result),
    }
