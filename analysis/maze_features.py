"""The three Maze features used in the paper, from retained candidate paths."""
from collections import Counter
import json
from pathlib import Path
import numpy as np

FEATURE_NAMES = ('turn_count', 'background_edge_load', 'mean_centered_goal_distance')


def canonical_edge(a, b):
    # All candidates are shortest paths with a common start and goal. Each edge
    # therefore has the same direction on every candidate that contains it.
    return tuple(sorted((tuple(a), tuple(b))))


def compute_features(puzzle):
    paths = np.asarray([np.asarray(r['coordinates'])-1 for r in puzzle['routes']], dtype=float)
    assert np.allclose(paths[:, 0], paths[0, 0]) and np.allclose(paths[:, -1], paths[0, -1])
    assert np.all(np.sum(abs(np.diff(paths, axis=1)), axis=2) == 1)
    distances = np.linalg.norm(paths-paths[0, -1], axis=2)
    centered = distances-distances.mean(axis=0)
    assert np.allclose(centered.mean(axis=0), 0)
    edges = [[canonical_edge(a, b) for a, b in zip(path, path[1:])] for path in paths]
    counts = Counter(e for route in edges for e in route)
    raw = []
    for j, route in enumerate(puzzle['routes']):
        turns = sum(a != b for a, b in zip(route['moves'], route['moves'][1:]))
        overlap = np.mean([(counts[e]-1)/(len(paths)-1) for e in edges[j]])
        raw.append([turns, overlap, centered[j].mean()])
    raw = np.asarray(raw)
    low, high = raw.min(axis=0), raw.max(axis=0)
    normalized = np.divide(raw-low, high-low, out=np.zeros_like(raw), where=(high-low) > 1e-12)
    return {'puzzle_id': puzzle['id'], 'classes': puzzle['classes'], 'feature_names': list(FEATURE_NAMES),
            'raw': raw.tolist(), 'normalized': normalized.tolist(), 'ranges': np.stack([low, high], axis=1).tolist(),
            'goal_distance_by_step': distances.tolist(), 'centered_goal_distance_by_step': centered.tolist()}


def load_maze_features(collection: Path):
    """Read the canonical main mazes; no experiment snapshots are required."""
    rows = []
    for line in (collection/'stimuli/all_puzzles.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        puzzle = json.loads(line)
        if puzzle['puzzle_type'] != 'maze':
            continue
        solutions = puzzle['solutions']
        rows.append(compute_features({'id': puzzle['id'],
            'classes': [s['abstract_solution_id'] for s in solutions],
            'routes': solutions}))
    rows.sort(key=lambda r: r['puzzle_id'])
    ids = [r['puzzle_id'] for r in rows]
    if len(ids) != 20 or len(set(ids)) != 20:
        raise ValueError('Expected 20 distinct main Maze puzzles')
    features = {r['puzzle_id']: dict(zip(r['classes'], map(tuple, r['normalized']))) for r in rows}
    return ids, features, {'candidate_count': sum(len(r['classes']) for r in rows),
                           'raw_features': rows}
