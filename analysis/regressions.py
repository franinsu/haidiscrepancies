#!/usr/bin/env python3
"""Rebuild only the 35 primary regressions in the current paper, offline.

Inputs are canonical collection puzzles plus retained/scored main-study rows.
No experimental snapshot, frozen estimate, network request or paper write is
needed. Checkpoints belong to one exact input/code/environment manifest.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

# Parallelism is across independent fits. Avoid an additional BLAS thread pool
# by default; explicit user settings are honored and included in the manifest.
THREAD_VARIABLES = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
for variable in THREAD_VARIABLES:
    os.environ.setdefault(variable, '1')

import numpy as np
from . import DEFAULT_L2_PENALTY
import scipy

ROOT = Path(__file__).resolve().parents[1]
WRANGLING = Path(__file__).resolve().parent

from . import regression_core as core
from . import grid_features as grid
from . import additional_features as additional
from . import maze_features
from .regression_optimizer import (finalize_regression_metadata, penalty_metadata,
                                   validate_l2_penalty)

FAMILIES = {
    'arithmetic': ('arithmetic24', 'Arithmetic', 20260827),
    'maze': ('maze', 'Maze', 20260913),
    'grid': ('grid_placement', 'Rooks', 20260826),
    'minesweeper': ('minesweeper_lite', 'Minesweeper', 20260827),
    'sudoku': ('mini_sudoku', 'Sudoku', 20260827),
}
SOURCES = {'Human': 'Human', 'GPT': 'GPT-5.6 Sol|low|plain',
           'Claude': 'Claude Opus 4.8|low|plain', 'Gemini': 'Gemini 3.5 Flash|low|plain'}
CODE_NAMES = ('__init__.py', 'regression_core.py', 'grid_features.py', 'additional_features.py',
              'maze_features.py', 'constant_feature_means.py', 'regression_optimizer.py', 'regression_bootstrap.py')


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


def numeric_build_config():
    """Freeze JSON-safe BLAS/LAPACK/compiler metadata, not only version labels."""
    return json.loads(encoded({'numpy': np.__config__.CONFIG, 'scipy': scipy.__config__.CONFIG}))


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def manifest(collection, responses, draws, l2_penalty=DEFAULT_L2_PENALTY):
    l2_penalty = validate_l2_penalty(l2_penalty)
    inputs = {'stimuli/all_puzzles.jsonl': digest(collection/'stimuli/all_puzzles.jsonl')}
    for name in ['human/main_retained.jsonl', *['ai/'+f for f in core.MODEL_FILES.values()]]:
        inputs['processed/'+name] = digest(responses/name)
    code = {'analysis/regressions.py': digest(Path(__file__))}
    code.update({'analysis/'+name: digest(WRANGLING/name) for name in CODE_NAMES})
    # Arithmetic's canonical class map imports common/probes helpers. Hash the
    # small complete collection puzzle package to cover those transitive imports.
    code.update({'puzzles/'+p.name: digest(p) for p in sorted((ROOT/'puzzles').glob('*.py'))})
    result = {'schema_version': 1, 'input_sha256': inputs, 'code_sha256': code,
            'parameters': {'draws': draws, 'seeds': {s:v[2] for s,v in FAMILIES.items()},
                           'condition': 'low effort, plain prompt', 'fits_per_family': 7},
            'environment': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                            'platform': platform.platform(),
                            'numeric_builds': numeric_build_config(),
                            'numeric_threads': {k:os.environ.get(k) for k in THREAD_VARIABLES}}}
    if l2_penalty:
        result['parameters']['l2_penalty'] = l2_penalty
    return result


def prepare_output(output, current_manifest):
    output.mkdir(parents=True, exist_ok=True)
    path = output/'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != current_manifest:
            raise ValueError('Output manifest differs in inputs, code, settings or environment; use a new output directory.')
    else:
        if any(output.iterdir()):
            raise ValueError('Nonempty output directory has no manifest; choose an empty directory.')
        save(path, current_manifest)
    return hashlib.sha256(encoded(current_manifest)).hexdigest()


def load_family(stem, collection, responses):
    family, display, seed = FAMILIES[stem]
    if stem == 'maze':
        names = maze_features.FEATURE_NAMES
        ids, features, diagnostics = maze_features.load_maze_features(collection)
    elif stem == 'grid':
        names = grid.FEATURE_NAMES
        ids, features, diagnostics = grid.load_grid_features(collection)
    else:
        names = additional.FAMILY_SPECS[family]['features']
        ids, features, diagnostics = additional.load_family_features(collection, family)
    observations, metadata = core.load_observations(responses, ids, features)
    # Maze's selected frozen payload uses the four display names. Preserve that
    # public schema while reading exactly the same retained/scored source rows.
    keys = {label:label if stem == 'maze' else source for label, source in SOURCES.items()}
    observed = {keys[label]:observations[source] for label, source in SOURCES.items()}
    meta = {keys[label]:metadata[source] for label, source in SOURCES.items()}
    header = {'family': family, 'family_display': display, 'feature_names': list(names),
              'puzzle_ids': ids, 'puzzle_count': len(ids), 'source_metadata': meta,
              'feature_diagnostics': diagnostics, 'seed': seed,
              'primary_condition': {'effort':'low', 'prompt':'plain'}}
    return ids, features, observed, header, keys


def initialize(ids, features, observations, names):
    core.FEATURE_NAMES = tuple(names)
    core.initialize_fit_worker(ids, features, observations)


def tasks_for(stem, keys, draws, l2_penalty=DEFAULT_L2_PENALTY):
    l2_penalty = validate_l2_penalty(l2_penalty)
    tasks = [{'family':stem, 'kind':kind, 'source':source, 'label':label,
             'draws':draws, 'seed':FAMILIES[stem][2]}
            for kind in ['choice', 'source'] for label, source in keys.items()
            if kind == 'choice' or label != 'Human']
    if l2_penalty:
        for task in tasks:
            task['l2_penalty'] = l2_penalty
    return tasks


def fit_task(task):
    _, fit = core.execute_fit_task((task['kind'], task['label'], (task['source'],), task['seed'], task['draws']),
                                  l2_penalty=task.get('l2_penalty', 0.0))
    return fit


def validate_fit(fit, task, names):
    if fit['bootstrap_validation_repeats'] != task['draws'] or fit['bootstrap_rejected_fits'] != 0:
        raise ValueError('Incomplete bootstrap checkpoint')
    if (not fit['bootstrap_shared_draws_for_coefficients_and_validation']
            or fit['optimizer']['penalty'] != penalty_metadata(task.get('l2_penalty', 0.0))):
        raise ValueError('Checkpoint does not implement the requested regression specification')
    if set(fit['feature_coefficients']) != set(names):
        raise ValueError('Checkpoint feature inventory differs')
    coefficients = np.asarray(fit['bootstrap_coefficients'])
    if coefficients.shape != (task['draws'], len(names)) or not np.isfinite(coefficients).all():
        raise ValueError('Incomplete or nonfinite bootstrap coefficients')
    if any(len(values) != task['draws'] or not np.isfinite(values).all()
           for values in fit['bootstrap_validation'].values()):
        raise ValueError('Incomplete or nonfinite bootstrap validation')
    encoded(fit)  # Reject nonfinite scalars anywhere in the record as well.


def read_checkpoint(path, fingerprint, task, names):
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get('run_fingerprint') != fingerprint or record.get('task') != task:
        raise ValueError(f'Checkpoint identity mismatch: {path}')
    fit = record['fit']
    if record.get('fit_sha256') != hashlib.sha256(encoded(fit)).hexdigest():
        raise ValueError(f'Checkpoint checksum mismatch: {path}')
    validate_fit(fit, task, names)
    return fit


def output_name(stem):
    return ('maze_three' if stem == 'maze' else stem)+'_feature_models.json'


def run(collection, responses, output_dir, draws=2000, workers=1, statistics_dir=None, *, l2_penalty=DEFAULT_L2_PENALTY):
    from types import SimpleNamespace
    l2_penalty = validate_l2_penalty(l2_penalty)
    args = SimpleNamespace(collection=Path(collection), responses=Path(responses), output_dir=Path(output_dir), draws=draws, workers=workers)

    if draws < 2 or workers < 1:
        raise ValueError('draws must be at least 2 and workers at least 1')
    args.collection, args.responses, args.output_dir = (p.resolve() for p in [args.collection, args.responses, args.output_dir])
    current_manifest = manifest(args.collection, args.responses, args.draws, l2_penalty)
    fingerprint = prepare_output(args.output_dir, current_manifest)
    start = time.monotonic()
    completed = reused = 0
    for stem in FAMILIES:
        ids, features, observations, data, keys = load_family(stem, args.collection, args.responses)
        names = data['feature_names']
        results = {'choice':{}, 'source':{}}
        pending = []
        for task in tasks_for(stem, keys, args.draws, l2_penalty):
            path = args.output_dir/'checkpoints'/stem/f"{task['kind']}-{task['label']}.json"
            fit = read_checkpoint(path, fingerprint, task, names)
            if fit is None:
                pending.append((task, path))
            else:
                results[task['kind']][task['source']] = fit
                reused += 1
        def retain(task, path, fit):
            nonlocal completed
            validate_fit(fit, task, names)
            save(path, {'run_fingerprint':fingerprint, 'task':task, 'fit':fit,
                        'fit_sha256':hashlib.sha256(encoded(fit)).hexdigest()})
            results[task['kind']][task['source']] = fit
            completed += 1
            print(f"{stem}: {task['kind']} {task['label']} saved", flush=True)
        if pending and args.workers == 1:
            initialize(ids, features, observations, names)
            for task, path in pending:
                retain(task, path, fit_task(task))
        elif pending:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                     initargs=(ids, features, observations, names)) as pool:
                jobs = {pool.submit(fit_task, task):(task,path) for task,path in pending}
                for future in as_completed(jobs):
                    task,path = jobs[future]
                    retain(task, path, future.result())
        data.update(bootstrap_repeats=args.draws, run_fingerprint=fingerprint,
                    constant_feature_replacement='training_mean')
        if stem == 'maze':
            data['fits'] = results
        else:
            data['selected_solution_prediction'] = {'conditional':{'fits':results['choice']}}
            data['source_prediction'] = {'binary_vs_human':{'conditional':{'fits':results['source']}}}
        data = finalize_regression_metadata(data, l2_penalty)
        save((Path(statistics_dir) if statistics_dir is not None else args.output_dir)/output_name(stem), data)
        print(f'{stem}: all 7 fits complete', flush=True)
    save(args.output_dir/'run.json', {'run_fingerprint':fingerprint, 'fits':35, 'computed':completed,
         'reused':reused, 'draws':args.draws, 'workers':args.workers,
         'elapsed_seconds':time.monotonic()-start, 'complete':True})
    print(f'Complete: 35 fits ({completed} computed, {reused} reused).', flush=True)
