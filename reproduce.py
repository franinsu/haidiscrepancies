#!/usr/bin/env python3
"""Run the study's explicit processing, analysis and presentation stages."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from analysis import DEFAULT_L2_PENALTY

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', MPLBACKEND='Agg',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', SOURCE_DATE_EPOCH='1790121600',
               MPLCONFIGDIR=str(ROOT / '.cache/matplotlib'), XDG_CACHE_HOME=str(ROOT / '.cache'))
    return env


def invoke(relative: str, arguments: list[str | Path]) -> None:
    subprocess.run([sys.executable, '-B', str(ROOT / relative), *map(str, arguments)],
                   cwd=ROOT, env=environment(), check=True)


def analysis_arguments(args, output: Path, processed: Path | None = None) -> list:
    result = ['--data-dir', args.data_dir, '--output-dir', output, '--mode', args.mode,
              '--bootstrap', args.bootstrap, '--regression-draws', args.regression_draws,
              '--permutations', args.permutations, '--mi-permutations', args.mi_permutations,
              '--workers', args.workers, '--l2-penalty', args.l2_penalty]
    if processed is not None:
        result += ['--processed-dir', processed]
    return result


def pipeline(args) -> None:
    """Fresh recorded observations → statistics → public figures and tables."""
    output = args.output_dir.resolve()
    intermediate = args.intermediate_dir.resolve()
    data = args.data_dir.resolve()
    for destination in (intermediate, output):
        if destination.is_relative_to(data) or data.is_relative_to(destination):
            raise ValueError('Full-run destinations must be separate from the input data directory')
    if output.is_relative_to(intermediate) or intermediate.is_relative_to(output):
        raise ValueError('Intermediate and final output directories must be separate')
    for destination in (intermediate, output):
        if destination.exists():
            raise FileExistsError(f'Choose a fresh full-run directory: {destination}')
    intermediate.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    sources = [p for group in ['analysis', 'processing', 'puzzles', 'figures']
               for p in (ROOT / group).rglob('*.py')]
    sources += list((ROOT / 'figures').rglob('*.json'))
    sources += [ROOT / 'reproduce.py', ROOT / 'requirements.lock', ROOT / 'configs/study.json']
    report = {'status': 'running', 'mode': 'fresh_raw_to_figures',
              'reference_paper_commit': json.loads((ROOT / 'configs/study.json').read_text())['reference_paper_commit'],
              'python': sys.version, 'platform': platform.platform(),
              'data_dir': str(args.data_dir.resolve()),
              'intermediate_dir': str(intermediate), 'output_dir': str(output),
              'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources},
              'settings': {k: getattr(args, k) for k in ['bootstrap', 'regression_draws', 'permutations', 'mi_permutations', 'workers', 'l2_penalty']},
              'completed_stages': []}
    manifest = intermediate / 'run.json'
    started = time.monotonic()
    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    save()
    try:
        print('Stage 1/3: reconstructing scored and retained observations', flush=True)
        invoke('processing/run.py', ['--data-dir', args.data_dir, '--output-dir', intermediate / 'processed', '--stage', 'all'])
        report['completed_stages'].append('processing'); save()
        print('Stage 2/3: calculating scientific results', flush=True)
        invoke('analysis/run.py', analysis_arguments(args, intermediate / 'statistics', intermediate / 'processed'))
        report['completed_stages'].append('analysis'); save()
        print('Stage 3/3: rendering figures, tables and display summaries', flush=True)
        invoke('figures/run.py', ['--input-dir', intermediate / 'statistics', '--output-dir', output, '--data-dir', args.data_dir])
        report['completed_stages'].append('figures')
        if any(sha(p) != report['source_sha256'][str(p.relative_to(ROOT))] for p in sources):
            raise ValueError('Scientific code or layout inputs changed during the run')
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'; report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        save()


def numerical_options(parser):
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--regression-draws', type=int, default=2000)
    parser.add_argument('--l2-penalty', type=float, default=DEFAULT_L2_PENALTY,
                        help='L2 strength for regression slopes, excluding the intercept (default: %(default)g; use 0 for unpenalized fits)')
    parser.add_argument('--permutations', type=int, default=10000)
    parser.add_argument('--mi-permutations', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=1)


def main():
    # Detailed outputs may contain participant records; public export is explicit.
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    anonymize = commands.add_parser('anonymize', help='Create a pseudonymized human replay archive and a separate private ID map')
    anonymize.add_argument('--input-dir', type=Path, required=True, help='Original archive containing main/ and module/')
    anonymize.add_argument('--output-dir', type=Path, required=True, help='Fresh export directory, separate from the input and ID map')
    anonymize.add_argument('--id-map', type=Path, default=ROOT / 'data/private/id_map.csv')
    demo = commands.add_parser('demo', help='Invented responses; no access to collected human/model records')
    demo.add_argument('--output-dir', type=Path, default=ROOT / 'checks/runs/demo')
    process = commands.add_parser('process', help='Fresh v4 model scoring and/or human retention')
    process.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    process.add_argument('--output-dir', type=Path, required=True)
    process.add_argument('--stage', choices=['human', 'models', 'all'], default='all')
    process.add_argument('--limit', type=int)
    analyze = commands.add_parser('analyze', help='Calculate numerical results from scored/retained observations')
    analyze.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    analyze.add_argument('--processed-dir', type=Path, required=True)
    analyze.add_argument('--output-dir', type=Path, required=True)
    analyze.add_argument('--mode', choices=['summaries', 'regressions', 'mi', 'all'], default='all')
    numerical_options(analyze)
    render = commands.add_parser('render', help='Render explicit saved numerical inputs; does not recompute analyses')
    render.add_argument('--input-dir', type=Path, default=ROOT / 'intermediate/statistics')
    render.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    render.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    render.add_argument('--reference', action='store_true', help='Explicitly render archived numerical references')
    render.add_argument('--only', nargs='+', help='Selected figure/table names from figures/manifest.json')
    render.add_argument('--previews', action='store_true', help='Also render PNG previews of PDF figures')
    full = commands.add_parser('full', help='Complete fresh processing, analysis and rendering; substantial computation')
    full.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    full.add_argument('--intermediate-dir', type=Path, default=ROOT / 'intermediate', help='Fresh directory for processed data, statistics and the run record')
    full.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    full.set_defaults(mode='all')
    numerical_options(full)
    args = parser.parse_args()
    if hasattr(args, 'l2_penalty') and (not math.isfinite(args.l2_penalty) or args.l2_penalty < 0):
        parser.error('--l2-penalty must be finite and nonnegative')
    # Interpret user paths at the invocation directory, before subprocesses
    # switch to the repository root.
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.command == 'anonymize':
        invoke('processing/anonymize_humans.py', ['--input-dir', args.input_dir,
               '--output-dir', args.output_dir, '--id-map', args.id_map])
    elif args.command == 'demo':
        os.environ.update(environment())
        from demo import run
        summary = run(args.output_dir)
        print(json.dumps({'synthetic': True, 'output': str(args.output_dir), 'total_variation': summary['total_variation']}, indent=2))
    elif args.command == 'process':
        options = ['--data-dir', args.data_dir, '--output-dir', args.output_dir, '--stage', args.stage]
        if args.limit is not None: options += ['--limit', args.limit]
        invoke('processing/run.py', options)
    elif args.command == 'analyze':
        invoke('analysis/run.py', analysis_arguments(args, args.output_dir, args.processed_dir))
    elif args.command == 'render':
        options = ['--input-dir', args.input_dir, '--output-dir', args.output_dir, '--data-dir', args.data_dir]
        if args.reference: options += ['--reference']
        if args.only: options += ['--only', *args.only]
        if args.previews: options += ['--previews']
        invoke('figures/run.py', options)
    elif args.command == 'full':
        pipeline(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
