#!/usr/bin/env python3
"""Reconstruct human and model analysis rows into a fresh output directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from processing import human, response_parser, score_responses as scoring


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def model_inputs(data_dir):
    for provider in ('openai', 'anthropic', 'gemini'):
        for cohort in ('main', 'module'):
            stimuli = data_dir / 'stimuli'
            data = stimuli / ('all_puzzles.jsonl' if cohort == 'main' else 'modules/all_module_trials.jsonl')
            catalog = stimuli / ('solution_catalog.jsonl' if cohort == 'main' else 'modules/solution_catalog_modules.jsonl')
            responses = data_dir / 'ai' / provider / ('formal' if provider == 'gemini' else cohort) / 'responses.jsonl'
            yield provider, cohort, data, catalog, responses


def process_models(data_dir, output, limit=None):
    reports = []
    for provider, cohort, data, catalog_path, responses in model_inputs(data_dir):
        destination = output / 'ai' / f'{provider}_{cohort}.jsonl'
        puzzles, catalog = scoring.load_scoring_inputs(str(data), str(responses), str(destination), str(catalog_path))
        requests = rows = valid = 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8') as stream:
            for request in json_rows(responses):
                parts = scoring._expand_response_row(request)
                if provider == 'gemini':
                    parts = [part for part in parts if part.get('puzzle_id') in puzzles]
                if not parts:
                    continue
                requests += 1
                for part in parts:
                    rows += 1
                    scored = scoring._score_one_response(part, response_index=rows, puzzles=puzzles, catalog=catalog)
                    valid += bool(scored['is_valid'])
                    stream.write(json.dumps(scored, sort_keys=True, ensure_ascii=False) + '\n')
                if limit is not None and requests >= limit:
                    break
        reports.append({'provider': provider, 'cohort': cohort, 'requests': requests,
                        'rows': rows, 'valid_rows': valid, 'output': str(destination),
                        'output_sha256': sha256(destination), 'raw_input': str(responses)})
        print(f'{provider}/{cohort}: {requests} requests, {rows} scored rows', flush=True)
    return reports


def process_humans(data_dir, output):
    destination = output / 'human'
    destination.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cohort in ('main', 'module'):
        archive = data_dir / 'private' / cohort
        participants = human.read_jsonl(archive / 'participants.jsonl')
        raw_rows = human.payloads(archive)
        retained = human.retained_users(cohort, participants, raw_rows)
        rows = human.deidentify(raw_rows, participants, retained)
        human.write_jsonl(destination / f'{cohort}_retained.jsonl', rows)
        summary[cohort] = {'retained_participants': len(retained), 'retained_rows': len(rows),
                           'correct_rows': sum(bool(row['is_correct']) for row in rows)}
    (destination / 'retention_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return summary


def run(data_dir, output_dir=None, stage='all', limit=None):
    data_dir = Path(data_dir).resolve()
    if output_dir is None:
        raise ValueError('Choose an explicit generated --output-dir, normally under intermediate/')
    output = Path(output_dir).resolve()
    if stage not in {'human', 'models', 'all'}:
        raise ValueError('stage must be human, models, or all')
    if limit is not None and (limit < 1 or stage != 'models'):
        raise ValueError('--limit requires stage models, a positive count, and a separate --output-dir')
    protected = [data_dir / name for name in ['private', 'ai', 'stimuli']]
    if any(output.is_relative_to(path.resolve()) or path.resolve().is_relative_to(output) for path in protected):
        raise ValueError('Output must be separate from raw human/model inputs and study materials')
    required = []
    if stage in {'human', 'all'}:
        required += [data_dir / 'private' / cohort / name for cohort in ['main', 'module']
                     for name in ['participants.jsonl', 'responses.jsonl']]
    if stage in {'models', 'all'}:
        required += [path for _, _, data, catalog, responses in model_inputs(data_dir)
                     for path in [data, catalog, responses]]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    # Reserve the run before any output is written; never replace saved baselines.
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    started = time.monotonic()
    manifest = {'status': 'running', 'stage': stage, 'data_dir': str(data_dir),
                'output_dir': str(output), 'smoke_only': limit is not None,
                'model_request_limit_per_file': limit, 'parser_version': response_parser.PARSER_VERSION,
                'parser_sha256': sha256(Path(response_parser.__file__)),
                'input_sha256': {str(path): sha256(path) for path in sorted(set(required))}}
    manifest_path = output / 'processing_manifest.json'
    try:
        if stage in {'human', 'all'}:
            manifest['human'] = process_humans(data_dir, output)
        if stage in {'models', 'all'}:
            manifest['models'] = process_models(data_dir, output, limit)
        manifest['status'] = 'passed'
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - started
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, required=True, help='New generated directory, normally under intermediate/')
    parser.add_argument('--stage', choices=['human', 'models', 'all'], default='all')
    parser.add_argument('--limit', type=int, help='Smoke only: maximum matching requests per model/cohort; requires separate output')
    args = parser.parse_args()
    result = run(args.data_dir, args.output_dir, args.stage, args.limit)
    print(f"{result['status']}: {result['output_dir']}/processing_manifest.json")


if __name__ == '__main__':
    main()
