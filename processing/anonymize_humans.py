#!/usr/bin/env python3
"""Create a de-identified human replay archive and a separate private ID map."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ('main', 'module')
MAP_FIELDS = ('cohort', 'new_id', 'old_username', 'old_analysis_id', 'prolific_pid')
TEXT_FIELDS = ('puzzle_type', 'module', 'family_id', 'block_id', 'condition',
               'sequence_id', 'solution_id', 'abstract_solution_id',
               'accepted_solution_id', 'accepted_abstract_solution_id')


def text(value, field, *, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or (not optional and not value.strip()):
        raise ValueError(f'{field} must be a string')
    return value


def integer(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{field} must be an integer >= {minimum}')
    return value


def flag(value, field):
    if type(value) not in (bool, int) or value not in (0, 1):
        raise ValueError(f'{field} must be boolean or 0/1')
    return bool(value)


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Archive records must be JSON objects')
                yield row


def load_inputs(directory):
    participants, responses, identities = {}, {}, {}
    for cohort in COHORTS:
        participants[cohort], responses[cohort] = [], []
        names, analysis_ids = set(), set()
        for index, row in enumerate(rows(directory/cohort/'participants.jsonl'), 1):
            username = text(row.get('username'), 'username')
            old_analysis_id = text(row.get('analysis_id'), 'analysis_id', optional=True) or f'anonymous-{index:04d}'
            if username in names or old_analysis_id in analysis_ids:
                raise ValueError(f'Duplicate participant identity in {cohort}')
            names.add(username); analysis_ids.add(old_analysis_id)
            identity = {'cohort': cohort, 'old_username': username, 'old_analysis_id': old_analysis_id,
                        'prolific_pid': text(row.get('prolific_pid'), 'prolific_pid', optional=True) or ''}
            identities[(cohort, username)] = identity
            source = text(row.get('participant_source'), 'participant_source', optional=True)
            participants[cohort].append({'username': username, 'assigned_count': integer(row.get('assigned_count'), 'assigned_count'),
                                         'participant_source': 'tester' if source == 'tester' else 'formal', 'cohort': cohort})
        database_ids, trials = set(), set()
        for index, row in enumerate(rows(directory/cohort/'responses.jsonl'), 1):
            username = text(row.get('username'), 'response username')
            puzzle_id = text(row.get('puzzle_id'), 'puzzle_id')
            database_id = integer(row.get('id'), 'response id', 1)
            if username not in names:
                raise ValueError(f'Response participant is unknown in {cohort}')
            if database_id in database_ids or (username, puzzle_id) in trials:
                raise ValueError(f'Duplicate response record in {cohort}')
            database_ids.add(database_id); trials.add((username, puzzle_id))
            payload = json.loads(text(row.get('row_json'), 'row_json'))
            if not isinstance(payload, dict):
                raise ValueError('row_json must encode an object')
            safe = {field: text(payload.get(field), field, optional=True) for field in TEXT_FIELDS}
            safe['puzzle_id'] = puzzle_id
            position = payload.get('sequence_position')
            safe['sequence_position'] = None if position is None else integer(position, 'sequence_position', 1)
            duration = row.get('response_time_ms')
            if duration is not None and (type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0):
                raise ValueError('response_time_ms must be a finite nonnegative number or null')
            cleaned = {'id': index, 'username': username, 'puzzle_id': puzzle_id,
                       'response_time_ms': duration, 'row_json': json.dumps(safe, sort_keys=True, allow_nan=False)}
            for field in ('is_correct', 'bad_record', 'timeout', 'gave_up'):
                cleaned[field] = flag(row.get(field), field)
            responses[cohort].append(cleaned)
    return participants, responses, identities


def identity_map(path, identities):
    existing = path.exists()
    if existing:
        if not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise ValueError('Existing ID map must be a regular file with mode 0600')
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(MAP_FIELDS):
                raise ValueError('ID map has incorrect columns')
            entries = list(reader)
    else:
        entries, allocated = [], set()
        for cohort in COHORTS:
            people = sorted((v for v in identities.values() if v['cohort'] == cohort), key=lambda v:v['old_analysis_id'])
            new_ids = set()
            while len(new_ids) < len(people):
                candidate = 'h_' + secrets.token_hex(12)
                if candidate not in allocated:
                    new_ids.add(candidate); allocated.add(candidate)
            entries.extend({**person, 'new_id': new_id} for person, new_id in zip(people, sorted(new_ids)))
    mapping, used = {}, set()
    for row in entries:
        if set(row) != set(MAP_FIELDS) or any(not isinstance(v, str) for v in row.values()):
            raise ValueError('Malformed ID map row')
        key = (row['cohort'], row['old_username'])
        identity = {field: row[field] for field in MAP_FIELDS if field != 'new_id'}
        if key in mapping or identity != identities.get(key):
            raise ValueError('ID map does not match the participant identity inventory')
        if not re.fullmatch(r'h_[0-9a-f]{24}', row['new_id']) or row['new_id'] in used:
            raise ValueError('ID map has an invalid or duplicate new ID')
        mapping[key] = row['new_id']; used.add(row['new_id'])
    if set(mapping) != set(identities):
        raise ValueError('ID map must cover the complete participant identity inventory')
    for cohort in COHORTS:
        ordered = sorted((v for v in entries if v['cohort'] == cohort), key=lambda v:v['old_analysis_id'])
        new_ids = [v['new_id'] for v in ordered]
        if new_ids != sorted(new_ids):
            raise ValueError('ID map changes the historical participant bootstrap ordering')
    return mapping, entries, existing


def exclusive_text(path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, 'w', encoding='utf-8', newline='')


def validate_id_map_path(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('The ID map must not be a symlink')
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT):
        raise ValueError('The ID map must be outside the repository')
    return resolved


def run(input_dir, output_dir, id_map):
    input_dir, requested_output = map(Path, (input_dir, output_dir))
    map_path = validate_id_map_path(id_map)
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError('Choose a fresh replay output directory')
    input_dir, output_dir = [p.resolve() for p in (input_dir, requested_output)]
    if input_dir.is_relative_to(output_dir) or output_dir.is_relative_to(input_dir):
        raise ValueError('Input and output directories must be separate')
    if any(map_path.is_relative_to(p) or p.is_relative_to(map_path) for p in (input_dir, output_dir)):
        raise ValueError('The ID map must be outside both input and output directories')
    participants, responses, identities = load_inputs(input_dir)
    mapping, entries, existing_map = identity_map(map_path, identities)
    # All records and the complete mapping are checked before any files are written.
    output_dir.mkdir(parents=True, mode=0o700)
    if not existing_map:
        map_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with exclusive_text(map_path) as stream:
            writer = csv.DictWriter(stream, fieldnames=MAP_FIELDS)
            writer.writeheader(); writer.writerows(entries)
    manifest = {'counts': {}, 'outputs': {}}
    for cohort in COHORTS:
        (output_dir/cohort).mkdir(mode=0o700)
        for row in participants[cohort]:
            row['username'] = row['analysis_id'] = mapping[(cohort, row['username'])]
        for row in responses[cohort]:
            row['username'] = mapping[(cohort, row['username'])]
        for name, records in [('participants', participants[cohort]), ('responses', responses[cohort])]:
            relative = f'{cohort}/{name}.jsonl'
            path = output_dir/relative
            with exclusive_text(path) as stream:
                for row in records:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            manifest['outputs'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest['counts'][cohort] = {'participants': len(participants[cohort]), 'responses': len(responses[cohort])}
    with exclusive_text(output_dir/'anonymization_manifest.json') as stream:
        stream.write(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True, help='Original archive containing main/ and module/')
    parser.add_argument('--output-dir', type=Path, required=True, help='Fresh de-identified replay directory')
    parser.add_argument('--id-map', type=Path, required=True, help='Private CSV outside the repository and both archives; reuse requires an exact identity match')
    args = parser.parse_args()
    try:
        result = run(args.input_dir, args.output_dir, args.id_map)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result['counts'], sort_keys=True))


if __name__ == '__main__':
    main()
