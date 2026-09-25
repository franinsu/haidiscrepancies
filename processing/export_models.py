#!/usr/bin/env python3
"""Export the recorded model responses without provider payloads or identifiers.

Only reviewed study fields are retained. New fields and populated API errors
require review rather than being silently copied into a public archive.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

ARCHIVES = ('openai/main', 'openai/module', 'anthropic/main',
            'anthropic/module', 'gemini/formal')
REMOVED_FIELDS = frozenset(('provider_response_json', 'provider_response_id'))
PUBLIC_FIELDS = frozenset('''
api_error api_model_condition api_provider attempt_all_valid attempt_index
attempt_invalid_reasons attempt_item_count attempt_solution_ids attempt_valid_count
cache_creation_input_tokens cache_read_input_tokens cache_write_input_tokens
cached_input_tokens collection_mode condition dataset_name feedback_mode
hidden_reasoning_tokens image_id image_path image_sha256 incomplete_reason
input_kind input_tokens latency_ms max_attempts max_output_tokens max_tokens
model module output_token_cap output_tokens prompt_condition prompt_file
prompt_file_used prompt_sha256 prompt_version provider_model provider_status
puzzle_id puzzle_type raw_answer raw_response reasoning_effort reasoning_tokens
render_version request_id request_start_time_iso response_end_time_iso retry_count
run_id sample_index sampling_profile sequence_id source stop_reason temperature
thinking_effort thinking_tokens thinking_type top_p total_tokens trial_ids
'''.split())


def public_response(row):
    if not isinstance(row, dict):
        raise ValueError('A model response must be an object')
    if row.keys() - PUBLIC_FIELDS - REMOVED_FIELDS:
        raise ValueError('Unreviewed response fields; review the export schema first')
    if row.get('source') != 'api':
        raise ValueError('Only model API responses can be exported')
    if not isinstance(row.get('request_id'), str) or not re.fullmatch(r'[0-9a-f]{20}', row['request_id']):
        raise ValueError('A study-generated request_id is required for paired analysis')
    if row.get('api_error') not in (None, ''):
        raise ValueError('API error text needs separate review before publication')
    for field in ('image_path', 'prompt_file', 'prompt_file_used'):
        value = row.get(field)
        if value and (not isinstance(value, str) or Path(value).is_absolute()
                      or '\\' in value or ':' in value or value.startswith('~')
                      or '..' in Path(value).parts):
            raise ValueError('Study asset paths must be relative')
    return {key: value for key, value in row.items() if key in PUBLIC_FIELDS}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(input_dir, output_dir):
    source = Path(input_dir).resolve()
    requested = Path(output_dir)
    if requested.exists() or requested.is_symlink():
        raise FileExistsError('Choose a fresh export directory')
    output = requested.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('Input and output directories must be separate')
    inputs = {name: source / name / 'responses.jsonl' for name in ARCHIVES}
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    before = {name: sha256(path) for name, path in inputs.items()}
    output.mkdir(parents=True)
    manifest = {'schema_version': 1, 'removed_fields': sorted(REMOVED_FIELDS),
                'description': 'Recorded model responses; provider payloads and run/batch manifests omitted.',
                'files': {}}
    for name, path in inputs.items():
        destination = output / name / 'responses.jsonl.gz'
        destination.parent.mkdir(parents=True)
        count = 0
        with path.open(encoding='utf-8') as incoming, destination.open('xb') as outgoing:
            # No source filename or timestamp is embedded in the gzip header.
            with gzip.GzipFile(filename='', mode='wb', fileobj=outgoing, mtime=0, compresslevel=6) as compressed:
                for line in incoming:
                    if not line.strip():
                        continue
                    row = public_response(json.loads(line))
                    compressed.write((json.dumps(row, sort_keys=True, ensure_ascii=False,
                                                 separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8'))
                    count += 1
        if count == 0:
            raise ValueError(f'Empty response archive: {name}')
        manifest['files'][f'{name}/responses.jsonl.gz'] = {
            'requests': count, 'bytes': destination.stat().st_size,
            'sha256': sha256(destination), 'original_sha256': before[name]}
    if any(sha256(path) != before[name] for name, path in inputs.items()):
        raise ValueError('Source responses changed during export')
    (output / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True, help='Original AI archive with provider/cohort folders')
    parser.add_argument('--output-dir', type=Path, required=True, help='Fresh directory for the reviewed public fields')
    args = parser.parse_args()
    manifest = run(args.input_dir, args.output_dir)
    print(f"Exported {sum(item['requests'] for item in manifest['files'].values()):,} model requests.")


if __name__ == '__main__':
    main()
