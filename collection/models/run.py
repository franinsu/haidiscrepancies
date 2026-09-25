#!/usr/bin/env python3
"""Study model acquisition: prepare a queue, mock locally, or explicitly call providers."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
STUDY_CONDITIONS = 'OAI_GPT56_SOL_LOW,OAI_GPT56_SOL_MEDIUM,ANT_OPUS48_LOW,ANT_OPUS48_MEDIUM,GEMINI_35_FLASH_LOW,GEMINI_35_FLASH_MEDIUM'


def main():
    choices = {'prepare': 'build_api_run_manifest', 'mock': 'run_api_collection',
               'live': 'run_api_collection', 'batch': 'run_api_batch',
               'render': 'render_api_images', 'score': 'score_api_run'}
    if len(sys.argv) < 2 or sys.argv[1] not in choices:
        print('Usage: python collection/models/run.py {prepare,mock,live,batch,render,score} [arguments]')
        return 0 if len(sys.argv) > 1 and sys.argv[1] in {'-h', '--help'} else 2
    import importlib
    command, arguments = sys.argv[1], sys.argv[2:]
    if command == 'prepare' and not any(arg.startswith('--api_model_conditions') for arg in arguments):
        arguments += ['--api_model_conditions', STUDY_CONDITIONS]
    if command in {'mock', 'live'}:
        if '--mode' in arguments:
            raise ValueError('Choose mock or live through the command name')
        arguments += ['--mode', command]
    os.chdir(ROOT)
    module = importlib.import_module('collection.models.' + choices[command])
    sys.argv = [module.__file__, *arguments]
    return module.main()


if __name__ == '__main__':
    raise SystemExit(main())
