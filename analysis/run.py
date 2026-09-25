#!/usr/bin/env python3
"""Recompute the current paper's numerical results from retained/scored data."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import os
# Keep numerical libraries single-threaded; independent fits can use --workers.
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis import DEFAULT_L2_PENALTY, pipeline, regressions
import hashlib
import json
import platform
import time
import numpy as np
import scipy


def file_hash(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('summaries','regressions','mi','all'),default='all')
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    parser.add_argument('--processed-dir',type=Path,required=True,help='Retained/scored rows from an explicit processing run')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'intermediate/statistics')
    parser.add_argument('--bootstrap',type=int,default=5000)
    parser.add_argument('--regression-draws',type=int,default=2000)
    parser.add_argument('--l2-penalty',type=float,default=DEFAULT_L2_PENALTY,
                        help='Nonnegative L2 strength on regression slopes; mean NLL + strength/2 * squared slopes (default: %(default)g; use 0 for unpenalized fits)')
    parser.add_argument('--permutations',type=int,default=10000)
    parser.add_argument('--mi-permutations',type=int,default=2000)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    args.data_dir=args.data_dir.resolve()
    args.processed_dir=args.processed_dir.resolve()
    args.output_dir=args.output_dir.resolve()
    if min(args.bootstrap,args.permutations,args.mi_permutations,args.workers)<1 or args.regression_draws<2:
        parser.error('Resampling counts/workers must be positive; regression draws must be at least two')
    if not np.isfinite(args.l2_penalty) or args.l2_penalty < 0:
        parser.error('--l2-penalty must be finite and nonnegative')
    for protected in (args.data_dir,args.processed_dir):
        if args.output_dir.is_relative_to(protected) or protected.is_relative_to(args.output_dir):
            parser.error('Output must be separate from input data directories')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=args.output_dir/'analysis_manifest.json'
    params={k:v for k,v in vars(args).items() if not isinstance(v,Path)}
    started = time.monotonic()
    print('Checking numerical inputs and recording provenance.',flush=True)
    code_paths = {p.name:p for p in Path(__file__).resolve().parent.glob('*.py')}
    code={name:file_hash(p) for name,p in code_paths.items()}
    inputs={}
    input_paths={}
    for prefix,directory in [('stimuli',args.data_dir/'stimuli'),('processed',args.processed_dir)]:
        for path in sorted(directory.rglob('*.jsonl')):
            key=prefix+'/'+str(path.relative_to(directory))
            inputs[key]=file_hash(path); input_paths[key]=path
    manifest=dict(schema_version=1,parameters=params,code_sha256=code,input_sha256=inputs,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        data_root=str(args.data_dir),processed_root=str(args.processed_dir),status='running',
        reproduction_scope='Retained/scored observations to statistics; processing freshness is recorded by the upstream processing manifest.')
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        for key in ('parameters','code_sha256','input_sha256','environment'):
            if old.get(key)!=manifest[key]: raise ValueError('Existing run differs; choose a fresh output directory')
    elif any(args.output_dir.iterdir()): raise ValueError('Nonempty statistics output lacks analysis_manifest.json')
    pipeline.write(manifest_path,manifest)
    if args.mode in ('summaries','all'):
        pipeline.run_summaries(args.data_dir,args.processed_dir,args.output_dir,args.bootstrap,args.permutations,args.mi_permutations)
    elif args.mode=='mi':
        pipeline.configure(args.bootstrap,args.permutations)
        pipeline.run_mi(args.data_dir,args.processed_dir,args.output_dir,args.bootstrap,args.mi_permutations)
    if args.mode in ('regressions','all'):
        # Fit checkpoints live below statistics; final five payloads use flat names.
        fit_dir=args.output_dir/'regression_checkpoints'
        regressions.run(args.data_dir,args.processed_dir,fit_dir,args.regression_draws,args.workers,
                        statistics_dir=args.output_dir,l2_penalty=args.l2_penalty)
    if any(file_hash(path)!=code[name] for name,path in code_paths.items()) or any(file_hash(path)!=inputs[name] for name,path in input_paths.items()):
        raise ValueError('Input or numerical code changed during the run; outputs are not accepted')
    manifest['status']='complete'
    manifest['elapsed_seconds']=time.monotonic()-started
    manifest['outputs_sha256']={p.name:file_hash(p) for p in sorted(args.output_dir.glob('*.json')) if p!=manifest_path}
    pipeline.write(manifest_path,manifest)
    print(f'Completed {args.mode}: {args.output_dir}',flush=True)


if __name__=='__main__':
    main()
