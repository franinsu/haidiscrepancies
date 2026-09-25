#!/usr/bin/env python3
"""Render selected paper figures/tables from explicit numerical inputs.

This command does not collect data, score responses, run inference, or read
individual records. Reference mode must be requested explicitly.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ''}:
    sys.path.insert(0, str(ROOT))
    __package__ = 'figures'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def save(fig, name, directory, *, previews=False):
    from .plot_style import plt
    for suffix in (['pdf', 'png'] if previews else ['pdf']):
        options={'metadata':{'CreationDate':None,'ModDate':None}} if suffix=='pdf' else {}
        fig.savefig(directory/f'{name}.{suffix}', facecolor='white', dpi=300, **options)
    plt.close(fig)


def run(input_dir, output_dir, stimuli_dir, *, only=None, reference=False, previews=False):
    from . import entropy_profile, tv_geometry, effort_matrix, effort_projection
    from . import entropy_correlation, effort_summary, context_mi, conditions, procedure, stimuli, tables
    from .plot_style import FONT_FAMILY
    from matplotlib import font_manager
    input_dir, output_dir, stimuli_dir = map(lambda p:Path(p).resolve(),[input_dir,output_dir,stimuli_dir])
    folders=[output_dir/name for name in ['figures','tables','source_data']]
    manifest_path=output_dir/'render_manifest.json'
    managed=[path.resolve() for path in [*folders,manifest_path]]
    for label,path in [('Numerical',input_dir),('Stimulus',stimuli_dir)]:
        if (output_dir.is_relative_to(path)
                or any(path.is_relative_to(target) or target.is_relative_to(path) for target in managed)):
            raise ValueError(f'{label} inputs overlap the renderer output destinations')
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f'Output directory is not a directory: {output_dir}')
    for path in folders:
        if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
            raise FileExistsError(f'Choose an empty renderer output directory: {path}')
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(f'Render manifest already exists: {manifest_path}')
    manifest=json.loads(Path(__file__).with_name('manifest.json').read_text())
    entries=manifest['outputs']
    if only:
        unknown=set(only)-{r['name'] for r in entries}
        if unknown:raise ValueError(f'Unknown requested outputs: {sorted(unknown)}')
        entries=[r for r in entries if r['name'] in only]
    names={r['name'] for r in entries}; hashes={}
    if (any(r['public_input_class']=='display_summary' for r in entries)
            and not reference and not (input_dir/'analysis_manifest.json').is_file()):
        raise ValueError('Saved statistical inputs without an analysis manifest require --reference')
    cache={}
    def read(stem):
        if stem not in cache:
            path=input_dir/(stem+'.json')
            hashes[path.name]=sha(path);cache[stem]=json.loads(path.read_text())
        return cache[stem]
    # Fail before writing if a requested numerical artifact is missing.
    for entry in entries:
        if entry['public_input_class']=='display_summary':
            for filename in entry['inputs']:read(Path(filename).stem)
    for folder in folders:folder.mkdir(parents=True,exist_ok=True)
    artwork, source_data = output_dir/'figures', output_dir/'source_data'
    board_names={name for name in names if name.startswith('fBboard')}
    if board_names:
        stimuli.stimulus_figures(stimuli_dir, artwork, only=board_names, previews=previews)
        for path in ['all_puzzles.jsonl','modules/all_module_trials.jsonl','modules/module_blocks.jsonl']:
            hashes['stimuli/'+path]=sha(stimuli_dir/path)
    for name in sorted(names-board_names):
        entry=next(r for r in entries if r['name']==name)
        if entry['kind']=='table':continue
        if name=='fig_procedure':
            data=procedure.pipeline(artwork/(name+'.pdf'), previews=previews)
            # The schematic exports design/sample totals and illustrative bars only.
            data={key:data[key] for key in ['kind','total_puzzles','core_puzzles','perturbation_puzzles','families','retained_humans_total','retained_core_humans','retained_module_humans','maximum_human_attempts_per_trial','model_requests_per_puzzle_per_condition','display']}
        elif name=='tv_source_geometry':data=tv_geometry.render(read('figure_statistics'),read('source_geometry'),artwork/(name+'.pdf'), previews=previews)
        elif name=='entropy_profile':
            fig,payload=entropy_profile.render(read('entropy_deficit_stats'))
            data={'summary':payload['summary']['records'],'density':payload['density'],'source_labels':payload['source_labels']}
            save(fig,name,artwork,previews=previews)
        else:
            if name=='fig_presentation_context':fig,data=conditions.presentation(read('perturbation_statistics'),read('gain_summaries'),read('stimulus_sensitivity_stats'))
            elif name=='fig6_effort_prompting':fig,data=conditions.conditions(read('figure_statistics'),read('condition_geometry'))
            elif name=='fig5_relative_difficulty':fig,data=effort_matrix.render(read('effort_difficulty_stats'))
            elif name=='effort_svd_projection':
                fig,payload=effort_projection.render(read('effort_difficulty_stats'))
                data={key:payload[key] for key in ['family_order','source_order','frobenius_energy_share','left_singular_vectors_top_two','source_scores_top_two']}
            elif name=='effort_difficulty_by_type':
                fig,payload=effort_summary.render(read('effort_difficulty_stats'))
                data={'points':[{key:r[key] for key in ['source','family','mean','ci95']} for r in payload['points']]}
            elif name=='entropy_correlation_pooled':fig,data=entropy_correlation.render(read('entropy_deficit_stats'))
            elif name=='context_mi_contrasts':fig,data=context_mi.render(read('context_mi_summary'))
            else:raise ValueError(name)
            save(fig,name,artwork,previews=previews)
        write_json(source_data/(name+'.json'),data)
    table_names=[r['name'] for r in entries if r['kind']=='table']
    texts, table_data=tables.generate(read,table_names)
    for name,text in texts.items():
        (output_dir/'tables'/(name+'.tex')).write_text(text)
        write_json(source_data/(name+'.json'),table_data[name])
    missing=[r['output'] for r in entries if not (output_dir/r['output']).is_file()]
    if missing:raise RuntimeError(f'Rendering incomplete: {missing}')
    def font_hashes(family):
        files={}
        for weight,style in [('normal','normal'),('bold','normal'),('normal','italic')]:
            path=Path(font_manager.findfont(font_manager.FontProperties(family=family,weight=weight,style=style),fallback_to_default=False))
            files[weight+'/'+style]={'filename':path.name,'sha256':sha(path)}
        return files
    analysis_manifest=input_dir/'analysis_manifest.json'
    analysis_parameters=json.loads(analysis_manifest.read_text()).get('parameters',{}) if analysis_manifest.is_file() else {}
    reduced_draws=any(analysis_parameters.get(key,expected)!=expected for key,expected in [('bootstrap',5000),('regression_draws',2000),('permutations',10000),('mi_permutations',2000)])
    result={'previews':previews, 'validation_only':reduced_draws, 'font_files':font_hashes(FONT_FAMILY), 'upstream_parameters':analysis_parameters, 'mode':'archived-reference-render' if reference else 'computed-statistics-render',
            'font_family':FONT_FAMILY,'numerical_input_sha256':hashes,
            'analysis_parameters':{name:{key:obj[key] for key in ['bootstrap','tv_bootstrap','global_tv_response_bootstrap','permutation','bootstrap_repeats'] if key in obj} for name,obj in cache.items()},
            'outputs':{r['output']:sha(output_dir/r['output']) for r in entries},
            'preview_outputs':{str(Path(r['output']).with_suffix('.png')):sha((output_dir/r['output']).with_suffix('.png'))
                               for r in entries if previews and r['kind']=='figure' and r['output'].endswith('.pdf')},
            'public_exports':'Only displayed summaries, per-puzzle plotted aggregates, design values and table cells; no individual records or bootstrap draws.'}
    write_json(manifest_path,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'intermediate/statistics')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--stimuli-dir',type=Path)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data',help='Public stimulus catalog is read from its stimuli subfolder')
    parser.add_argument('--only',nargs='+')
    parser.add_argument('--previews',action='store_true',help='Also render PNG previews of the ten PDF figures')
    parser.add_argument('--reference',action='store_true',help='Explicitly render archived inputs; this is not a fresh reproduction')
    args=parser.parse_args()
    result=run(args.input_dir,args.output_dir,args.stimuli_dir or args.data_dir/'stimuli',only=args.only,reference=args.reference,previews=args.previews)
    print(f"Rendered {len(result['outputs'])} selected paper assets ({result['mode']}).")


if __name__=='__main__':main()
