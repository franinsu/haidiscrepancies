"""Compose numerical calculations; never read reference results as inputs."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from . import main_statistics as core
from . import entropy, entropy_correlation, effort, geometry
from . import stimulus_modules as sm
from . import perturbation_summaries as ps
from . import perturbation_counts, perturbation_effects
from . import gain, mi, context_dependence


def configure(bootstrap=5000, permutations=10000):
    core.TV_BOOTSTRAP_REPEATS = bootstrap
    for module in (entropy, effort, sm):
        module.BOOTSTRAP_REPEATS = bootstrap
    perturbation_effects.NBOOT = gain.NBOOT = bootstrap
    perturbation_effects.NPERM = permutations


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_geometry(stats):
    global_tv = np.asarray(stats['global_tv_primary']['matrix'])
    family_tv = np.asarray([stats['pairwise_tv_by_type_primary'][f]['matrix'] for f in geometry.FAMILY_ORDER])
    return geometry.serializable_result(stats, global_tv, family_tv,
        geometry.mean_tv_embedding(global_tv), geometry.family_geometry_cka(family_tv))


def condition_geometry(stats, source):
    """Supplementary MDS projection; primary reference points remain fixed."""
    base = stats['condition_source_geometry']
    order = ['Uniform', *base['source_order']]
    metadata = [dict(source='Uniform',provider='Uniform',effort='-',prompt='-'), *base['source_metadata']]
    distance = np.zeros((len(order),len(order)))
    distance[1:,1:] = base['tv_matrix']
    primary = stats['global_tv_primary']
    u,h = [primary['order'].index(s) for s in ('Uniform','Human')]
    uniform = dict(estimate=primary['matrix'][u][h],ci95=primary['ci95_matrix'][u][h],source='Uniform',reference='Human')
    distance[0,1] = distance[1,0] = uniform['estimate']
    for i,meta in enumerate(metadata[2:],2):
        key = meta['effort']+'|'+('as-human' if meta['prompt']=='persona' else meta['prompt'])
        cell = stats['global_tv_by_condition'][key]
        distance[0,i] = distance[i,0] = cell['matrix'][cell['order'].index('Uniform')][cell['order'].index(meta['provider'])]
    reference = source['mean_tv_mds']
    coords = np.asarray(reference['coordinates'])
    indices = [order.index(s if s in ('Uniform','Human') else s+'|low|plain') for s in primary['order']]
    if not np.allclose(distance[np.ix_(indices,indices)],primary['matrix'],atol=1e-12,rtol=0): raise ValueError('Reference distances disagree')
    squared = np.asarray(primary['matrix'])**2
    query = distance[:,indices]**2
    kernel = -.5*(query-query.mean(axis=1,keepdims=True)-squared.mean(axis=0)+squared.mean())
    eigenvalues = np.asarray(reference['eigenvalues'])
    coordinates = kernel@coords/eigenvalues[:2]
    if not np.allclose(coordinates[indices],coords,atol=1e-12,rtol=0): raise ValueError('Reference projection moved')
    coordinates[indices] = coords
    limits = [[float(x.min()-max(.06,.30*np.ptp(x))),float(x.max()+max(.06,.30*np.ptp(x)))] for x in coords.T]
    upper = np.triu_indices(len(distance),1)
    fitted = np.linalg.norm(coordinates[:,None]-coordinates[None,:],axis=-1)
    stress = float(np.sqrt(np.sum((distance[upper]-fitted[upper])**2)/np.sum(distance[upper]**2)))
    return dict(geometry_input=dict(source_order=order,source_metadata=metadata,tv_matrix=distance.tolist(),n_puzzles=base['n_puzzles'],aggregation=base['aggregation']),
        reference_geometry=reference, uniform_reference=uniform,
        embedding=dict(coordinates=coordinates.tolist(),eigenvalues=eigenvalues.tolist(),shares=reference['shares'],reference_indices=indices,reference_limits=limits,stress1=stress))


def mi_summary(records, bootstrap=5000):
    """MI level contrasts use paired whole-puzzle bootstrap, with no sign-flip tests."""
    sources = ['Human','GPT','Claude','Gemini']
    display = {'Human':'Human', **{core.condition_key(p,'low','direct_solve'):label for p,label in zip(core.PROVIDERS,sources[1:])}}
    contexts = {'related':'Related','unrelated_control':'Unrelated'}
    tests = [dict(source=display[r['source']],context=contexts[r['arm']],family=core.TYPE_LABELS[r['family']],
        puzzle_id=r['block'],mi_bits=r['mi_bits'],n_complete_valid_pairs=r['n_pairs'],p_unadjusted=r['permutation_p'],permutations=r['permutations']) for r in records]
    ids = sorted({r['puzzle_id'] for r in tests}); families={r['puzzle_id']:r['family'] for r in tests}
    strata=[(family,[pid for pid in ids if families[pid]==family]) for family in sorted(set(families.values()))]
    values={(r['source'],r['context'],r['puzzle_id']):r['mi_bits'] for r in tests}
    draws=sm.bootstrap_draws(tuple(len(v) for _,v in strata),repeats=bootstrap,seed=0)
    contrasts=[]
    def add(kind,source,context,differences):
        result=sm.summarize_blocks(dict(zip(ids,differences)),strata,draws)
        contrasts.append(dict(comparison=kind,source=source,context=context,estimate_bits=result['estimate'],ci95_low=result['ci95'][0],ci95_high=result['ci95'][1],n_puzzles=len(ids)))
    for source in sources:
        add('Related minus unrelated',source,'Related minus unrelated',[values[source,'Related',p]-values[source,'Unrelated',p] for p in ids])
    for context in contexts.values():
        for source in sources[1:]:
            add('Human minus model',source,context,[values['Human',context,p]-values[source,context,p] for p in ids])
    means=[]
    for source in sources:
        for context in contexts.values():
            selected=[r for r in tests if r['source']==source and r['context']==context]
            means.append(dict(source=source,context=context,mean_p=float(np.mean([r['p_unadjusted'] for r in selected])),
                mean_mi_bits=float(np.mean([r['mi_bits'] for r in selected])),n_puzzles=len(selected),p_below_0_05=sum(r['p_unadjusted']<.05 for r in selected)))
    return dict(contrasts=contrasts,individual_independence_tests=tests,source_means=means,
        method='Uncorrected empirical MI in bits; 2,000 conditional permutations by default; paired whole-puzzle percentile intervals stratified by family; no sign-flip contrasts.',bootstrap_repeats=bootstrap)


def run_mi(data_dir, processed_dir, output_dir, bootstrap=5000, mi_permutations=2000, counts=None):
    counts = counts if counts is not None else perturbation_counts.extract_counts(data_dir,processed_dir)
    records = mi.compute_tests(counts,draws=mi_permutations,base_seed=0)
    write(output_dir/'context_mi_tests.json',dict(tests=records,n_tests=len(records),randomization=dict(namespace=mi.NAMESPACE,base_seed=0,draws_per_test=mi_permutations,generator='PCG64',seed='SHA-256',canonical_pair_order=True)))
    summary=mi_summary(records,bootstrap)
    write(output_dir/'context_mi_summary.json',summary)
    return records


def run_summaries(data_dir, processed_dir, output_dir, bootstrap=5000, permutations=10000, mi_permutations=2000):
    configure(bootstrap,permutations)
    print('Computing main distributions, condition comparisons and uncertainty.',flush=True)
    stats=core.compute(data_dir,processed_dir)
    write(output_dir/'figure_statistics.json',stats)
    puzzles,blocks=core.puzzle_metadata(data_dir)
    print('Computing normalized entropy, effort and geometry.',flush=True)
    ent=entropy.compute_normalized_entropy(stats,puzzles,stats['main_puzzle_ids'],blocks)
    write(output_dir/'entropy_deficit_stats.json',ent)
    write(output_dir/'entropy_correlation_stats.json',entropy_correlation.compute_entropy_correlations(ent))
    write(output_dir/'effort_difficulty_stats.json',effort.compute(data_dir,processed_dir))
    geom=source_geometry(stats)
    write(output_dir/'source_geometry.json',geom)
    write(output_dir/'condition_geometry.json',condition_geometry(stats,geom))
    print('Computing perturbation counts, TV tests and MI permutations.',flush=True)
    counts=perturbation_counts.extract_counts(data_dir,processed_dir)
    write(output_dir/'perturbation_counts.json',counts)
    effects=perturbation_effects.compute(counts)
    effects['paper_tests']=[]
    for row in [r for r in effects['row_order'] if r!='pair_control']:
        for source in effects['sources']:
            cells=[r for r in effects['records'] if r['row']==row and r['source']==source]
            effects['paper_tests'].append(dict(row=row,module=cells[0]['module'],source=source,
                mean_unadjusted_p=float(np.mean([r['p_value'] for r in cells])),n_comparisons=len(cells),
                holm_below_005=sum(r['p_holm']<=.05 for r in cells)))
    write(output_dir/'perturbation_statistics.json',effects)
    write(output_dir/'gain_summaries.json',gain.compute(effects))
    write(output_dir/'context_dependence_table.json',context_dependence.compute(effects,repeats=bootstrap))
    tests=run_mi(data_dir,processed_dir,output_dir,bootstrap,mi_permutations,counts)
    print('Computing displayed module summaries and family intervals.',flush=True)
    design,_ = sm.load_design(data_dir)
    summary, normalized = ps.compute(design,effects,tests,repeats=bootstrap)
    write(output_dir/'stimulus_sensitivity_stats.json',summary)
    write(output_dir/'normalized_attraction_stats.json',normalized)
    print('Summary calculations complete.',flush=True)
