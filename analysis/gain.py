"""Normalize directional change within each comparison, then average blocks."""
import random
import numpy as np
NBOOT = 5000
EPS = 1e-12
def mean_and_bootstrap(groups):
    """Equal family means; resample whole blocks, retaining all arm-derived columns."""
    sizes=[len(g) for g in groups];assert all(sizes)
    offsets=np.cumsum([0]+sizes[:-1]);rng=random.Random(0)
    draws=np.array([[o+rng.randrange(n) for o,n in zip(offsets,sizes) for _ in range(n)] for _ in range(NBOOT)])
    values=np.concatenate(groups);cuts=np.cumsum([0]+sizes)
    sample=np.mean([values[draws[:,cuts[i]:cuts[i+1]]].mean(axis=1) for i in range(len(groups))],axis=0)
    return np.mean([np.mean(g,axis=0) for g in groups],axis=0),sample

def compute(data):
    summaries = {}
    rows = ['cue', 'pair_related', 'transfer']
    for row in rows:
        summaries[row] = {}
        for source in data['sources']:
            selected = [r for r in data['records'] if r['row'] == row and r['source'] == source]
            groups = []
            used = at_target = departed = 0
            for family in sorted({r['family'] for r in selected}):
                values = []
                for block in sorted({r['block'] for r in selected if r['family'] == family}):
                    ratios = []
                    for r in selected:
                        if r['block'] != block: continue
                        p = r['reference_repeat'] if row == 'pair_related' else r['baseline_mass']
                        q = r['repeat'] if row == 'pair_related' else r['perturbed_mass']
                        gap = max(0., 1-p)
                        if gap > EPS:
                            ratios.append((q-p)/gap); used += 1
                        else:
                            at_target += 1; departed += q < 1-EPS
                    if ratios: values.append([float(np.mean(ratios))])
                if not values: raise ValueError('No eligible normalization blocks in a family')
                groups.append(np.asarray(values))
            estimate, samples = mean_and_bootstrap(groups)
            summaries[row][source] = dict(estimate=float(estimate[0]),
                ci95=np.quantile(samples[:,0], [.025,.975]).tolist(),
                comparisons_used=used, comparisons_total=len(selected),
                n_blocks=sum(len(g) for g in groups), initially_at_target=at_target,
                initially_at_target_then_departed=departed,
                bootstrap_defined=NBOOT, bootstrap_undefined=0)
    return dict(sources=data['sources'], summaries={'b': summaries},
        method='Per-comparison directional change divided by available upward change; eligible arms within block, equal blocks within family, equal families.',
        bootstrap_repeats=NBOOT, bootstrap_seed=0)
