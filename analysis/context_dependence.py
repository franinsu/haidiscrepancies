"""Complete-pair context table, using the same effects as the main figure."""
from __future__ import annotations
from collections import defaultdict
from .normalized_attraction import equal_family_summary


def compute(effects, repeats=5000):
    blocks = defaultdict(dict)
    for row in effects['records']:
        if row['row'] not in ('pair_related', 'pair_control'):
            continue
        record = blocks[row['source']].setdefault(row['block'], {'family': row['family']})
        arm = 'related' if row['row'] == 'pair_related' else 'unrelated'
        record[arm] = row['movement']
        record[arm+'_n'] = row['n_complete_pairs']
        if arm == 'related':
            p, q = row['reference_repeat'], row['repeat']
            gap, remaining = max(0., 1-p), max(0., 1-q)
            record['related_normalized_change'] = (gap-remaining)/gap if gap > 1e-12 else None
            record['related_repetition'] = q
            record['related_independent_repetition'] = p
    summaries = {}
    family_order = ['minesweeper_lite', 'mini_sudoku']
    for source, records in blocks.items():
        if len(records) != 10:
            raise ValueError('Context summary requires all ten matched blocks')
        for record in records.values():
            record['difference'] = record['related']-record['unrelated']
        summaries[source] = {}
        for family in [*family_order, 'Mean']:
            families = family_order if family == 'Mean' else [family]
            strata = [(f, sorted(k for k,r in records.items() if r['family']==f)) for f in families]
            summary = {}
            for metric in ('related','unrelated','difference','related_normalized_change'):
                values = {k:r[metric] for k,r in records.items() if r['family'] in families and r[metric] is not None}
                result = equal_family_summary(values, strata, repeats=repeats, seed=0)
                summary[metric] = {k:result[k] for k in ('estimate','ci95')}
                if metric == 'related_normalized_change':
                    summary[metric].update(comparisons_used=len(values),comparisons_total=sum(len(ids) for _,ids in strata))
            summaries[source][family] = summary
    return dict(definition='TV(Q(A,B), Q_A(A) Q_B(B)) among complete valid pairs; differences are matched related-minus-unrelated blocks.',
        bootstrap=dict(resamples=repeats,seed=0,unit='Matched whole blocks within family',family_weight='equal'),
        normalized_probability_change=dict(definition='Mapped-repetition increase divided by available upward probability, within comparison before averaging; exclude denominators at most 1e-12.'),
        blocks=dict(blocks),summaries=summaries)
