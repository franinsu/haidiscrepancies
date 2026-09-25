"""Six pooled entropy correlations quoted in the supplement."""
from itertools import combinations
import numpy as np
from scipy.stats import pearsonr
from .entropy import SOURCES, PUZZLE_TYPES


def compute_entropy_correlations(entropy_stats):
    ids = entropy_stats['puzzle_order']
    types = entropy_stats['puzzle_types']
    if len(ids)!=100 or len(set(ids))!=100:
        raise ValueError('Expected 100 distinct main puzzles')
    if any(sum(types[p]==family for p in ids)!=20 for family,_ in PUZZLE_TYPES):
        raise ValueError('Expected 20 main puzzles per family')
    order = [p for family,_ in PUZZLE_TYPES for p in ids if types[p]==family]
    results = {}
    for left,right in combinations(SOURCES,2):
        x,y = [np.asarray([entropy_stats['values_by_puzzle'][source][p] for p in order]) for source in (left,right)]
        if (not np.isfinite(x).all() or not np.isfinite(y).all()
                or np.ptp(x) == 0 or np.ptp(y) == 0):
            raise ValueError('Undefined pooled entropy correlation')
        results[left+'|'+right] = {'sources':[left,right],'pooled':{'estimate':float(pearsonr(x,y).statistic),'n_puzzles':100}}
    return {'metric':'Pearson correlation across all 100 paired puzzle entropies','results':results,
            'primary_condition':{'effort':'low','prompt':'plain'},
            'uniform_excluded':'Constant normalized entropy gives undefined correlation'}
